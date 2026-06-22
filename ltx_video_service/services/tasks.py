"""
services/tasks.py — Celery 异步任务队列

为什么需要异步任务队列？
  视频生成耗时 ~60s（Wan 2.1），HTTP 请求无法等待这么长时间：
  - 客户端超时（nginx/CDN 默认 60s）
  - 无法给用户展示进度
  - GPU 排队无法公平调度

架构：
  POST /generate/async   → 提交任务 → 立即返回 task_id
  GET  /tasks/{id}       → 查询状态（pending/processing/done/failed）
  GET  /tasks/{id}/download → 下载结果 MP4 / ZIP

Celery 配置：
  broker:  Redis（任务队列）
  backend: Redis（任务状态/结果存储）
  queue:   gpu_queue（单队列串行，防止 GPU OOM）
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from celery import Celery
from celery.result import AsyncResult
from loguru import logger

from core.config import get_settings

# ── Celery 应用实例 ────────────────────────────────────────────────────────

def _make_celery() -> Celery:
    settings = get_settings()
    redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")

    app = Celery(
        "ltx_video_tasks",
        broker=redis_url,
        backend=redis_url,
    )
    app.conf.update(
        # 任务序列化
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        # 时区
        timezone="Asia/Shanghai",
        enable_utc=True,
        # 结果保留 24 小时（足够客户端下载）
        result_expires=86400,
        # GPU 队列：单 worker 单并发，防止 OOM
        task_routes={
            "services.tasks.generate_shot_task":     {"queue": "gpu_queue"},
            "services.tasks.generate_storyboard_task": {"queue": "gpu_queue"},
        },
        # worker 每次只取 1 个任务（prefetch=1），GPU 任务不预取
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        # 任务超时：单分镜 15min，分镜脚本 2h
        task_soft_time_limit=900,
        task_time_limit=920,
    )
    return app


celery_app = _make_celery()


# ── Worker 懒加载模型 ──────────────────────────────────────────────────────
# 不在 worker_process_init 中加载模型！
#
# 原因：worker_process_init 信号在 billiard 等待 UP 握手消息之前触发，
# 而 Wan 2.2 A14B 加载需要 ~3min，billiard 默认只等待 ~2s 就 SIGKILL 新进程，
# 导致 Celery 陷入"启动 → 超时 → SIGKILL → 再次启动"的无限死循环。
#
# 解决方案：worker 进程先正常发送 UP 消息（不阻塞），第一个任务到来时才加载模型。
# _worker_models_loaded 标志保证每个 worker 进程只加载一次。

_worker_models_loaded: bool = False


def _ensure_models_loaded() -> None:
    """懒加载：第一次被任务调用时加载模型，之后直接返回。"""
    global _worker_models_loaded
    if _worker_models_loaded:
        return
    from services.engine import load_model
    logger.info("Celery worker 首次推理：开始加载模型（懒加载）...")
    try:
        asyncio.run(load_model())
        _worker_models_loaded = True
        logger.info("Celery worker 模型加载完成")
    except Exception as e:
        logger.opt(exception=True).error("Celery worker 模型加载失败: {}", e)
        raise  # 加载失败时让任务失败，而非继续用空模型


# ── 任务状态常量 ───────────────────────────────────────────────────────────

class TaskStatus:
    PENDING    = "pending"      # 等待 worker 领取
    PROCESSING = "processing"   # GPU 推理中
    DONE       = "done"         # 完成，可下载
    FAILED     = "failed"       # 失败


# ── Celery 任务定义 ────────────────────────────────────────────────────────

@celery_app.task(
    name="services.tasks.generate_shot_task",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def generate_shot_task(self, shot_spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    单分镜异步推理任务。

    在独立 worker 进程中运行，完全隔离于 FastAPI 主进程。
    使用 asyncio.run() 执行异步的 engine.generate_shot()。

    Args:
        shot_spec_dict: ShotSpec 的字典序列化（JSON 兼容）

    Returns:
        {"status": "done", "output_path": str, "frames": int}
    """
    import asyncio
    from services.engine import ShotSpec, generate_shot

    # 懒加载模型（第一次调用时加载，之后直接返回）
    _ensure_models_loaded()

    # 更新任务状态为 processing
    self.update_state(state="PROGRESS", meta={"status": TaskStatus.PROCESSING, "progress": 0})
    logger.info("任务 {} 开始推理 | shot={}", self.request.id, shot_spec_dict.get("prompt", "")[:60])

    try:
        spec = ShotSpec(**shot_spec_dict)
        output_path: Path = asyncio.run(generate_shot(spec))

        result = {
            "status": TaskStatus.DONE,
            "output_path": str(output_path),
            "task_id": self.request.id,
        }
        logger.info("任务 {} 完成 | 输出: {}", self.request.id, output_path)
        return result

    except Exception as exc:
        logger.opt(exception=True).error("任务 {} 失败: {}", self.request.id, exc)
        # 自动重试（最多 2 次）
        raise self.retry(exc=exc)


@celery_app.task(
    name="services.tasks.generate_storyboard_task",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
    # 分镜脚本任务最多包含多个分镜，每镜最长 15min，整体给 2h 软限制 / 2h10min 硬限制
    soft_time_limit=7200,   # 2h  — 超时后抛 SoftTimeLimitExceeded，可做清理
    time_limit=7800,        # 2h10min — 硬限制，兜底防止 worker 僵死
)
def generate_storyboard_task(self, shot_spec_dicts: list) -> Dict[str, Any]:
    """
    分镜脚本异步推理任务（批量）。

    逐镜推理，每镜完成后更新进度。

    Args:
        shot_spec_dicts: List[ShotSpec 字典]

    Returns:
        {"status": "done", "output_paths": [str, ...], "total": int}
    """
    import asyncio
    from services.engine import ShotSpec, generate_shot

    # 懒加载模型（第一次调用时加载，之后直接返回）
    _ensure_models_loaded()

    total = len(shot_spec_dicts)
    self.update_state(
        state="PROGRESS",
        meta={"status": TaskStatus.PROCESSING, "progress": 0, "total": total, "done": 0},
    )
    logger.info("分镜脚本任务 {} 开始，共 {} 个分镜", self.request.id, total)

    output_paths = []
    try:
        for idx, spec_dict in enumerate(shot_spec_dicts, start=1):
            spec = ShotSpec(**spec_dict)
            path = asyncio.run(generate_shot(spec))
            output_paths.append(str(path))

            # 更新进度
            progress = round(idx / total * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "status": TaskStatus.PROCESSING,
                    "progress": progress,
                    "total": total,
                    "done": idx,
                },
            )
            logger.info("分镜任务 {} 进度: {}/{}", self.request.id, idx, total)

        result = {
            "status": TaskStatus.DONE,
            "output_paths": output_paths,
            "total": total,
            "task_id": self.request.id,
        }
        logger.info("分镜脚本任务 {} 完成，共 {} 个视频", self.request.id, total)
        return result

    except Exception as exc:
        logger.opt(exception=True).error("分镜脚本任务 {} 失败: {}", self.request.id, exc)
        raise self.retry(exc=exc)


# ── 任务状态查询 ───────────────────────────────────────────────────────────

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    查询任务状态，返回标准化状态字典。

    Returns:
        {
            "task_id": str,
            "status": "pending" | "processing" | "done" | "failed",
            "progress": int (0-100),        # 仅 processing 时有意义
            "total": int,                   # 分镜脚本任务的总分镜数
            "done": int,                    # 已完成分镜数
            "output_path": str | None,      # 单分镜完成后可用
            "output_paths": list | None,    # 分镜脚本完成后可用
            "error": str | None,
        }
    """
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"task_id": task_id, "status": TaskStatus.PENDING, "progress": 0}

    if result.state == "PROGRESS":
        meta = result.info or {}
        return {
            "task_id":  task_id,
            "status":   TaskStatus.PROCESSING,
            "progress": meta.get("progress", 0),
            "total":    meta.get("total"),
            "done":     meta.get("done"),
        }

    if result.state == "SUCCESS":
        info = result.result or {}
        return {
            "task_id":      task_id,
            "status":       TaskStatus.DONE,
            "progress":     100,
            "output_path":  info.get("output_path"),
            "output_paths": info.get("output_paths"),
            "total":        info.get("total"),
        }

    if result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status":  TaskStatus.FAILED,
            "error":   str(result.info),
        }

    # RETRY / REVOKED 等
    return {"task_id": task_id, "status": result.state.lower()}
