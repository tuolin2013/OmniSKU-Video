"""
api/async_endpoints.py — 异步任务 API 端点（P1-1）

接口流程：
  1. POST /api/v1/generate/async          → 提交单分镜任务，立即返回 task_id
  2. POST /api/v1/generate/storyboard/async → 提交分镜脚本任务，立即返回 task_id
  3. GET  /api/v1/tasks/{task_id}         → 查询任务状态 + 进度
  4. GET  /api/v1/tasks/{task_id}/download → 下载结果（MP4 或 ZIP）

客户端轮询建议：
  - 每 3s 轮询一次 /tasks/{id}
  - status=done 时调用 /tasks/{id}/download 下载
  - status=failed 时读取 error 字段展示给用户
"""

import io
import zipfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from api.endpoints import ShotRequest, StoryboardRequest
from services.tasks import TaskStatus, celery_app, generate_shot_task, generate_storyboard_task, get_task_status

async_router = APIRouter()


# ── 响应体 ────────────────────────────────────────────────────────────────

class TaskSubmitResponse(BaseModel):
    """任务提交成功响应"""
    task_id: str
    status: str = TaskStatus.PENDING
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态查询响应"""
    task_id: str
    status: str
    progress: int = 0
    total: Optional[int] = None
    done: Optional[int] = None
    output_path: Optional[str] = None
    output_paths: Optional[List[str]] = None
    error: Optional[str] = None


# ── 提交接口 ──────────────────────────────────────────────────────────────

@async_router.post(
    "/generate/async",
    response_model=TaskSubmitResponse,
    summary="异步提交单分镜生成任务",
    description=(
        "立即返回 task_id，不等待 GPU 推理完成。\n"
        "客户端每 3s 轮询 GET /tasks/{task_id} 查询进度，\n"
        "status=done 后调用 GET /tasks/{task_id}/download 下载 MP4。"
    ),
)
async def submit_generate_async(request: ShotRequest) -> TaskSubmitResponse:
    """提交单分镜异步生成任务"""
    spec = request.to_shot_spec()
    # 将 dataclass 序列化为 dict（Celery JSON 序列化要求）
    spec_dict = {
        "prompt":                spec.prompt,
        "negative_prompt":       spec.negative_prompt,
        "num_frames":            spec.num_frames,
        "num_inference_steps":   spec.num_inference_steps,
        "height":                spec.height,
        "width":                 spec.width,
        "fps":                   spec.fps,
        "guidance_scale":        spec.guidance_scale,
        "seed":                  spec.seed,
        "reference_images_b64":  spec.reference_images_b64,
        "fast":                  spec.fast,
        "background_style":      spec.background_style,
    }

    try:
        task = generate_shot_task.apply_async(
            args=[spec_dict],
            queue="gpu_queue",
        )
    except Exception as e:
        logger.opt(exception=True).error("任务提交失败: {}", e)
        raise HTTPException(status_code=503, detail=f"任务队列不可用: {e}")

    has_ref = bool(request.reference_images or request.reference_image)
    logger.info("异步任务已提交 | task_id={} | mode={}", task.id, "i2v" if has_ref else "t2v")
    return TaskSubmitResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"任务已加入 GPU 队列，预计耗时 {'~10s（预览）' if request.fast else '~60s（正式出片）'}",
    )


@async_router.post(
    "/generate/storyboard/async",
    response_model=TaskSubmitResponse,
    summary="异步提交分镜脚本批量生成任务",
    description=(
        "立即返回 task_id，不等待所有分镜完成。\n"
        "客户端每 5s 轮询 GET /tasks/{task_id}，progress 字段显示当前进度（0-100）。\n"
        "status=done 后调用 GET /tasks/{task_id}/download 下载 ZIP。"
    ),
)
async def submit_storyboard_async(request: StoryboardRequest) -> TaskSubmitResponse:
    """提交分镜脚本批量异步生成任务"""
    shot_count = len(request.shots)
    spec_dicts = []
    for shot in request.shots:
        spec = shot.to_shot_spec()
        spec_dicts.append({
            "prompt":                spec.prompt,
            "negative_prompt":       spec.negative_prompt,
            "num_frames":            spec.num_frames,
            "num_inference_steps":   spec.num_inference_steps,
            "height":                spec.height,
            "width":                 spec.width,
            "fps":                   spec.fps,
            "guidance_scale":        spec.guidance_scale,
            "seed":                  spec.seed,
            "reference_images_b64":  spec.reference_images_b64,
            "fast":                  spec.fast,
            "background_style":      spec.background_style,
        })

    try:
        task = generate_storyboard_task.apply_async(
            args=[spec_dicts],
            queue="gpu_queue",
        )
    except Exception as e:
        logger.opt(exception=True).error("分镜脚本任务提交失败: {}", e)
        raise HTTPException(status_code=503, detail=f"任务队列不可用: {e}")

    fast_mode = all(getattr(s, "fast", False) for s in request.shots)
    est = shot_count * (10 if fast_mode else 60)
    logger.info("分镜脚本异步任务已提交 | task_id={} | {} 个分镜", task.id, shot_count)
    return TaskSubmitResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message=f"{shot_count} 个分镜已加入队列，预计总耗时 ~{est}s",
    )


# ── 状态查询 ──────────────────────────────────────────────────────────────

@async_router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="查询任务状态",
    description="轮询此接口获取生成进度。建议间隔：预览模式 2s，正式出片模式 5s。",
)
async def query_task_status(task_id: str) -> TaskStatusResponse:
    """查询单个任务状态"""
    try:
        status_dict = get_task_status(task_id)
    except Exception as e:
        logger.warning("任务状态查询失败 | task_id={} | {}", task_id, e)
        raise HTTPException(status_code=500, detail=f"状态查询失败: {e}")

    return TaskStatusResponse(**status_dict)


# ── 结果下载 ──────────────────────────────────────────────────────────────

@async_router.get(
    "/tasks/{task_id}/download",
    summary="下载任务结果",
    description=(
        "任务 status=done 后可调用此接口下载结果。\n"
        "单分镜任务返回 MP4，分镜脚本任务返回 ZIP。\n"
        "文件下载后 24 小时内可重复下载，超时后自动清理。"
    ),
    responses={
        200: {"content": {"video/mp4": {}, "application/zip": {}}},
        404: {"description": "任务不存在或文件已过期"},
        425: {"description": "任务尚未完成"},
    },
)
async def download_task_result(
    task_id: str,
    background_tasks: BackgroundTasks,
):
    """下载任务生成结果"""
    try:
        status_dict = get_task_status(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"状态查询失败: {e}")

    if status_dict["status"] == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"任务失败: {status_dict.get('error')}")

    if status_dict["status"] != TaskStatus.DONE:
        raise HTTPException(
            status_code=425,   # Too Early
            detail=f"任务尚未完成，当前状态: {status_dict['status']}，进度: {status_dict.get('progress', 0)}%",
        )

    # ── 单分镜：返回 MP4 ────────────────────────────────────────────
    if status_dict.get("output_path"):
        output_path = Path(status_dict["output_path"])
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="视频文件已过期或被清理")

        # 下载完成后后台清理（延迟 60s，给客户端重试机会）
        background_tasks.add_task(_delayed_cleanup, output_path, delay=60)
        logger.info("下载单分镜结果 | task_id={} | file={}", task_id, output_path.name)
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"video_{task_id[:8]}.mp4",
        )

    # ── 分镜脚本：打包 ZIP 返回 ─────────────────────────────────────
    if status_dict.get("output_paths"):
        output_paths = [Path(p) for p in status_dict["output_paths"]]
        missing = [p for p in output_paths if not p.exists()]
        if missing:
            raise HTTPException(status_code=404, detail=f"{len(missing)} 个视频文件已过期")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for idx, path in enumerate(output_paths, start=1):
                zf.write(str(path), arcname=f"shot_{idx:03d}.mp4")
        zip_buffer.seek(0)

        background_tasks.add_task(_delayed_cleanup_list, output_paths, delay=60)
        logger.info("下载分镜脚本结果 | task_id={} | {} 个视频", task_id, len(output_paths))
        return StreamingResponse(
            content=zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=storyboard_{task_id[:8]}.zip"},
        )

    raise HTTPException(status_code=404, detail="任务结果不存在")


# ── 后台清理 ──────────────────────────────────────────────────────────────

import asyncio as _asyncio


async def _delayed_cleanup(path: Path, delay: int = 60) -> None:
    """延迟 delay 秒后删除文件（给客户端重试下载的时间窗口）"""
    await _asyncio.sleep(delay)
    try:
        path.unlink(missing_ok=True)
        logger.debug("延迟清理完成: {}", path)
    except Exception as e:
        logger.warning("延迟清理失败: {} | {}", path, e)


async def _delayed_cleanup_list(paths: List[Path], delay: int = 60) -> None:
    """延迟批量清理"""
    await _asyncio.sleep(delay)
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    logger.debug("延迟批量清理完成，共 {} 个文件", len(paths))
