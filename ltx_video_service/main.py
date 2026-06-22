"""
main.py — FastAPI 应用入口、生命周期管理与路由挂载

设计要点：
1. @asynccontextmanager lifespan：替代已废弃的 on_event("startup/shutdown")，
   是 FastAPI 0.93+ 推荐的生命周期管理方式，代码更清晰，异常传播更可靠。

2. 启动阶段预热双模型：同时加载文生视频（LTXPipeline）和图生视频
   （LTXImageToVideoPipeline），确保第一个请求无需等待冷启动。

3. 关闭阶段清理显存：优雅退出时显式调用 torch.cuda.empty_cache()，
   防止容器/进程重启场景下显存被上一个进程残留占用。

4. API 版本前缀（/api/v1）：商用服务的标准实践，便于后续无缝升级 API 版本。

5. 全局异常处理器：捕获所有未被路由层处理的异常，统一返回 JSON 格式错误，
   防止 Python traceback 泄露给客户端。
"""

import asyncio
import torch
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from api.endpoints import router
from api.async_endpoints import async_router
from core.config import get_settings
from core.logger import setup_logging
from services.engine import load_model, unload_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 应用生命周期管理器。
    yield 之前：服务启动阶段（初始化日志、加载双模型）
    yield 之后：服务关闭阶段（释放 GPU 显存）
    """
    settings = get_settings()

    # ── 启动阶段 ──────────────────────────────────────────────────
    setup_logging(log_dir="logs", log_level="INFO")
    logger.info("=" * 60)
    logger.info("Wan2.2 + LTX-Video 分镜脚本微服务启动中...")
    logger.info("监听地址: {}:{}", settings.APP_HOST, settings.APP_PORT)
    logger.info("模型来源: {}", settings.model_source)
    logger.info("=" * 60)

    try:
        # 加载文生视频 + 图生视频双 Pipeline（阻塞等待，确保就绪后再接受请求）
        await load_model()
        logger.info("✅ 服务就绪，开始接受请求")
    except Exception as e:
        logger.opt(exception=True).critical("模型加载失败，服务无法启动: {}", e)
        raise

    try:
        yield  # ← 服务运行中，处理所有 HTTP 请求
    except asyncio.CancelledError:
        logger.info("Lifespan 被取消 (CancelledError)")
    finally:
        # ── 关闭阶段 ──────────────────────────────────────────────────
        logger.info("收到关闭信号，开始优雅退出...")
        await unload_model()
        torch.cuda.empty_cache()
        logger.info("✅ 服务已安全关闭")


# ── FastAPI 应用实例 ──────────────────────────────────────────────────
app = FastAPI(
    title="Wan2.2 + LTX-Video 分镜脚本视频生成微服务",
    description=(
        "基于 Wan-AI Wan2.2 和 Lightricks LTX-Video 模型的商用级私有化视频生成 API。\n\n"
        "### 功能\n"
        "- **POST /api/v1/generate/storyboard** — 提交分镜脚本（JSON），"
        "返回包含每个分镜 MP4 的 ZIP 压缩包\n"
        "- **POST /api/v1/generate** — 单分镜快捷接口（向后兼容）\n"
        "- **GET  /api/v1/health** — 服务健康检查\n\n"
        "### 模型架构（fast=false 正式出片）\n"
        "- **无参考图** → Wan2.2-T2V-A14B（文生视频，14B 参数，顶级质量）\n"
        "- **有参考图** → Wan2.2-TI2V-5B（文本+图像生视频，5B 参数，轻量高效）\n"
        "  商品实拍图经背景移除+色彩校正+锐化处理，商品形态不变形\n\n"
        "### 模型架构（fast=true 快速预览）\n"
        "- LTX-Video（~8s/clip，用于构图/动作草稿确认）\n\n"
        "### ZIP 内文件命名\n"
        "`shot_001.mp4`, `shot_002.mp4` … 与请求中分镜顺序严格对应"
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── 挂载路由（带版本前缀）────────────────────────────────────────────
app.include_router(router,       prefix="/api/v1", tags=["视频生成（同步）"])
app.include_router(async_router, prefix="/api/v1", tags=["视频生成（异步任务队列）"])


# ── 全局异常兜底处理器 ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    捕获所有未被路由层 try-except 处理的异常。
    商用服务不能把 Python traceback 暴露给客户端，
    统一返回通用错误消息，详细堆栈只写入服务器日志。
    """
    logger.opt(exception=True).error(
        "未捕获异常 | {} {} | {}",
        request.method, request.url.path, exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请联系管理员"},
    )


# ── 开发模式直接运行入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,   # 生产环境务必关闭（会导致模型重复加载）
        workers=1,      # 单 worker：GPU 推理不适合多进程 fork
        log_config=None,  # 禁用 uvicorn 默认日志，交由 loguru 统一管理
    )
