"""
core/logger.py — 基于 Loguru 的结构化日志初始化

商用级设计要点：
- 同时输出到控制台（彩色）和滚动文件（按天切割、自动压缩、保留 30 天）
- 通过 enqueue=True 使文件写入在独立线程中异步完成，不阻塞 FastAPI 事件循环
- 统一日志格式，包含时间戳、级别、模块名、行号，方便 ELK/Loki 等日志系统采集
- 拦截标准库 logging（uvicorn/asyncio 等使用 stdlib logging），统一由 loguru 处理
"""

import logging
import sys
from pathlib import Path

from loguru import logger


class _InterceptHandler(logging.Handler):
    """
    将 Python 标准库 logging 的所有记录转发给 Loguru。
    这样 uvicorn、sqlalchemy 等第三方库的日志也会走统一格式。
    """

    def emit(self, record: logging.LogRecord) -> None:
        # 将 stdlib 日志级别映射到 loguru 级别名称
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到真正的调用帧，确保日志显示正确的源文件/行号
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    初始化日志系统，在 FastAPI 生命周期启动阶段调用一次。

    Args:
        log_dir:   日志文件存放目录
        log_level: 最低记录级别（DEBUG/INFO/WARNING/ERROR）
    """
    # 先移除 loguru 默认的 stderr sink，避免重复输出
    logger.remove()

    # ── Sink 1: 控制台（带颜色，适合开发调试）────────────────────
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )

    # ── Sink 2: 滚动文件（适合生产归档）─────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{log_dir}/ltx_service_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        rotation="00:00",       # 每天零点切割
        retention="30 days",    # 保留最近 30 天
        compression="gz",       # 旧文件 gzip 压缩，节省磁盘
        enqueue=True,           # 异步写入，不阻塞主线程
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{line} — {message}"
        ),
    )

    # ── 拦截标准库 logging ────────────────────────────────────────
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # 确保 uvicorn 的子 logger 也被拦截
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

    logger.info("日志系统初始化完成，级别={}, 文件目录={}", log_level, log_dir)
