"""
core/config.py — 基于 Pydantic-Settings 的集中式配置管理

商用级设计要点：
- 所有配置项均从 .env 文件或系统环境变量读取，避免硬编码
- 使用 Pydantic V2 的 model_validator 做启动时自检，提前暴露配置错误
- 单例模式：全局只实例化一次 Settings，通过 get_settings() 注入依赖
"""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，字段顺序即优先级：环境变量 > .env 文件 > 默认值"""

    model_config = SettingsConfigDict(
        # 指定 .env 文件位置（相对于项目根目录启动）
        env_file=".env",
        env_file_encoding="utf-8",
        # 允许额外字段，防止 .env 中有注释行导致解析报错
        extra="ignore",
    )

    # ── 服务网络配置 ──────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # ── LTX-Video 预览模型配置 ────────────────────────────────────
    MODEL_ID: str = "Lightricks/LTX-Video"
    MODEL_LOCAL_PATH: str = ""          # 本地路径优先级高于 MODEL_ID

    # ── Wan 2.2 主力出片模型配置（-Diffusers 格式，原生 diffusers 支持）───
    # T2V（文生视频）：Wan2.2-T2V-A14B-Diffusers（14B 参数）
    WAN_T2V_MODEL: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    # I2V（图生视频）：Wan2.2-I2V-A14B-Diffusers（14B 参数）
    WAN_I2V_MODEL: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    # TI2V（文本+图像生视频）：Wan2.2-TI2V-5B-Diffusers（5B 参数，轻量高效）
    WAN_TI2V_MODEL: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # ── 模型加载开关（内存优化）────────────────────────────────────
    # I2V A14B 约占 28GB RAM，与 T2V A14B 同时加载容易 OOM
    # TI2V 5B 已可处理图生视频场景，默认关闭 I2V A14B
    LOAD_WAN_I2V: bool = False
    # LTX-Video 预览模型额外占用约 4GB RAM，默认关闭，按需开启
    LOAD_LTX_PREVIEW: bool = False

    # ── LLM / Agent API（提示词增强 Skill）──────────────────────────
    # 调用 LLM 将简短分镜描述增强为结构化 Wan 专业提示词
    RIGHT_CODE_API_KEY: str = ""
    TEXT_CHAT_URL: str = "https://right.codes/codex/v1/chat/completions"
    CODEX_BASE_URL: str = "https://right.codes/codex/v1"
    GEMINI_BASE_URL: str = "https://right.codes/gemini/v1"
    # 是否在提交分镜脚本时自动调用 LLM 增强提示词（可通过 .env 关闭）
    ENABLE_PROMPT_ENHANCEMENT: bool = True
    # 提示词增强最大并发数（防止 LLM API 限流）
    PROMPT_ENHANCE_CONCURRENCY: int = 5

    # ── Redis / Celery 任务队列 ────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── 文件系统配置 ──────────────────────────────────────────────
    OUTPUT_DIR: str = "outputs"

    # ── 推理设备 ──────────────────────────────────────────────────
    DEVICE: str = "cuda"

    @model_validator(mode="after")
    def validate_output_dir(self) -> "Settings":
        """启动时自动创建输出目录，确保写入权限正常"""
        output_path = Path(self.OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def model_source(self) -> str:
        """返回实际加载模型的路径或 Hub ID（优先本地路径）"""
        if self.MODEL_LOCAL_PATH and Path(self.MODEL_LOCAL_PATH).exists():
            return self.MODEL_LOCAL_PATH
        return self.MODEL_ID


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    单例工厂函数（lru_cache 保证全局只解析一次 .env）
    在 FastAPI 中通过 Depends(get_settings) 注入，便于单元测试时替换
    """
    return Settings()
