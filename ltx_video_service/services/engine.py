"""
services/engine.py — 核心 AI 推理引擎（生产级，双模型 + 视频超分）

架构设计：
1. 双模型策略：
   - 预览模式（fast=True）：LTX-Video（速度快，~8s/clip）用于草稿确认
   - 正式出片（fast=False，默认）：Wan 2.2（顶级质量）用于最终交付

2. Wan 2.2 Pipeline 架构（diffusers 0.38+，标准稳定 Pipeline）：
   - WanPipeline                → 文生视频（T2V，加载 A14B 权重）
   - WanImageToVideoPipeline    → 图生视频（I2V，加载 A14B 权重）
   - WanImageToVideoPipeline    → 文本+图像生视频（TI2V，加载 5B 权重，轻量高效）
   - bfloat16 + CPU offload：RTX 5090 32GB 轻松运行
   注：Wan22ModularPipeline 是实验性 API，在 accelerate 版本不匹配时崩溃，
       已统一改用 WanPipeline / WanImageToVideoPipeline 标准 Pipeline。

3. 商品图增强流水线（services/image_enhance.py）：
   - 背景移除（rembg u2net）
   - 色彩校正（白平衡 + 亮度标准化 + 对比度/饱和度微增）
   - 锐化（Unsharp Mask）
   - 投影合成（底部柔和阴影）
   - 专业背景合成（支持 white/gradient/warm/dark）

4. 视频后处理（Real-ESRGAN 超分）：
   - 生成帧 2x 超分（如 576→1152，720→1440），再 resize 到目标分辨率
   - 仅在正式出片模式（fast=False）启用，预览模式跳过以节省时间

5. asyncio.Lock 全局推理锁：防止多请求 GPU OOM

6. asyncio.to_thread：所有阻塞操作在线程池执行，不阻塞事件循环
"""

import asyncio
import base64
import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
from loguru import logger

# ── cuSolver 兼容性修复 ───────────────────────────────────────────────────────
# 多进程共用 GPU（Celery worker + FastAPI）时 cusolverDnCreate 可能失败。
# 强制 PyTorch 使用 MAGMA 作为 linalg 后端，绕过 cuSolver 初始化问题。
try:
    # "magma" bypasses cuSolver entirely, avoiding cusolverDnCreate failures
    # that can occur when Celery worker and FastAPI share a CUDA context.
    torch.backends.cuda.preferred_linalg_library("magma")
except Exception:
    pass
from PIL import Image

from core.config import get_settings
from services.image_enhance import enhance_product_image
from services.image_selector import select_best_image
from services.quality_check import QualityReport, check_video_quality, should_retry

# ── 全局状态 ────────────────────────────────────────────────────────────────
_state: dict = {
    # 正式出片：Wan 2.2（标准稳定 Pipeline）
    "wan_t2v_pipe": None,   # WanPipeline             — 文生视频（T2V-A14B 权重）
    "wan_i2v_pipe": None,   # WanImageToVideoPipeline — 图生视频（I2V-A14B 权重）
    "wan_ti2v_pipe": None,  # WanImageToVideoPipeline — 文本+图像生视频（TI2V-5B 权重）
    # 预览/草稿：LTX-Video
    "ltx_t2v_pipe": None,
    "ltx_i2v_pipe": None,
    # 视频超分
    "upsampler": None,
    # 推理锁
    "lock": None,
}


# ── 分镜数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class ShotSpec:
    """
    单个分镜的完整参数。

    Attributes:
        prompt:               正向提示词（已含写实前缀 + 构图语）
        negative_prompt:      负向提示词（已含系统默认词）
        num_frames:           帧数（Wan 要求满足 4N+1，LTX 要求 8N+1）
        num_inference_steps:  去噪步数（正式出片默认 50，预览 20）
        height:               视频高度（须为 32 的倍数）
        width:                视频宽度（须为 32 的倍数）
        fps:                  输出帧率
        guidance_scale:       CFG 引导强度
        seed:                 随机种子（None=随机）
        reference_image_b64:  base64 编码商品参考图（None → 文生视频）
        fast:                 True=预览模式(LTX)，False=正式出片(Wan 2.2)
        background_style:     商品背景样式（white/gradient/warm/dark）
    """
    prompt: str
    negative_prompt: str
    num_frames: int
    num_inference_steps: int
    height: int
    width: int
    fps: int
    guidance_scale: float = field(default=7.5)
    seed: Optional[int] = field(default=None)
    reference_images_b64: Optional[List[str]] = field(default=None)  # 多图列表
    fast: bool = field(default=False)
    background_style: str = field(default="gradient")


# ── Pipeline 访问（健康检查用）────────────────────────────────────────────────

def get_pipeline():
    """返回主推理 Pipeline（非 None 即服务就绪）"""
    return _state["wan_t2v_pipe"] or _state["ltx_t2v_pipe"]


# ── 超分模型加载 ──────────────────────────────────────────────────────────────

def _load_realesrgan():
    """
    加载 Real-ESRGAN x2plus 超分模型。
    x2plus：针对真实世界图像优化，2x 放大，比 x4 快且足够电商使用。
    失败时返回 None（超分降级跳过，不影响主流程）。
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=model,
            tile=512,          # 分块推理，防止显存溢出
            tile_pad=10,
            pre_pad=0,
            half=True,         # FP16 加速
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        logger.info("✅ Real-ESRGAN x2plus 超分模型加载完成")
        return upsampler
    except Exception as e:
        logger.warning("Real-ESRGAN 加载失败，将跳过视频超分: {}", e)
        return None


# ── 模型生命周期 ──────────────────────────────────────────────────────────────

async def load_model() -> None:
    """
    加载所有模型（启动阶段调用）：
    1. Wan 2.2（主力出片，T2V A14B + I2V A14B + TI2V 5B）
    2. LTX-Video（快速预览）
    3. Real-ESRGAN（视频超分，失败不阻断启动）
    """
    settings = get_settings()
    _state["lock"] = asyncio.Lock()

    logger.info("=" * 60)
    logger.info("开始加载生产级双模型...")
    logger.info("主力模型: Wan 2.2（T2V A14B + I2V A14B + TI2V 5B）  |  预览模型: LTX-Video")
    logger.info("=" * 60)

    def _load_wan():
        """
        加载 Wan 2.2 T2V + I2V + TI2V 三条 Pipeline。

        Pipeline 选型（使用 diffusers 标准稳定 Pipeline，兼容 diffusers 0.38+）：
          - T2V：WanPipeline（文生视频，加载 T2V-A14B 权重）
          - I2V：WanImageToVideoPipeline（图生视频，加载 I2V-A14B 权重）
          - TI2V：WanImageToVideoPipeline（文本+图像生视频，加载 TI2V-5B 权重）

        注意：Wan22ModularPipeline / Wan22Image2VideoModularPipeline 是
              实验性 Modular Pipeline API，在 accelerate 版本不匹配时
              会抛出 UnboundLocalError: blocks_class，改用标准稳定 Pipeline。
        """
        from diffusers import WanPipeline, WanImageToVideoPipeline

        wan_t2v_model  = getattr(settings, "WAN_T2V_MODEL",  "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        wan_i2v_model  = getattr(settings, "WAN_I2V_MODEL",  "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        wan_ti2v_model = getattr(settings, "WAN_TI2V_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

        # ── T2V（文生视频，A14B）──────────────────────────────────────────
        logger.info("加载 Wan 2.2 T2V (A14B): {}", wan_t2v_model)
        t2v = WanPipeline.from_pretrained(
            wan_t2v_model,
            torch_dtype=torch.bfloat16,
        )
        t2v.enable_model_cpu_offload()
        if hasattr(t2v, "vae") and hasattr(t2v.vae, "enable_slicing"):
            t2v.vae.enable_slicing()
        logger.info("✅ Wan 2.2 T2V 加载完成")

        # ── I2V（图生视频，A14B）— 默认关闭，LOAD_WAN_I2V=true 才加载 ──────
        i2v = None
        if getattr(settings, "LOAD_WAN_I2V", False):
            logger.info("加载 Wan 2.2 I2V (A14B): {}", wan_i2v_model)
            i2v = WanImageToVideoPipeline.from_pretrained(
                wan_i2v_model,
                torch_dtype=torch.bfloat16,
            )
            i2v.enable_model_cpu_offload()
            if hasattr(i2v, "vae") and hasattr(i2v.vae, "enable_slicing"):
                i2v.vae.enable_slicing()
            logger.info("✅ Wan 2.2 I2V 加载完成")
        else:
            logger.info("⏭ Wan 2.2 I2V 已跳过（LOAD_WAN_I2V=false，TI2V 5B 覆盖图生视频需求）")

        # ── TI2V（文本+图像生视频，5B）──────────────────────────────────
        logger.info("加载 Wan 2.2 TI2V (5B): {}", wan_ti2v_model)
        ti2v = WanImageToVideoPipeline.from_pretrained(
            wan_ti2v_model,
            torch_dtype=torch.bfloat16,
        )
        ti2v.enable_model_cpu_offload()
        if hasattr(ti2v, "vae") and hasattr(ti2v.vae, "enable_slicing"):
            ti2v.vae.enable_slicing()
        logger.info("✅ Wan 2.2 TI2V 加载完成")

        return t2v, i2v, ti2v

    def _load_ltx():
        """加载 LTX-Video 预览模型"""
        from diffusers import LTXImageToVideoPipeline, LTXPipeline

        ltx_model = getattr(settings, "MODEL_LOCAL_PATH", "") or settings.MODEL_ID
        logger.info("加载 LTX-Video 预览模型: {}", ltx_model)

        common = dict(
            pretrained_model_name_or_path=ltx_model,
            torch_dtype=torch.bfloat16,
        )
        t2v = LTXPipeline.from_pretrained(**common)
        t2v.enable_model_cpu_offload()
        if hasattr(t2v, "vae") and hasattr(t2v.vae, "enable_slicing"):
            t2v.vae.enable_slicing()

        i2v = LTXImageToVideoPipeline.from_pretrained(**common)
        i2v.enable_model_cpu_offload()
        if hasattr(i2v, "vae") and hasattr(i2v.vae, "enable_slicing"):
            i2v.vae.enable_slicing()

        logger.info("✅ LTX-Video 预览模型加载完成")
        return t2v, i2v

    # 串行加载：先 Wan 2.2（主力），再 LTX（预览）
    # CPU offload 机制保护，不会显存溢出
    try:
        wan_t2v, wan_i2v, wan_ti2v = await asyncio.to_thread(_load_wan)
        _state["wan_t2v_pipe"]  = wan_t2v
        _state["wan_i2v_pipe"]  = wan_i2v
        _state["wan_ti2v_pipe"] = wan_ti2v
    except Exception as e:
        logger.opt(exception=True).error("Wan 2.2 加载失败: {}", e)
        logger.warning("⚠ Wan 2.2 不可用，将仅使用 LTX-Video")

    if getattr(settings, "LOAD_LTX_PREVIEW", False):
        try:
            ltx_t2v, ltx_i2v = await asyncio.to_thread(_load_ltx)
            _state["ltx_t2v_pipe"] = ltx_t2v
            _state["ltx_i2v_pipe"] = ltx_i2v
        except Exception as e:
            logger.opt(exception=True).error("LTX-Video 加载失败: {}", e)
            if _state["wan_t2v_pipe"] is None:
                raise RuntimeError("所有模型均加载失败，服务无法启动") from e
    else:
        logger.info("⏭ LTX-Video 预览模型已跳过（LOAD_LTX_PREVIEW=false）")
        if _state["wan_t2v_pipe"] is None:
            raise RuntimeError("Wan 2.2 加载失败且 LTX_PREVIEW 未启用，服务无法启动")

    # 超分模型（失败不阻断启动）
    try:
        _state["upsampler"] = await asyncio.to_thread(_load_realesrgan)
    except Exception as e:
        logger.warning("Real-ESRGAN 加载异常: {}", e)

    logger.info("✅ 所有模型加载完成，服务就绪")


async def unload_model() -> None:
    """释放所有模型显存"""
    for key in ("wan_t2v_pipe", "wan_i2v_pipe", "wan_ti2v_pipe", "ltx_t2v_pipe", "ltx_i2v_pipe", "upsampler"):
        if _state.get(key) is not None:
            del _state[key]
            _state[key] = None
    torch.cuda.empty_cache()
    logger.info("所有模型已卸载，GPU 显存已释放")


# ── 商品参考图预处理 ──────────────────────────────────────────────────────────

def prepare_reference_image(
    image_b64: str,
    width: int,
    height: int,
    background_style: str = "gradient",
) -> Image.Image:
    """
    商品参考图完整增强流水线：
      base64解码 → 背景移除 → 色彩校正 → 锐化 → 等比缩放 → 投影合成 → 背景合成

    Args:
        image_b64:        base64 字符串（支持 data URI）
        width, height:    目标帧尺寸（32 的倍数）
        background_style: 背景样式（white/gradient/warm/dark）

    Returns:
        RGB 模式 PIL Image，尺寸 = width × height
    """
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    raw_bytes = base64.b64decode(image_b64)
    img_rgb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    return enhance_product_image(
        img_rgb=img_rgb,
        target_width=width,
        target_height=height,
        padding_ratio=0.10,
        background_style=background_style,
        enable_color_correct=True,
        enable_sharpen=True,
        enable_shadow=True,
    )


# ── 视频超分 ──────────────────────────────────────────────────────────────────

def _upscale_frames(frames: List[Image.Image]) -> List[Image.Image]:
    """
    使用 Real-ESRGAN x2plus 对每一帧做 2x 超分，然后 resize 回原始分辨率。

    为什么先放大再缩小？
    Real-ESRGAN 会在放大过程中修复压缩伪影、噪点、模糊——
    这些修复效果在 resize 回原始分辨率后仍然保留，
    最终得到比直接生成更清晰、细节更丰富的帧。

    失败时原样返回（降级策略）。
    """
    upsampler = _state.get("upsampler")
    if upsampler is None:
        return frames

    try:
        import cv2
        target_w, target_h = frames[0].size
        upscaled = []
        for i, frame in enumerate(frames):
            arr = np.array(frame)
            # Real-ESRGAN 使用 BGR 格式
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            output_bgr, _ = upsampler.enhance(arr_bgr, outscale=2)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            # resize 回原始分辨率
            pil = Image.fromarray(output_rgb).resize(
                (target_w, target_h), Image.LANCZOS
            )
            upscaled.append(pil)
            if (i + 1) % 10 == 0:
                logger.debug("超分进度: {}/{}", i + 1, len(frames))
        logger.info("✅ 视频超分完成，共处理 {} 帧", len(upscaled))
        return upscaled
    except Exception as e:
        logger.warning("视频超分失败，使用原始帧: {}", e)
        return frames


# ── 帧数对齐工具 ──────────────────────────────────────────────────────────────

def _align_frames_wan(n: int) -> int:
    """Wan 2.2 要求帧数满足 4N+1（如 17, 21, 25...97...）"""
    if (n - 1) % 4 == 0:
        return n
    # 向上对齐到最近的 4N+1
    adjusted = ((n - 1 + 3) // 4) * 4 + 1
    logger.debug("Wan 帧数对齐: {} → {}", n, adjusted)
    return adjusted


def _align_frames_ltx(n: int) -> int:
    """LTX 要求帧数满足 8N+1（如 9, 17, 25...65...）"""
    if (n - 1) % 8 == 0:
        return n
    adjusted = ((n - 1 + 7) // 8) * 8 + 1
    logger.debug("LTX 帧数对齐: {} → {}", n, adjusted)
    return adjusted


# ── 单分镜推理 ────────────────────────────────────────────────────────────────

async def generate_shot(shot: ShotSpec) -> Path:
    """
    单分镜视频生成。

    路由逻辑：
      shot.fast=True  → LTX-Video（预览，~8s）
      shot.fast=False → Wan 2.2（正式出片，~60s）
        - 无参考图          → T2V A14B
        - 有参考图 + prompt → TI2V 5B（文本+图像生视频）
        - TI2V 不可用       → I2V A14B 回退

    回退逻辑：
      Wan 2.2 不可用 → 自动回退到 LTX-Video

    Returns:
        生成视频的 Path（MP4）
    """
    lock = _state.get("lock")
    if lock is None:
        raise RuntimeError("Pipeline 未初始化，服务尚未就绪")

    settings = get_settings()
    output_path = Path(settings.OUTPUT_DIR) / f"{uuid.uuid4().hex}.mp4"

    generator = (
        torch.Generator(device="cpu").manual_seed(shot.seed)
        if shot.seed is not None else None
    )

    # ── 选择 Pipeline ──────────────────────────────────────────────────────
    # 优先级：fast=True → LTX；LTX 不可用时自动回退到 Wan 2.2
    ltx_available = _state["ltx_t2v_pipe"] is not None
    wan_available = _state["wan_t2v_pipe"] is not None

    want_ltx = shot.fast and ltx_available
    use_wan  = wan_available and not want_ltx

    if want_ltx:
        t2v_pipe  = _state["ltx_t2v_pipe"]
        i2v_pipe  = _state["ltx_i2v_pipe"]
        ti2v_pipe = None   # LTX-Video 无 TI2V，退化到 i2v_pipe
        num_frames = _align_frames_ltx(shot.num_frames)
        model_name = "LTX-Video（预览）"
    elif use_wan:
        t2v_pipe  = _state["wan_t2v_pipe"]
        i2v_pipe  = _state["wan_i2v_pipe"]
        ti2v_pipe = _state["wan_ti2v_pipe"]
        num_frames = _align_frames_wan(shot.num_frames)
        model_name = "Wan 2.2（T2V A14B / I2V A14B / TI2V 5B）"
        if shot.fast:
            logger.warning("LTX-Video 未加载，fast=true 请求自动回退到 Wan 2.2")
    else:
        raise RuntimeError("所有 Pipeline 均未就绪，请检查模型加载状态")

    if t2v_pipe is None:
        raise RuntimeError(f"{model_name} Pipeline 未就绪")

    has_ref = bool(shot.reference_images_b64)
    mode = "i2v（图生视频）" if has_ref else "t2v（文生视频）"
    img_count = len(shot.reference_images_b64) if has_ref else 0
    logger.info(
        "分镜推理 | 模型={} 模式={} 参考图={}张 frames={} steps={} cfg={} fps={} res={}×{} seed={}",
        model_name, mode, img_count, num_frames, shot.num_inference_steps,
        shot.guidance_scale, shot.fps, shot.width, shot.height, shot.seed,
    )
    logger.info("Prompt: {}", shot.prompt[:100])

    # ── GPU 推理 ────────────────────────────────────────────────────────────
    async with lock:
        logger.debug("已获取 GPU 推理锁，开始推理...")

        def _infer() -> List[Image.Image]:
            with torch.inference_mode():
                if use_wan:
                    return _infer_wan(
                        t2v_pipe, i2v_pipe, ti2v_pipe, shot, num_frames, generator
                    )
                else:
                    return _infer_ltx(
                        t2v_pipe, i2v_pipe, shot, num_frames, generator
                    )

        frames = await asyncio.to_thread(_infer)
        logger.info("推理完成，共 {} 帧", len(frames))

    # ── 质量校验 + 自动重试（最多 3 次）────────────────────────────────────
    ref_frame = None
    if shot.reference_images_b64:
        try:
            import base64 as _b64, io as _io
            b64 = shot.reference_images_b64[0].split(",", 1)[-1]
            ref_frame = Image.open(_io.BytesIO(_b64.b64decode(b64))).convert("RGB")
        except Exception:
            pass

    quality_report: QualityReport = check_video_quality(
        frames,
        is_i2v=has_ref,
        reference_frame=ref_frame,
    )

    retry_attempt = 0
    while True:
        do_retry, reason = should_retry(quality_report, retry_attempt)
        if not do_retry:
            logger.info("质量校验决策: {}", reason)
            break

        logger.warning("质量不合格，触发重试 #{} | 原因: {}", retry_attempt + 1, reason)
        retry_attempt += 1

        # 换一个随机种子重试（避免生成相同的坏结果）
        new_seed = (shot.seed or 0) + retry_attempt * 1000
        retry_shot = ShotSpec(
            prompt=shot.prompt,
            negative_prompt=shot.negative_prompt,
            num_frames=shot.num_frames,
            num_inference_steps=min(shot.num_inference_steps + 10, 80),
            height=shot.height,
            width=shot.width,
            fps=shot.fps,
            guidance_scale=max(shot.guidance_scale - 0.5, 5.0),
            seed=new_seed,
            reference_images_b64=shot.reference_images_b64,
            fast=shot.fast,
            background_style=shot.background_style,
        )

        async with lock:
            def _retry_infer() -> List[Image.Image]:
                gen = torch.Generator(device="cpu").manual_seed(new_seed)
                with torch.inference_mode():
                    if use_wan:
                        return _infer_wan(t2v_pipe, i2v_pipe, ti2v_pipe, retry_shot,
                                          _align_frames_wan(retry_shot.num_frames), gen)
                    else:
                        return _infer_ltx(t2v_pipe, i2v_pipe, retry_shot,
                                          _align_frames_ltx(retry_shot.num_frames), gen)

            frames = await asyncio.to_thread(_retry_infer)
            logger.info("重试 #{} 推理完成，共 {} 帧", retry_attempt, len(frames))

        quality_report = check_video_quality(
            frames,
            is_i2v=has_ref,
            reference_frame=ref_frame,
        )

    if not quality_report.passed:
        logger.warning(
            "经 {} 次重试后质量仍不合格（{}），强制交付",
            retry_attempt, quality_report.error_codes(),
        )

    # ── 视频后处理（超分，仅正式出片模式）──────────────────────────────────
    if use_wan and _state.get("upsampler") is not None:
        logger.info("开始视频超分（Real-ESRGAN x2plus）...")
        frames = await asyncio.to_thread(_upscale_frames, frames)

    # ── LUT 色彩分级（P2-2，仅正式出片模式）────────────────────────────────
    if use_wan:
        frames = _apply_lut_grading(frames)

    # ── 视频写入 ────────────────────────────────────────────────────────────
    def _write_video() -> None:
        np_frames = [np.array(f) for f in frames]
        writer = imageio.get_writer(
            str(output_path),
            fps=shot.fps,
            codec="libx264",
            quality=9,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()

    await asyncio.to_thread(_write_video)
    logger.info(
        "分镜视频已写入: {} ({} 帧, {}fps, 质量={})",
        output_path, len(frames), shot.fps,
        "合格" if quality_report.passed else "强制交付",
    )
    return output_path


def _apply_lut_grading(frames: List[Image.Image]) -> List[Image.Image]:
    """
    P2-2: LUT 色彩分级后处理（纯 NumPy 实现，无需 GPU）。

    使用 S 曲线（Sigmoid 色调映射）模拟专业电影 LUT 效果：
      - 暗部提亮（消除死黑）
      - 高光压制（消除过曝）
      - 对比度自然增强
      - 色彩整体偏暖 (+轻微黄色调，商品更有质感)

    这是一个轻量近似方案，完整 LUT 需要 .cube 文件 + colour-science 库（P2后续）。
    """
    try:
        import numpy as np

        # 预计算 S 曲线 LUT（256 级）
        x = np.arange(256, dtype=np.float32) / 255.0
        # Sigmoid S 曲线: 增强对比度，保留高低光细节
        curve = 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))
        # 归一化到 [0, 1]
        curve = (curve - curve.min()) / (curve.max() - curve.min())
        # 轻微暖调：R 通道 +2%，B 通道 -1%
        lut_r = np.clip(curve * 1.02, 0, 1)
        lut_g = curve
        lut_b = np.clip(curve * 0.99, 0, 1)
        # 转为 0-255 整数 LUT
        lut_r_u8 = (lut_r * 255).astype(np.uint8)
        lut_g_u8 = (lut_g * 255).astype(np.uint8)
        lut_b_u8 = (lut_b * 255).astype(np.uint8)

        result = []
        for frame in frames:
            arr = np.array(frame)
            arr[:, :, 0] = lut_r_u8[arr[:, :, 0]]
            arr[:, :, 1] = lut_g_u8[arr[:, :, 1]]
            arr[:, :, 2] = lut_b_u8[arr[:, :, 2]]
            result.append(Image.fromarray(arr))

        logger.debug("LUT 色彩分级完成，处理 {} 帧", len(result))
        return result
    except Exception as e:
        logger.warning("LUT 色彩分级失败，使用原始帧: {}", e)
        return frames


def _resolve_reference_image(shot: ShotSpec) -> Optional[Image.Image]:
    """
    多图智能选择 + 增强处理。

    流程：
      1. 若有多张图 → CLIP 从中选出与 prompt 最匹配的一张
      2. 单张图 → 直接使用
      3. 无图 → 返回 None（文生视频模式）
      4. 选定图 → 经完整增强流水线处理（背景移除/色彩校正/锐化/投影/背景合成）
         严格等比缩放，商品绝不变形

    Returns:
        增强后的 PIL Image（RGB），或 None
    """
    images = shot.reference_images_b64
    if not images:
        return None

    # CLIP 智能选图
    if len(images) > 1:
        select_result = select_best_image(images, shot.prompt, shot_index=0)
        chosen_b64 = select_result.selected_image_b64
        logger.info(
            "CLIP 选图完成 | 选第{}张 | method={} | score={:.4f} | reason={}",
            select_result.selected_index + 1,
            select_result.method,
            select_result.best_score,
            select_result.reason,
        )
    else:
        chosen_b64 = images[0]
        logger.debug("单张参考图，直接使用")

    # 完整增强流水线（不变形保证）
    return prepare_reference_image(
        chosen_b64,
        shot.width,
        shot.height,
        background_style=shot.background_style,
    )


def _infer_wan(t2v_pipe, i2v_pipe, ti2v_pipe, shot: ShotSpec, num_frames: int, generator) -> List[Image.Image]:
    """
    Wan 2.2 推理逻辑。

    路由规则：
      - 无参考图          → T2V（WanPipeline，A14B 权重，文生视频）
      - 有参考图 + prompt → TI2V（WanImageToVideoPipeline，TI2V-5B 权重，文本+图像生视频）
                            同时接受 image 和 prompt，比纯 I2V 更精准控制运动
      - TI2V 不可用时     → I2V 回退（WanImageToVideoPipeline，I2V-A14B 权重）
    """
    common_kwargs = dict(
        prompt=shot.prompt,
        negative_prompt=shot.negative_prompt,
        num_frames=num_frames,
        num_inference_steps=shot.num_inference_steps,
        height=shot.height,
        width=shot.width,
        guidance_scale=shot.guidance_scale,
        generator=generator,
    )

    ref_image = _resolve_reference_image(shot)

    if ref_image is None:
        # 纯文生视频：使用 T2V（A14B，顶级质量）
        result = t2v_pipe(**common_kwargs)
    elif ti2v_pipe is not None:
        # 文本+图像生视频：使用 TI2V-5B（轻量高效，同时接受文本和参考图）
        result = ti2v_pipe(image=ref_image, **common_kwargs)
        logger.debug("使用 TI2V-5B Pipeline（文本+图像生视频）")
    else:
        # TI2V 不可用时回退到 I2V-A14B
        result = i2v_pipe(image=ref_image, **common_kwargs)
        logger.debug("TI2V 不可用，回退到 I2V-A14B Pipeline")

    return _normalize_frames(result.frames[0])


def _normalize_frames(frames) -> List[Image.Image]:
    """
    统一将 pipeline 输出的帧序列转为 PIL Image 列表。

    diffusers 不同版本/模型返回格式不一致：
      - 旧版：List[PIL.Image]
      - 新版（0.33+）：numpy array，shape (N, H, W, C)，值域 [0, 1]
    """
    import numpy as _np
    if isinstance(frames, _np.ndarray):
        frames_u8 = (_np.clip(frames, 0, 1) * 255).astype(_np.uint8)
        return [Image.fromarray(frames_u8[i]) for i in range(frames_u8.shape[0])]
    # 已经是 list，逐项检查是否为 numpy array
    result = []
    for f in frames:
        if isinstance(f, _np.ndarray):
            # 单帧 (H, W, C)，值域可能是 [0,1] 或 [0,255]
            if f.dtype != _np.uint8:
                f = (_np.clip(f, 0, 1) * 255).astype(_np.uint8)
            result.append(Image.fromarray(f))
        else:
            result.append(f)
    return result


def _infer_ltx(t2v_pipe, i2v_pipe, shot: ShotSpec, num_frames: int, generator) -> List[Image.Image]:
    """LTX-Video 推理逻辑（保留原有参数）"""
    common_kwargs = dict(
        prompt=shot.prompt,
        negative_prompt=shot.negative_prompt,
        num_frames=num_frames,
        num_inference_steps=shot.num_inference_steps,
        height=shot.height,
        width=shot.width,
        frame_rate=shot.fps,
        guidance_scale=shot.guidance_scale,
        decode_timestep=0.05,
        max_sequence_length=256,
        generator=generator,
    )

    ref_image = _resolve_reference_image(shot)
    if ref_image is None:
        result = t2v_pipe(**common_kwargs)
    else:
        result = i2v_pipe(image=ref_image, **common_kwargs)

    return _normalize_frames(result.frames[0])


# ── 分镜脚本批量推理 ──────────────────────────────────────────────────────────

async def generate_storyboard(shots: List[ShotSpec]) -> List[Path]:
    """
    按顺序处理完整分镜脚本，返回与 shots 等长的视频路径列表。
    每个分镜独立获取/释放 GPU 锁，不同请求的分镜公平排队。
    """
    total = len(shots)
    logger.info("开始处理分镜脚本，共 {} 个分镜", total)
    output_paths: List[Path] = []

    for idx, shot in enumerate(shots, start=1):
        model_tag = "Wan2.2" if (not shot.fast and _state["wan_t2v_pipe"]) else "LTX"
        mode_tag  = "i2v" if shot.reference_images_b64 else "t2v"
        logger.info("▶ 分镜 {}/{} | {}[{}]", idx, total, model_tag, mode_tag)
        path = await generate_shot(shot)
        output_paths.append(path)

    logger.info("✅ 分镜脚本处理完成，共生成 {} 个视频", len(output_paths))
    return output_paths
