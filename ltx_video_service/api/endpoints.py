"""
api/endpoints.py — HTTP 路由控制器（分镜脚本版）

职责分层：
- 本层（Controller）：校验请求 → 构造 ShotSpec → 调用 engine → 组装响应
- engine 层（Service）：GPU 推理、图像预处理、文件写入
- 控制器不含任何 AI/图像业务逻辑，仅负责协议适配

端点一览：
  GET  /api/v1/health                — 服务健康检查（K8s liveness probe）
  POST /api/v1/generate/storyboard   — 提交分镜脚本，返回 ZIP（每镜一个 MP4）
  POST /api/v1/generate              — 单分镜快捷接口（向后兼容，支持可选参考图）
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from services import engine
from services.engine import ShotSpec

router = APIRouter()

# ── 公共常量 ──────────────────────────────────────────────────────────────

# 写实风格强制前缀：注入到每个分镜的正向提示词头部，确保模型生成写实/高清画面
_REALISM_PREFIX = (
    "photorealistic, ultra-realistic, hyperrealistic, 8K UHD, "
    "sharp focus, high detail, professional photography, "
    "cinematic lighting, RAW photo, DSLR quality, "
)

# 强化负向提示词：排除卡通/动画 + 商品变形词（P0-3 生产级）
_DEFAULT_NEG = (
    # 风格排除
    "cartoon, anime, illustration, drawing, painting, sketch, 3D render, "
    "CGI, animated, toon, flat design, vector art, clipart, comic, "
    # 画质排除
    "worst quality, low quality, blurry, jittery, distorted, "
    "inconsistent motion, watermark, text overlay, logo, "
    "noise, grain, pixelated, compression artifacts, "
    # 商品变形专用词（P0-3）
    "morphing, shape-shifting, product deformation, wrong proportions, "
    "label text distortion, logo corruption, color shift on product, "
    "flickering product, temporal inconsistency, ghosting artifacts, "
    "product melting, object warping, size inconsistency, "
    # 光照排除
    "overexposed, underexposed, harsh shadows, flat lighting, "
    # 通用排除
    "deformed, ugly, plastic, artificial, fake, unrealistic, "
    "multiple products, duplicate, mirrored incorrectly"
)


# ── 宽高比自适应配置 ──────────────────────────────────────────────────────

@dataclass
class AspectConfig:
    """
    针对某种宽高比的电商标准配置。

    Attributes:
        name:           宽高比名称（用于日志）
        std_width:      标准分辨率宽（32的倍数）
        std_height:     标准分辨率高（32的倍数）
        prompt_suffix:  自动追加到正向提示词的构图/运镜补充语
        fps:            推荐帧率（竖版短视频用 30，横版品牌用 24）
    """
    name: str
    std_width: int
    std_height: int
    prompt_suffix: str
    fps: int


# 电商主流宽高比配置表（比例容差 ±5%）
_ASPECT_CONFIGS = [
    # 16:9 — 横版，PC 详情页 / 品牌旗舰视频
    AspectConfig(
        name="16:9（横版）",
        std_width=1280, std_height=720,
        prompt_suffix=(
            "horizontal composition, wide-angle product showcase, "
            "slow 360-degree rotation, centered product, "
            "cinematic widescreen, studio lighting from left and right"
        ),
        fps=24,
    ),
    # 9:16 — 竖版，短视频/直播/移动端全屏
    AspectConfig(
        name="9:16（竖版）",
        std_width=576, std_height=1024,
        prompt_suffix=(
            "vertical video composition, portrait orientation, "
            "product centered with generous top and bottom space, "
            "mobile-first framing, close-up product detail, "
            "smooth upward camera movement"
        ),
        fps=30,
    ),
    # 1:1 — 方形，天猫/淘宝/亚马逊主图，移动端 feed
    AspectConfig(
        name="1:1（方形）",
        std_width=768, std_height=768,
        prompt_suffix=(
            "square format composition, product perfectly centered, "
            "pure white seamless background, product occupies 85% of frame, "
            "slow gentle rotation, e-commerce main image standard"
        ),
        fps=30,
    ),
    # 4:3 — 通用电商，传统详情页
    AspectConfig(
        name="4:3（通用）",
        std_width=960, std_height=720,
        prompt_suffix=(
            "standard product photography composition, "
            "product centered with slight angle, "
            "professional studio lighting, clean background"
        ),
        fps=24,
    ),
    # 3:4 — 竖版通用（小红书/Pinterest）
    AspectConfig(
        name="3:4（竖版通用）",
        std_width=576, std_height=768,
        prompt_suffix=(
            "portrait product photography, vertical composition, "
            "product in upper two-thirds of frame, "
            "lifestyle background or clean studio, elegant lighting"
        ),
        fps=30,
    ),
]


def _snap_to_32(value: int) -> int:
    """将整数向上对齐到 32 的倍数"""
    return ((value + 31) // 32) * 32


def _detect_aspect_config(width: int, height: int) -> Optional[AspectConfig]:
    """
    根据请求的宽高比检测最匹配的电商配置。
    容差 ±5%：ratio 落在 [target*(1-0.05), target*(1+0.05)] 内即匹配。
    无法匹配时返回 None（使用用户原始尺寸）。
    """
    ratio = width / height
    targets = {
        16 / 9: _ASPECT_CONFIGS[0],   # ≈ 1.778
        9 / 16: _ASPECT_CONFIGS[1],   # ≈ 0.5625
        1 / 1:  _ASPECT_CONFIGS[2],   # = 1.0
        4 / 3:  _ASPECT_CONFIGS[3],   # ≈ 1.333
        3 / 4:  _ASPECT_CONFIGS[4],   # ≈ 0.75
    }
    TOLERANCE = 0.05
    for target_ratio, config in targets.items():
        if abs(ratio - target_ratio) / target_ratio <= TOLERANCE:
            return config
    return None


# ── 请求体数据模型 ────────────────────────────────────────────────────────

class ShotRequest(BaseModel):
    """
    单个分镜的参数。

    - reference_image: base64 编码的商品实拍图（可选）
      · 缺省 → 文生视频（LTXPipeline）
      · 传入 → 图生视频（LTXImageToVideoPipeline），engine 自动等比缩放保证不变形
    - 支持标准 base64 字符串或 data URI（"data:image/jpeg;base64,..."）
    """

    prompt: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="正向提示词，描述该分镜希望呈现的画面内容",
            examples=["A luxury perfume bottle rotating slowly on a white marble surface, "
                      "product photography, cinematic lighting, 4K"],
        ),
    ]

    negative_prompt: Annotated[
        str,
        Field(
            default=_DEFAULT_NEG,
            max_length=2000,
            description="负向提示词，描述希望避免的内容",
        ),
    ]

    num_frames: Annotated[
        int,
        Field(
            default=97,   # P0-1：生产级默认 97 帧（约 4 秒@24fps，原 65 帧仅 2.7 秒不够用）
            ge=9,
            le=257,
            description=(
                "帧数（服务自动对齐到模型要求）。"
                "Wan 2.2 要求 4N+1（17,21,25...97...），LTX 要求 8N+1（9,17,25...65...）。"
                "正式出片推荐 97（4s），预览推荐 25（1s）"
            ),
        ),
    ]

    num_inference_steps: Annotated[
        int,
        Field(
            default=50,   # P0-1：生产级默认 50 步（原 30 步质量不足）
            ge=1,
            le=100,
            description="扩散去噪步数，越大质量越高。正式出片推荐 50，预览模式推荐 20",
        ),
    ]

    height: Annotated[
        int,
        Field(
            default=480,
            ge=256,
            le=1280,
            description=(
                "视频高度（像素），须为 32 的倍数。"
                "服务会根据宽高比自动对齐到电商标准分辨率："
                "16:9→720, 9:16→1024, 1:1→768, 4:3→720, 3:4→768"
            ),
        ),
    ]

    width: Annotated[
        int,
        Field(
            default=704,
            ge=256,
            le=1280,
            description=(
                "视频宽度（像素），须为 32 的倍数。"
                "服务会根据宽高比自动对齐到电商标准分辨率："
                "16:9→1280, 9:16→576, 1:1→768, 4:3→960, 3:4→576"
            ),
        ),
    ]

    fps: Annotated[
        int,
        Field(default=24, ge=8, le=60, description="输出视频帧率（宽高比自适应时会覆盖为推荐值）"),
    ]

    reference_images: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            min_length=1,
            max_length=10,
            description=(
                "商品实拍参考图列表，base64 编码（JPEG / PNG / WEBP 均可），最多 10 张。\n"
                "• 传入 1 张：直接使用该图作为参考帧\n"
                "• 传入多张：服务用 CLIP 自动从中选出与当前分镜 prompt 最匹配的图\n"
                "  （如'侧面展示'会选侧面图，'特写材质'会选细节图）\n"
                "• 选定图片经背景移除+色彩校正+锐化+投影合成处理，商品绝不变形\n"
                "• 支持标准 base64 或 data URI（data:image/jpeg;base64,...）"
            ),
        ),
    ]

    # ── 向后兼容：保留旧的单图字段（deprecated）──────────────────────────
    reference_image: Annotated[
        Optional[str],
        Field(
            default=None,
            description="[已废弃] 请使用 reference_images（数组）。保留此字段仅为向后兼容。",
        ),
    ]

    fast: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "False（默认）= 正式出片模式，使用 Wan 2.2（T2V A14B / TI2V 5B），~60s/clip，顶级质量。\n"
                "True = 快速预览模式，使用 LTX-Video，~8s/clip，用于确认构图/动作。"
            ),
        ),
    ]

    background_style: Annotated[
        str,
        Field(
            default="gradient",
            description=(
                "商品背景样式（仅在传入 reference_image 时生效）：\n"
                "• gradient — 径向渐变（近白→浅蓝灰，品牌感，默认）\n"
                "• white    — 纯白背景（天猫/亚马逊主图合规）\n"
                "• warm     — 温暖渐变（奶白→米色，适合美妆/食品）\n"
                "• dark     — 深色渐变（深灰→近黑，适合科技/3C）"
            ),
        ),
    ]

    # ── 校验器 ────────────────────────────────────────────────────────────

    @field_validator("height", "width")
    @classmethod
    def must_be_multiple_of_32(cls, v: int) -> int:
        """LTX-Video 架构要求分辨率必须是 32 的倍数，否则推理报错"""
        if v % 32 != 0:
            raise ValueError(
                f"分辨率 {v} 不是 32 的倍数，请调整（如 480, 512, 704, 768）"
            )
        return v

    @field_validator("num_frames")
    @classmethod
    def must_be_valid_frames(cls, v: int) -> int:
        """
        帧数兼容性校验：接受 4N+1（Wan 2.2）或 8N+1（LTX-Video）格式。
        engine 层会根据实际使用的模型自动对齐到正确格式。
        """
        is_wan_valid = (v - 1) % 4 == 0
        is_ltx_valid = (v - 1) % 8 == 0
        if not (is_wan_valid or is_ltx_valid):
            raise ValueError(
                f"num_frames={v} 无效。"
                "正式出片（Wan 2.2）请使用 4N+1：17, 21, 25, 29...97...；"
                "预览（LTX-Video）请使用 8N+1：9, 17, 25...65...257"
            )
        return v

    def to_shot_spec(self) -> ShotSpec:
        """
        将 Pydantic 请求模型转换为 engine 层的 ShotSpec（解耦 HTTP 与业务层）。

        宽高比自适应策略：
        - 根据客户端传入的 width/height 检测宽高比（容差 ±5%）
        - 匹配到电商标准配置后，自动将分辨率升级到该宽高比的标准值
          （如 16:9 → 1280×720，9:16 → 576×1024，1:1 → 768×768）
        - 同时将推荐帧率和构图/运镜 prompt 补充语注入
        - 客户端若传入自定义尺寸且不符合任何标准比例，原样使用

        写实增强策略：
        - 正向提示词头部注入写实风格前缀（_REALISM_PREFIX）
        - 尾部追加宽高比专属构图语（如旋转方向、取景比例）
        - 负向提示词合并系统默认（卡通/动画排除词）
        """
        # ── 向后兼容：将旧的单图字段合并到 reference_images ──────────
        ref_images = self.reference_images
        if ref_images is None and self.reference_image is not None:
            ref_images = [self.reference_image]

        # ── 宽高比检测 & 标准分辨率对齐 ──────────────────────────────
        aspect_cfg = _detect_aspect_config(self.width, self.height)

        if aspect_cfg is not None:
            final_w = aspect_cfg.std_width
            final_h = aspect_cfg.std_height
            final_fps = self.fps if self.fps != 24 else aspect_cfg.fps  # 用户显式改了 fps 则尊重
            composition_suffix = ", " + aspect_cfg.prompt_suffix
            logger.info(
                "宽高比自适应 | 输入={}×{} → 识别为 {} → 标准={}×{} fps={}",
                self.width, self.height, aspect_cfg.name,
                final_w, final_h, final_fps,
            )
        else:
            # 自定义尺寸：确保是 32 的倍数即可
            final_w = _snap_to_32(self.width)
            final_h = _snap_to_32(self.height)
            final_fps = self.fps
            composition_suffix = ""
            if final_w != self.width or final_h != self.height:
                logger.info(
                    "自定义尺寸对齐 32 倍数 | {}×{} → {}×{}",
                    self.width, self.height, final_w, final_h,
                )
            else:
                logger.info(
                    "自定义尺寸 {}×{}，不匹配标准宽高比，原样使用",
                    final_w, final_h,
                )

        # ── 合并写实前缀 + 用户提示词 + 构图补充语 ──────────────────
        enhanced_prompt = _REALISM_PREFIX + self.prompt + composition_suffix

        # ── 合并系统负向词 + 用户负向词 ──────────────────────────────
        user_neg = self.negative_prompt.strip()
        if user_neg and user_neg != _DEFAULT_NEG:
            enhanced_neg = _DEFAULT_NEG + ", " + user_neg
        else:
            enhanced_neg = _DEFAULT_NEG

        return ShotSpec(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_neg,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            height=final_h,
            width=final_w,
            fps=final_fps,
            reference_images_b64=ref_images,   # 传入全部图片，engine 层做选择
            fast=self.fast,
            background_style=self.background_style,
        )


class StoryboardRequest(BaseModel):
    """
    分镜脚本请求体。

    客户端将完整的分镜脚本以 JSON 数组形式提交，每个元素对应一个分镜（ShotRequest）。
    服务按顺序逐镜推理，以 ZIP 压缩包返回所有分镜的 MP4 文件。

    ZIP 内文件命名规则：shot_001.mp4, shot_002.mp4 ...（方便客户端按序拼接）
    """

    shots: Annotated[
        List[ShotRequest],
        Field(
            min_length=1,
            max_length=50,
            description="分镜列表，顺序即为最终出片顺序，最多 50 个分镜",
        ),
    ]


# ── 响应体数据模型 ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool


class ShotResult(BaseModel):
    """单分镜推理结果（用于 JSON 响应中描述每个分镜）"""
    shot_index: int
    filename: str
    mode: str   # "t2v" 或 "i2v"


# ── 路由定义 ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="服务健康检查")
async def health_check() -> HealthResponse:
    """
    健康检查端点，用于容器编排平台（K8s / Docker Compose）的存活探针。
    模型未加载时返回 model_loaded=false，HTTP 状态码仍为 200，
    区分"服务存活但模型还在加载"与"服务崩溃"两种状态。
    """
    return HealthResponse(
        status="ok",
        model_loaded=engine.get_pipeline() is not None,
    )


@router.post(
    "/generate/storyboard",
    summary="提交分镜脚本，批量生成视频",
    response_description="ZIP 压缩包，包含每个分镜对应的 MP4 文件（shot_001.mp4…）",
)
async def generate_storyboard(
    request: StoryboardRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """
    接收完整分镜脚本，逐镜调用 LTX-Video 推理，打包 ZIP 后流式返回。

    - 无 reference_image 的分镜 → 文生视频（Wan2.2 T2V A14B / LTX 预览）
    - 有 reference_image 的分镜 → 文本+图像生视频（Wan2.2 TI2V 5B / LTX I2V 预览），
      商品图像自动背景移除+色彩校正+等比缩放，保证商品不变形
    - 多分镜按顺序串行推理，GPU 锁在每个分镜间公平释放，不阻塞其他并发请求
    - ZIP 在内存中构建，发送完毕后后台任务清理所有临时 MP4 文件
    """
    shot_count = len(request.shots)
    mode_summary = ", ".join(
        f"#{i + 1}={'i2v' if (s.reference_images or s.reference_image) else 't2v'}"
        for i, s in enumerate(request.shots)
    )
    logger.info("POST /generate/storyboard | {} 个分镜 | {}", shot_count, mode_summary)

    # ── 构造 engine 层数据结构 ────────────────────────────────────
    shot_specs = [shot.to_shot_spec() for shot in request.shots]

    try:
        output_paths = await engine.generate_storyboard(shot_specs)
    except RuntimeError as e:
        logger.warning("服务未就绪: {}", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.opt(exception=True).error("分镜脚本推理失败: {}", e)
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")

    # ── 在内存中打包 ZIP ──────────────────────────────────────────
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for idx, path in enumerate(output_paths, start=1):
            arcname = f"shot_{idx:03d}.mp4"
            zf.write(str(path), arcname=arcname)
    zip_buffer.seek(0)

    # 注册后台任务：响应发送完毕后清理所有临时 MP4 文件
    background_tasks.add_task(_cleanup_files, output_paths)

    logger.info("ZIP 打包完成，共 {} 个视频，开始流式传输", len(output_paths))
    return StreamingResponse(
        content=zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=storyboard_videos.zip"},
    )


@router.post(
    "/generate",
    summary="单分镜视频生成（支持文生视频 / 图生视频）",
    response_description="生成完成的 MP4 视频文件（video/mp4）",
)
async def generate_single(
    request: ShotRequest,
    background_tasks: BackgroundTasks,
) -> FileResponse:
    """
    单分镜快捷接口，与旧版 /generate 向后兼容。

    - 不传 reference_image → 文生视频
    - 传入 reference_image → 图生视频（商品参考图等比缩放，不变形）

    文件发送完毕后，后台任务自动删除临时文件，防止磁盘爆满。
    """
    mode = "i2v" if (request.reference_images or request.reference_image) else "t2v"
    logger.info(
        "POST /generate | mode={} | prompt='{}'", mode, request.prompt[:60]
    )

    try:
        output_path = await engine.generate_shot(request.to_shot_spec())
    except RuntimeError as e:
        logger.warning("服务未就绪: {}", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.opt(exception=True).error("视频生成失败: {}", e)
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")

    background_tasks.add_task(_cleanup_file, output_path)

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_path.name,
    )


# ── 后台清理任务 ──────────────────────────────────────────────────────────

def _cleanup_file(path: Path) -> None:
    """删除单个临时视频文件，防止磁盘爆满"""
    try:
        path.unlink(missing_ok=True)
        logger.debug("临时文件已清理: {}", path)
    except Exception as e:
        logger.warning("清理临时文件失败: {} | {}", path, e)


def _cleanup_files(paths: List[Path]) -> None:
    """批量删除临时视频文件（分镜脚本模式使用）"""
    for path in paths:
        _cleanup_file(path)
