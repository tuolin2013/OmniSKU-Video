"""
services/image_enhance.py — 商品图像增强流水线（生产级）

处理链：
  原始实拍图
    → 背景移除（rembg）
    → 色彩校正（白平衡 + 曝光标准化 + 饱和度微增）
    → 锐化（Unsharp Mask，突出商品边缘细节）
    → 投影合成（底部软阴影，增强立体感）
    → 合成到专业背景（径向渐变，商品居中，10% 边距）

设计原则：
  - 每个步骤独立函数，便于单测和参数调整
  - 所有操作在 PIL / NumPy 层完成，无需 GPU
  - 失败时优雅降级（跳过该步骤，不中断整体流程）
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from PIL import Image, ImageEnhance, ImageFilter

# ── rembg 懒加载 ──────────────────────────────────────────────────────────
_rembg_session = None


def _get_rembg_session():
    """懒加载 rembg 推理会话（线程安全：仅在推理线程内调用）"""
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            _rembg_session = new_session("u2net")
            logger.info("✅ rembg 背景移除模型已加载（u2net）")
        except BaseException as e:
            # 注意：rembg/bg.py 在 onnxruntime 缺失时直接 sys.exit(1)，
            # 抛出 SystemExit（BaseException 子类），普通 except Exception 捕获不到。
            # 必须用 BaseException 才能拦截，避免进程崩溃。
            logger.warning("rembg 加载失败，将跳过背景移除: {}", e)
            _rembg_session = False
    return _rembg_session if _rembg_session is not False else None


# ── Step 1: 背景移除 ──────────────────────────────────────────────────────

def remove_background(img_rgb: Image.Image) -> Image.Image:
    """
    使用 rembg u2net 移除商品图背景，返回 RGBA 图像。
    失败时返回原图转 RGBA（回退策略）。
    """
    session = _get_rembg_session()
    if session is None:
        logger.warning("rembg 不可用，跳过背景移除")
        return img_rgb.convert("RGBA")
    try:
        from rembg import remove as rembg_remove
        result = rembg_remove(img_rgb, session=session)
        return result.convert("RGBA") if result.mode != "RGBA" else result
    except Exception as e:
        logger.warning("rembg 推理失败，跳过背景移除: {}", e)
        return img_rgb.convert("RGBA")


# ── Step 2: 色彩校正 ──────────────────────────────────────────────────────

def color_correct(img_rgba: Image.Image) -> Image.Image:
    """
    对商品主体做色彩校正：
      1. 亮度标准化：让商品主体平均亮度落在目标区间 [160, 210]
      2. 对比度微增（×1.05），让商品层次更丰富
      3. 饱和度微增（×1.15），颜色更鲜亮但不过饱和
      4. 白平衡校正：通过 RGB 通道均值平衡消除偏色

    Args:
        img_rgba: RGBA 模式的商品图（已去背景）

    Returns:
        色彩校正后的 RGBA 图像
    """
    # 分离 RGB 和 Alpha 通道，仅对 RGB 做处理
    r, g, b, alpha = img_rgba.split()
    img_rgb = Image.merge("RGB", (r, g, b))

    # ── 白平衡校正（灰世界算法：让三通道均值趋向一致）──────────────
    try:
        arr = np.array(img_rgb, dtype=np.float32)
        # 仅统计非透明像素（alpha > 10）的颜色
        alpha_arr = np.array(alpha)
        mask = alpha_arr > 10
        if mask.sum() > 100:
            mean_r = arr[:, :, 0][mask].mean()
            mean_g = arr[:, :, 1][mask].mean()
            mean_b = arr[:, :, 2][mask].mean()
            mean_all = (mean_r + mean_g + mean_b) / 3
            if mean_all > 0:
                scale_r = np.clip(mean_all / (mean_r + 1e-6), 0.85, 1.20)
                scale_g = np.clip(mean_all / (mean_g + 1e-6), 0.85, 1.20)
                scale_b = np.clip(mean_all / (mean_b + 1e-6), 0.85, 1.20)
                arr[:, :, 0] = np.clip(arr[:, :, 0] * scale_r, 0, 255)
                arr[:, :, 1] = np.clip(arr[:, :, 1] * scale_g, 0, 255)
                arr[:, :, 2] = np.clip(arr[:, :, 2] * scale_b, 0, 255)
                img_rgb = Image.fromarray(arr.astype(np.uint8), "RGB")
                logger.debug(
                    "白平衡校正 | mean_r={:.1f} g={:.1f} b={:.1f} → scale r={:.3f} g={:.3f} b={:.3f}",
                    mean_r, mean_g, mean_b, scale_r, scale_g, scale_b,
                )
    except Exception as e:
        logger.warning("白平衡校正失败，跳过: {}", e)

    # ── 亮度标准化 ──────────────────────────────────────────────────
    try:
        arr = np.array(img_rgb, dtype=np.float32)
        alpha_arr = np.array(alpha)
        mask = alpha_arr > 10
        if mask.sum() > 100:
            # 转灰度计算亮度
            lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            current_lum = lum[mask].mean()
            target_lum = 185.0  # 目标亮度（0-255），偏亮适合白背景展示
            if current_lum > 10:
                scale = np.clip(target_lum / current_lum, 0.7, 1.5)
                arr = np.clip(arr * scale, 0, 255)
                img_rgb = Image.fromarray(arr.astype(np.uint8), "RGB")
                logger.debug(
                    "亮度标准化 | 当前亮度={:.1f} → 目标={:.1f} scale={:.3f}",
                    current_lum, target_lum, scale,
                )
    except Exception as e:
        logger.warning("亮度标准化失败，跳过: {}", e)

    # ── 对比度微增 ──────────────────────────────────────────────────
    try:
        img_rgb = ImageEnhance.Contrast(img_rgb).enhance(1.08)
    except Exception as e:
        logger.warning("对比度增强失败，跳过: {}", e)

    # ── 饱和度微增 ──────────────────────────────────────────────────
    try:
        img_rgb = ImageEnhance.Color(img_rgb).enhance(1.15)
    except Exception as e:
        logger.warning("饱和度增强失败，跳过: {}", e)

    # 重新合并 Alpha 通道
    r2, g2, b2 = img_rgb.split()
    return Image.merge("RGBA", (r2, g2, b2, alpha))


# ── Step 3: 锐化 ──────────────────────────────────────────────────────────

def sharpen(img_rgba: Image.Image) -> Image.Image:
    """
    对商品主体做 Unsharp Mask 锐化，突出材质细节和商品边缘。
    锐化量（amount=1.3）适中，避免过锐产生光晕。
    仅对 RGB 通道锐化，Alpha 通道保持不变，避免边缘出现白边。
    """
    try:
        r, g, b, alpha = img_rgba.split()
        img_rgb = Image.merge("RGB", (r, g, b))
        # UnsharpMask: radius=1.5（作用范围），percent=130（锐化强度），threshold=3（最小对比度差）
        sharpened = img_rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
        r2, g2, b2 = sharpened.split()
        return Image.merge("RGBA", (r2, g2, b2, alpha))
    except Exception as e:
        logger.warning("锐化处理失败，跳过: {}", e)
        return img_rgba


# ── Step 4: 投影合成 ──────────────────────────────────────────────────────

def add_drop_shadow(
    img_rgba: Image.Image,
    shadow_offset_y: int = 12,
    shadow_blur: int = 18,
    shadow_opacity: float = 0.35,
    shadow_color: tuple = (30, 30, 40),
) -> Image.Image:
    """
    在商品底部添加柔和的下落阴影，增强立体感和材质真实感。

    实现原理：
      1. 从商品 alpha 通道提取轮廓
      2. 向下平移 shadow_offset_y 像素
      3. 高斯模糊（shadow_blur 半径）营造柔和效果
      4. 按 shadow_opacity 透明度叠加到结果图层下方

    Args:
        img_rgba:         待处理的商品图（RGBA）
        shadow_offset_y:  阴影向下偏移像素数（默认 12px）
        shadow_blur:      阴影模糊半径（默认 18，越大越柔）
        shadow_opacity:   阴影不透明度 0~1（默认 0.35）
        shadow_color:     阴影 RGB 颜色（默认深蓝灰，比纯黑更自然）

    Returns:
        包含商品 + 阴影的 RGBA 图像（尺寸与输入相同）
    """
    try:
        w, h = img_rgba.size
        alpha = img_rgba.split()[3]

        # 创建阴影层：以商品 alpha 为形状，颜色为 shadow_color
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow_color_img = Image.new("RGB", (w, h), shadow_color)
        # 将 alpha 通道缩放到 shadow_opacity
        shadow_alpha = alpha.point(lambda p: int(p * shadow_opacity))
        shadow_layer.paste(shadow_color_img, mask=shadow_alpha)

        # 向下偏移
        offset_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        offset_layer.paste(shadow_layer, (0, shadow_offset_y))

        # 高斯模糊
        blurred_shadow = offset_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

        # 合成：先放阴影，再叠商品
        composite = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        composite = Image.alpha_composite(composite, blurred_shadow)
        composite = Image.alpha_composite(composite, img_rgba)

        logger.debug(
            "投影合成完成 | offset_y={} blur={} opacity={}",
            shadow_offset_y, shadow_blur, shadow_opacity,
        )
        return composite
    except Exception as e:
        logger.warning("投影合成失败，跳过: {}", e)
        return img_rgba


# ── Step 5: 背景合成 ──────────────────────────────────────────────────────

def make_gradient_background(width: int, height: int, style: str = "gradient") -> Image.Image:
    """
    生成专业商品展示背景。

    Args:
        width, height: 目标画布尺寸
        style:
          "white"    — 纯白（天猫/亚马逊主图合规）
          "gradient" — 径向渐变（中心近白 → 边缘浅蓝灰，品牌感）
          "warm"     — 温暖渐变（中心奶白 → 边缘浅米色，美妆/食品）
          "dark"     — 深色渐变（中心深灰 → 边缘近黑，科技/3C）

    Returns:
        RGB 模式背景图
    """
    if style == "white":
        return Image.new("RGB", (width, height), (255, 255, 255))

    cx, cy = width / 2, height / 2
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    dist = np.sqrt(((x_coords - cx) / cx) ** 2 + ((y_coords - cy) / cy) ** 2)
    dist = np.clip(dist, 0, 1)

    if style == "warm":
        # 奶白 #FAF7F0 → 浅米色 #E8DDD0
        center = (250, 247, 240)
        edge   = (232, 221, 208)
    elif style == "dark":
        # 深灰 #2A2A2E → 近黑 #14141A
        center = (42, 42, 46)
        edge   = (20, 20, 26)
    else:  # gradient（默认）
        # 近白 #F5F5F5 → 浅蓝灰 #D0D8E8
        center = (245, 245, 245)
        edge   = (208, 216, 232)

    r = np.clip(center[0] - (center[0] - edge[0]) * dist, 0, 255).astype(np.uint8)
    g = np.clip(center[1] - (center[1] - edge[1]) * dist, 0, 255).astype(np.uint8)
    b = np.clip(center[2] - (center[2] - edge[2]) * dist, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


# ── 主入口：完整增强流水线 ────────────────────────────────────────────────

def enhance_product_image(
    img_rgb: Image.Image,
    target_width: int,
    target_height: int,
    padding_ratio: float = 0.10,
    background_style: str = "gradient",
    enable_color_correct: bool = True,
    enable_sharpen: bool = True,
    enable_shadow: bool = True,
) -> Image.Image:
    """
    商品图像完整增强流水线。

    流程：
      背景移除 → 色彩校正 → 锐化 → 等比缩放 → 投影合成 → 背景合成

    Args:
        img_rgb:          原始商品图（RGB）
        target_width:     目标帧宽度（pipeline 要求，32的倍数）
        target_height:    目标帧高度
        padding_ratio:    商品四周留白比例（默认 10%）
        background_style: 背景样式（white/gradient/warm/dark）
        enable_color_correct: 是否做色彩校正
        enable_sharpen:   是否做锐化
        enable_shadow:    是否添加投影

    Returns:
        RGB 模式，尺寸恰好为 target_width × target_height 的商品展示帧
    """
    orig_w, orig_h = img_rgb.size
    logger.info(
        "商品图增强流水线启动 | 原始={}×{} → 目标={}×{} | "
        "色彩校正={} 锐化={} 投影={} 背景={}",
        orig_w, orig_h, target_width, target_height,
        enable_color_correct, enable_sharpen, enable_shadow, background_style,
    )

    # ── Step 1: 背景移除 ────────────────────────────────────────────
    img_rgba = remove_background(img_rgb)
    logger.debug("Step1 背景移除完成")

    # ── Step 2: 色彩校正 ────────────────────────────────────────────
    if enable_color_correct:
        img_rgba = color_correct(img_rgba)
        logger.debug("Step2 色彩校正完成")

    # ── Step 3: 锐化 ────────────────────────────────────────────────
    if enable_sharpen:
        img_rgba = sharpen(img_rgba)
        logger.debug("Step3 锐化完成")

    # ── Step 4: 等比缩放（严格保持商品比例，绝不变形）────────────────
    inner_w = round(target_width  * (1 - 2 * padding_ratio))
    inner_h = round(target_height * (1 - 2 * padding_ratio))
    scale = min(inner_w / orig_w, inner_h / orig_h)
    scaled_w = round(orig_w * scale)
    scaled_h = round(orig_h * scale)
    img_rgba = img_rgba.resize((scaled_w, scaled_h), Image.LANCZOS)
    logger.debug(
        "Step4 等比缩放 | {}×{} → {}×{} (scale={:.4f})",
        orig_w, orig_h, scaled_w, scaled_h, scale,
    )

    # ── Step 5: 投影合成 ────────────────────────────────────────────
    if enable_shadow:
        img_rgba = add_drop_shadow(img_rgba)
        logger.debug("Step5 投影合成完成")

    # ── Step 6: 合成到背景画布（居中） ──────────────────────────────
    canvas = make_gradient_background(target_width, target_height, style=background_style)
    canvas = canvas.convert("RGBA")
    offset_x = (target_width  - scaled_w) // 2
    offset_y = (target_height - scaled_h) // 2
    canvas.paste(img_rgba, (offset_x, offset_y), mask=img_rgba.split()[3])
    result = canvas.convert("RGB")

    logger.info(
        "商品图增强完成 | 商品居中 offset=({},{}) 最终={}×{}",
        offset_x, offset_y, target_width, target_height,
    )
    return result
