"""
services/image_selector.py — 基于 CLIP 的智能图片选择器

功能：
  客户端上传多张商品实拍图（最多 10 张），服务根据每个分镜的 prompt
  自动选出最匹配的那张图作为参考帧，再送入图生视频 Pipeline。

算法：
  使用 OpenAI CLIP（openai/clip-vit-base-patch32）做零样本图文匹配：
  - 将每张图片提取视觉 embedding
  - 将分镜 prompt 提取文本 embedding
  - 计算余弦相似度，选相似度最高的图片

设计：
  - CLIP 模型懒加载（首次调用时初始化，不影响服务启动速度）
  - 纯 CPU 运行（CLIP 小模型，图片数量少，CPU 足够快）
  - 失败时回退到"选第一张"策略（服务不中断）
  - 选择结果详细记录（便于排查和调试）

为什么用 CLIP？
  - 不需要图片标注，零样本理解商品语义
  - "香水瓶正面" 会匹配正面图，"香水瓶侧面" 会匹配侧面图
  - "商品特写材质" 会匹配高清细节图
  - transformers 已安装，无需额外依赖
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

# ── CLIP 懒加载 ──────────────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None
_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def _get_clip():
    """懒加载 CLIP 模型（线程安全：仅在请求处理时调用）"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            logger.info("加载 CLIP 模型: {}", _CLIP_MODEL_ID)
            _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
            _clip_model = CLIPModel.from_pretrained(
                _CLIP_MODEL_ID,
                torch_dtype=torch.float32,   # CPU 用 float32
            )
            _clip_model.eval()
            logger.info("✅ CLIP 模型加载完成")
        except Exception as e:
            logger.warning("CLIP 加载失败，将回退到选第一张策略: {}", e)
            _clip_model = False  # False = 明确失败，不再重试
    return (_clip_model, _clip_processor) if _clip_model is not False else (None, None)


# ── 图片选择结果 ──────────────────────────────────────────────────────────────

@dataclass
class ImageSelectResult:
    """图片选择结果"""
    selected_index: int          # 选中图片在输入列表中的索引（0-based）
    selected_image_b64: str      # 选中图片的 base64 字符串
    similarity_scores: List[float]  # 每张图片与 prompt 的相似度分数（0~1）
    method: str                  # 选择方法："clip" 或 "fallback_first"
    reason: str                  # 选择原因描述

    @property
    def best_score(self) -> float:
        return self.similarity_scores[self.selected_index] if self.similarity_scores else 0.0


# ── 图片解码工具 ──────────────────────────────────────────────────────────────

def _decode_image(image_b64: str) -> Image.Image:
    """将 base64 字符串解码为 PIL Image（支持 data URI）"""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ── 图片内容分析（辅助 CLIP 提升准确度）────────────────────────────────────────

def _analyze_image_properties(img: Image.Image) -> dict:
    """
    快速分析图片基本属性，用于辅助选择（当 CLIP 相似度相近时作为次级排序依据）。

    返回：
      - aspect_ratio: 宽高比
      - brightness: 平均亮度（越高越亮）
      - is_landscape: 是否横向（宽 > 高）
      - is_portrait: 是否纵向（高 > 宽）
      - is_square: 是否接近方形
    """
    w, h = img.size
    arr = np.array(img, dtype=np.float32)
    brightness = float(arr.mean())
    ratio = w / h
    return {
        "aspect_ratio": ratio,
        "brightness": brightness,
        "is_landscape": ratio > 1.1,
        "is_portrait": ratio < 0.9,
        "is_square": 0.9 <= ratio <= 1.1,
        "width": w,
        "height": h,
    }


# ── 核心选择逻辑 ──────────────────────────────────────────────────────────────

def select_best_image(
    images_b64: List[str],
    prompt: str,
    shot_index: int = 0,
) -> ImageSelectResult:
    """
    从多张商品实拍图中，根据分镜 prompt 选出最匹配的一张。

    算法优先级：
      1. CLIP 图文相似度（主要依据）
      2. 若最高分与次高分差距 < 0.02（太接近），优先选亮度更高的图
      3. CLIP 不可用时，回退到选第一张

    Args:
        images_b64:  List[str]，base64 编码的商品图（1~10 张）
        prompt:      当前分镜的正向提示词（含场景、动作、视角描述）
        shot_index:  分镜序号（仅用于日志）

    Returns:
        ImageSelectResult，含选中图片和所有相似度分数
    """
    n = len(images_b64)

    # 只有一张图，直接返回
    if n == 1:
        logger.debug("分镜#{} 只有 1 张参考图，直接使用", shot_index)
        return ImageSelectResult(
            selected_index=0,
            selected_image_b64=images_b64[0],
            similarity_scores=[1.0],
            method="single",
            reason="仅有 1 张图，直接使用",
        )

    # 解码所有图片
    try:
        pil_images = [_decode_image(b64) for b64 in images_b64]
    except Exception as e:
        logger.warning("分镜#{} 图片解码失败，选第一张: {}", shot_index, e)
        return ImageSelectResult(
            selected_index=0,
            selected_image_b64=images_b64[0],
            similarity_scores=[0.0] * n,
            method="fallback_first",
            reason=f"图片解码失败: {e}",
        )

    # ── CLIP 相似度计算 ────────────────────────────────────────────────────
    model, processor = _get_clip()

    if model is not None and processor is not None:
        try:
            import torch

            # 构建匹配用的提示词（去掉写实前缀，保留核心场景描述）
            # 截取前 77 个 token（CLIP 上限），聚焦关键词
            match_prompt = _extract_match_prompt(prompt)

            # 编码文本
            text_inputs = processor(
                text=[match_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            # 编码所有图片
            image_inputs = processor(
                images=pil_images,
                return_tensors="pt",
            )

            with torch.no_grad():
                # 提取 embedding
                text_embeds = model.get_text_features(**text_inputs)
                image_embeds = model.get_image_features(**image_inputs)

                # L2 归一化
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                # 余弦相似度
                similarities = (image_embeds @ text_embeds.T).squeeze(-1)
                scores = similarities.tolist()

            # 选出最高分
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]

            # 若最高分与次高分差距很小（<0.02），用亮度作为次级排序
            sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            if len(sorted_scores) >= 2:
                top1_score = sorted_scores[0][1]
                top2_score = sorted_scores[1][1]
                if top1_score - top2_score < 0.02:
                    # 在得分相近的候选中，选亮度更高的（更适合商品展示）
                    top_candidates = [i for i, s in sorted_scores if top1_score - s < 0.02]
                    props = [_analyze_image_properties(pil_images[i]) for i in top_candidates]
                    bright_idx = top_candidates[int(np.argmax([p["brightness"] for p in props]))]
                    if bright_idx != best_idx:
                        logger.debug(
                            "分镜#{} CLIP 分数相近（差<0.02），改选亮度更高的图 #{}",
                            shot_index, bright_idx,
                        )
                        best_idx = bright_idx

            scores_rounded = [round(s, 4) for s in scores]
            logger.info(
                "分镜#{} CLIP 图片选择 | prompt='{}' | 分数={} | 选第{}张（score={:.4f}）",
                shot_index, match_prompt[:50], scores_rounded,
                best_idx, scores[best_idx],
            )

            return ImageSelectResult(
                selected_index=best_idx,
                selected_image_b64=images_b64[best_idx],
                similarity_scores=scores_rounded,
                method="clip",
                reason=(
                    f"CLIP 余弦相似度最高: {scores[best_idx]:.4f}（共 {n} 张图）"
                    f"，匹配词: '{match_prompt[:40]}'"
                ),
            )

        except Exception as e:
            logger.warning("分镜#{} CLIP 推理失败，回退到选第一张: {}", shot_index, e)

    # ── 回退策略：选第一张 ─────────────────────────────────────────────────
    logger.warning("分镜#{} 使用回退策略（CLIP 不可用），选第一张图", shot_index)
    return ImageSelectResult(
        selected_index=0,
        selected_image_b64=images_b64[0],
        similarity_scores=[0.0] * n,
        method="fallback_first",
        reason="CLIP 不可用，默认选第一张",
    )


# ── 分镜脚本批量选图 ──────────────────────────────────────────────────────────

def select_images_for_storyboard(
    images_b64: List[str],
    prompts: List[str],
) -> List[ImageSelectResult]:
    """
    为整个分镜脚本批量选图。

    每个分镜独立选图，同一张图可以被多个分镜选中（允许重复使用）。

    Args:
        images_b64: 所有参考图（共享给所有分镜）
        prompts:    每个分镜的 prompt 列表

    Returns:
        与 prompts 等长的 ImageSelectResult 列表
    """
    results = []
    logger.info(
        "批量选图 | {} 张参考图 → {} 个分镜",
        len(images_b64), len(prompts),
    )
    for idx, prompt in enumerate(prompts):
        result = select_best_image(images_b64, prompt, shot_index=idx + 1)
        results.append(result)
        logger.debug(
            "分镜#{} → 选择图片#{} (method={}, score={:.4f})",
            idx + 1, result.selected_index + 1,
            result.method, result.best_score,
        )
    return results


# ── Prompt 预处理（提取匹配关键词）──────────────────────────────────────────────

def _extract_match_prompt(prompt: str) -> str:
    """
    从完整 prompt 中提取用于图文匹配的关键词。

    去掉写实前缀（photorealistic, ultra-realistic 等通用词），
    保留真正描述商品场景、角度、动作的核心描述。
    这样 CLIP 能更精准匹配图片内容，而不是被通用质量词干扰。

    示例：
      输入: "photorealistic, 8K UHD, sharp focus, a perfume bottle on white background,
             rotating slowly, side view, horizontal composition..."
      输出: "a perfume bottle on white background, rotating slowly, side view"
    """
    # 常见的质量/风格前缀词（需要过滤）
    _QUALITY_PREFIXES = {
        "photorealistic", "ultra-realistic", "hyperrealistic",
        "8k uhd", "8k", "4k", "sharp focus", "high detail",
        "professional photography", "cinematic lighting",
        "raw photo", "dslr quality", "studio lighting",
        "horizontal composition", "vertical video composition",
        "portrait orientation", "square format composition",
        "slow 360-degree rotation", "centered product",
        "cinematic widescreen", "mobile-first framing",
        "smooth upward camera movement", "slow gentle rotation",
        "e-commerce main image standard",
        "standard product photography composition",
        "portrait product photography", "vertical composition",
    }

    # 按逗号分割，过滤掉质量词，保留有意义的描述片段
    parts = [p.strip() for p in prompt.split(",")]
    meaningful = []
    for part in parts:
        low = part.lower()
        if any(prefix in low for prefix in _QUALITY_PREFIXES):
            continue
        if len(part) > 3:  # 过滤太短的碎片
            meaningful.append(part)

    result = ", ".join(meaningful[:8])  # 最多保留 8 段，避免超 CLIP token 限制
    return result if result else prompt[:200]
