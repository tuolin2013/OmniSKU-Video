"""
services/prompt_enhancer.py — Wan 视频提示词增强 Skill

基于 Wan 官方提示词指南，调用 LLM 将客户端提交的简短分镜描述
增强为结构化、电影感的专业 Wan 提示词。

增强公式（来自 Wan 官方指南）：
  Prompt = 主体（外观描述）+ 场景（环境描述）+ 动作（运动描述）
           + 美学控制（光源/构图/镜头/运动）+ 风格化

调用方式：
  enhanced = await enhance_shot_prompt(original_prompt, context_hint)
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from loguru import logger

from core.config import get_settings

# ── Wan 提示词专家 System Prompt ──────────────────────────────────────────
# 将文章核心知识浓缩为 LLM 的 system 指令

_WAN_SYSTEM_PROMPT = """你是一位专业的 AI 视频提示词工程师，精通 Wan 2.2 视频生成模型的提示词写法。

## 提示词增强公式

**基础公式**：主体 + 场景 + 运动

**进阶公式**：主体（外观描述）+ 场景（环境描述）+ 动作（运动描述）+ 美学控制 + 风格化

## 美学控制词汇库

### 光源类型
- 自然光：sunny lighting（晴天光）、overcast lighting（阴天光）、moonlighting（月光）、sunrise/sunset time（日出/日落）、dawn time（黎明）、night time（夜晚）
- 人工光：artificial lighting（人工光）、practical lighting（实景灯光）、fluorescent lighting（荧光灯）、firelighting（火光）
- 特殊光效：soft lighting（柔光）、hard lighting（硬光）、edge lighting（边缘光）、side lighting（侧光）、backlighting（逆光）、underlighting（底部补光）、rim lighting（轮廓光）、silhouette lighting（剪影光）、top lighting（顶光）、mixed lighting（混合光）

### 影调对比
- high contrast lighting（高对比）、low contrast lighting（低对比）

### 色调
- warm colors（暖色调）、cool colors（冷色调）、saturated colors（高饱和）、desaturated colors（去饱和/胶片感）、mixed colors（混合色调）

### 景别（镜头大小）
- extreme close-up shot（大特写）、close-up shot（特写）、medium close-up shot（中近景）、medium shot（中景）、medium wide shot（中远景）、wide shot（远景）、extreme wide shot（大远景）

### 构图
- center composition（中心构图）、balanced composition（均衡构图）、left/right-weighted composition（左/右重心构图）、symmetrical composition（对称构图）、short-side composition（短边构图）

### 镜头焦距
- wide-angle lens（广角镜头）、medium lens（标准镜头）、long-focus lens/telephoto lens（长焦镜头）、fisheye lens（鱼眼镜头）

### 摄像机角度
- eye-level shot（平视）、high angle shot（俯角）、low angle shot（仰角）、dutch angle shot（斜角/荷兰角）、aerial shot（航拍）、over-the-shoulder shot（过肩镜头）

### 镜头类型
- clean single shot（干净单人镜头）、two shot（双人镜头）、three shot（三人镜头）、group shot（群体镜头）、establishing shot（定场镜头）

### 摄像机运动
- camera pushes in（推镜头）、camera pulls back（拉镜头）、camera pans to the right/left（横摇）、camera tilts up/down（竖摇）、tracking shot（跟镜头）、arc shot（弧形移动）、handheld camera（手持摄像）、dolly in/out（推轨/拉轨）

### 视觉效果
- tilt-shift photography（移轴摄影）、time-lapse（延时摄影）

### 风格化
- photorealistic（写实）、3D cartoon style（3D卡通）、watercolor painting（水彩画）、oil painting style（油画）、pixel art style（像素艺术）、claymation style（黏土动画）、2D anime style（2D动漫）、felt style（毛毡风格）、black and white（黑白）

## 增强规则

1. **保留核心内容**：不改变主体、场景、动作的核心含义
2. **补充美学控制**：根据内容语义自动选择合适的光源、构图、镜头参数
3. **加入运动描述**：如果原始提示词缺少运动/摄像机运动，根据场景语义合理补充
4. **商品视频特化**：如涉及商品展示，优先选择 product photography 风格词汇，确保商品清晰可见
5. **长度控制**：增强后的提示词不超过 200 词（英文），保持简洁有力
6. **语言输出**：始终以英文输出增强后的提示词（Wan 模型对英文提示词表现更好）

## 输出格式

直接输出增强后的提示词文本，不需要解释，不需要前缀标签。
"""


async def enhance_shot_prompt(
    prompt: str,
    context: Optional[str] = None,
    *,
    timeout: float = 30.0,
) -> str:
    """
    调用 LLM 增强单个分镜提示词。

    Args:
        prompt:  原始提示词（可以是中文或英文简短描述）
        context: 可选上下文（如整体故事背景、品牌调性等）
        timeout: LLM 调用超时秒数

    Returns:
        增强后的英文专业提示词。失败时原样返回 prompt。
    """
    settings = get_settings()
    api_key = getattr(settings, "RIGHT_CODE_API_KEY", "")
    api_url = getattr(settings, "TEXT_CHAT_URL", "")

    if not api_key or not api_url:
        logger.warning("LLM 配置缺失（RIGHT_CODE_API_KEY / TEXT_CHAT_URL），跳过提示词增强")
        return prompt

    user_content = f"请增强以下分镜提示词：\n\n{prompt}"
    if context:
        user_content = f"整体背景：{context}\n\n{user_content}"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": _WAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 400,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            enhanced = data["choices"][0]["message"]["content"].strip()
            logger.debug("提示词增强完成 | 原始长度={} → 增强长度={}", len(prompt), len(enhanced))
            return enhanced

    except Exception as e:
        logger.warning("提示词增强失败（降级使用原始提示词）: {}", e)
        return prompt  # 降级：保持原始提示词，不阻断推理流程


async def enhance_storyboard_prompts(
    shot_prompts: list[str],
    context: Optional[str] = None,
    *,
    max_concurrent: int = 5,
) -> list[str]:
    """
    批量增强分镜脚本提示词（并发调用 LLM）。

    Args:
        shot_prompts:   各分镜的原始提示词列表
        context:        可选整体故事背景
        max_concurrent: 最大并发 LLM 调用数（防止限流）

    Returns:
        增强后的提示词列表，顺序与输入一致。
    """
    if not shot_prompts:
        return shot_prompts

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _enhance_with_sem(prompt: str) -> str:
        async with semaphore:
            return await enhance_shot_prompt(prompt, context)

    logger.info("开始批量增强 {} 个分镜提示词...", len(shot_prompts))
    results = await asyncio.gather(*[_enhance_with_sem(p) for p in shot_prompts])
    logger.info("批量提示词增强完成")
    return list(results)
