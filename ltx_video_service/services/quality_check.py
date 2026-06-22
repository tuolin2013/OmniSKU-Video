"""
services/quality_check.py — 视频质量自动校验（P2-1）

生产环境中 AI 生成的视频可能出现以下质量问题，需要自动检测并触发重试：

1. 黑帧（Black Frame）
   - 原因：模型推理异常、显存不足导致帧生成失败
   - 检测：帧平均亮度 < 阈值（15/255）

2. 纯白/过曝帧（Overexposed Frame）
   - 原因：guidance_scale 过高、CFG 梯度爆炸
   - 检测：帧平均亮度 > 阈值（245/255）

3. 静态视频（Static/Frozen Video）
   - 原因：i2v 模型未能产生运动，视频退化为静止图
   - 检测：相邻帧 MSE 均值 < 阈值（图像完全没有变化）

4. 高噪点/伪影（High Noise/Artifacts）
   - 原因：步数不足、decode_timestep 参数异常
   - 检测：帧间 MSE 方差极高（噪点导致帧间剧烈抖动）

5. 商品主体缺失（Product Missing）
   - 仅适用于 i2v 模式
   - 原因：抠图后商品区域太小，模型未能保留商品
   - 检测：第一帧与参考帧相似度过低（SSIM < 阈值）

设计原则：
  - 所有检测在 CPU / NumPy 层完成，无需 GPU
  - 每项检测独立，失败信息详细记录便于排查
  - 返回 QualityReport，调用方决定是否重试
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image


@dataclass
class QualityIssue:
    """单个质量问题描述"""
    code: str           # 问题代码（便于程序判断）
    message: str        # 人类可读描述
    severity: str       # "warning" 或 "error"
    value: float = 0.0  # 检测到的实际值


@dataclass
class QualityReport:
    """视频质量检测报告"""
    passed: bool                            # True = 质量合格，可交付
    issues: List[QualityIssue] = field(default_factory=list)
    frame_count: int = 0
    avg_brightness: float = 0.0
    avg_motion: float = 0.0                 # 相邻帧平均 MSE
    motion_variance: float = 0.0            # 帧间 MSE 方差（噪点指标）

    def error_codes(self) -> List[str]:
        return [i.code for i in self.issues if i.severity == "error"]

    def summary(self) -> str:
        if self.passed:
            return (
                f"✅ 质量合格 | 帧数={self.frame_count} "
                f"亮度={self.avg_brightness:.1f} 运动={self.avg_motion:.2f}"
            )
        codes = ", ".join(self.error_codes())
        return f"❌ 质量不合格 | 问题: {codes}"


# ── 检测阈值（可通过环境变量覆盖）────────────────────────────────────────

class QualityThresholds:
    BLACK_FRAME_BRIGHTNESS   = 15.0    # 帧平均亮度低于此值判为黑帧（0-255）
    WHITE_FRAME_BRIGHTNESS   = 245.0   # 帧平均亮度高于此值判为过曝
    STATIC_VIDEO_MOTION      = 0.5     # 帧间平均 MSE 低于此值判为静态视频
    HIGH_NOISE_VARIANCE      = 150000.0  # 帧间 MSE 方差高于此值判为高噪点
    # 注：Wan 2.2 在低步数（<10）时帧间方差正常在 5000-80000，
    # 150000 以上才代表真正的噪点/伪影问题（如显存溢出导致的花屏）。
    BLACK_FRAME_RATIO        = 0.15    # 黑帧比例超过此值才报 error（允许少量黑帧过渡）
    WHITE_FRAME_RATIO        = 0.10    # 过曝帧比例阈值
    MIN_FRAMES               = 8      # 最少帧数，低于此值认为生成不完整


# ── 核心检测逻辑 ──────────────────────────────────────────────────────────

def check_video_quality(
    frames: List[Image.Image],
    is_i2v: bool = False,
    reference_frame: Optional[Image.Image] = None,
) -> QualityReport:
    """
    对生成的视频帧序列做全面质量检测。

    Args:
        frames:          生成的帧列表（PIL.Image，RGB 模式）
        is_i2v:          是否为图生视频模式（启用商品主体检测）
        reference_frame: i2v 模式下的参考帧（用于主体相似度检测）

    Returns:
        QualityReport，passed=True 表示质量合格
    """
    issues: List[QualityIssue] = []

    if frames is None or len(frames) == 0:
        return QualityReport(
            passed=False,
            issues=[QualityIssue("NO_FRAMES", "未生成任何帧", "error")],
        )

    # diffusers 新版本可能直接返回 numpy array（值域 [0,1]），统一转为 PIL Image 列表
    if isinstance(frames, np.ndarray):
        # shape: (N, H, W, C)，值域 [0,1] → [0,255]
        frames_np = (frames * 255).clip(0, 255).astype(np.uint8)
        frames = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]

    n = len(frames)
    t = QualityThresholds

    # ── 帧数检查 ────────────────────────────────────────────────────
    if n < t.MIN_FRAMES:
        issues.append(QualityIssue(
            "TOO_FEW_FRAMES",
            f"生成帧数过少: {n} < {t.MIN_FRAMES}（可能生成中断）",
            "error", value=float(n),
        ))

    # ── 转换为 numpy 数组（float32，加速批量计算）──────────────────
    try:
        arr = np.stack([np.array(f, dtype=np.float32) for f in frames], axis=0)
        # arr shape: (N, H, W, 3)
    except Exception as e:
        return QualityReport(
            passed=False,
            issues=[QualityIssue("ARRAY_CONV_ERROR", f"帧转换失败: {e}", "error")],
        )

    # ── 亮度分析 ────────────────────────────────────────────────────
    # 每帧亮度 = 0.299R + 0.587G + 0.114B（感知亮度）
    brightness = (
        0.299 * arr[:, :, :, 0]
        + 0.587 * arr[:, :, :, 1]
        + 0.114 * arr[:, :, :, 2]
    ).mean(axis=(1, 2))   # shape: (N,)

    avg_brightness = float(brightness.mean())

    # 黑帧检测
    black_frames = int((brightness < t.BLACK_FRAME_BRIGHTNESS).sum())
    black_ratio = black_frames / n
    if black_ratio > t.BLACK_FRAME_RATIO:
        issues.append(QualityIssue(
            "BLACK_FRAMES",
            f"黑帧比例过高: {black_frames}/{n} ({black_ratio:.1%})，可能是推理异常或显存不足",
            "error", value=black_ratio,
        ))
    elif black_frames > 0:
        issues.append(QualityIssue(
            "FEW_BLACK_FRAMES",
            f"少量黑帧: {black_frames}/{n}，可能是场景过渡，建议检查",
            "warning", value=float(black_frames),
        ))

    # 过曝帧检测
    white_frames = int((brightness > t.WHITE_FRAME_BRIGHTNESS).sum())
    white_ratio = white_frames / n
    if white_ratio > t.WHITE_FRAME_RATIO:
        issues.append(QualityIssue(
            "OVEREXPOSED_FRAMES",
            f"过曝帧比例过高: {white_frames}/{n} ({white_ratio:.1%})，"
            "建议降低 guidance_scale 或增加负向词 overexposed",
            "error", value=white_ratio,
        ))

    # ── 运动分析（帧间 MSE）──────────────────────────────────────────
    avg_motion = 0.0
    motion_variance = 0.0
    if n >= 2:
        # 相邻帧差的 MSE
        frame_diffs = arr[1:] - arr[:-1]   # (N-1, H, W, 3)
        mse_per_pair = (frame_diffs ** 2).mean(axis=(1, 2, 3))  # (N-1,)
        avg_motion = float(mse_per_pair.mean())
        motion_variance = float(mse_per_pair.var())

        # 静态视频检测
        if avg_motion < t.STATIC_VIDEO_MOTION:
            issues.append(QualityIssue(
                "STATIC_VIDEO",
                f"视频几乎无运动（帧间 MSE={avg_motion:.3f} < {t.STATIC_VIDEO_MOTION}），"
                "图生视频退化为静止图，建议增加运动相关 prompt 词汇",
                "error", value=avg_motion,
            ))

        # 高噪点检测（帧间 MSE 方差极高）
        if motion_variance > t.HIGH_NOISE_VARIANCE:
            issues.append(QualityIssue(
                "HIGH_NOISE",
                f"帧间 MSE 方差过高（{motion_variance:.0f} > {t.HIGH_NOISE_VARIANCE}），"
                "可能存在严重噪点或伪影，建议增加推理步数或检查参数",
                "error", value=motion_variance,
            ))

    # ── i2v 商品主体相似度检测 ───────────────────────────────────────
    if is_i2v and reference_frame is not None and n > 0:
        try:
            ref_arr = np.array(
                reference_frame.resize(frames[0].size, Image.LANCZOS),
                dtype=np.float32,
            )
            first_arr = arr[0]
            # 简单 NMSE（归一化 MSE）作为相似度指标
            nmse = float(((ref_arr - first_arr) ** 2).mean()) / (255.0 ** 2)
            if nmse > 0.15:   # 差异超过 15% 认为商品主体丢失
                issues.append(QualityIssue(
                    "PRODUCT_SHAPE_LOST",
                    f"首帧与参考图差异过大（NMSE={nmse:.3f}），"
                    "商品主体可能在生成中变形或丢失",
                    "warning", value=nmse,
                ))
        except Exception as e:
            logger.warning("商品相似度检测失败: {}", e)

    # ── 汇总 ─────────────────────────────────────────────────────────
    has_error = any(i.severity == "error" for i in issues)
    report = QualityReport(
        passed=not has_error,
        issues=issues,
        frame_count=n,
        avg_brightness=avg_brightness,
        avg_motion=avg_motion,
        motion_variance=motion_variance,
    )
    logger.info("质量检测 | {}", report.summary())
    for issue in issues:
        log_fn = logger.warning if issue.severity == "warning" else logger.error
        log_fn("质量问题 [{}] {}", issue.code, issue.message)

    return report


# ── 带自动重试的生成包装器 ────────────────────────────────────────────────

def should_retry(report: QualityReport, attempt: int, max_attempts: int = 3) -> Tuple[bool, str]:
    """
    根据质量报告决定是否重试。

    策略：
      - 黑帧、过曝、无帧 → 重试（随机种子换一个）
      - 静态视频 → 重试（增加运动 prompt）
      - 高噪点 → 重试（增加推理步数建议）
      - 商品变形（warning）→ 不重试（warning 级别，交付但记录）
      - 已达最大重试次数 → 不再重试，交付当前结果并告警

    Returns:
        (should_retry: bool, reason: str)
    """
    if attempt >= max_attempts:
        return False, f"已达最大重试次数 {max_attempts}，交付当前结果"

    if report.passed:
        return False, "质量合格，无需重试"

    error_codes = report.error_codes()
    retry_codes = {"BLACK_FRAMES", "OVEREXPOSED_FRAMES", "NO_FRAMES",
                   "TOO_FEW_FRAMES", "STATIC_VIDEO", "HIGH_NOISE"}

    needs_retry = bool(set(error_codes) & retry_codes)
    if needs_retry:
        return True, f"检测到质量问题 {error_codes}，触发第 {attempt + 1} 次重试"

    return False, f"质量问题 {error_codes} 不触发重试（非致命错误）"
