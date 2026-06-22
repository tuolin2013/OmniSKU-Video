#!/usr/bin/env python3
"""
test_client.py — 模拟客户端请求的集成测试脚本

测试覆盖：
  1. 健康检查（等待服务就绪）
  2. 同步单分镜生成（t2v）
  3. 同步分镜脚本批量生成（多分镜，返回 ZIP）
  4. 异步单分镜提交 + 轮询 + 下载
  5. 异步分镜脚本提交 + 轮询 + 下载

用法：
  pip install requests
  python test_client.py                          # 全量测试（默认 low-cost 参数）
  python test_client.py --url http://host:8000   # 指定服务地址
  python test_client.py --only sync              # 只跑同步测试
  python test_client.py --only async             # 只跑异步测试
  python test_client.py --steps 5                # 推理步数（越小越快，用于快速冒烟）
"""

import argparse
import base64
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

# ── 默认配置 ──────────────────────────────────────────────────────────────────
BASE_URL      = "http://localhost:8000/api/v1"
POLL_INTERVAL = 5      # 秒
TIMEOUT_ASYNC = 3600   # 秒，异步任务最长等待
TIMEOUT_SYNC  = 900    # 秒，同步请求超时


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _ok(msg: str):
    print(f"  ✅ {msg}")

def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)

def _info(msg: str):
    print(f"  ℹ  {msg}")

def _header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _make_tiny_image_b64() -> str:
    """
    生成一张 64×64 纯蓝色 PNG 并返回 base64，
    用于在无真实商品图时模拟参考图上传。
    需要 Pillow（requirements.txt 已包含）。
    """
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(30, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── 1. 等待服务就绪 ───────────────────────────────────────────────────────────

def wait_for_ready(base_url: str, max_wait: int = 600) -> None:
    _header("等待服务就绪")
    deadline = time.time() + max_wait
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.ok:
                data = r.json()
                if data.get("model_loaded"):
                    _ok(f"服务就绪（尝试 #{attempt}）")
                    return
                else:
                    _info(f"模型仍在加载，等待中...（尝试 #{attempt}）")
            else:
                _info(f"HTTP {r.status_code}，等待中...")
        except requests.ConnectionError:
            _info(f"连接失败，等待中...（尝试 #{attempt}）")
        time.sleep(10)
    _fail(f"服务在 {max_wait}s 内未就绪")


# ── 2. 健康检查 ───────────────────────────────────────────────────────────────

def test_health(base_url: str) -> None:
    _header("测试 1：健康检查 GET /health")
    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, f"期望 200，实际 {r.status_code}"
    data = r.json()
    assert "status" in data,        "响应缺少 status 字段"
    assert "model_loaded" in data,  "响应缺少 model_loaded 字段"
    _ok(f"status={data['status']}  model_loaded={data['model_loaded']}")


# ── 3. 同步单分镜（文生视频） ─────────────────────────────────────────────────

def test_sync_single_t2v(base_url: str, steps: int) -> None:
    _header(f"测试 2：同步单分镜 POST /generate（t2v, steps={steps}）")
    payload = {
        "prompt": "luxury tea product rotating on white pedestal, studio lighting",
        "num_frames": 9,
        "num_inference_steps": steps,
        "height": 480,
        "width": 704,
        "fps": 24,
        "fast": False,
    }
    _info(f"请求中（timeout={TIMEOUT_SYNC}s）...")
    r = requests.post(f"{base_url}/generate", json=payload, timeout=TIMEOUT_SYNC)
    assert r.status_code == 200, f"期望 200，实际 {r.status_code}\n{r.text[:500]}"
    ct = r.headers.get("Content-Type", "")
    assert "video" in ct or "octet-stream" in ct, f"期望视频 Content-Type，实际: {ct}"
    size_kb = len(r.content) / 1024
    _ok(f"返回视频 {size_kb:.1f} KB，Content-Type={ct}")

    out = Path("test_output")
    out.mkdir(exist_ok=True)
    (out / "single_t2v.mp4").write_bytes(r.content)
    _ok(f"已保存到 test_output/single_t2v.mp4")


# ── 4. 同步单分镜（图生视频） ─────────────────────────────────────────────────

def test_sync_single_i2v(base_url: str, steps: int, ref_b64: str) -> None:
    _header(f"测试 3：同步单分镜 POST /generate（i2v, steps={steps}）")
    payload = {
        "prompt": "product bottle slowly rotating, cinematic soft lighting",
        "reference_images": [ref_b64],
        "num_frames": 9,
        "num_inference_steps": steps,
        "height": 480,
        "width": 704,
        "fps": 24,
        "fast": False,
        "background_style": "gradient",
    }
    _info(f"请求中（timeout={TIMEOUT_SYNC}s）...")
    r = requests.post(f"{base_url}/generate", json=payload, timeout=TIMEOUT_SYNC)
    assert r.status_code == 200, f"期望 200，实际 {r.status_code}\n{r.text[:500]}"
    size_kb = len(r.content) / 1024
    _ok(f"返回视频 {size_kb:.1f} KB")

    out = Path("test_output")
    out.mkdir(exist_ok=True)
    (out / "single_i2v.mp4").write_bytes(r.content)
    _ok(f"已保存到 test_output/single_i2v.mp4")


# ── 5. 同步分镜脚本（多分镜，返回 ZIP） ──────────────────────────────────────

def test_sync_storyboard(base_url: str, steps: int, ref_b64: str) -> None:
    _header(f"测试 4：同步分镜脚本 POST /generate/storyboard（2 分镜, steps={steps}）")
    shots = [
        {
            "prompt": "luxury product on white background, clean studio shot",
            "num_frames": 9,
            "num_inference_steps": steps,
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": False,
        },
        {
            "prompt": "product close-up detail shot, soft lighting, premium feel",
            "reference_images": [ref_b64],
            "num_frames": 9,
            "num_inference_steps": steps,
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": False,
            "background_style": "white",
        },
    ]
    _info(f"请求中（timeout={TIMEOUT_SYNC}s）...")
    r = requests.post(
        f"{base_url}/generate/storyboard",
        json={"shots": shots},
        timeout=TIMEOUT_SYNC,
    )
    assert r.status_code == 200, f"期望 200，实际 {r.status_code}\n{r.text[:500]}"
    ct = r.headers.get("Content-Type", "")
    assert "zip" in ct, f"期望 ZIP Content-Type，实际: {ct}"

    out = Path("test_output")
    out.mkdir(exist_ok=True)
    zip_path = out / "storyboard.zip"
    zip_path.write_bytes(r.content)
    _ok(f"ZIP 已下载 ({len(r.content)/1024:.1f} KB)")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out / "storyboard")
        names = zf.namelist()
    _ok(f"ZIP 内包含 {len(names)} 个文件: {names}")
    assert len(names) == 2, f"期望 2 个 MP4，实际 {len(names)}"


# ── 6. 异步单分镜 ─────────────────────────────────────────────────────────────

def test_async_single(base_url: str, steps: int) -> None:
    _header(f"测试 5：异步单分镜 POST /generate/async（steps={steps}）")
    payload = {
        "prompt": "sneaker product rotating on white platform, professional photography",
        "num_frames": 9,
        "num_inference_steps": steps,
        "height": 480,
        "width": 704,
        "fps": 24,
        "fast": False,
    }

    # 提交
    r = requests.post(f"{base_url}/generate/async", json=payload, timeout=30)
    assert r.status_code == 200, f"提交失败 {r.status_code}\n{r.text[:500]}"
    data = r.json()
    task_id = data["task_id"]
    _ok(f"任务提交成功 | task_id={task_id}")

    # 轮询
    task_data = _poll_task(base_url, task_id)

    # 下载
    _download_task(base_url, task_id, "async_single.mp4")


# ── 7. 异步分镜脚本 ───────────────────────────────────────────────────────────

def test_async_storyboard(base_url: str, steps: int, ref_b64: str) -> None:
    _header(f"测试 6：异步分镜脚本 POST /generate/storyboard/async（2 分镜, steps={steps}）")
    shots = [
        {
            "prompt": "luxury watch on marble surface, product photography",
            "num_frames": 9,
            "num_inference_steps": steps,
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": False,
        },
        {
            "prompt": "watch detail close-up, reflective surface, studio light",
            "reference_images": [ref_b64],
            "num_frames": 9,
            "num_inference_steps": steps,
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": False,
            "background_style": "dark",
        },
    ]

    # 提交
    r = requests.post(
        f"{base_url}/generate/storyboard/async",
        json={"shots": shots},
        timeout=30,
    )
    assert r.status_code == 200, f"提交失败 {r.status_code}\n{r.text[:500]}"
    data = r.json()
    task_id = data["task_id"]
    _ok(f"任务提交成功 | task_id={task_id}")

    # 轮询
    _poll_task(base_url, task_id)

    # 下载
    _download_task(base_url, task_id, "async_storyboard.zip", is_zip=True)


# ── 轮询 + 下载公用函数 ───────────────────────────────────────────────────────

def _poll_task(base_url: str, task_id: str) -> dict:
    """轮询直到 done 或 failed"""
    _info(f"开始轮询（间隔 {POLL_INTERVAL}s，最长 {TIMEOUT_ASYNC}s）...")
    deadline = time.time() + TIMEOUT_ASYNC
    while time.time() < deadline:
        r = requests.get(f"{base_url}/tasks/{task_id}", timeout=10)
        assert r.status_code == 200, f"查询任务失败 {r.status_code}"
        data = r.json()
        status   = data.get("status", "unknown")
        progress = data.get("progress", 0)
        done_n   = data.get("done")
        total_n  = data.get("total")

        if total_n:
            _info(f"status={status}  progress={progress}%  ({done_n}/{total_n})")
        else:
            _info(f"status={status}  progress={progress}%")

        if status == "done":
            _ok("任务完成")
            return data
        if status == "failed":
            _fail(f"任务失败: {data.get('error')}")

        time.sleep(POLL_INTERVAL)

    _fail(f"任务在 {TIMEOUT_ASYNC}s 内未完成")


def _download_task(
    base_url: str,
    task_id: str,
    filename: str,
    is_zip: bool = False,
) -> None:
    """下载并验证任务结果"""
    _info(f"下载结果...")
    r = requests.get(f"{base_url}/tasks/{task_id}/download", timeout=120)
    assert r.status_code == 200, f"下载失败 {r.status_code}\n{r.text[:300]}"

    out = Path("test_output")
    out.mkdir(exist_ok=True)
    path = out / filename
    path.write_bytes(r.content)
    size_kb = len(r.content) / 1024
    _ok(f"已保存 {filename} ({size_kb:.1f} KB)")

    if is_zip:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        _ok(f"ZIP 内包含: {names}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LTX-Video 服务集成测试")
    parser.add_argument("--url",   default="http://localhost:8000/api/v1",
                        help="服务 base URL（含 /api/v1）")
    parser.add_argument("--steps", type=int, default=5,
                        help="推理步数（越少越快，用于冒烟测试，默认 5）")
    parser.add_argument("--only",  choices=["sync", "async", "health"],
                        help="只运行指定分组的测试")
    parser.add_argument("--no-wait", action="store_true",
                        help="跳过等待服务就绪（服务已知就绪时使用）")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    steps    = args.steps
    only     = args.only

    print(f"\n{'='*60}")
    print(f"  LTX-Video 集成测试")
    print(f"  服务地址: {base_url}")
    print(f"  推理步数: {steps}")
    print(f"  测试范围: {only or '全量'}")
    print(f"{'='*60}")

    # 等待服务
    if not args.no_wait:
        wait_for_ready(base_url)

    # 准备参考图（64×64 蓝色色块，真实场景请替换为商品实拍图）
    ref_b64 = _make_tiny_image_b64()
    _info("已生成模拟参考图（64×64 PNG）")

    passed = 0
    failed = 0

    def run(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  ❌ 测试失败: {e}")
            failed += 1

    # ── 健康检查（始终执行） ────────────────────────────────────────
    run("health", lambda: test_health(base_url))

    # ── 同步测试 ───────────────────────────────────────────────────
    if only in (None, "sync"):
        run("sync_t2v",        lambda: test_sync_single_t2v(base_url, steps))
        run("sync_i2v",        lambda: test_sync_single_i2v(base_url, steps, ref_b64))
        run("sync_storyboard", lambda: test_sync_storyboard(base_url, steps, ref_b64))

    # ── 异步测试 ───────────────────────────────────────────────────
    if only in (None, "async"):
        run("async_single",     lambda: test_async_single(base_url, steps))
        run("async_storyboard", lambda: test_async_storyboard(base_url, steps, ref_b64))

    # ── 汇总 ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  测试结果：{passed} 通过  {failed} 失败")
    print(f"  输出文件：test_output/")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
