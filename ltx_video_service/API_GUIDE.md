# LTX-Video 分镜脚本服务 — 客户端请求文档

> 服务地址默认：`http://localhost:8000`  
> 所有业务端点均带版本前缀 `/api/v1`  
> 文档版本：**v3.0.0**（Wan 2.2 + LTX-Video 双模型架构）

---

## 端点总览

| 方法 | 路径 | 功能 | 响应类型 |
|------|------|------|----------|
| `GET`  | `/api/v1/health` | 健康检查 | `application/json` |
| `POST` | `/api/v1/generate/storyboard` | 分镜脚本批量生成（同步） | `application/zip` |
| `POST` | `/api/v1/generate` | 单分镜生成（向后兼容，同步） | `video/mp4` |
| `POST` | `/api/v1/generate/async` | 单分镜异步提交 | `application/json` |
| `POST` | `/api/v1/generate/storyboard/async` | 分镜脚本批量异步提交 | `application/json` |
| `GET`  | `/api/v1/tasks/{task_id}` | 查询异步任务状态 | `application/json` |
| `GET`  | `/api/v1/tasks/{task_id}/download` | 下载异步任务结果 | `video/mp4` 或 `application/zip` |

---

## 模型架构

服务集成两套模型，通过请求字段 `fast` 切换：

```
fast=false（默认）— 正式出片模式
  无参考图  →  Wan2.2-T2V-A14B（文生视频，14B 参数，顶级质量，~60s/clip）
  有参考图  →  Wan2.2-TI2V-5B（文本+图像生视频，5B 参数，轻量高效）
              TI2V 不可用时自动回退到 Wan2.2-I2V-A14B

fast=true — 快速预览模式
  无参考图  →  LTX-Video LTXPipeline（文生视频，~8s/clip，用于构图草稿确认）
  有参考图  →  LTX-Video LTXImageToVideoPipeline（图生视频）
```

> Wan 2.2 不可用时，服务自动回退到 LTX-Video。

---

## 自动路由逻辑（图生视频 vs 文生视频）

每个分镜根据 `reference_images` 字段自动选择推理模式：

```
reference_images 未填写 / null  →  文生视频（T2V）
reference_images 传入 1 张      →  图生视频（I2V / TI2V），直接使用该图
reference_images 传入多张       →  图生视频（I2V / TI2V）
                                    ↓
                               CLIP 自动从多张图中选出与当前分镜
                               prompt 最匹配的一张
                               （如"侧面展示"会选侧面图，
                                 "特写材质"会选细节图）
                                    ↓
                               选定图片经完整增强流水线处理：
                               背景移除 → 色彩校正 → 锐化
                               → 等比缩放 → 投影合成 → 背景合成
                               商品形态不变形、不裁切
```

---

## 写实增强（自动注入）

提交请求时，服务会自动对提示词做如下增强，**无需客户端手动处理**：

- **正向提示词**：头部注入写实风格前缀（`photorealistic, ultra-realistic, 8K UHD…`），尾部追加宽高比专属构图/运镜语
- **负向提示词**：合并系统级排除词（卡通/动画/画质劣化/商品变形专用词），客户端自定义 `negative_prompt` 将追加在系统词之后

---

## 视频后处理（正式出片模式自动启用）

`fast=false` 正式出片时，服务在推理完成后自动执行：

1. **Real-ESRGAN x2plus 视频超分**：逐帧 2× 放大修复压缩伪影/模糊，再 resize 回目标分辨率，画质显著提升
2. **LUT 色彩分级**：S 曲线色调映射（暗部提亮 + 高光压制 + 轻微暖调），使商品视频质感更接近专业摄影

---

## 宽高比自适应

服务根据客户端传入的 `width / height` 自动识别宽高比（容差 ±5%），并将分辨率对齐到电商标准：

| 宽高比 | 标准分辨率（宽×高） | 推荐 fps | 适用场景 |
|--------|---------------------|----------|----------|
| 16:9 | 1280 × 720 | 24 | PC 详情页 / 品牌旗舰视频 |
| 9:16 | 576 × 1024 | 30 | 短视频 / 直播 / 移动端全屏 |
| 1:1 | 768 × 768 | 30 | 天猫/淘宝/亚马逊主图 |
| 4:3 | 960 × 720 | 24 | 传统详情页 |
| 3:4 | 576 × 768 | 30 | 小红书 / Pinterest |

> 不匹配任何标准比例时，原样使用客户端传入的尺寸（确保是 32 的倍数即可）。  
> 用户显式传入非 24 的 fps 时，服务尊重用户设定，不覆盖。

---

## 请求字段说明

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `prompt` | `string` | **必填** | 1–2000 字符 | 正向提示词，描述该分镜画面内容（服务自动在头部注入写实前缀） |
| `negative_prompt` | `string` | 系统默认（含卡通/变形排除词） | ≤ 2000 字符 | 负向提示词（将与系统默认词合并，无需手动填写基础排除词） |
| `reference_images` | `string[] \| null` | `null` | base64 编码，最多 10 张 | 商品实拍参考图列表（JPEG/PNG/WEBP），多张时 CLIP 自动选最匹配的一张 |
| `reference_image` | `string \| null` | `null` | base64 编码 | **[已废弃]** 单图兼容字段，请改用 `reference_images`（自动合并） |
| `num_frames` | `int` | `97` | 9–257，须满足 **4N+1** 或 **8N+1** | 帧数（97≈4s@24fps）。Wan 2.2 要求 4N+1，LTX 要求 8N+1，服务自动对齐 |
| `num_inference_steps` | `int` | `50` | 1–100 | 扩散去噪步数。正式出片推荐 50，预览推荐 20 |
| `height` | `int` | `480` | 256–1280，须为 **32 的倍数** | 视频高度（像素），服务根据宽高比自动对齐到标准分辨率 |
| `width` | `int` | `704` | 256–1280，须为 **32 的倍数** | 视频宽度（像素），服务根据宽高比自动对齐到标准分辨率 |
| `fps` | `int` | `24` | 8–60 | 输出视频帧率（宽高比自适应时推荐值会自动覆盖，除非显式设置） |
| `fast` | `bool` | `false` | — | `false` = 正式出片（Wan 2.2，~60s）；`true` = 快速预览（LTX-Video，~8s） |
| `background_style` | `string` | `"gradient"` | — | 商品背景样式（仅在传入参考图时生效）：`gradient`（径向渐变，默认）/ `white`（纯白）/ `warm`（暖色）/ `dark`（深色） |

> `reference_images` 支持标准 base64 字符串或 data URI（`data:image/jpeg;base64,...`）。  
> **帧数说明**：正式出片推荐 `97`（约 4 秒@24fps）；快速预览推荐 `25`（约 1 秒）。

---

## 1. 健康检查

### 请求

```bash
curl http://localhost:8000/api/v1/health
```

### 响应

```json
{
  "status": "ok",
  "model_loaded": true
}
```

- `model_loaded: false` → 模型仍在加载（Wan 2.2 + LTX-Video 双模型，启动较慢），请稍后重试
- `model_loaded: true`  → 服务就绪，可以提交任务

---

## 2. 分镜脚本接口 `POST /api/v1/generate/storyboard`（同步）

### 请求体结构

```json
{
  "shots": [
    { /* 分镜 1 参数 */ },
    { /* 分镜 2 参数 */ },
    ...
  ]
}
```

- `shots` 数组长度：1–50
- 服务按数组顺序逐镜串行推理
- **响应**：`storyboard_videos.zip`，内含 `shot_001.mp4`、`shot_002.mp4`……

> ⚠️ 同步接口会阻塞直到所有分镜生成完毕再返回，多分镜正式出片耗时较长（每镜 ~60s）。  
> 超过 5 个分镜或需要后台排队时，建议使用 [异步接口](#4-异步任务接口)。

### 示例：混合脚本（分镜1 文生视频 + 分镜2/3 图生视频，多角度参考图）

#### cURL

```bash
#!/usr/bin/env bash
# 将商品实拍图转为 base64（传入多张，服务用 CLIP 自动选最匹配的）
IMG_FRONT_B64=$(base64 -w 0 product_front.jpg)   # Linux
IMG_SIDE_B64=$(base64 -w 0 product_side.jpg)
IMG_DETAIL_B64=$(base64 -w 0 product_detail.jpg)
# macOS 用户请改用：base64 -i product_front.jpg

curl -s -X POST http://localhost:8000/api/v1/generate/storyboard \
  -H "Content-Type: application/json" \
  --max-time 900 \
  -o storyboard_videos.zip \
  -d "{
    \"shots\": [
      {
        \"prompt\": \"Brand logo animation on a dark background, glowing gold letters, cinematic\",
        \"num_frames\": 33,
        \"num_inference_steps\": 25,
        \"height\": 480,
        \"width\": 704,
        \"fps\": 24,
        \"fast\": true
      },
      {
        \"prompt\": \"Luxury perfume bottle emerging from morning mist, product photography, dramatic lighting\",
        \"reference_images\": [\"${IMG_FRONT_B64}\", \"${IMG_SIDE_B64}\", \"${IMG_DETAIL_B64}\"],
        \"num_frames\": 97,
        \"num_inference_steps\": 50,
        \"height\": 480,
        \"width\": 704,
        \"fps\": 24,
        \"fast\": false,
        \"background_style\": \"gradient\"
      },
      {
        \"prompt\": \"Perfume bottle rotating slowly, side angle, soft studio light, white background\",
        \"reference_images\": [\"${IMG_FRONT_B64}\", \"${IMG_SIDE_B64}\", \"${IMG_DETAIL_B64}\"],
        \"num_frames\": 97,
        \"num_inference_steps\": 50,
        \"height\": 480,
        \"width\": 704,
        \"fps\": 24,
        \"fast\": false,
        \"background_style\": \"white\"
      }
    ]
  }"

echo "下载完成，解压中..."
unzip -q storyboard_videos.zip -d storyboard_output/
ls storyboard_output/
# shot_001.mp4  shot_002.mp4  shot_003.mp4
```

#### Python

```python
#!/usr/bin/env python3
"""
示例：提交含 3 个分镜的脚本（1 文生视频预览 + 2 正式出片图生视频），下载 ZIP 并解压。
支持多张参考图，CLIP 自动选最匹配的一张。
"""

import base64
import io
import zipfile
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000/api/v1"


def image_to_b64(path: str) -> str:
    """将本地图片文件编码为 base64 字符串"""
    return base64.b64encode(Path(path).read_bytes()).decode()


def submit_storyboard(shots: list, output_dir: str = "storyboard_output") -> None:
    """
    提交分镜脚本，下载 ZIP 并将每个分镜 MP4 写入 output_dir。
    """
    print(f"提交 {len(shots)} 个分镜...")

    resp = requests.post(
        f"{BASE_URL}/generate/storyboard",
        json={"shots": shots},
        timeout=900,  # 正式出片耗时较长，务必设置足够的超时时间
    )
    resp.raise_for_status()

    # 解压 ZIP 到目标目录
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out)

    mp4_files = sorted(out.glob("shot_*.mp4"))
    print(f"✅ 生成完成，共 {len(mp4_files)} 个视频片段：")
    for f in mp4_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # 多角度参考图（服务用 CLIP 为每个分镜自动选最匹配的一张）
    ref_images = [
        image_to_b64("product_front.jpg"),
        image_to_b64("product_side.jpg"),
        image_to_b64("product_detail.jpg"),
    ]

    shots = [
        # ── 分镜 1：快速预览（文生视频，无参考图）──────────────────────
        {
            "prompt": "Brand logo animation on a dark background, glowing gold letters, cinematic",
            "num_frames": 33,
            "num_inference_steps": 25,
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": True,   # 预览模式，LTX-Video，~8s
        },
        # ── 分镜 2：正式出片（图生视频，多角度参考图，CLIP 自动选）────
        {
            "prompt": (
                "Luxury perfume bottle emerging from morning mist, "
                "product photography, dramatic lighting, 4K"
            ),
            "reference_images": ref_images,
            "num_frames": 97,           # 约 4 秒@24fps
            "num_inference_steps": 50,  # 正式出片质量
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": False,              # 正式出片，Wan 2.2 TI2V，~60s
            "background_style": "gradient",
        },
        # ── 分镜 3：正式出片（图生视频，侧面展示）──────────────────────
        {
            "prompt": "Perfume bottle rotating slowly, side angle, soft studio light, white background",
            "reference_images": ref_images,
            "num_frames": 97,
            "num_inference_steps": 50,
            "height": 480,
            "width": 704,
            "fps": 24,
            "fast": False,
            "background_style": "white",
        },
    ]

    submit_storyboard(shots, output_dir="storyboard_output")
```

---

## 3. 单分镜接口 `POST /api/v1/generate`（向后兼容，同步）

### 3a. 文生视频（无参考图，预览模式）

```bash
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  --max-time 300 \
  -o output.mp4 \
  -d '{
    "prompt": "A luxury sneaker rotating on a white pedestal, studio lighting, 4K",
    "num_frames": 33,
    "num_inference_steps": 25,
    "height": 480,
    "width": 704,
    "fps": 24,
    "fast": true
  }'
```

### 3b. 图生视频（商品实拍参考图，正式出片）

```bash
# 多张参考图（CLIP 自动选最匹配的一张）
IMG_FRONT_B64=$(base64 -w 0 sneaker_front.jpg)
IMG_SIDE_B64=$(base64 -w 0 sneaker_side.jpg)

curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  --max-time 300 \
  -o output.mp4 \
  -d "{
    \"prompt\": \"The sneaker slowly rotates showcasing all angles, cinematic lighting, 4K\",
    \"reference_images\": [\"${IMG_FRONT_B64}\", \"${IMG_SIDE_B64}\"],
    \"num_frames\": 97,
    \"num_inference_steps\": 50,
    \"height\": 480,
    \"width\": 704,
    \"fps\": 24,
    \"fast\": false,
    \"background_style\": \"gradient\"
  }"
```

---

## 4. 异步任务接口

同步接口在正式出片时每镜约 60s，多分镜脚本总等待时间较长，容易超时。  
**异步接口**立即返回 `task_id`，客户端轮询状态后再下载结果，适合前端集成。

### 流程图

```
POST /generate/async  或  POST /generate/storyboard/async
           ↓  立即返回 { task_id, status: "pending" }
    
每 3–5 秒轮询  GET /tasks/{task_id}
           ↓  返回 { status, progress }
    
status = "done"
           ↓  
    GET /tasks/{task_id}/download
           ↓  返回 MP4 或 ZIP
```

---

### 4a. 异步提交单分镜 `POST /api/v1/generate/async`

#### 请求体

与同步接口 `/api/v1/generate` 相同（见 [请求字段说明](#请求字段说明)）。

#### 响应

```json
{
  "task_id": "abc123-...",
  "status": "pending",
  "message": "任务已加入 GPU 队列，预计耗时 ~60s（正式出片）"
}
```

#### cURL 示例

```bash
TASK=$(curl -s -X POST http://localhost:8000/api/v1/generate/async \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A luxury sneaker rotating on a white pedestal, studio lighting, 4K",
    "num_frames": 97,
    "num_inference_steps": 50,
    "height": 480,
    "width": 704,
    "fps": 24,
    "fast": false
  }')

TASK_ID=$(echo $TASK | jq -r '.task_id')
echo "任务 ID: $TASK_ID"
```

---

### 4b. 异步提交分镜脚本 `POST /api/v1/generate/storyboard/async`

#### 请求体

与同步接口 `/api/v1/generate/storyboard` 相同（`{ "shots": [...] }`）。

#### 响应

```json
{
  "task_id": "def456-...",
  "status": "pending",
  "message": "3 个分镜已加入队列，预计总耗时 ~180s"
}
```

---

### 4c. 查询任务状态 `GET /api/v1/tasks/{task_id}`

#### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | `string` | 任务 ID |
| `status` | `string` | 任务状态：`pending` / `running` / `done` / `failed` |
| `progress` | `int` | 进度 0–100 |
| `total` | `int \| null` | 分镜总数（分镜脚本任务） |
| `done` | `int \| null` | 已完成分镜数 |
| `output_path` | `string \| null` | 单分镜结果路径（`done` 后可读） |
| `output_paths` | `string[] \| null` | 分镜脚本结果路径列表（`done` 后可读） |
| `error` | `string \| null` | 失败原因（`status=failed` 时） |

#### 示例响应（进行中）

```json
{
  "task_id": "def456-...",
  "status": "running",
  "progress": 33,
  "total": 3,
  "done": 1,
  "output_path": null,
  "output_paths": null,
  "error": null
}
```

#### cURL 轮询示例

```bash
while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/tasks/$TASK_ID | jq -r '.status')
  PROGRESS=$(curl -s http://localhost:8000/api/v1/tasks/$TASK_ID | jq -r '.progress')
  echo "状态: $STATUS | 进度: $PROGRESS%"
  if [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 5
done
```

---

### 4d. 下载任务结果 `GET /api/v1/tasks/{task_id}/download`

- `status=done` 后调用此接口
- 单分镜任务返回 `video/mp4`
- 分镜脚本任务返回 `application/zip`（内含 `shot_001.mp4`…）
- 文件下载后 60 秒内可重复下载，之后自动清理

#### cURL 下载示例

```bash
curl -s http://localhost:8000/api/v1/tasks/$TASK_ID/download \
  -o result.zip   # 或 result.mp4
```

#### Python 异步轮询完整示例

```python
#!/usr/bin/env python3
"""
示例：提交分镜脚本异步任务，轮询进度，下载并解压结果。
"""

import base64
import io
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000/api/v1"


def image_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def submit_storyboard_async(shots: list) -> str:
    """提交异步分镜脚本任务，返回 task_id"""
    resp = requests.post(
        f"{BASE_URL}/generate/storyboard/async",
        json={"shots": shots},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"✅ 任务已提交 | task_id={data['task_id']} | {data['message']}")
    return data["task_id"]


def wait_for_task(task_id: str, poll_interval: int = 5, timeout: int = 3600) -> dict:
    """轮询任务状态直到完成或超时"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/tasks/{task_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        progress = data.get("progress", 0)
        done = data.get("done", 0)
        total = data.get("total")

        if total:
            print(f"⏳ 状态: {status} | 进度: {progress}% ({done}/{total} 分镜)")
        else:
            print(f"⏳ 状态: {status} | 进度: {progress}%")

        if status == "done":
            print("✅ 任务完成！")
            return data
        if status == "failed":
            raise RuntimeError(f"任务失败: {data.get('error')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"任务在 {timeout}s 内未完成")


def download_result(task_id: str, output_dir: str = "async_output") -> None:
    """下载并解压分镜脚本 ZIP 结果"""
    resp = requests.get(
        f"{BASE_URL}/tasks/{task_id}/download",
        timeout=120,
    )
    resp.raise_for_status()

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # 根据 Content-Type 判断是 ZIP 还是 MP4
    if "zip" in resp.headers.get("Content-Type", ""):
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(out)
        mp4_files = sorted(out.glob("shot_*.mp4"))
        print(f"📦 ZIP 已解压，共 {len(mp4_files)} 个视频：")
        for f in mp4_files:
            print(f"   {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        output_file = out / f"video_{task_id[:8]}.mp4"
        output_file.write_bytes(resp.content)
        print(f"🎬 视频已保存: {output_file}")


if __name__ == "__main__":
    ref_images = [
        image_to_b64("product_front.jpg"),
        image_to_b64("product_side.jpg"),
    ]

    shots = [
        {
            "prompt": "Brand logo animation on a dark background, glowing gold letters",
            "num_frames": 33,
            "num_inference_steps": 25,
            "height": 480,
            "width": 704,
            "fast": True,
        },
        {
            "prompt": "Luxury perfume bottle emerging from morning mist, dramatic lighting",
            "reference_images": ref_images,
            "num_frames": 97,
            "num_inference_steps": 50,
            "height": 480,
            "width": 704,
            "fast": False,
            "background_style": "gradient",
        },
    ]

    task_id = submit_storyboard_async(shots)
    wait_for_task(task_id, poll_interval=5)
    download_result(task_id, output_dir="async_output")
```

---

## 5. 多个分镜拼接为完整视频

服务只负责逐镜生成，最终拼接由客户端完成（推荐 `ffmpeg`）：

```bash
# 生成 filelist.txt（ffmpeg concat 格式）
printf "file '%s'\n" storyboard_output/shot_*.mp4 > filelist.txt

# 无损拼接（所有分镜分辨率和帧率必须一致）
ffmpeg -f concat -safe 0 -i filelist.txt -c copy final_ad.mp4

echo "完整广告视频：final_ad.mp4"
```

---

## 6. 错误处理

| HTTP 状态码 | 原因 | 处理建议 |
|-------------|------|----------|
| `422 Unprocessable Entity` | 请求字段校验失败（帧数不满足 4N+1 或 8N+1、分辨率不是 32 倍数等） | 检查 `detail` 字段中的错误描述并修正参数 |
| `425 Too Early` | 异步任务尚未完成，提前调用 `/download` | 等待 `status=done` 后再下载 |
| `503 Service Unavailable` | 模型仍在加载（Wan 2.2 三模型加载时间较长）或任务队列不可用 | 轮询 `/health` 直到 `model_loaded: true` 后重试 |
| `500 Internal Server Error` | 推理过程异常（OOM、模型错误等） | 查看服务端日志 `logs/` 目录获取详细堆栈 |
| `404 Not Found` | 异步任务结果文件已过期（超过 60s 清理窗口） | 重新提交任务 |

### Python 客户端推荐的错误处理模板

```python
import time
import requests


def wait_for_ready(base_url: str, max_wait: int = 300) -> None:
    """等待服务就绪（Wan 2.2 双模型加载较慢，建议等待上限设 300s）"""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.ok and r.json().get("model_loaded"):
                print("✅ 服务就绪")
                return
        except requests.ConnectionError:
            pass
        print("⏳ 等待模型加载...")
        time.sleep(10)
    raise TimeoutError("服务在规定时间内未就绪")


def safe_generate(shots: list, base_url: str = "http://localhost:8000/api/v1") -> bytes:
    wait_for_ready(base_url)
    try:
        resp = requests.post(
            f"{base_url}/generate/storyboard",
            json={"shots": shots},
            timeout=900,
        )
        resp.raise_for_status()
        return resp.content   # ZIP 二进制内容
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        raise RuntimeError(f"生成失败（{e.response.status_code}）: {detail}") from e
```

---

## 7. 商品参考图建议规格

| 项目 | 建议 |
|------|------|
| 格式 | JPEG / PNG / WEBP |
| 分辨率 | 不限（服务自动等比缩放） |
| 商品占比 | 商品居中，占画面 60–80% 效果最佳 |
| 背景 | 纯色或渐变背景效果优于复杂背景 |
| 图片大小 | 建议每张 ≤ 5 MB，避免 base64 字符串过长 |
| 多图建议 | 提供 2–5 张不同角度/场景（正面、侧面、细节特写），CLIP 自动为每个分镜选最匹配的一张 |

> **商品图增强流水线**（服务内部自动处理，无需客户端预处理）：  
> 原图 → **背景移除**（rembg u2net）→ **色彩校正**（白平衡+亮度标准化+对比度/饱和度微调）  
> → **锐化**（Unsharp Mask）→ **等比缩放**（`min(目标宽/原宽, 目标高/原高)`）  
> → **居中粘贴**到目标尺寸画布（letterbox）→ **投影合成**（底部柔和阴影）→ **专业背景合成**  
> 商品比例 **100% 保留**，零变形、零裁切。

---

## 8. 环境变量配置

服务通过 `.env` 文件或系统环境变量配置，以下为关键参数：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `APP_HOST` | `0.0.0.0` | 服务监听地址 |
| `APP_PORT` | `8000` | 服务监听端口 |
| `WAN_T2V_MODEL` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Wan 2.2 文生视频模型 |
| `WAN_I2V_MODEL` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Wan 2.2 图生视频模型 |
| `WAN_TI2V_MODEL` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Wan 2.2 文本+图像生视频模型 |
| `MODEL_ID` | `Lightricks/LTX-Video` | LTX-Video 预览模型 Hub ID |
| `MODEL_LOCAL_PATH` | `""` | LTX-Video 本地模型路径（优先于 MODEL_ID） |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery 任务队列 Redis 地址（异步接口依赖） |
| `OUTPUT_DIR` | `outputs` | 生成视频临时存储目录 |
| `DEVICE` | `cuda` | 推理设备 |
