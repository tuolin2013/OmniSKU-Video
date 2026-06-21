#!/usr/bin/env bash
# ============================================================
# /workspace/setup.sh — 换 Pod 一键恢复完整开发环境
#
# 包含：
#   A. 系统工具（bun / redis / uv）
#   B. OmniVoice-Studio（Python uv 后端 + bun 前端）
#   C. ltx_video_service（pip 依赖）
#
# 用法：
#   bash /workspace/setup.sh            # 安装所有环境
#   bash /workspace/setup.sh --omni     # 仅 OmniVoice-Studio
#   bash /workspace/setup.sh --ltx      # 仅 ltx_video_service
#   bash /workspace/setup.sh --tools    # 仅安装系统工具
# ============================================================

set -euo pipefail

# ── 颜色 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log()     { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[⚠]${NC} $*"; }
err()     { echo -e "${RED}[✗]${NC} $*" >&2; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

WORKSPACE="/workspace"
OMNI_DIR="$WORKSPACE/OmniVoice-Studio"
LTX_DIR="$WORKSPACE/ltx_video_service"

# ── 参数解析 ──────────────────────────────────────────────────────────────
DO_TOOLS=true
DO_OMNI=true
DO_LTX=true

if [[ $# -gt 0 ]]; then
    DO_TOOLS=false; DO_OMNI=false; DO_LTX=false
    for arg in "$@"; do
        case $arg in
            --tools) DO_TOOLS=true ;;
            --omni)  DO_TOOLS=true; DO_OMNI=true ;;
            --ltx)   DO_TOOLS=true; DO_LTX=true ;;
            --help|-h)
                echo "用法: bash setup.sh [--omni|--ltx|--tools]"
                echo "  (无参数)  安装全部"
                echo "  --omni    OmniVoice-Studio 环境"
                echo "  --ltx     ltx_video_service 环境"
                echo "  --tools   仅系统工具（bun/redis/uv）"
                exit 0 ;;
            *) err "未知参数: $arg"; exit 1 ;;
        esac
    done
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════╗"
echo "║     Workspace 一键环境初始化脚本                   ║"
echo "║     OmniVoice-Studio + ltx_video_service         ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. 系统工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ "$DO_TOOLS" == "true" ]]; then
    section "A. 系统工具"

    # ── apt 基础包 ──────────────────────────────────────
    log "更新 apt 并安装基础包..."
    apt-get update -qq
    apt-get install -y -qq \
        curl wget unzip git redis-server \
        build-essential libssl-dev \
        ffmpeg \
        2>/dev/null || warn "部分 apt 包安装失败（可能已存在）"

    # ── bun ─────────────────────────────────────────────
    if ! command -v bun &>/dev/null; then
        log "安装 bun..."
        curl -fsSL https://bun.sh/install | bash
        # 加入当前 shell PATH
        export BUN_INSTALL="$HOME/.bun"
        export PATH="$BUN_INSTALL/bin:$PATH"
        log "bun $(bun --version) 安装完成"
    else
        log "bun 已存在: $(bun --version)"
    fi

    # ── uv ──────────────────────────────────────────────
    if ! command -v uv &>/dev/null; then
        log "安装 uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        log "uv $(uv --version) 安装完成"
    else
        log "uv 已存在: $(uv --version)"
    fi

    # ── HuggingFace 缓存路径统一（防止配额问题）────────────
    # RunPod 对 /workspace/.cache 有时有额外配额限制
    # 统一用 /workspace/huggingface_cache，并建软链接兼容两种路径
    mkdir -p /workspace/huggingface_cache/hub /workspace/huggingface_cache/xet
    if [[ ! -L /workspace/.cache/huggingface ]]; then
        mkdir -p /workspace/.cache
        if [[ -d /workspace/.cache/huggingface && ! -L /workspace/.cache/huggingface ]]; then
            # 迁移已有内容
            for d in /workspace/.cache/huggingface/hub/*/; do
                [[ -d "$d" ]] || continue
                name=$(basename "$d")
                if [[ ! -d "/workspace/huggingface_cache/hub/$name" ]]; then
                    mv "$d" "/workspace/huggingface_cache/hub/$name"
                    log "迁移模型缓存: $name"
                fi
            done
            rm -rf /workspace/.cache/huggingface
        fi
        ln -s /workspace/huggingface_cache /workspace/.cache/huggingface
        log "HF 缓存软链接已创建: .cache/huggingface -> huggingface_cache"
    else
        log "HF 缓存软链接已存在"
    fi
    # 写入 ~/.bashrc 永久生效
    grep -q "HF_HOME" /root/.bashrc 2>/dev/null || cat >> /root/.bashrc << 'BASHEOF'
export HF_HOME="/workspace/huggingface_cache"
export HF_HUB_CACHE="/workspace/huggingface_cache/hub"
export HF_XET_CACHE="/workspace/huggingface_cache/xet"
BASHEOF
    export HF_HOME="/workspace/huggingface_cache"
    export HF_HUB_CACHE="/workspace/huggingface_cache/hub"
    export HF_XET_CACHE="/workspace/huggingface_cache/xet"

    # ── Redis 启动 ───────────────────────────────────────
    if ! redis-cli ping &>/dev/null 2>&1; then
        log "启动 Redis..."
        redis-server --daemonize yes --logfile /tmp/redis.log
        sleep 1
        redis-cli ping && log "Redis 就绪" || warn "Redis 启动失败，请手动检查"
    else
        log "Redis 已在运行"
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. OmniVoice-Studio
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ "$DO_OMNI" == "true" ]]; then
    section "B. OmniVoice-Studio"

    if [[ ! -d "$OMNI_DIR" ]]; then
        err "目录不存在: $OMNI_DIR"
        warn "请先 git clone 项目到 $OMNI_DIR"
    else
        cd "$OMNI_DIR"

        # ── Python 后端（uv）────────────────────────────
        log "安装 Python 依赖（uv sync）..."
        # 确保 PATH 包含 uv
        export PATH="$HOME/.local/bin:$PATH"

        uv sync --no-dev 2>&1 | tail -5 || {
            warn "uv sync 失败，尝试使用系统 Python..."
            uv sync --no-dev --python-preference only-system 2>&1 | tail -5
        }
        log "Python 依赖安装完成"

        # ── 前端（bun）──────────────────────────────────
        export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
        export PATH="$BUN_INSTALL/bin:$PATH"

        if command -v bun &>/dev/null; then
            log "安装前端依赖（bun install）..."
            cd "$OMNI_DIR/frontend"
            bun install 2>&1 | tail -5
            log "前端依赖安装完成"
            cd "$OMNI_DIR"
        else
            warn "bun 不可用，跳过前端依赖安装"
        fi

        # ── 数据库迁移 ───────────────────────────────────
        log "运行数据库迁移..."
        uv run alembic upgrade head 2>&1 | tail -3 || warn "数据库迁移失败（可能是首次运行正常）"

        log "OmniVoice-Studio 环境就绪 ✅"
        echo -e "  启动后端: ${GREEN}cd $OMNI_DIR && uv run python backend/main.py${NC}"
        echo -e "  启动前端: ${GREEN}cd $OMNI_DIR/frontend && bun run dev${NC}"
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. ltx_video_service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ "$DO_LTX" == "true" ]]; then
    section "C. ltx_video_service"

    if [[ ! -d "$LTX_DIR" ]]; then
        err "目录不存在: $LTX_DIR"
    else
        cd "$LTX_DIR"

        log "安装 Python 依赖（pip install -r requirements.txt）..."
        pip install -r requirements.txt -q 2>&1 | tail -5 || \
            warn "部分依赖安装失败，可能已存在或需要手动处理"

        # 创建必要目录
        mkdir -p outputs logs .pids

        # 检查 .env
        if [[ ! -f ".env" ]]; then
            warn ".env 不存在，正在从模板创建..."
            cat > .env << 'EOF'
# ltx_video_service 环境变量
# 模型路径（留空则从 HuggingFace 自动下载）
MODEL_ID=Lightricks/LTX-Video
MODEL_LOCAL_PATH=

# Wan 2.2 模型（可选，留空则跳过 Wan）
# WAN_T2V_MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers
# WAN_I2V_MODEL=Wan-AI/Wan2.2-I2V-A14B-Diffusers
# WAN_TI2V_MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers

# 服务配置
APP_HOST=0.0.0.0
APP_PORT=8000
REDIS_URL=redis://localhost:6379/0
OUTPUT_DIR=outputs
LOG_LEVEL=INFO

# HuggingFace（如需下载私有/受限模型）
# HF_TOKEN=hf_xxx
EOF
            log ".env 模板已创建，请按需修改"
        fi

        log "ltx_video_service 环境就绪 ✅"
        echo -e "  启动服务: ${GREEN}cd $LTX_DIR && bash start.sh${NC}"
        echo -e "  后台启动: ${GREEN}cd $LTX_DIR && bash start.sh --bg${NC}"
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 汇总
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ 环境初始化完成！${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}快速启动命令：${NC}"
echo ""
echo -e "  # OmniVoice-Studio"
echo -e "  ${GREEN}cd $OMNI_DIR && uv run python backend/main.py &${NC}"
echo -e "  ${GREEN}cd $OMNI_DIR/frontend && bun run dev${NC}"
echo ""
echo -e "  # ltx_video_service"
echo -e "  ${GREEN}cd $LTX_DIR && bash start.sh --bg${NC}"
echo ""
echo -e "  # 查看 PATH 提示（若 bun/uv 命令找不到）："
echo -e "  ${YELLOW}source ~/.bashrc  或  export PATH=\"\$HOME/.bun/bin:\$HOME/.local/bin:\$PATH\"${NC}"
echo ""
