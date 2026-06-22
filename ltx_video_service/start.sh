#!/usr/bin/env bash
# ============================================================
# start.sh — 一键启动脚本（生产级）
#
# 启动顺序：
#   1. 检查依赖（Redis / Python 包）
#   2. 启动 Redis（如未运行）
#   3. 启动 Celery GPU worker（gpu_queue，单并发）
#   4. 启动 FastAPI 主服务（uvicorn）
#
# 用法：
#   chmod +x start.sh
#   ./start.sh              # 前台运行（Ctrl+C 退出所有进程）
#   ./start.sh --bg         # 后台运行（PID 写入 .pids/ 目录）
#   ./start.sh --stop       # 停止所有后台进程
#   ./start.sh --status     # 查看进程状态
#
# 多 GPU（P2-3）：
#   CUDA_VISIBLE_DEVICES=0 ./start.sh &
#   CUDA_VISIBLE_DEVICES=1 APP_PORT=8001 ./start.sh &
# ============================================================

set -euo pipefail

# ── HuggingFace 缓存路径（必须在任何 Python import 之前生效）────────────────
# 统一指向 /workspace/huggingface_cache，避免写入 .cache/ 触发 RunPod 配额限制
export HF_HOME="${HF_HOME:-/workspace/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/huggingface_cache/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-/workspace/huggingface_cache/xet}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"

# ── 配置（可通过环境变量覆盖）──────────────────────────────────────────────
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
WORKERS="${WORKERS:-1}"             # Celery 并发数（GPU 任务必须为 1）
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PID_DIR="${PID_DIR:-.pids}"
LOG_DIR="${LOG_DIR:-logs}"

# 颜色输出
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*" >&2; }
info() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }

# ── 帮助 ──────────────────────────────────────────────────────────────────
usage() {
    echo "用法: $0 [--bg|--stop|--status|--help]"
    echo ""
    echo "  (无参数)   前台运行，Ctrl+C 退出所有进程"
    echo "  --bg       后台运行，PID 存入 $PID_DIR/"
    echo "  --stop     停止所有后台进程"
    echo "  --status   查看进程状态"
    echo ""
    echo "环境变量:"
    echo "  APP_PORT=8000            FastAPI 监听端口"
    echo "  REDIS_URL=redis://...    Redis 连接 URL"
    echo "  CUDA_VISIBLE_DEVICES=0   指定 GPU（多 GPU 扩展用）"
    echo "  WORKERS=1                Celery GPU 并发数（生产建议保持 1）"
    exit 0
}

# ── 停止命令 ──────────────────────────────────────────────────────────────
cmd_stop() {
    log "停止所有后台进程..."
    for pidfile in "$PID_DIR"/*.pid; do
        [[ -f "$pidfile" ]] || continue
        name=$(basename "$pidfile" .pid)
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && log "已停止 $name (PID=$pid)"
        else
            warn "$name (PID=$pid) 已不在运行"
        fi
        rm -f "$pidfile"
    done
    log "所有进程已停止"
}

# ── 状态命令 ──────────────────────────────────────────────────────────────
cmd_status() {
    info "=== 进程状态 ==="
    for pidfile in "$PID_DIR"/*.pid; do
        [[ -f "$pidfile" ]] || continue
        name=$(basename "$pidfile" .pid)
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $name (PID=$pid) — 运行中"
        else
            echo -e "  ${RED}●${NC} $name (PID=$pid) — 已停止"
        fi
    done
    echo ""
    # 检查 API 健康
    if curl -sf "http://localhost:${APP_PORT}/api/v1/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}●${NC} API http://localhost:${APP_PORT} — 健康"
    else
        echo -e "  ${RED}●${NC} API http://localhost:${APP_PORT} — 不可达"
    fi
}

# ── 依赖检查 ──────────────────────────────────────────────────────────────
check_deps() {
    log "检查依赖..."

    # Python
    if ! command -v python &>/dev/null; then
        err "未找到 python，请先安装 Python 3.10+"
        exit 1
    fi

    # Redis CLI（用于检查 Redis 状态）
    REDIS_CLI=""
    if command -v redis-cli &>/dev/null; then
        REDIS_CLI="redis-cli"
    fi

    # GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        log "GPU: $GPU_INFO"
    else
        warn "未检测到 NVIDIA GPU，将使用 CPU 运行（速度极慢）"
    fi

    # 检查关键 Python 包
    python -c "import fastapi, celery, redis, diffusers" 2>/dev/null || {
        err "缺少 Python 依赖，请先运行: pip install -r requirements.txt"
        exit 1
    }

    log "依赖检查完成 ✅"
}

# ── 启动 Redis ────────────────────────────────────────────────────────────
start_redis() {
    # 提取 Redis host/port
    REDIS_HOST=$(echo "$REDIS_URL" | sed 's|redis://||' | cut -d: -f1)
    REDIS_PORT=$(echo "$REDIS_URL" | sed 's|redis://||' | cut -d: -f2 | cut -d/ -f1)
    REDIS_PORT="${REDIS_PORT:-6379}"

    # 检查 Redis 是否已在运行
    if [[ -n "$REDIS_CLI" ]] && $REDIS_CLI -h "$REDIS_HOST" -p "$REDIS_PORT" ping &>/dev/null 2>&1; then
        log "Redis 已在运行 ($REDIS_HOST:$REDIS_PORT)"
        return 0
    fi

    if ! command -v redis-server &>/dev/null; then
        log "Redis server 未安装，正在安装..."
        apt-get update -qq > /dev/null 2>&1
        apt-get install -y redis-server > /dev/null 2>&1 && log "Redis 安装完成" || {
            warn "Redis 安装失败，Celery 任务队列将不可用"
        }
    fi

    if command -v redis-server &>/dev/null; then
        log "启动 Redis server..."
        if [[ "$BACKGROUND" == "true" ]]; then
            mkdir -p "$PID_DIR" "$LOG_DIR"
            redis-server --daemonize yes \
                --pidfile "$(pwd)/$PID_DIR/redis.pid" \
                --logfile "$(pwd)/$LOG_DIR/redis.log" \
                --port "$REDIS_PORT"
            log "Redis 已在后台启动 (port=$REDIS_PORT)"
        else
            redis-server --port "$REDIS_PORT" &
            REDIS_PID=$!
            log "Redis 已启动 (PID=$REDIS_PID, port=$REDIS_PORT)"
        fi
        sleep 1
    else
        warn "Redis server 未安装，假设 Redis 已在运行"
        warn "若 Celery 无法连接，请先安装 Redis: apt-get install redis-server"
    fi
}

# ── 启动 Celery Worker ─────────────────────────────────────────────────────
start_celery() {
    log "启动 Celery GPU worker (concurrency=$WORKERS)..."

    GPU_TAG=""
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPU_TAG="_gpu${CUDA_VISIBLE_DEVICES}"
    fi

    CELERY_CMD="celery -A services.tasks:celery_app worker \
        --queues=gpu_queue \
        --concurrency=$WORKERS \
        --loglevel=$LOG_LEVEL \
        --hostname=gpu_worker${GPU_TAG}@%h \
        --without-gossip \
        --without-mingle"

    if [[ "$BACKGROUND" == "true" ]]; then
        mkdir -p "$PID_DIR" "$LOG_DIR"
        eval "$CELERY_CMD \
            --pidfile=$(pwd)/$PID_DIR/celery.pid \
            --logfile=$(pwd)/$LOG_DIR/celery.log \
            --detach"
        log "Celery worker 已在后台启动"
    else
        eval "$CELERY_CMD" &
        CELERY_PID=$!
        log "Celery worker 已启动 (PID=$CELERY_PID)"
    fi
}

# ── 启动 FastAPI ──────────────────────────────────────────────────────────
start_fastapi() {
    log "启动 FastAPI 服务 (host=$APP_HOST port=$APP_PORT)..."

    UVICORN_CMD="uvicorn main:app \
        --host $APP_HOST \
        --port $APP_PORT \
        --workers 1 \
        --log-level $(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]') \
        --no-access-log"

    if [[ "$BACKGROUND" == "true" ]]; then
        mkdir -p "$PID_DIR" "$LOG_DIR"
        eval "$UVICORN_CMD" >> "$LOG_DIR/fastapi.log" 2>&1 &
        echo $! > "$PID_DIR/fastapi.pid"
        log "FastAPI 已在后台启动 (PID=$(cat $PID_DIR/fastapi.pid))"
        log "日志: $LOG_DIR/fastapi.log"
        log "API 文档: http://localhost:$APP_PORT/docs"
    else
        eval "$UVICORN_CMD"
    fi
}

# ── 前台模式退出清理 ──────────────────────────────────────────────────────
cleanup() {
    echo ""
    log "收到退出信号，正在停止所有进程..."
    # 杀掉所有子进程
    jobs -p | xargs -r kill 2>/dev/null || true
    log "✅ 已安全退出"
    exit 0
}

# ── 主逻辑 ──────────────────────────────────────────────────────────────
BACKGROUND="false"
ACTION="start"

for arg in "$@"; do
    case $arg in
        --bg)     BACKGROUND="true" ;;
        --stop)   ACTION="stop" ;;
        --status) ACTION="status" ;;
        --help|-h) usage ;;
        *) err "未知参数: $arg"; usage ;;
    esac
done

cd "$(dirname "$0")"

case $ACTION in
    stop)   cmd_stop;   exit 0 ;;
    status) cmd_status; exit 0 ;;
esac

# ── 启动流程 ────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    LTX-Video 生产级电商视频生成服务 v2.0          ║${NC}"
echo -e "${BLUE}║    Wan 2.1 14B + LTX-Video + Real-ESRGAN         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

check_deps
start_redis
start_celery

# 等待 Celery 就绪
sleep 2

start_fastapi

if [[ "$BACKGROUND" == "true" ]]; then
    echo ""
    log "所有服务已在后台启动："
    echo -e "  ${GREEN}➜${NC} API 文档:       http://localhost:${APP_PORT}/docs"
    echo -e "  ${GREEN}➜${NC} 健康检查:       http://localhost:${APP_PORT}/api/v1/health"
    echo -e "  ${GREEN}➜${NC} 查看状态:       ./start.sh --status"
    echo -e "  ${GREEN}➜${NC} 停止服务:       ./start.sh --stop"
    echo -e "  ${GREEN}➜${NC} 日志目录:       $LOG_DIR/"
    echo ""
else
    # 前台模式：捕获退出信号
    trap cleanup SIGINT SIGTERM
    echo ""
    log "所有服务已启动（前台模式，Ctrl+C 退出）"
    echo -e "  ${GREEN}➜${NC} API 文档: http://localhost:${APP_PORT}/docs"
    echo ""
    wait
fi
