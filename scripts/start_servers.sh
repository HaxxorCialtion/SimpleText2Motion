#!/bin/bash
# start_servers.sh
#
# Start both backend servers for SimpleLove on macOS:
#   - fused_server (LLM hidden + simpletool) on 127.0.0.1:8421 / :8422
#   - t2m_infer    (motion service)          on 127.0.0.1:8423
#
# Both run in the background; this script becomes their supervisor.
# Press Ctrl-C to bring everything down cleanly.
#
# Logs:  ./logs/fused_server.log, ./logs/t2m_infer.log
# PIDs:  ./logs/fused_server.pid, ./logs/t2m_infer.pid
#
# Usage:
#   ./scripts/start_servers.sh                  # normal start
#   ./scripts/start_servers.sh --tail           # also tail t2m_infer log
#   ./scripts/start_servers.sh --skip-fused     # only t2m_infer (debugging)
#   ./scripts/start_servers.sh --skip-t2m      # only fused_server
#
# Override paths via env vars:
#   FUSED_BIN=./llama.cpp/fused_server
#   FUSED_MODEL=./models/qwen3-4b-q8_0.gguf
#   T2M_BIN=./build/t2m_infer
#   T2M_ONNX_DIR=./models/SimpleT2M

set -u   # error on undefined vars; intentionally NOT -e (we handle errors)

# ---------- Locate project root --------------------------------------------
# This script may be invoked from anywhere; resolve to project root.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT"

# ---------- Config (overridable via env) -----------------------------------
FUSED_BIN="${FUSED_BIN:-./llama.cpp/fused_server}"
FUSED_MODEL="${FUSED_MODEL:-./models/qwen3-4b-q8_0.gguf}"
FUSED_PORT_HIDDEN="${FUSED_PORT_HIDDEN:-8421}"
FUSED_PORT_SIMPLETOOL="${FUSED_PORT_SIMPLETOOL:-8422}"
FUSED_LAYER="${FUSED_LAYER:-5}"

T2M_BIN="${T2M_BIN:-./build/t2m_infer}"
T2M_ONNX_DIR="${T2M_ONNX_DIR:-./models/SimpleT2M}"
T2M_PORT="${T2M_PORT:-8423}"

LOG_DIR="${LOG_DIR:-./logs}"

# ---------- Parse args -----------------------------------------------------
SKIP_FUSED=0
SKIP_T2M=0
TAIL_LOG=0
for arg in "$@"; do
    case "$arg" in
        --skip-fused)  SKIP_FUSED=1 ;;
        --skip-t2m)    SKIP_T2M=1 ;;
        --tail)        TAIL_LOG=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | head -30
            exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ---------- Pretty print helpers -------------------------------------------
c_reset=$'\033[0m'; c_b=$'\033[1m'; c_g=$'\033[32m'; c_r=$'\033[31m'; c_y=$'\033[33m'; c_d=$'\033[2m'
log()  { printf "%s[start_servers]%s %s\n" "$c_b" "$c_reset" "$*"; }
ok()   { printf "%s[start_servers]%s %s✓%s %s\n" "$c_b" "$c_reset" "$c_g" "$c_reset" "$*"; }
err()  { printf "%s[start_servers]%s %s✗%s %s\n" "$c_b" "$c_reset" "$c_r" "$c_reset" "$*" >&2; }
warn() { printf "%s[start_servers]%s %s!%s %s\n" "$c_b" "$c_reset" "$c_y" "$c_reset" "$*"; }
dim()  { printf "%s%s%s\n" "$c_d" "$*" "$c_reset"; }

# ---------- TCP port probe (bash builtin, no nc/lsof dependency) -----------
# /dev/tcp/HOST/PORT is a magic bash device; opening it tries connect.
port_open() {  # returns 0 if something is listening
    local port="$1"
    (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null
}

wait_for_port() {  # wait_for_port <port> <name> <timeout_sec>
    local port="$1" name="$2" timeout="$3"
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if port_open "$port"; then return 0; fi
        sleep 0.2
        elapsed=$((elapsed + 1))   # 0.2s ticks; treat as "0.2 unit"
        # we want real seconds; recompute below
    done
    return 1
}

# More precise: count by 0.2s, give up after ceil(timeout * 5) ticks
wait_for_port() {
    local port="$1" name="$2" timeout_sec="$3"
    local ticks=$(( timeout_sec * 5 ))
    local i=0
    while [[ $i -lt $ticks ]]; do
        if port_open "$port"; then return 0; fi
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

# ---------- Preflight checks -----------------------------------------------
preflight() {
    local fail=0
    if [[ $SKIP_FUSED -eq 0 ]]; then
        if [[ ! -x "$FUSED_BIN" ]]; then
            err "fused_server binary not found or not executable: $FUSED_BIN"
            fail=1
        fi
        if [[ ! -f "$FUSED_MODEL" ]]; then
            err "fused_server model not found: $FUSED_MODEL"
            fail=1
        fi
        if port_open "$FUSED_PORT_HIDDEN"; then
            err "port $FUSED_PORT_HIDDEN already in use (hidden)"
            err "  → run scripts/stop_servers.sh first, or kill the offender"
            fail=1
        fi
        if port_open "$FUSED_PORT_SIMPLETOOL"; then
            err "port $FUSED_PORT_SIMPLETOOL already in use (simpletool)"
            fail=1
        fi
    fi
    if [[ $SKIP_T2M -eq 0 ]]; then
        if [[ ! -x "$T2M_BIN" ]]; then
            err "t2m_infer binary not found or not executable: $T2M_BIN"
            fail=1
        fi
        if [[ ! -d "$T2M_ONNX_DIR" ]]; then
            err "t2m onnx dir not found: $T2M_ONNX_DIR"
            fail=1
        fi
        if port_open "$T2M_PORT"; then
            err "port $T2M_PORT already in use (t2m)"
            fail=1
        fi
    fi
    if [[ $fail -ne 0 ]]; then
        err "preflight failed; see errors above"
        exit 1
    fi
}

# ---------- Process group / cleanup ----------------------------------------
# Track PIDs so we can kill them on exit.
FUSED_PID=""
T2M_PID=""
SHUTTING_DOWN=0

cleanup() {
    [[ $SHUTTING_DOWN -eq 1 ]] && return
    SHUTTING_DOWN=1
    echo  # newline after ^C
    log "shutting down..."

    # Send SIGTERM first
    for pid in "$T2M_PID" "$FUSED_PID"; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait up to 3 seconds for graceful exit
    local i
    for i in 1 2 3 4 5 6; do
        local any_alive=0
        for pid in "$T2M_PID" "$FUSED_PID"; do
            [[ -z "$pid" ]] && continue
            if kill -0 "$pid" 2>/dev/null; then any_alive=1; fi
        done
        [[ $any_alive -eq 0 ]] && break
        sleep 0.5
    done

    # Anything still alive -> SIGKILL
    for pid in "$T2M_PID" "$FUSED_PID"; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            warn "force-killing pid $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    rm -f "$LOG_DIR/fused_server.pid" "$LOG_DIR/t2m_infer.pid"
    ok "all servers stopped"
}

trap cleanup INT TERM EXIT

# ---------- Launch fused_server --------------------------------------------
start_fused() {
    log "starting fused_server..."
    dim "  bin:   $FUSED_BIN"
    dim "  model: $FUSED_MODEL"
    dim "  ports: hidden=$FUSED_PORT_HIDDEN, simpletool=$FUSED_PORT_SIMPLETOOL"
    dim "  log:   $LOG_DIR/fused_server.log"

    # fused_server expects to be launched from llama.cpp/ for its RPATH
    # (@loader_path/build_mac/bin). We chdir into its parent dir so the
    # relative model path keeps working.
    local fused_dir
    fused_dir="$( cd "$(dirname "$FUSED_BIN")" && pwd )"
    local fused_exe
    fused_exe="$(basename "$FUSED_BIN")"

    # Compute model path relative to fused_dir (or absolute)
    local model_abs
    model_abs="$( cd "$(dirname "$FUSED_MODEL")" && pwd )/$(basename "$FUSED_MODEL")"

    (
        cd "$fused_dir"
        exec "./$fused_exe" "$model_abs" \
            --layer "$FUSED_LAYER" \
            --port-hidden "$FUSED_PORT_HIDDEN" \
            --port-simpletool "$FUSED_PORT_SIMPLETOOL"
    ) >> "$LOG_DIR/fused_server.log" 2>&1 &
    FUSED_PID=$!
    echo "$FUSED_PID" > "$LOG_DIR/fused_server.pid"
    dim "  pid:   $FUSED_PID"

    # Wait for hidden port. Cold load of Qwen3-4B q8 GGUF + warmup ~ 3-8s.
    if wait_for_port "$FUSED_PORT_HIDDEN" "fused_server" 60; then
        ok "fused_server ready (hidden:$FUSED_PORT_HIDDEN, simpletool:$FUSED_PORT_SIMPLETOOL)"
    else
        err "fused_server failed to open port $FUSED_PORT_HIDDEN within 60s"
        err "  last lines of $LOG_DIR/fused_server.log:"
        tail -30 "$LOG_DIR/fused_server.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 1
    fi

    # Sanity: if process died, ports won't open, but double-check
    if ! kill -0 "$FUSED_PID" 2>/dev/null; then
        err "fused_server died after starting"
        tail -30 "$LOG_DIR/fused_server.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 1
    fi
}

# ---------- Launch t2m_infer -----------------------------------------------
start_t2m() {
    log "starting t2m_infer..."
    dim "  bin:      $T2M_BIN"
    dim "  onnx_dir: $T2M_ONNX_DIR"
    dim "  port:     $T2M_PORT"
    dim "  log:      $LOG_DIR/t2m_infer.log"

    "$T2M_BIN" --server \
        --server-port "$T2M_PORT" \
        --onnx-dir "$T2M_ONNX_DIR" \
        --host 127.0.0.1 \
        --port "$FUSED_PORT_HIDDEN" \
        >> "$LOG_DIR/t2m_infer.log" 2>&1 &
    T2M_PID=$!
    echo "$T2M_PID" > "$LOG_DIR/t2m_infer.pid"
    dim "  pid:      $T2M_PID"

    # ONNX load + dummy warmup ~ 1-3s on Apple Silicon
    if wait_for_port "$T2M_PORT" "t2m_infer" 30; then
        ok "t2m_infer ready (port $T2M_PORT)"
    else
        err "t2m_infer failed to open port $T2M_PORT within 30s"
        err "  last lines of $LOG_DIR/t2m_infer.log:"
        tail -30 "$LOG_DIR/t2m_infer.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 1
    fi

    if ! kill -0 "$T2M_PID" 2>/dev/null; then
        err "t2m_infer died after starting"
        tail -30 "$LOG_DIR/t2m_infer.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 1
    fi
}

# ---------- Monitor loop ---------------------------------------------------
monitor_loop() {
    log "all servers ready. press Ctrl-C to stop."
    log ""
    log "to test:"
    dim "  python t2m_client.py \"A person walks forward\""
    log ""
    log "to follow logs in another terminal:"
    dim "  tail -f $LOG_DIR/t2m_infer.log"
    dim "  tail -f $LOG_DIR/fused_server.log"
    log ""

    if [[ $TAIL_LOG -eq 1 ]]; then
        log "tailing t2m_infer.log (logs will appear below):"
        tail -F "$LOG_DIR/t2m_infer.log" &
        local tail_pid=$!
        # ensure tail dies with us
        trap "kill $tail_pid 2>/dev/null; cleanup" INT TERM EXIT
    fi

    # Keep the script alive; check liveness every 2s. If any backend dies
    # unexpectedly, log it and exit (trap will clean up the survivor).
    while true; do
        if [[ -n "$FUSED_PID" ]] && ! kill -0 "$FUSED_PID" 2>/dev/null; then
            err "fused_server (pid $FUSED_PID) died unexpectedly"
            err "  last lines of $LOG_DIR/fused_server.log:"
            tail -20 "$LOG_DIR/fused_server.log" | sed 's/^/    /' >&2
            exit 1
        fi
        if [[ -n "$T2M_PID" ]] && ! kill -0 "$T2M_PID" 2>/dev/null; then
            err "t2m_infer (pid $T2M_PID) died unexpectedly"
            err "  last lines of $LOG_DIR/t2m_infer.log:"
            tail -20 "$LOG_DIR/t2m_infer.log" | sed 's/^/    /' >&2
            exit 1
        fi
        sleep 2
    done
}

# ---------- Main -----------------------------------------------------------
preflight
[[ $SKIP_FUSED -eq 0 ]] && start_fused
[[ $SKIP_T2M -eq 0 ]] && start_t2m
monitor_loop
