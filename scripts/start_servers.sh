#!/bin/bash
# scripts/start_servers.sh
#
# Start backend servers for SimpleT2M. Linux + macOS.
#   - fused_server (LLM hidden + simpletool) on :8421 / :8422
#   - t2m_infer    (motion service)          on :8423
#
# All paths and ports come from config.toml (project root by default).
# Override the config file with:
#   CONFIG=path/to/other.toml ./scripts/start_servers.sh
#
# Logs:  $LOG_DIR/{fused_server,t2m_infer}.log
# PIDs:  $LOG_DIR/{fused_server,t2m_infer}.pid
#
# Usage:
#   ./scripts/start_servers.sh                 # normal start
#   ./scripts/start_servers.sh --tail          # also tail t2m_infer log
#   ./scripts/start_servers.sh --skip-fused    # only t2m_infer
#   ./scripts/start_servers.sh --skip-t2m      # only fused_server

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT"

CONFIG="${CONFIG:-$ROOT/config.toml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "[start_servers] config not found: $CONFIG" >&2
    exit 1
fi
CONFIG_DIR="$( cd "$(dirname "$CONFIG")" && pwd )"

# ---------- mini TOML reader -----------------------------------------------
# Reads scalar [section].key from the config. Strips quotes and inline
# comments. NOT a real parser (no arrays, no nesting) — enough for paths
# and ports. Anything more complex stays inside C++ binaries that use toml++.
toml_get() {
    local section="$1" key="$2"
    awk -v sect="[$section]" -v key="$key" '
        $0 ~ /^\[.*\]/ { in_sect = ($0 == sect); next }
        in_sect && $0 ~ "^[[:space:]]*"key"[[:space:]]*=" {
            sub(/^[^=]*=[[:space:]]*/, "")
            sub(/[[:space:]]*#.*$/, "")
            gsub(/^"|"$/, "")
            gsub(/^'\''|'\''$/, "")
            print; exit
        }
    ' "$CONFIG"
}

# Resolve a path relative to CONFIG_DIR (matches AppConfig::resolve in C++).
resolve_path() {
    local p="$1"
    [[ -z "$p" ]] && { echo ""; return; }
    if [[ "$p" = /* ]]; then echo "$p"; return; fi
    echo "$CONFIG_DIR/$p"
}

# ---------- Platform detection ---------------------------------------------
case "$(uname -s)" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      PLATFORM="unknown" ;;
esac

# ---------- Pull config ----------------------------------------------------
FUSED_BIN="$(resolve_path "$(toml_get fused_server binary)")"
FUSED_MODEL="$(resolve_path "$(toml_get paths llama_model)")"
FUSED_PORT_HIDDEN="$(toml_get fused_server port_hidden)"
FUSED_PORT_SIMPLETOOL="$(toml_get fused_server port_simpletool)"
FUSED_LAYER="$(toml_get fused_server hidden_layer)"

T2M_BIN="$(resolve_path "$(toml_get t2m_infer binary)")"
T2M_ONNX_DIR="$(resolve_path "$(toml_get paths t2m_onnx_dir)")"
T2M_PORT="$(toml_get t2m_infer server_port)"

LOG_DIR="$(resolve_path "$(toml_get paths logs_dir)")"
[[ -z "$LOG_DIR" ]] && LOG_DIR="$ROOT/logs"

# ORT lib dir is auto-discovered (matches CMake's preference rule:
# pick the GPU variant if present, otherwise newest CPU).
ORT_LIB_DIR=""
if [[ "$PLATFORM" == "macos" ]]; then
    ORT_LIB_DIR="$(ls -d "$ROOT"/third_party/onnxruntime-osx-arm64-*/lib 2>/dev/null | sort | tail -n 1 || true)"
else
    ORT_LIB_DIR="$(ls -d "$ROOT"/third_party/onnxruntime-linux-x64-gpu-*/lib 2>/dev/null | sort | tail -n 1 || true)"
    [[ -z "$ORT_LIB_DIR" ]] && \
        ORT_LIB_DIR="$(ls -d "$ROOT"/third_party/onnxruntime-linux-x64-*/lib 2>/dev/null | sort | tail -n 1 || true)"
fi

# ---------- Args -----------------------------------------------------------
SKIP_FUSED=0
SKIP_T2M=0
TAIL_LOG=0
for arg in "$@"; do
    case "$arg" in
        --skip-fused) SKIP_FUSED=1 ;;
        --skip-t2m)   SKIP_T2M=1 ;;
        --tail)       TAIL_LOG=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \?//' | head -30; exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ---------- Pretty print ---------------------------------------------------
c_reset=$'\033[0m'; c_b=$'\033[1m'; c_g=$'\033[32m'; c_r=$'\033[31m'; c_y=$'\033[33m'; c_d=$'\033[2m'
log()  { printf "%s[start_servers]%s %s\n" "$c_b" "$c_reset" "$*"; }
ok()   { printf "%s[start_servers]%s %s\xe2\x9c\x93%s %s\n" "$c_b" "$c_reset" "$c_g" "$c_reset" "$*"; }
err()  { printf "%s[start_servers]%s %s\xe2\x9c\x97%s %s\n" "$c_b" "$c_reset" "$c_r" "$c_reset" "$*" >&2; }
warn() { printf "%s[start_servers]%s %s!%s %s\n" "$c_b" "$c_reset" "$c_y" "$c_reset" "$*"; }
dim()  { printf "%s%s%s\n" "$c_d" "$*" "$c_reset"; }

# ---------- TCP port probe -------------------------------------------------
port_open() { (echo >"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
wait_for_port() {
    local port="$1" name="$2" timeout_sec="$3"
    local ticks=$(( timeout_sec * 5 )) i=0
    while [[ $i -lt $ticks ]]; do
        if port_open "$port"; then return 0; fi
        sleep 0.2; i=$((i + 1))
    done
    return 1
}

# ---------- Preflight ------------------------------------------------------
preflight() {
    local fail=0
    if [[ $SKIP_FUSED -eq 0 ]]; then
        [[ -x "$FUSED_BIN"   ]] || { err "fused_server bin missing: $FUSED_BIN"; fail=1; }
        [[ -f "$FUSED_MODEL" ]] || { err "fused_server model missing: $FUSED_MODEL"; fail=1; }
        port_open "$FUSED_PORT_HIDDEN"     && { err "port $FUSED_PORT_HIDDEN in use (hidden)"; fail=1; }
        port_open "$FUSED_PORT_SIMPLETOOL" && { err "port $FUSED_PORT_SIMPLETOOL in use (simpletool)"; fail=1; }
    fi
    if [[ $SKIP_T2M -eq 0 ]]; then
        [[ -x "$T2M_BIN"      ]] || { err "t2m_infer bin missing: $T2M_BIN"; fail=1; }
        [[ -d "$T2M_ONNX_DIR" ]] || { err "t2m onnx dir missing: $T2M_ONNX_DIR"; fail=1; }
        port_open "$T2M_PORT" && { err "port $T2M_PORT in use (t2m)"; fail=1; }
    fi
    [[ $fail -ne 0 ]] && { err "preflight failed"; exit 1; }
}

# ---------- Cleanup --------------------------------------------------------
FUSED_PID=""; T2M_PID=""; SHUTTING_DOWN=0
cleanup() {
    [[ $SHUTTING_DOWN -eq 1 ]] && return
    SHUTTING_DOWN=1
    echo; log "shutting down..."
    for pid in "$T2M_PID" "$FUSED_PID"; do
        [[ -z "$pid" ]] && continue
        kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in 1 2 3 4 5 6; do
        local any=0
        for pid in "$T2M_PID" "$FUSED_PID"; do
            [[ -z "$pid" ]] && continue
            kill -0 "$pid" 2>/dev/null && any=1
        done
        [[ $any -eq 0 ]] && break
        sleep 0.5
    done
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
    dim "  bin:    $FUSED_BIN"
    dim "  model:  $FUSED_MODEL"
    dim "  ports:  hidden=$FUSED_PORT_HIDDEN  simpletool=$FUSED_PORT_SIMPLETOOL  layer=$FUSED_LAYER"
    dim "  log:    $LOG_DIR/fused_server.log"

    # Chdir into binary's own dir so $ORIGIN rpath (Linux) /
    # @loader_path (macOS) resolve correctly. Model path is absolute.
    local fused_dir; fused_dir="$( cd "$(dirname "$FUSED_BIN")" && pwd )"
    local fused_exe; fused_exe="$(basename "$FUSED_BIN")"

    (
        cd "$fused_dir"
        exec "./$fused_exe" "$FUSED_MODEL" \
            --config "$CONFIG" \
            --layer "$FUSED_LAYER" \
            --port-hidden "$FUSED_PORT_HIDDEN" \
            --port-simpletool "$FUSED_PORT_SIMPLETOOL"
    ) >> "$LOG_DIR/fused_server.log" 2>&1 &
    FUSED_PID=$!
    echo "$FUSED_PID" > "$LOG_DIR/fused_server.pid"
    dim "  pid:    $FUSED_PID"

    if wait_for_port "$FUSED_PORT_HIDDEN" "fused_server" 60; then
        ok "fused_server ready"
    else
        err "fused_server failed to open port $FUSED_PORT_HIDDEN within 60s"
        tail -30 "$LOG_DIR/fused_server.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 1
    fi
    kill -0 "$FUSED_PID" 2>/dev/null || { err "fused_server died"; exit 1; }
}

# ---------- Launch t2m_infer -----------------------------------------------
start_t2m() {
    log "starting t2m_infer..."
    dim "  bin:      $T2M_BIN"
    dim "  onnx_dir: $T2M_ONNX_DIR"
    dim "  port:     $T2M_PORT"
    dim "  log:      $LOG_DIR/t2m_infer.log"

    (
        if [[ "$PLATFORM" == "linux" && -n "$ORT_LIB_DIR" ]]; then
            export LD_LIBRARY_PATH="${ORT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
        exec "$T2M_BIN" --server --config "$CONFIG"
    ) >> "$LOG_DIR/t2m_infer.log" 2>&1 &
    T2M_PID=$!
    echo "$T2M_PID" > "$LOG_DIR/t2m_infer.pid"
    dim "  pid:      $T2M_PID"

    if wait_for_port "$T2M_PORT" "t2m_infer" 60; then
        ok "t2m_infer ready"
    else
        err "t2m_infer failed to open port $T2M_PORT within 30s"
        tail -30 "$LOG_DIR/t2m_infer.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 1
    fi
    kill -0 "$T2M_PID" 2>/dev/null || { err "t2m_infer died"; exit 1; }
}

# ---------- Monitor --------------------------------------------------------
monitor_loop() {
    log "all servers ready. press Ctrl-C to stop."
    log ""
    log "to bench:"
    dim "  ./build/bench_t2m --config $CONFIG"
    log ""

    if [[ $TAIL_LOG -eq 1 ]]; then
        tail -F "$LOG_DIR/t2m_infer.log" &
        local tail_pid=$!
        trap "kill $tail_pid 2>/dev/null; cleanup" INT TERM EXIT
    fi

    while true; do
        if [[ -n "$FUSED_PID" ]] && ! kill -0 "$FUSED_PID" 2>/dev/null; then
            err "fused_server (pid $FUSED_PID) died"
            tail -20 "$LOG_DIR/fused_server.log" | sed 's/^/    /' >&2
            exit 1
        fi
        if [[ -n "$T2M_PID" ]] && ! kill -0 "$T2M_PID" 2>/dev/null; then
            err "t2m_infer (pid $T2M_PID) died"
            tail -20 "$LOG_DIR/t2m_infer.log" | sed 's/^/    /' >&2
            exit 1
        fi
        sleep 2
    done
}

# ---------- Main -----------------------------------------------------------
log "platform: $PLATFORM"
log "config:   $CONFIG"
preflight
[[ $SKIP_FUSED -eq 0 ]] && start_fused
[[ $SKIP_T2M -eq 0 ]]   && start_t2m
monitor_loop