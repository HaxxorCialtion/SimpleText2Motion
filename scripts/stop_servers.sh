#!/bin/bash
# stop_servers.sh
#
# Stop servers started by start_servers.sh.
# Useful when start_servers.sh didn't get a chance to clean up
# (terminal closed, kernel panic, ssh disconnect, etc).
#
# Strategy:
#   1. Read PIDs from logs/*.pid, kill them.
#   2. As a backup, find anything listening on our ports and kill it.

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-./logs}"
PORTS=("${FUSED_PORT_HIDDEN:-8421}" "${FUSED_PORT_SIMPLETOOL:-8422}" "${T2M_PORT:-8423}")

killed_anything=0

kill_pid() {  # kill_pid <pid> <label>
    local pid="$1" label="$2"
    [[ -z "$pid" ]] && return
    if kill -0 "$pid" 2>/dev/null; then
        echo "stopping $label (pid $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        # wait 3s
        local i
        for i in 1 2 3 4 5 6; do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 0.5
        done
        echo "  → still alive, sending SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
        killed_anything=1
    fi
}

# ---- pass 1: PID files ----
for name in fused_server t2m_infer; do
    pidfile="$LOG_DIR/$name.pid"
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile" 2>/dev/null)
        kill_pid "$pid" "$name"
        rm -f "$pidfile"
        killed_anything=1
    fi
done

# ---- pass 2: anything still listening on our ports ----
# lsof comes with macOS by default; quiet if nothing found.
for port in "${PORTS[@]}"; do
    pids=$(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            kill_pid "$pid" "(orphan on :$port)"
        done
    fi
done

if [[ $killed_anything -eq 0 ]]; then
    echo "no running servers found"
else
    echo "done"
fi
