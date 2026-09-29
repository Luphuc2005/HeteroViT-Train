#!/usr/bin/env bash
# ==============================================================================
# HeteroViT-MPI: Helper to inject / stop background load on cluster worker nodes
#
# Usage:
#   ./scripts/inject_slowdown.sh start lab03 300    # Slow down lab03 for 300 seconds
#   ./scripts/inject_slowdown.sh stop lab03         # Stop slow down and recover lab03
#   ./scripts/inject_slowdown.sh status             # Check status on all nodes
# ==============================================================================

set -euo pipefail

NODE_IPS_lab01="192.168.1.131"
NODE_IPS_lab02="192.168.1.132"
NODE_IPS_lab03="192.168.1.133"
NODE_IPS_lab04="192.168.1.134"
NODE_IPS_lab05="192.168.1.135"

ACTION="${1:-help}"

get_ip() {
  local node="$1"
  case "$node" in
    lab01|iciplab01) echo "$NODE_IPS_lab01" ;;
    lab02|iciplab02) echo "$NODE_IPS_lab02" ;;
    lab03|iciplab03) echo "$NODE_IPS_lab03" ;;
    lab04|iciplab04) echo "$NODE_IPS_lab04" ;;
    lab05|iciplab05) echo "$NODE_IPS_lab05" ;;
    *) echo "" ;;
  esac
}

case "$ACTION" in
  start)
    TARGET="${2:-lab03}"
    DURATION="${3:-240}"
    IP=$(get_ip "$TARGET")
    if [ -z "$IP" ]; then
      echo "[ERROR] Unknown node: $TARGET. Choose from lab02, lab03, lab04, lab05."
      exit 1
    fi
    echo "===================================================================="
    echo ">>> [INJECT SLOWDOWN] Node: $TARGET ($IP) for ${DURATION}s..."
    echo "===================================================================="
    ssh icip@"$IP" "nohup python3 -c \"
import multiprocessing, time, os
def burn():
    while True:
        _ = 999999 * 999999
if __name__ == '__main__':
    cores = os.cpu_count() or 4
    procs = [multiprocessing.Process(target=burn) for _ in range(cores)]
    for p in procs: p.start()
    time.sleep($DURATION)
    for p in procs: p.terminate()
\" > /dev/null 2>&1 &"
    echo ">>> Injected CPU stress on $TARGET ($IP) for ${DURATION}s."
    echo ">>> In the next epoch boundary, Master will detect $TARGET slowdown and rebalance!"
    ;;

  stop)
    TARGET="${2:-all}"
    if [ "$TARGET" = "all" ]; then
      NODES="lab02 lab03 lab04 lab05"
    else
      NODES="$TARGET"
    fi
    for n in $NODES; do
      IP=$(get_ip "$n")
      if [ -n "$IP" ]; then
        echo ">>> [STOP SLOWDOWN] Cleaning burner on $n ($IP)..."
        ssh icip@"$IP" "pkill -f 'burn' || true" 2>/dev/null || true
      fi
    done
    echo ">>> Done. Node(s) restored to normal."
    ;;

  status)
    echo "===================================================================="
    echo ">>> Checking CPU load on worker nodes:"
    echo "===================================================================="
    for n in lab02 lab03 lab04 lab05; do
      IP=$(get_ip "$n")
      echo -n "  $n ($IP): "
      ssh -o ConnectTimeout=2 icip@"$IP" "uptime | awk -F'load average:' '{print \$2}'" 2>/dev/null || echo "offline"
    done
    ;;

  *)
    echo "Usage:"
    echo "  $0 start <node> [duration_seconds]   (e.g., $0 start lab03 300)"
    echo "  $0 stop <node|all>                   (e.g., $0 stop lab03)"
    echo "  $0 status"
    exit 1
    ;;
esac
