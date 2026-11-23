#!/usr/bin/env bash
# Log nvidia-smi and host RAM every second for the whole session.
# Start once, in its own tmux window:   bash scripts/gpu/monitor.sh
# Files: results/logs/nvsmi_<ts>.csv, results/logs/hostmem_<ts>.csv
set -uo pipefail
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p results/logs
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active \
  --format=csv -l 1 > "results/logs/nvsmi_$TS.csv" &
NV=$!
echo "time,host_used_mib,host_avail_mib,cgroup_mib" > "results/logs/hostmem_$TS.csv"
trap 'kill $NV 2>/dev/null' EXIT
while true; do
  read -r used avail < <(free -m | awk '/Mem:/ {print $3, $7}')
  cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  echo "$(date -Is),$used,$avail,$((cg / 1048576))" >> "results/logs/hostmem_$TS.csv"
  sleep 1
done
