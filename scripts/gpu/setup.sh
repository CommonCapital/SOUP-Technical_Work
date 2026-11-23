#!/usr/bin/env bash
# One-time setup on the rented GPU host. Records the environment (plan step 6:
# "record GPU, CPU, link width and bandwidth") to results/logs/env_<ts>.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
ENV=results/logs/env_$TS.txt
mkdir -p results/logs

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync

{
  echo "== date"; date -Is
  echo "== nvidia-smi"; nvidia-smi
  echo "== gpu query"; nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.sm --format=csv
  echo "== cpu"; lscpu | head -20
  echo "== memory"; free -g
  echo "== cgroup memory limit"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo n/a
  echo "== disk"; df -h .
  echo "== torch"; uv run python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'bf16', torch.cuda.is_bf16_supported())"
  echo "== git"; git rev-parse HEAD 2>/dev/null || echo "no git"
} | tee "$ENV"

# Hard requirements from the plan
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA"
cap = torch.cuda.get_device_capability(0)
assert cap >= (8, 0), f"need Ampere or newer for bf16, got sm_{cap[0]}{cap[1]}"
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU total memory {total:.2f} GiB",
      "(physical 8 GB card: cap enforced by hardware)" if total < 8.5 else
      "(LARGER than 8 GB: the 8,192 MiB cap must be proven from the nvidia-smi log)")
EOF

echo "Downloading Qwen/Qwen3-4B (~8 GB) ..."
uv run python -c "import sys; sys.path.insert(0,'src'); from soupgrpo.runtime import resolve_model_dir; print(resolve_model_dir('Qwen/Qwen3-4B'))"
echo "env written to $ENV"
