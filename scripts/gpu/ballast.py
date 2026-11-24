"""Enforce an 8 GB cap on a bigger card (Kaggle T4 = 16 GB) for every other
process, the way a physical 8 GB card would.

This separate process allocates GPU memory until exactly CAP_MIB remain free,
then sleeps. Any later process (our trainer) physically cannot use more than
CAP_MIB, CUDA context included: if it tried, it would hit out-of-memory.
An allocator limit (set_per_process_memory_fraction) would not count, because
it does not cover the CUDA context and is a soft limit inside our own process.

Writes results/logs/ballast.json with the memory it holds, so the monitor can
report our usage as  nvidia-smi used - ballast_used.
Usage: python scripts/gpu/ballast.py --cap-mib 8192 &   (then wait for the file)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def smi_used(dev: int) -> int:
    o = subprocess.run(["nvidia-smi", f"--id={dev}", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    return int(o[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mib", type=int, default=8192)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "logs" / "ballast.json"))
    a = ap.parse_args()

    before = smi_used(a.device)
    torch.cuda.set_device(a.device)
    torch.empty(1, device="cuda")  # create the context
    free, total = torch.cuda.mem_get_info()
    hold = free - a.cap_mib * 2**20
    if hold <= 0:
        raise SystemExit(f"card has only {free / 2**20:.0f} MiB free; nothing to hold")
    # One big block, allocated in pieces of 256 MiB to avoid a single huge request failing.
    blocks, left = [], hold
    while left > 0:
        n = min(left, 256 * 2**20)
        blocks.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
        left -= n
    torch.cuda.synchronize()
    time.sleep(1)
    free_after, _ = torch.cuda.mem_get_info()
    after = smi_used(a.device)
    info = {"cap_mib": a.cap_mib, "total_mib": total // 2**20, "free_left_mib": free_after // 2**20,
            "ballast_used_mib": after - before, "smi_used_before_mib": before, "smi_used_after_mib": after,
            "pid": __import__("os").getpid(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
