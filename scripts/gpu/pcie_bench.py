"""PCIe host-to-device bandwidth (plan step 6) and CUDA context cost (H13).

Copies one Qwen3-4B layer's worth (202 MB) from pinned and from pageable host
memory, 20 times each, timed with CUDA events. Also verifies is_pinned() and
measures nvidia-smi memory after CUDA init + one matmul, before any weights.
Writes results/logs/pcie_<ts>.json.
"""

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def smi():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split()
    return int(o[0])


def main():
    before = smi()
    torch.cuda.init()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    torch.cuda.synchronize()
    ctx = smi()
    res = {"smi_mib_before_init": before, "smi_mib_after_init_matmul": ctx,
           "context_plus_workspace_gib": (ctx - before) / 1024}

    n = 101_000_000  # ~202 MB in bf16 = one Qwen3-4B decoder layer
    dst = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for kind in ("pinned", "pageable"):
        src = torch.empty(n, dtype=torch.bfloat16)
        if kind == "pinned":
            src = src.pin_memory()
        res[f"{kind}_is_pinned"] = src.is_pinned()
        times = []
        for i in range(22):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            dst.copy_(src, non_blocking=True)
            e.record()
            e.synchronize()
            if i >= 2:
                times.append(s.elapsed_time(e) / 1000)
        gbs = [n * 2 / t / 1e9 for t in times]
        res[f"{kind}_GBps_median"] = statistics.median(gbs)
        res[f"{kind}_GBps_min_max"] = [min(gbs), max(gbs)]
    res["predicted_decode_pass_s"] = 7.26e9 / (res["pinned_GBps_median"] * 1e9)
    res["gpu"] = torch.cuda.get_device_name(0)
    out = ROOT / "results" / "logs" / f"pcie_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    if not res["pinned_is_pinned"]:
        sys.exit("pinned allocation fell back to pageable memory")


if __name__ == "__main__":
    main()
