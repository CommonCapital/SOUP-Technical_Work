"""Export the trained LoRA adapter in fp16 (half the size: fits in a normal git repo)
and measure what that costs. Unlike the bf16 *merge* (H29: lost 87% of the change,
because dW is tiny next to W), here A and B are rounded on their own, so the error
is relative to dW itself: fp16 has ~11 bits of mantissa -> expected ~1e-3.

  python scripts/export_adapter.py results/kaggle/main/adapters/main model/countdown-lora-main
Writes the adapter + adapter_config.json + export_check.json into the output folder.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main(src: str, dst: str) -> None:
    src, dst = Path(src), Path(dst)
    sd = load_file(str(src / "adapter_model.safetensors"))
    half = {k: v.to(torch.float16).contiguous() for k, v in sd.items()}
    assert all(torch.isfinite(v).all() for v in half.values()), "fp16 overflow"
    worst, total_err, total_ref = 0.0, 0.0, 0.0
    n = 0
    for ka in sd:
        if ".lora_A." not in ka:
            continue
        kb = ka.replace(".lora_A.", ".lora_B.")
        ref = sd[kb].double() @ sd[ka].double()
        got = half[kb].double() @ half[ka].double()
        err, nrm = float((got - ref).norm()), float(ref.norm())
        worst = max(worst, err / max(nrm, 1e-30))
        total_err += err ** 2
        total_ref += nrm ** 2
        n += 1
    rep = {"modules": n, "dW_rel_error_overall": (total_err / total_ref) ** 0.5, "dW_rel_error_worst_module": worst,
           "bytes_fp32": (src / "adapter_model.safetensors").stat().st_size}
    dst.mkdir(parents=True, exist_ok=True)
    save_file(half, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    rep["bytes_fp16"] = (dst / "adapter_model.safetensors").stat().st_size
    (dst / "export_check.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
