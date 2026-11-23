"""Merge a LoRA adapter into the base weights, one tensor at a time.

W_new = W + (alpha / r) * B @ A, computed in fp32 on the GPU, cast to bf16,
written to new safetensors shards with the original names. Peak GPU memory is
one weight tensor in fp32 plus A and B (~0.2 GB for Qwen3-4B's largest MLP
matrix). CPU side: one shard at a time via mmap.

Usage: uv run python scripts/merge.py --model Qwen/Qwen3-4B --adapter runs/main/ckpt/step_00040 --out merged/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.runtime import nvidia_smi_used_mib, resolve_model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dtype", default="bf16", choices=["bf16", "fp32"],
                    help="bf16 = same size as the base model (the shipped artifact); fp32 = diagnostic, 2x size")
    a = ap.parse_args()

    mdir = resolve_model_dir(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(Path(a.adapter) / "adapter_model.safetensors"))
    # base_model.model.model.layers.3.mlp.up_proj.lora_A.weight -> model.layers.3.mlp.up_proj.weight
    pairs = {}
    for k in lora:
        if k.endswith(".lora_A.weight"):
            base = k[len("base_model.model."):-len(".lora_A.weight")] + ".weight"
            pairs[base] = (lora[k], lora[k.replace("lora_A", "lora_B")])

    idx = mdir / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    merged, smi_peak, max_rel = 0, 0, 0.0
    d2, err2, lost, n_nz = 0.0, 0.0, 0, 0
    out_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.out_dtype]
    for sh in shards:
        new = {}
        with safe_open(str(mdir / sh), framework="pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in pairs:
                    A, B = (x.to(a.device, torch.float32) for x in pairs[k])
                    w0 = t.to(a.device, torch.float32)
                    delta = scale * (B @ A)
                    assert delta.shape == w0.shape, f"{k}: delta {tuple(delta.shape)} vs W {tuple(w0.shape)}"
                    w = (w0 + delta).to(out_dtype)
                    # How much of the adapter's change survives storage in out_dtype:
                    # realized = stored(W + d) - W. In bf16, changes below half a
                    # rounding step of W vanish ("lost"), so a merged bf16 model can
                    # behave like the base model even though the math is right (E07).
                    realized = w.float() - w0
                    d2 += float(delta.pow(2).sum())
                    err2 += float((realized - delta).pow(2).sum())
                    nz = delta != 0
                    lost += int(((realized == 0) & nz).sum())
                    n_nz += int(nz.sum())
                    new[k] = w.cpu()
                    max_rel = max(max_rel, float(delta.norm() / w0.norm()))
                    merged += 1
                    del A, B, w, w0, delta, realized
                    s = nvidia_smi_used_mib()
                    smi_peak = max(smi_peak, s or 0)
                else:
                    new[k] = t
        save_file(new, str(out / sh), metadata=meta or {"format": "pt"})
        del new
    assert merged == len(pairs), f"merged {merged} tensors but adapter has {len(pairs)} (module name mismatch?)"
    for f in mdir.iterdir():
        if f.suffix in (".json", ".txt") or f.name.startswith("tokenizer") or f.name == "merges.txt":
            if not (out / f.name).exists():
                shutil.copy(f, out / f.name)
    rep = {"merged_tensors": merged, "scale": scale, "seconds": time.time() - t0,
           "gpu_peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
           "smi_peak_mib": smi_peak or None, "max_relative_delta": max_rel, "out_dtype": a.out_dtype,
           "delta_lost_to_rounding_share": lost / max(1, n_nz),
           "delta_relative_rounding_error": (err2 / d2) ** 0.5 if d2 else None}
    (out / "merge_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
