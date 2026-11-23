"""LoRA adapters for all decoder layers, resident on the GPU in fp32.

Per layer and target projection: A [r, in] (Kaiming-uniform, as in PEFT),
B [out, r] (zeros), scale = alpha / r. Output = W x + scale * B A x.
With B = 0 at init, the adapted model starts identical to the base model.

Saved in PEFT-compatible names so the adapter can be loaded elsewhere:
  base_model.model.model.layers.{i}.{self_attn|mlp}.{proj}.lora_{A|B}.weight
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .qwen3_math import PROJ_SHAPES, Qwen3Dims

_PARENT = {p: ("self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp") for p in PROJ_SHAPES}


class LoRA:
    def __init__(self, dims: Qwen3Dims, r: int = 16, alpha: float = 32.0, targets=PROJ_SHAPES,
                 device="cpu", seed: int = 0, b_init_std: float = 0.0):
        self.dims, self.r, self.alpha, self.targets = dims, r, alpha, tuple(targets)
        self.scale = alpha / r
        g = torch.Generator().manual_seed(seed)
        io = dims.proj_io()
        self.layers: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
        for _ in range(dims.n_layers):
            d = {}
            for name in self.targets:
                fin, fout = io[name]
                bound = 1.0 / math.sqrt(fin)  # kaiming_uniform_(a=sqrt(5)) bound
                a = (torch.rand(r, fin, generator=g) * 2 - 1) * bound
                b = torch.randn(fout, r, generator=g) * b_init_std if b_init_std else torch.zeros(fout, r)
                d[name] = (a.to(device).requires_grad_(), b.to(device).requires_grad_())
            self.layers.append(d)
        self.enabled = True

    def layer(self, i: int):
        return self.layers[i] if self.enabled else None

    def parameters(self) -> list[torch.Tensor]:
        return [t for d in self.layers for a, b in d.values() for t in (a, b)]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def delta_norms(self) -> list[float]:
        """Per-layer Frobenius norm of scale * B A summed over projections
        (silent-failure check: adapter really changes)."""
        out = []
        with torch.no_grad():
            for d in self.layers:
                out.append(float(sum((self.scale * b @ a).norm() ** 2 for a, b in d.values()) ** 0.5))
        return out

    def named_tensors(self) -> dict[str, torch.Tensor]:
        out = {}
        for i, d in enumerate(self.layers):
            for name, (a, b) in d.items():
                base = f"base_model.model.model.layers.{i}.{_PARENT[name]}.{name}"
                out[f"{base}.lora_A.weight"] = a.detach().cpu().contiguous()
                out[f"{base}.lora_B.weight"] = b.detach().cpu().contiguous()
        return out

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(self.named_tensors(), str(out / "adapter_model.safetensors"))
        (out / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": self.r, "lora_alpha": self.alpha,
            "target_modules": list(self.targets), "lora_dropout": 0.0, "bias": "none",
            "task_type": "CAUSAL_LM",
        }, indent=2))

    def load(self, in_dir: str | Path) -> None:
        sd = load_file(str(Path(in_dir) / "adapter_model.safetensors"))
        with torch.no_grad():
            for i, d in enumerate(self.layers):
                for name, (a, b) in d.items():
                    base = f"base_model.model.model.layers.{i}.{_PARENT[name]}.{name}"
                    a.copy_(sd[f"{base}.lora_A.weight"])
                    b.copy_(sd[f"{base}.lora_B.weight"])
