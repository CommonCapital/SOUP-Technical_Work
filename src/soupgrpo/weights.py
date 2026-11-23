"""Weight store: model weights live in CPU RAM (pinned when CUDA is present),
one flat contiguous buffer per decoder layer, so bringing a layer to the GPU
is a single copy. Embedding, final norm (and LM head if untied) stay resident
on the GPU.

Loads HF safetensors shards one at a time with mmap (safe_open), so peak CPU
memory is the store itself plus one tensor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .qwen3_math import LAYER_TENSORS, Qwen3Dims


@dataclass
class TensorSlot:
    name: str
    offset: int  # elements
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


class WeightStore:
    def __init__(self, model_dir: str | Path, dtype=torch.bfloat16, pin: bool | None = None):
        self.model_dir = Path(model_dir)
        cfg = json.loads((self.model_dir / "config.json").read_text())
        self.config = cfg
        self.dims = Qwen3Dims.from_hf_config(cfg)
        self.dtype = dtype
        self.pin = torch.cuda.is_available() if pin is None else pin

        d = self.dims
        shapes = {
            "ln1": (d.hidden,), "ln2": (d.hidden,), "q_norm": (d.head_dim,), "k_norm": (d.head_dim,),
            **{n: (o, i) for n, (i, o) in d.proj_io().items()},
        }
        self.layout: list[TensorSlot] = []
        off = 0
        for short in LAYER_TENSORS.values():
            s = TensorSlot(short, off, shapes[short])
            self.layout.append(s)
            off += s.numel
        self.layer_numel = off
        self.layer_bytes = off * torch.tensor([], dtype=dtype).element_size()

        self.layers: list[torch.Tensor] = []
        for _ in range(d.n_layers):
            t = torch.empty(off, dtype=dtype)
            self.layers.append(t.pin_memory() if self.pin else t)
        self.resident: dict[str, torch.Tensor] = {}
        self._load()
        self.checksums = [float(t.float().sum()) for t in self.layers]

    def _shards(self) -> list[Path]:
        idx = self.model_dir / "model.safetensors.index.json"
        if idx.exists():
            files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
            return [self.model_dir / f for f in files]
        return [self.model_dir / "model.safetensors"]

    def _load(self) -> None:
        slot = {s.name: s for s in self.layout}
        seen = set()
        for shard in self._shards():
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    if key.startswith("model.layers."):
                        rest = key[len("model.layers."):]
                        li, suffix = rest.split(".", 1)
                        if suffix not in LAYER_TENSORS:
                            raise KeyError(f"unexpected layer tensor {key}")
                        s = slot[LAYER_TENSORS[suffix]]
                        if tuple(t.shape) != s.shape:
                            raise ValueError(f"{key}: shape {tuple(t.shape)} != {s.shape}")
                        self.layers[int(li)][s.offset:s.offset + s.numel].copy_(t.reshape(-1))
                    elif key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                        self.resident[key] = t.to(self.dtype).contiguous()
                    else:
                        raise KeyError(f"unexpected tensor {key}")
                    seen.add(key)
        expected = {f"model.layers.{i}.{s}" for i in range(self.dims.n_layers) for s in LAYER_TENSORS}
        missing = expected - seen
        if missing:
            raise KeyError(f"missing {len(missing)} layer tensors, e.g. {sorted(missing)[:3]}")
        if "lm_head.weight" not in self.resident and not self.dims.tie_embeddings:
            raise KeyError("lm_head.weight missing and embeddings are not tied")

    def views(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Named tensor views into a flat layer buffer (CPU or GPU)."""
        return {s.name: flat[s.offset:s.offset + s.numel].view(s.shape) for s in self.layout}

    def layer_views(self, i: int) -> dict[str, torch.Tensor]:
        return self.views(self.layers[i])
