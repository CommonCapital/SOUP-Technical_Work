"""Shared runtime helpers: building the engine, memory/time probes, rollouts
with rewards, JSONL logging."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch

from .l2l import L2LEngine
from .lora import LoRA
from .prompts import completion_has_thinking, get_tokenizer, render_prompt
from .reward import countdown_reward
from .weights import WeightStore

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_model_dir(model: str) -> Path:
    p = Path(model)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt",
                                                         "vocab.json", "tokenizer*"]))


def build_engine(model: str, device: str, dtype: str = "bf16", lora_r: int = 0, lora_alpha: float = 32.0,
                 adapter: str | None = None, seed: int = 0, **engine_kw) -> tuple[L2LEngine, Path]:
    mdir = resolve_model_dir(model)
    t0 = time.time()
    store = WeightStore(mdir, dtype=DTYPES[dtype])
    lora = None
    if lora_r or adapter:
        if adapter:
            cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
            lora_r, lora_alpha = cfg["r"], cfg["lora_alpha"]
        lora = LoRA(store.dims, r=lora_r, alpha=lora_alpha, device=device, seed=seed)
        if adapter:
            lora.load(adapter)
    eng = L2LEngine(store, device=device, lora=lora, **engine_kw)
    eng.timers["load"] = time.time() - t0
    return eng, mdir


def stop_ids(tok) -> tuple[int, ...]:
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    return tuple(sorted(i for i in ids if i is not None and i >= 0))


def encode_prompts(tok, puzzles: list[dict], version: str | None = None) -> list[list[int]]:
    from .prompts import DEFAULT_PROMPT
    v = version or DEFAULT_PROMPT
    return [tok(render_prompt(p["numbers"], p["target"], tok, v), add_special_tokens=False).input_ids
            for p in puzzles]


def score_completions(tok, puzzles, outs, stops) -> list[dict]:
    rows = []
    for p, o in zip(puzzles, outs):
        toks = [t for t in o.tokens if t not in stops]
        text = tok.decode(toks, skip_special_tokens=False)
        r = countdown_reward(text, p["numbers"], p["target"])
        rows.append({
            "id": p["id"], "numbers": p["numbers"], "target": p["target"], "completion": text,
            "n_tokens": len(o.tokens), "finish": o.finish, "thinking": completion_has_thinking(text),
            "tokens": list(o.tokens),   # exact generated ids (KL on held-out answers, H28)
            **r.as_dict(),
        })
    return rows


# ---------------- probes

def gpu_mem() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {"alloc_gb": torch.cuda.memory_allocated() / 2**30, "reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def host_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if os.uname().sysname == "Darwin" else r / 2**20  # bytes on mac, KB on linux


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()


def load_puzzles(path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit else rows


def tokenizer_for(mdir: Path):
    return get_tokenizer(str(mdir)) if (mdir / "tokenizer_config.json").exists() else get_tokenizer()
