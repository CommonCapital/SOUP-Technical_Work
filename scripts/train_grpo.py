"""GRPO + LoRA training with the L2L engine.

One step: 16 prompts x G=8 answers -> rewards -> group advantages -> one
training pass (forward + backward, streamed) -> one AdamW update -> atomic
checkpoint. Resumable after kill -9 (same mechanism as scripts/resume_test.py,
plus adapter, optimizer and sampling-RNG state in the same commit).

Control run: --shuffle-rewards (rewards permuted across the batch) or --lr 0.

Example (GPU):
  uv run python scripts/train_grpo.py --model Qwen/Qwen3-4B --data data/train_n4.jsonl \
      --run-dir runs/main --steps 40
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soupgrpo.data_stream import (PromptStream, append_id_log, atomic_write_json, load_state,  # noqa: E402
                                  reconcile_id_log, save_state)
from soupgrpo.grpo import group_advantages, make_loss_fn, pack_sequences, shuffled_rewards  # noqa: E402
from soupgrpo.runtime import (JsonlLog, build_engine, encode_prompts, gpu_mem, host_rss_gb,  # noqa: E402
                              nvidia_smi_used_mib, score_completions, stop_ids, tokenizer_for)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--train-seqs-per-pass", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--log-kl-every", type=int, default=5, help="compute KL(policy||base) every k steps when beta=0")
    ap.add_argument("--no-std-scale", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--shuffle-rewards", action="store_true", help="negative control")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-buffers", type=int, default=3)
    ap.add_argument("--resident-layers", type=int, default=0)
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--logit-chunk", type=int, default=512)
    ap.add_argument("--keep-ckpts", type=int, default=2)
    ap.add_argument("--save-samples", type=int, default=4, help="completions per step written to samples.jsonl")
    return ap.parse_args()


def main():
    a = parse()
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    # One trainer per run directory. The OS releases the lock when the process
    # dies (even on kill -9), so a resume works, but a second live trainer
    # (e.g. a resume started while the old process is still running) fails.
    lock = open(run / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another trainer is running in {run}")
    state_path, ids_log = run / "state.json", run / "ids.jsonl"
    metrics, samples = JsonlLog(run / "metrics.jsonl"), JsonlLog(run / "samples.jsonl")

    loaded = load_state(state_path)
    if loaded is None:
        (run / "config.json").write_text(json.dumps(vars(a), indent=2))
    else:
        old = json.loads((run / "config.json").read_text())
        diff = {k: (old.get(k), v) for k, v in vars(a).items() if old.get(k) != v and k not in ("steps",)}
        if diff:
            raise SystemExit(f"config differs from the run being resumed: {diff}")

    torch.manual_seed(a.seed)
    eng, mdir = build_engine(a.model, a.device, a.dtype, lora_r=a.lora_r, lora_alpha=a.lora_alpha, seed=a.seed,
                             n_buffers=a.n_buffers, resident_layers=a.resident_layers,
                             prefetch=not a.no_prefetch, logit_chunk=a.logit_chunk)
    tok = tokenizer_for(mdir)
    stops = stop_ids(tok)
    pad_id = tok.pad_token_id
    params = eng.lora.parameters()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    gen = torch.Generator(device=a.device).manual_seed(a.seed + 1)
    shuf_rng = np.random.default_rng(a.seed + 2)
    print(f"LoRA trainable params: {eng.lora.n_params():,}  (r={a.lora_r}, alpha={a.lora_alpha}, "
          f"scale={eng.lora.scale})", flush=True)
    assert eng.lora.n_params() > 0 and all(p.requires_grad for p in params)

    st = None
    if loaded is not None:
        st, extra = loaded
        reconcile_id_log(ids_log, st.step)
        ck = run / "ckpt" / extra["ckpt"]
        eng.lora.load(ck)
        opt.load_state_dict(torch.load(ck / "optimizer.pt", map_location=a.device))
        gen.set_state(torch.load(ck / "gen_state.pt"))
        shuf_rng.bit_generator.state = extra["shuffle_rng"]
        print(f"resumed at step {st.step} from {ck}", flush=True)
    stream = PromptStream(a.data, seed=a.seed, state=st)
    prev_delta = sum(x ** 2 for x in eng.lora.delta_norms()) ** 0.5

    while stream.state.step < a.steps:
        step = stream.state.step
        m: dict = {"step": step}
        T = {}
        t0 = time.perf_counter()
        copies0, bytes0 = eng.streamer.stats.copies, eng.streamer.stats.bytes
        eng.timers.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- rollouts (on-policy: current LoRA is active)
        batch = stream.next_batch(a.prompts_per_step)
        append_id_log(ids_log, step, [r["id"] for r in batch], {"epochs": sorted({r["_epoch"] for r in batch})})
        pids = encode_prompts(tok, batch)
        rep_p = [p for p in pids for _ in range(a.group)]
        rep_z = [z for z in batch for _ in range(a.group)]
        outs = eng.generate(rep_p, a.max_new, a.temperature, stops, pad_id, a.gen_batch, gen)
        rows = score_completions(tok, rep_z, outs, stops)
        T["generate"] = time.perf_counter() - t0
        m["gpu_gen"] = gpu_mem()
        m["smi_mib_after_gen"] = nvidia_smi_used_mib()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- rewards and advantages
        rewards = np.array([r["reward"] for r in rows])
        r_train = shuffled_rewards(rewards, shuf_rng) if a.shuffle_rewards else rewards
        adv = group_advantages(r_train, a.group, scale_by_std=not a.no_std_scale)
        grp = r_train.reshape(-1, a.group)
        zero_groups = float(np.mean(grp.min(1) == grp.max(1)))
        sel = np.arange(len(rows)) if a.beta > 0 else np.nonzero(adv != 0)[0]

        # ---- training pass(es)
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        lr_now = a.lr * min(1.0, (step + 1) / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr_now
        stats: dict = {}
        loss_total = 0.0
        want_ref = a.beta > 0 or (a.log_kl_every and step % a.log_kl_every == 0)
        if len(sel):
            ids, attn, lm = pack_sequences([rep_p[i] for i in sel], [outs[i].tokens for i in sel], pad_id)
            adv_t = torch.tensor(adv[sel], dtype=torch.float32, device=a.device)
            norm = float(len(rows) * a.max_new)
            for c in range(0, len(sel), a.train_seqs_per_pass):
                sl = slice(c, c + a.train_seqs_per_pass)
                fn = make_loss_fn(adv_t[sl], a.beta, norm, stats)
                out = eng.train_pass(ids[sl], attn[sl], lm[sl], fn, a.micro_batch, with_ref=bool(want_ref))
                loss_total += out["loss"]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip)) if len(sel) else 0.0
        T["train"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        if len(sel):
            opt.step()
        T["optimizer"] = time.perf_counter() - t2
        m["gpu_train"] = gpu_mem()
        m["smi_mib_after_train"] = nvidia_smi_used_mib()

        # ---- checkpoint (commit point): files first, then state.json atomically
        t3 = time.perf_counter()
        name = f"step_{step + 1:05d}"
        ck = run / "ckpt" / name
        tmp = run / "ckpt" / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        eng.lora.save(tmp)
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        torch.save(gen.get_state(), tmp / "gen_state.pt")
        if ck.exists():
            shutil.rmtree(ck)
        tmp.rename(ck)
        stream.state.step = step + 1
        save_state(state_path, stream.state, {"ckpt": name, "shuffle_rng": shuf_rng.bit_generator.state})
        for old in sorted((run / "ckpt").glob("step_*"))[:-a.keep_ckpts]:
            if not old.name.endswith(".tmp"):
                shutil.rmtree(old)
        T["checkpoint"] = time.perf_counter() - t3

        # ---- metrics
        delta = sum(x ** 2 for x in eng.lora.delta_norms()) ** 0.5
        n = len(rows)
        m.update({
            "reward": float(rewards.mean()), "reward_train_signal": float(r_train.mean()),
            "tag_rate": float(np.mean([r["tag_present"] for r in rows])),
            "parsed_rate": float(np.mean([r["parsed"] for r in rows])),
            "numbers_rate": float(np.mean([r["numbers_correct"] for r in rows])),
            "value_rate": float(np.mean([r["value_correct"] for r in rows])),
            "zero_adv_group_share": zero_groups, "n_trained_seqs": int(len(sel)),
            "len_mean": float(np.mean([r["n_tokens"] for r in rows])),
            "truncated_share": float(np.mean([r["finish"] == "length" for r in rows])),
            "thinking_share": float(np.mean([r["thinking"] for r in rows])),
            "loss": loss_total, "grad_norm": grad_norm, "lr": lr_now,
            "adapter_delta_norm": delta, "adapter_delta_change": delta - prev_delta,
            "kl_mean": (stats["kl_sum"] / stats["n_tok"]) if "kl_sum" in stats and stats.get("n_tok") else None,
            "copies": eng.streamer.stats.copies - copies0,
            "copied_gb": (eng.streamer.stats.bytes - bytes0) / 2**30,
            "time_s": {**T, "total": time.perf_counter() - t0},
            "engine_timers_s": dict(eng.timers), "host_rss_gb": host_rss_gb(),
            "epoch": stream.state.epoch, "position": stream.state.position,
        })
        prev_delta = delta
        metrics.write(m)
        for r in rows[:a.save_samples]:
            samples.write({"step": step, **r})
        print(f"step {step:3d} reward {m['reward']:.3f} value {m['value_rate']:.3f} tag {m['tag_rate']:.3f} "
              f"zero-adv {zero_groups:.2f} trained {len(sel)}/{n} len {m['len_mean']:.0f} trunc "
              f"{m['truncated_share']:.2f} gnorm {grad_norm:.3g} dnorm {delta:.4g} lr {lr_now:.2g} "
              f"t {m['time_s']['total']:.1f}s (gen {T['generate']:.1f} train {T['train']:.1f})", flush=True)

    atomic_write_json(run / "DONE.json", {"steps": stream.state.step})
    print("done", flush=True)


if __name__ == "__main__":
    main()
