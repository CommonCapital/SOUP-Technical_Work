"""Run the L2L equivalence suite (H17, H18, H4) and write a timestamped log.

Usage: uv run python scripts/equivalence_test.py [n_layers]
"""

import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.equivalence import run_all  # noqa: E402


def main():
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t0 = time.time()
    res = run_all(n_layers=n_layers, verbose=False)
    bf = res["bf16"]
    res["bf16_PASS"] = all(bf[m]["l2l_logits"] <= 1.5 * bf[m]["noise_logits"]
                           and bf[m]["l2l_grads"] <= 1.5 * bf[m]["noise_grads"] for m in ("nmax", "rfro"))
    res["padding_PASS"] = res["padding_invariance_nmax"] <= 1e-5
    res["PASS"] = (res["fp32"]["PASS"] and res["bf16_PASS"] and res["generation_greedy"]["PASS"]
                   and res["copies"]["PASS"] and res["padding_PASS"]
                   and res["gen_vs_train_logp_nmax"] <= 1e-5 and res["ref_logp_vs_adapter_off"] <= 1e-6
                   and max(res["variants_nmax"].values()) <= 1e-5)
    res["env"] = {"torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu",
                  "platform": platform.platform(), "seconds": round(time.time() - t0, 1)}
    out = ROOT / "results" / "logs" / f"equivalence_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("fp32", "copies", "generation_greedy", "padding_invariance_nmax",
                                          "gen_vs_train_logp_nmax", "bf16_PASS", "PASS")}, indent=1, default=str))
    print("log:", out)
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
