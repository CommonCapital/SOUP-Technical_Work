# countdown-lora-main: Qwen3-4B + LoRA, trained with GRPO on Countdown

This is the adapter from my main run: 40 GRPO steps on 5-number Countdown puzzles, under
an 8 GB GPU cap. **My verdict on it is DON'T SHIP** (see `report/report.pdf`). It beats the
base model, but a control trained with shuffled rewards beats it too. I include it so you
can load it and check the numbers yourself, not as a model to use.

| Held-out, 500 puzzles, greedy, 512 tokens | Accuracy |
|---|---|
| Base Qwen3-4B | 19.2% |
| **This adapter** | **24.8%** (+5.6 [+1.2, +10.0]) |
| Control adapter (shuffled rewards) | 28.8% |

**Files.** `adapter_model.safetensors` is the LoRA (rank 16, α 32, on q/k/v/o/gate/up/down,
33.0 M parameters), in PEFT format. Training ran in fp32. I store it in **fp16** so it fits
in a git repo (66 MB instead of 132 MB). `export_check.json` shows what that costs: the
change to the weights (ΔW = B·A) moves by **0.03%** (`scripts/export_adapter.py`). That
check is on the weights only; I did not re-run the evaluation with the fp16 copy. The fp32
original is available on request.

**Use it as an adapter, do not merge it into bf16 weights.** ΔW is so small that rounding
W + ΔW to bf16 throws away 87% of it (H29 in the report). The merged bf16 model behaves like
a different model.

**Load it** (Hugging Face + PEFT):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "model/countdown-lora-main")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
```

Use the same prompt as in training (`src/soupgrpo/prompts.py`, version v2, thinking off),
otherwise the numbers above don't apply. In this repo,
`scripts/sample_eval.py --adapter model/countdown-lora-main --data data/heldout_n5.jsonl --samples 1
--temperature 0` runs the held-out evaluation with my own engine (needs a GPU).
