# Running on Kaggle

## Step 1: baseline pilot (`soup_pilot.ipynb`, ~2 hours)

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `kaggle/soup_pilot.ipynb`.
2. Right panel → **Session options**:
   - **Accelerator: GPU T4 x2** (or a single T4). Not P100.
   - **Internet: On** (needs a phone-verified account). It downloads Qwen3-4B (~8 GB) and installs packages.
3. **Run All**. Keep the tab open, or use **Save Version → Save & Run All** to run in the background
   (up to 12 h; outputs are kept with the version).
4. When the last cell prints `DOWNLOAD: /kaggle/working/soup_results.zip`, download it from the
   **Output** panel and put it in `soup-grpo/kaggle/` on the Mac (or send it).

The notebook stops on its own if a safety check fails (8 GB cap not in place, engine not equivalent
on CUDA, fp16 changes the model). The reason is printed, and logs are still in the zip.

Rebuild after code changes: `uv run python kaggle/build_notebook.py`. The notebook embeds the code,
so an old notebook keeps running old code; check the git rev printed by cell 1.
