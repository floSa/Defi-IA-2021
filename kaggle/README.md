# Kaggle GPU workflow (Step C transformer)

Reproducible remote fine-tuning on a Kaggle T4 (16 GB), driven entirely from
the CLI. The competition data is already hosted on Kaggle, so nothing large is
uploaded.

## One-time setup (needs your token)

1. kaggle.com → **Settings** → **API** → **Create New Token** → downloads
   `kaggle.json` (`{"username": "...", "key": "..."}`).
2. Place it so the CLI finds it:
   ```bash
   mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
3. Put your username into `kernel-metadata.json` (`id` field, replacing
   `KAGGLE_USERNAME`).
4. Install the CLI in the venv: `.venv/bin/pip install kaggle`.

> The competition data source requires that the account has joined the
> competition (Rules → *I understand and accept*). If the data isn't attachable,
> we upload it once as a private Kaggle Dataset and switch `competition_sources`
> for `dataset_sources`.

## Run

```bash
# Push the kernel and start it on Kaggle's GPU:
.venv/bin/kaggle kernels push -p kaggle/

# Poll status:
.venv/bin/kaggle kernels status KAGGLE_USERNAME/defi-ia-2021-transformer

# Pull outputs when complete (submission.csv, test_logits.npy, valid_metrics.json):
.venv/bin/kaggle kernels output KAGGLE_USERNAME/defi-ia-2021-transformer -p models/kaggle_out/
```

Swap the backbone via the `MODEL_NAME` knob at the top of
`train_transformer_kernel.py` (`answerdotai/ModernBERT-base`,
`microsoft/deberta-v3-base`, …), then push again.

## Expected time

ModernBERT-base / DeBERTa-v3-base, 217k samples, seq 256, 2–3 epochs on T4:
~2–4 h. Session limit ~9–12 h, weekly GPU quota 30 h.
