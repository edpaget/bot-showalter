# Cloud GPU Training Guide

Parent doc: [contextual-event-embeddings.md](../archive/contextual-event-embeddings.md)

Training the contextual transformer locally on MPS (Apple Silicon) or CPU is impractical for the full dataset (~5.6M training pitches, 30 epochs). This guide covers how to run training on Modal's serverless GPUs.

---

## What We're Working With

The training code is already cloud-ready:

- **Device auto-detection** (`contextual/cli.py`): Selects CUDA > MPS > CPU automatically.
- **Checkpoint resume**: `--resume-from` flag restores model, optimizer, and scheduler state. Training can survive preemptions.
- **CLI entry point**: `uv run fantasy-baseball-manager contextual pretrain --seasons ...` runs the full pipeline.
- **File-based persistence**: Checkpoints go to `~/.fantasy_baseball/models/contextual/`. Data loads from `~/.fantasy_baseball/statcast/`.
- **AMP**: Automatic mixed precision is enabled by default on CUDA, roughly doubling throughput.
- **DataLoader workers**: Auto-detected on CUDA (`num_workers=4`, `pin_memory=True`, `persistent_workers=True`).

---

## Modal Setup

Modal provides zero-infrastructure GPU training — define the environment in Python, and Modal handles provisioning, billing per-second, and scaling to zero when idle. The `scripts/modal_train.py` wrapper delegates to the existing CLI so the training code needs no changes.

Modal gives $30/month free credit. The default T4 GPU costs $0.59/hr (~50 hours/month free).

### Prerequisites

```bash
# Install Modal (optional dependency group)
uv sync --group modal

# Authenticate with Modal
modal setup
```

### Data Upload (One-Time)

```bash
# Create the persistent volume
modal volume create fantasy-baseball-data

# Upload Statcast parquet files (~841 MB)
modal volume put fantasy-baseball-data ~/.fantasy_baseball/statcast/ statcast/

# Verify upload
modal run scripts/modal_train.py --command check
```

---

## Prepare Data (CPU-Only)

Preparing data on CPU first saves GPU-seconds by skipping parquet I/O, game-sequence building, and tensorization during training. This saves ~2-5 minutes per training run.

**Option A: Prepare locally and upload** (recommended — avoids paying for cloud CPU time):

```bash
# Prepare on your local machine
uv run fantasy-baseball-manager contextual prepare-data --mode all

# Upload prepared data to the Modal volume
modal volume put fantasy-baseball-data ~/.fantasy_baseball/prepared_data/ prepared_data/
```

**Option B: Prepare on Modal** (CPU-only, no GPU):

```bash
# Prepare all data (pretrain + finetune)
modal run scripts/modal_train.py --command prepare

# Prepare only pretrain data
modal run scripts/modal_train.py --command prepare --mode pretrain

# Prepare only finetune data for a specific perspective
modal run scripts/modal_train.py --command prepare --mode finetune --perspective pitcher
```

If you use a non-default `--max-seq-len`, pass it to `prepare-data` as well — the prepared data metadata must match training parameters or it will rebuild from scratch.

---

## Pre-Training

```bash
# Default: T4 GPU, 30 epochs, batch size 64
# Automatically loads prepared data if available
modal run scripts/modal_train.py --command pretrain

# Resume after interruption
modal run scripts/modal_train.py --command pretrain --resume-from pretrain_latest

# Skip prepared data and build from scratch
modal run scripts/modal_train.py --command pretrain --no-prepared-data
```

---

## Fine-Tuning

After pre-training completes, fine-tune using the best checkpoint.

**Important:** If you used a non-default `--max-seq-len` during pre-training, pass the same value to fine-tuning. The pre-trained model's positional embeddings must match. The `--batch-size` can differ (fine-tuning uses smaller context windows so you can increase it), but `--max-seq-len` must stay consistent.

```bash
# Fine-tune for pitcher prediction (default max-seq-len=512 matches default pretrain)
modal run scripts/modal_train.py --command finetune --perspective pitcher

# Fine-tune for batter prediction with frozen backbone
modal run scripts/modal_train.py --command finetune --perspective batter --freeze-backbone

# If you pre-trained with --max-seq-len 2048, pass it here too
modal run scripts/modal_train.py --command finetune --perspective pitcher --max-seq-len 2048
```

---

## Download Checkpoints

```bash
# Download all checkpoints (pretrain + finetune)
modal volume get fantasy-baseball-data models/contextual/ ~/.fantasy_baseball/models/contextual/

# List available checkpoints
modal volume ls fantasy-baseball-data models/contextual/
```

---

## Selecting a GPU

Set the `MODAL_GPU` environment variable before `modal run`. It's read at module scope and passed to `@app.function(gpu=...)`:

```bash
# A10G mid-tier option
MODAL_GPU=A10G modal run scripts/modal_train.py --command pretrain --epochs 30

# A100 80GB — use max_seq_len=2048 to see near-full seasons (~128 games)
# and drop batch size to 64 to fit in VRAM (attention is O(n²))
MODAL_GPU=A100-80GB modal run scripts/modal_train.py \
    --command pretrain --max-seq-len 2048 --batch-size 64 --learning-rate 1e-4 --epochs 30

# Fine-tune on A100 80GB (context_window=10 games, so 512 is fine — bump batch)
# Remember: --max-seq-len must match what you used for pretrain
MODAL_GPU=A100-80GB modal run scripts/modal_train.py \
    --command finetune --perspective pitcher --batch-size 128
```

**Sequence length vs. batch size tradeoff.** The T4 default (seq 512, batch 64) only sees ~32 recent games per player. On an A100 80GB, `--max-seq-len 2048` captures a near-full season (~128 games) but requires a smaller batch size. Fine-tuning uses a 10-game context window, so 512 is sufficient regardless of GPU.

### Budget Guidance

| GPU | Cost/hr | Free hours/mo ($30) | Use Case |
|-----|---------|---------------------|----------|
| T4 (default) | $0.59 | ~50 hrs | Budget pre-training/fine-tuning |
| A10G | $1.10 | ~27 hrs | Good throughput, reasonable cost |
| A100 40GB | $2.10 | ~14 hrs | Longer sequences (max_seq_len=1024) |
| A100 80GB | $2.50 | ~12 hrs | Full season context (max_seq_len=2048) |
| H100 | $3.95 | ~7.5 hrs | Overkill for this model size |

---

## Code Changes Worth Making

### Logging to a Remote Tracker

Training on a serverless function means you can't watch `rich` console output easily. Adding optional Weights & Biases or TensorBoard logging would let you monitor loss curves from your laptop.

This is a nice-to-have, not a blocker — Modal streams stdout to your terminal during `modal run`.
