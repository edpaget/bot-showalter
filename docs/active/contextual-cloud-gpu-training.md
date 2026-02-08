# Cloud GPU Training Guide

Parent doc: [contextual-event-embeddings.md](contextual-event-embeddings.md)

Training the contextual transformer locally on MPS (Apple Silicon) or CPU is impractical for the full dataset (~5.6M training pitches, 30 epochs). This guide covers how to run training on cloud GPUs with minimal changes to the existing codebase.

---

## What We're Working With

The training code is already cloud-ready in several respects:

- **Device auto-detection** (`contextual/cli.py:188-193`): Selects CUDA > MPS > CPU automatically. A CUDA GPU on a cloud instance is picked up with zero changes.
- **Checkpoint resume** (`contextual/training/pretrain.py:101-110`): `--resume-from` flag restores model, optimizer, and scheduler state. Training can survive preemptions.
- **CLI entry point**: `uv run python -m fantasy_baseball_manager contextual pretrain --seasons ...` runs the full pipeline from data loading through training.
- **File-based persistence**: Checkpoints go to `~/.fantasy_baseball/models/contextual/`. Data loads from `~/.fantasy_baseball/statcast/`. Both paths are configurable or can be symlinked.

The main gap: **data needs to be uploaded to the cloud instance** (~841 MB of Statcast parquet files).

---

## Provider Comparison

| Provider | A100 80GB/hr | Setup | Reliability | Notes |
|---|---|---|---|---|
| **Vast.ai** | ~$0.75 | Medium | Variable | Cheapest; marketplace with variable hosts |
| **Lambda Cloud** | ~$1.10 | Low | Medium | Pre-configured ML instances; availability can be spotty |
| **RunPod** | ~$1.75 | Low | High | Good balance of price and reliability |
| **Modal** | ~$2.50 | Very low | High | Python-native serverless; pay-per-second |
| **Google Colab Pro+** | ~$50/month | None | Medium | 24-hour session limit; fine for shorter runs |
| **AWS EC2 p4d** | ~$4.10 | High | Very high | Expensive but rock-solid; spot instances cut cost 60-70% |

**Estimated total cost for full pre-training** (assuming ~20-40 hours on a single A100):

| Provider | Estimated Cost |
|---|---|
| Vast.ai | $15 - $30 |
| Lambda Cloud | $22 - $44 |
| RunPod | $35 - $70 |
| Modal | $50 - $100 |

For a single BERT-sized model, even the pricier options are under $100. The fine-tuning phase (Phase 4) is cheaper — smaller dataset, fewer epochs, partially frozen model.

---

## Recommended Approach: RunPod or Lambda Cloud

These offer the best tradeoff of cost, setup friction, and reliability for this workload. Both provide bare VM instances with NVIDIA drivers and CUDA pre-installed.

### Step-by-Step Setup

#### 1. Provision an Instance

Pick an A100 40GB or 80GB instance. The model is small enough for 40GB; 80GB lets you increase batch size for faster training.

On **RunPod**: Create a GPU Pod with the `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` template (or similar PyTorch base image). Attach a network volume for persistent storage.

On **Lambda Cloud**: Launch an A100 instance. It comes with PyTorch, CUDA, and conda pre-installed.

#### 2. Clone the Repo and Install Dependencies

```bash
git clone <your-repo-url> ~/fantasy-baseball
cd ~/fantasy-baseball
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install project dependencies
uv sync
```

#### 3. Upload Statcast Data

The training data lives locally at `~/.fantasy_baseball/statcast/`. Upload it to the cloud instance.

From your local machine:
```bash
# Compress the data first (~841 MB → ~200 MB compressed)
tar czf statcast-data.tar.gz -C ~/.fantasy_baseball statcast

# Upload via scp (Lambda/RunPod both provide SSH access)
scp statcast-data.tar.gz user@<instance-ip>:~/

# On the cloud instance:
mkdir -p ~/.fantasy_baseball
tar xzf ~/statcast-data.tar.gz -C ~/.fantasy_baseball
```

Alternatively, use `rsync` for resumable transfers:
```bash
rsync -avz --progress ~/.fantasy_baseball/statcast/ user@<instance-ip>:~/.fantasy_baseball/statcast/
```

#### 4. Run Pre-Training

```bash
cd ~/fantasy-baseball

# Verify CUDA is detected
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run pre-training (same CLI as local, CUDA auto-detected)
uv run python -m fantasy_baseball_manager contextual pretrain \
    --seasons 2015,2016,2017,2018,2019,2020,2021,2022 \
    --val-seasons 2023 \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 1e-4
```

Increase `--batch-size` on A100 80GB (try 128 or 256 — the model is small, so batch size is limited by sequence length, not model parameters).

#### 5. Run in Background (Survive Disconnects)

```bash
# Use tmux or screen to keep training alive after SSH disconnect
tmux new -s train
uv run python -m fantasy_baseball_manager contextual pretrain \
    --seasons 2015,2016,2017,2018,2019,2020,2021,2022 \
    --val-seasons 2023 \
    --epochs 30 \
    --batch-size 64
# Detach: Ctrl-B, then D
# Reattach later: tmux attach -t train
```

Or use `nohup`:
```bash
nohup uv run python -m fantasy_baseball_manager contextual pretrain \
    --seasons 2015,2016,2017,2018,2019,2020,2021,2022 \
    --val-seasons 2023 \
    --epochs 30 \
    --batch-size 64 > training.log 2>&1 &

tail -f training.log
```

#### 6. Download Checkpoints When Done

```bash
# From your local machine:
scp user@<instance-ip>:~/.fantasy_baseball/models/contextual/pretrain_best.pt \
    ~/.fantasy_baseball/models/contextual/
scp user@<instance-ip>:~/.fantasy_baseball/models/contextual/pretrain_best_meta.json \
    ~/.fantasy_baseball/models/contextual/
```

Then shut down the instance.

#### 7. Resume After Preemption

If the instance is interrupted (spot instance reclaimed, network drop, etc.):

```bash
# Re-provision instance, re-upload data if needed, then:
uv run python -m fantasy_baseball_manager contextual pretrain \
    --resume-from pretrain_latest \
    --seasons 2015,2016,2017,2018,2019,2020,2021,2022 \
    --val-seasons 2023 \
    --epochs 30 \
    --batch-size 64
```

The `--resume-from` flag restores model weights, optimizer state, epoch counter, and best validation loss. Training continues from the last saved checkpoint.

---

## Alternative: Modal (Serverless)

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

### Pre-Training

```bash
# Default: T4 GPU, 30 epochs, batch size 64
modal run scripts/modal_train.py --command pretrain

# Override GPU and epochs
modal run scripts/modal_train.py --command pretrain --gpu A10G --epochs 50

# Resume after interruption
modal run scripts/modal_train.py --command pretrain --resume-from pretrain_latest
```

### Fine-Tuning

```bash
# Fine-tune for pitcher prediction
modal run scripts/modal_train.py --command finetune --perspective pitcher

# Fine-tune for batter prediction with frozen backbone
modal run scripts/modal_train.py --command finetune --perspective batter --freeze-backbone
```

### Download Checkpoints

```bash
modal volume get fantasy-baseball-data models/contextual/ ~/.fantasy_baseball/models/contextual/
```

### Budget Guidance

| GPU | Cost/hr | Use Case |
|-----|---------|----------|
| T4 (default) | $0.59 | Pre-training/fine-tuning with free credits |
| A10G | $1.10 | Faster training when free credits run out |
| A100 | $2.78 | Full dataset, large batch sizes |

---

## Alternative: Google Colab Pro+

Lowest friction for a quick experiment. Useful for validating the pipeline on a subset of data before committing to a full training run.

1. Upload `statcast-data.tar.gz` to Google Drive
2. In a Colab notebook:

```python
# Mount Drive and extract data
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p ~/.fantasy_baseball
!tar xzf /content/drive/MyDrive/statcast-data.tar.gz -C ~/.fantasy_baseball

# Clone repo and install
!git clone <repo-url> /content/fantasy-baseball
%cd /content/fantasy-baseball
!pip install uv && uv sync

# Verify GPU
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# Train (shorter run for validation)
!uv run python -m fantasy_baseball_manager contextual pretrain \
    --seasons 2020,2021,2022 \
    --val-seasons 2023 \
    --epochs 5 \
    --batch-size 32
```

**Limitations**: 24-hour session limit (Pro+), GPU type is not guaranteed (may get T4 instead of A100), data must be re-uploaded if runtime resets. Use `--resume-from` to recover from session timeouts. Not recommended for the full 30-epoch run.

---

## Code Changes Worth Making

The existing codebase works as-is on cloud GPUs. These optional improvements would make cloud training smoother:

### Mixed Precision Training (AMP)

Roughly doubles throughput on A100 with no accuracy loss for this model size. Requires wrapping the training loop with `torch.amp.autocast` and using a `GradScaler`.

Changes needed in `MGMTrainer._train_epoch`:
- Wrap `self._model(tb)` and loss computation in `torch.amp.autocast("cuda")`
- Scale loss with `torch.amp.GradScaler` before `.backward()`
- Unscale before gradient clipping

This is the single highest-impact optimization — roughly halves training time.

### DataLoader Workers

Both `MGMTrainer` and `FineTuneTrainer` now auto-detect CUDA and set `num_workers=4`, `pin_memory=True`, and `persistent_workers=True` on their DataLoaders. On CPU/MPS these are omitted (defaults only).

### Logging to a Remote Tracker

Training on a remote instance means you can't watch `rich` console output easily. Adding optional Weights & Biases or TensorBoard logging would let you monitor loss curves from your laptop.

This is a nice-to-have, not a blocker — `tail -f training.log` via SSH works fine.

---

## Data Transfer Checklist

Before starting a cloud training run:

- [ ] Statcast parquet files uploaded to `~/.fantasy_baseball/statcast/` on the instance
- [ ] Verify with: `ls ~/.fantasy_baseball/statcast/2015/` (should see monthly parquet files)
- [ ] CUDA detected: `uv run python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Short smoke test: `uv run python -m fantasy_baseball_manager contextual pretrain --seasons 2022 --val-seasons 2023 --epochs 1 --batch-size 16`
- [ ] `tmux` or `nohup` set up so training survives SSH disconnect
- [ ] Plan to download checkpoints from `~/.fantasy_baseball/models/contextual/` when done
