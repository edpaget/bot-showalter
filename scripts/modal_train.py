"""Modal serverless GPU training for the contextual transformer.

Wraps the existing CLI (``fantasy-baseball-manager contextual pretrain/finetune``)
so the training code needs zero changes.  Data and checkpoints persist on a Modal
Volume mounted at ``/data``; symlinks redirect the app's default paths there.

Usage::

    # One-time setup
    modal volume create fantasy-baseball-data
    modal volume put fantasy-baseball-data ~/.fantasy_baseball/statcast/ statcast/

    # Pre-train on T4 (default, $0.59/hr)
    modal run scripts/modal_train.py --command pretrain --epochs 30

    # Pre-train on A100 80GB with large batch size
    modal run scripts/modal_train.py --command pretrain --gpu A100-80GB --batch-size 256 --learning-rate 3e-4

    # Fine-tune
    modal run scripts/modal_train.py --command finetune --perspective pitcher

    # Fine-tune on A100 80GB
    modal run scripts/modal_train.py --command finetune --gpu A100-80GB --perspective pitcher --batch-size 128

    # Check data/checkpoints on volume
    modal run scripts/modal_train.py --command check

    # Download checkpoints locally
    modal volume get fantasy-baseball-data models/contextual/ ~/.fantasy_baseball/models/contextual/
"""

from __future__ import annotations

import modal

app = modal.App("fantasy-baseball-train")

vol = modal.Volume.from_name("fantasy-baseball-data", create_if_missing=True)

DATA_DIR = "/data"
HOME_DIR = "/root"
APP_DIR = "/app"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .copy_local_dir(".", APP_DIR)
    .run_commands(
        f"cd {APP_DIR} && /root/.local/bin/uv sync --no-dev --frozen",
        gpu="T4",
    )
)


def _setup_symlinks() -> None:
    """Symlink app data paths to the Modal Volume mount."""
    import os

    fantasy_dir = os.path.join(HOME_DIR, ".fantasy_baseball")
    os.makedirs(fantasy_dir, exist_ok=True)

    for name in ("statcast", "models"):
        src = os.path.join(DATA_DIR, name)
        dst = os.path.join(fantasy_dir, name)
        os.makedirs(src, exist_ok=True)
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.exists(dst):
            os.rename(dst, f"{dst}.bak")
        os.symlink(src, dst)


@app.function(
    image=image,
    gpu="T4",
    timeout=21600,
    volumes={DATA_DIR: vol},
)
def pretrain(
    seasons: str = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: str = "2023",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    resume_from: str | None = None,
    max_seq_len: int = 512,
) -> None:
    """Run MGM pre-training on a cloud GPU."""
    import subprocess

    _setup_symlinks()

    cmd = [
        "uv", "run", "fantasy-baseball-manager",
        "contextual", "pretrain",
        "--seasons", seasons,
        "--val-seasons", val_seasons,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--max-seq-len", str(max_seq_len),
    ]
    if resume_from is not None:
        cmd.extend(["--resume-from", resume_from])

    subprocess.run(cmd, cwd=APP_DIR, check=True)
    vol.commit()


@app.function(
    image=image,
    gpu="T4",
    timeout=21600,
    volumes={DATA_DIR: vol},
)
def finetune(
    perspective: str = "pitcher",
    base_model: str = "pretrain_best",
    seasons: str = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: str = "2023",
    epochs: int = 30,
    batch_size: int = 32,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    freeze_backbone: bool = False,
    resume_from: str | None = None,
    max_seq_len: int = 512,
) -> None:
    """Run fine-tuning on a cloud GPU."""
    import subprocess

    _setup_symlinks()

    cmd = [
        "uv", "run", "fantasy-baseball-manager",
        "contextual", "finetune",
        "--perspective", perspective,
        "--base-model", base_model,
        "--seasons", seasons,
        "--val-seasons", val_seasons,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--head-lr", str(head_lr),
        "--backbone-lr", str(backbone_lr),
        "--max-seq-len", str(max_seq_len),
    ]
    if freeze_backbone:
        cmd.append("--freeze-backbone")
    if resume_from is not None:
        cmd.extend(["--resume-from", resume_from])

    subprocess.run(cmd, cwd=APP_DIR, check=True)
    vol.commit()


@app.function(
    image=image,
    volumes={DATA_DIR: vol},
)
def check_data() -> None:
    """List data and checkpoint files on the volume."""
    import os

    for root, _dirs, files in os.walk(DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {os.path.relpath(path, DATA_DIR):60s} {size_mb:8.1f} MB")
    if not any(os.scandir(DATA_DIR)):
        print("Volume is empty. Upload data first:")
        print("  modal volume put fantasy-baseball-data ~/.fantasy_baseball/statcast/ statcast/")


@app.local_entrypoint()
def main(
    command: str = "pretrain",
    gpu: str = "T4",
    timeout: int = 21600,
    # pretrain / finetune shared
    seasons: str = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: str = "2023",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    resume_from: str | None = None,
    max_seq_len: int = 512,
    # finetune only
    perspective: str = "pitcher",
    base_model: str = "pretrain_best",
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    freeze_backbone: bool = False,
) -> None:
    """Local entrypoint — dispatches to pretrain, finetune, or check.

    Examples::

        modal run scripts/modal_train.py --command pretrain --gpu T4 --epochs 30
        modal run scripts/modal_train.py --command finetune --perspective pitcher
        modal run scripts/modal_train.py --command check
    """
    if command == "check":
        check_data.remote()
        return

    if command == "pretrain":
        pretrain.with_options(gpu=gpu, timeout=timeout).remote(
            seasons=seasons,
            val_seasons=val_seasons,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            resume_from=resume_from,
            max_seq_len=max_seq_len,
        )
    elif command == "finetune":
        finetune.with_options(gpu=gpu, timeout=timeout).remote(
            perspective=perspective,
            base_model=base_model,
            seasons=seasons,
            val_seasons=val_seasons,
            epochs=epochs,
            batch_size=batch_size,
            head_lr=head_lr,
            backbone_lr=backbone_lr,
            freeze_backbone=freeze_backbone,
            resume_from=resume_from,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Unknown command: {command!r}. Use 'pretrain', 'finetune', or 'check'.")
