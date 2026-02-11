"""Modal serverless GPU training for the contextual transformer.

Wraps the existing CLI (``fantasy-baseball-manager contextual pretrain/finetune``)
so the training code needs zero changes.  Data and checkpoints persist on a Modal
Volume mounted at ``/data``; symlinks redirect the app's default paths there.

Set the GPU type via the ``MODAL_GPU`` environment variable (default: T4).

Usage::

    # One-time setup
    modal volume create fantasy-baseball-data
    modal volume put fantasy-baseball-data ~/.fantasy_baseball/statcast/ statcast/

    # Pre-train on T4 (default, $0.59/hr)
    modal run scripts/modal_train.py --command pretrain --epochs 30

    # Pre-train on A100 80GB — full-season context with smaller batch
    MODAL_GPU=A100-80GB modal run scripts/modal_train.py --command pretrain --max-seq-len 2048 --batch-size 64

    # Fine-tune
    modal run scripts/modal_train.py --command finetune --perspective pitcher

    # Fine-tune on A100 80GB (context window is small, so bump batch instead)
    MODAL_GPU=A100-80GB modal run scripts/modal_train.py --command finetune --perspective pitcher --batch-size 128

    # Build identity (archetypes) — CPU-only, no GPU needed
    modal run scripts/modal_train.py --command build-identity --perspective pitcher --n-archetypes 8

    # Prepare hierarchical fine-tune data on CPU (avoids 60GB+ memory on GPU)
    modal run scripts/modal_train.py --command prepare --mode hier-finetune --perspective pitcher

    # Hierarchical fine-tune (uses prepared data by default)
    modal run scripts/modal_train.py --command hier-finetune --perspective pitcher
    modal run scripts/modal_train.py --command hier-finetune --perspective pitcher --archetype-model pitcher_archetypes

    # Check data/checkpoints on volume
    modal run scripts/modal_train.py --command check

    # Download checkpoints locally
    modal volume get fantasy-baseball-data models/contextual/ ~/.fantasy_baseball/models/contextual/
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("MODAL_GPU", "T4")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "21600"))

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
    # Copy dependency files first so `uv sync` layer is cached across code changes
    .add_local_file("pyproject.toml", f"{APP_DIR}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{APP_DIR}/uv.lock", copy=True)
    .run_commands(
        f"cd {APP_DIR} && /root/.local/bin/uv sync --no-dev --frozen --no-install-project",
        gpu="T4",
    )
    # Now copy the full source tree and install the project itself
    .add_local_dir(
        ".", APP_DIR, copy=True,
        ignore=[
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".git",
            ".ruff_cache",
            "*.py[oc]",
            "*.egg-info",
        ],
    )
    .run_commands(
        f"cd {APP_DIR} && /root/.local/bin/uv sync --no-dev --frozen",
        gpu="T4",
    )
)


def _setup_symlinks() -> None:
    """Symlink app data paths to the Modal Volume mount."""
    fantasy_dir = os.path.join(HOME_DIR, ".fantasy_baseball")
    os.makedirs(fantasy_dir, exist_ok=True)

    for name in ("statcast", "models", "prepared_data"):
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
    gpu=GPU,
    timeout=TIMEOUT,
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
    gpu=GPU,
    timeout=TIMEOUT,
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
    max_seq_len: int | None = None,
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
    ]
    if max_seq_len is not None:
        cmd.extend(["--max-seq-len", str(max_seq_len)])
    if freeze_backbone:
        cmd.append("--freeze-backbone")
    if resume_from is not None:
        cmd.extend(["--resume-from", resume_from])

    subprocess.run(cmd, cwd=APP_DIR, check=True)
    vol.commit()


@app.function(
    image=image,
    timeout=TIMEOUT,
    volumes={DATA_DIR: vol},
)
def prepare_data(
    mode: str = "all",
    seasons: str = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: str = "2023",
    perspectives: str = "batter,pitcher",
    perspective: str = "pitcher",
    context_window: int = 10,
    max_seq_len: int = 512,
    # hier-finetune mode options
    n_archetypes: int = 8,
    archetype_model: str | None = None,
    min_opportunities: float = 50.0,
    profile_year: int | None = None,
) -> None:
    """Build tensorized training data on CPU (no GPU needed)."""
    import subprocess

    _setup_symlinks()

    cmd = [
        "uv", "run", "fantasy-baseball-manager",
        "contextual", "prepare-data",
        "--mode", mode,
        "--seasons", seasons,
        "--val-seasons", val_seasons,
        "--perspectives", perspectives,
        "--perspective", perspective,
        "--context-window", str(context_window),
        "--max-seq-len", str(max_seq_len),
        "--n-archetypes", str(n_archetypes),
        "--min-opportunities", str(min_opportunities),
    ]
    if archetype_model is not None:
        cmd.extend(["--archetype-model", archetype_model])
    if profile_year is not None:
        cmd.extend(["--profile-year", str(profile_year)])

    subprocess.run(cmd, cwd=APP_DIR, check=True)
    vol.commit()


@app.function(
    image=image,
    volumes={DATA_DIR: vol},
)
def check_data() -> None:
    """List data and checkpoint files on the volume."""
    for root, _dirs, files in os.walk(DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {os.path.relpath(path, DATA_DIR):60s} {size_mb:8.1f} MB")
    if not any(os.scandir(DATA_DIR)):
        print("Volume is empty. Upload data first:")
        print("  modal volume put fantasy-baseball-data ~/.fantasy_baseball/statcast/ statcast/")


@app.function(
    image=image,
    timeout=TIMEOUT,
    volumes={DATA_DIR: vol},
)
def build_identity(
    perspective: str = "pitcher",
    n_archetypes: int = 8,
    min_opportunities: float = 50.0,
    profile_year: int = 2023,
    name: str | None = None,
) -> None:
    """Build player stat profiles and fit archetype model (CPU-only)."""
    import subprocess

    _setup_symlinks()

    cmd = [
        "uv", "run", "fantasy-baseball-manager",
        "contextual", "build-identity",
        "--perspective", perspective,
        "--n-archetypes", str(n_archetypes),
        "--min-opportunities", str(min_opportunities),
        "--profile-year", str(profile_year),
    ]
    if name is not None:
        cmd.extend(["--name", name])

    subprocess.run(cmd, cwd=APP_DIR, check=True)
    vol.commit()


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={DATA_DIR: vol},
)
def hier_finetune(
    perspective: str = "pitcher",
    base_model: str = "pretrain_best",
    seasons: str = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: str = "2023",
    epochs: int = 30,
    batch_size: int = 32,
    identity_lr: float = 1e-3,
    level3_lr: float = 5e-4,
    head_lr: float = 1e-3,
    n_archetypes: int = 8,
    archetype_model: str | None = None,
    min_opportunities: float = 50.0,
    profile_year: int | None = None,
    max_seq_len: int | None = None,
    context_window: int | None = None,
    prepared_data: bool = True,
) -> None:
    """Run hierarchical fine-tuning on a cloud GPU."""
    import subprocess

    _setup_symlinks()

    cmd = [
        "uv", "run", "fantasy-baseball-manager",
        "contextual", "hier-finetune",
        "--perspective", perspective,
        "--base-model", base_model,
        "--seasons", seasons,
        "--val-seasons", val_seasons,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--identity-lr", str(identity_lr),
        "--level3-lr", str(level3_lr),
        "--head-lr", str(head_lr),
        "--n-archetypes", str(n_archetypes),
        "--min-opportunities", str(min_opportunities),
    ]
    if prepared_data:
        cmd.append("--prepared-data")
    else:
        cmd.append("--no-prepared-data")
    if archetype_model is not None:
        cmd.extend(["--archetype-model", archetype_model])
    if profile_year is not None:
        cmd.extend(["--profile-year", str(profile_year)])
    if max_seq_len is not None:
        cmd.extend(["--max-seq-len", str(max_seq_len)])
    if context_window is not None:
        cmd.extend(["--context-window", str(context_window)])

    subprocess.run(cmd, cwd=APP_DIR, check=True)
    vol.commit()


@app.local_entrypoint()
def main(
    command: str = "pretrain",
    # pretrain / finetune shared
    seasons: str = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: str = "2023",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    resume_from: str | None = None,
    max_seq_len: int | None = None,
    # finetune / prepare / hier-finetune
    perspective: str = "pitcher",
    base_model: str = "pretrain_best",
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    freeze_backbone: bool = False,
    # prepare only
    mode: str = "all",
    perspectives: str = "batter,pitcher",
    context_window: int | None = None,
    # hier-finetune / build-identity
    identity_lr: float = 1e-3,
    level3_lr: float = 5e-4,
    n_archetypes: int = 8,
    archetype_model: str | None = None,
    min_opportunities: float = 50.0,
    profile_year: int | None = None,
    name: str | None = None,
) -> None:
    """Local entrypoint — dispatches to pretrain, finetune, hier-finetune, build-identity, prepare, or check.

    Set GPU via env var: ``MODAL_GPU=A100-80GB modal run scripts/modal_train.py ...``

    Examples::

        modal run scripts/modal_train.py --command prepare --mode all
        modal run scripts/modal_train.py --command pretrain --epochs 30
        MODAL_GPU=A100-80GB modal run scripts/modal_train.py --command pretrain --max-seq-len 2048 --batch-size 64
        modal run scripts/modal_train.py --command finetune --perspective pitcher
        modal run scripts/modal_train.py --command build-identity --perspective pitcher --n-archetypes 8
        modal run scripts/modal_train.py --command hier-finetune --perspective pitcher
        modal run scripts/modal_train.py --command check
    """
    if command == "check":
        check_data.remote()
        return

    if command == "prepare":
        prepare_data.remote(
            mode=mode,
            seasons=seasons,
            val_seasons=val_seasons,
            perspectives=perspectives,
            perspective=perspective,
            context_window=context_window if context_window is not None else 10,
            max_seq_len=max_seq_len if max_seq_len is not None else 512,
            n_archetypes=n_archetypes,
            archetype_model=archetype_model,
            min_opportunities=min_opportunities,
            profile_year=profile_year,
        )
    elif command == "pretrain":
        pretrain.remote(
            seasons=seasons,
            val_seasons=val_seasons,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            resume_from=resume_from,
            max_seq_len=max_seq_len if max_seq_len is not None else 512,
        )
    elif command == "finetune":
        finetune.remote(
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
    elif command == "build-identity":
        build_identity.remote(
            perspective=perspective,
            n_archetypes=n_archetypes,
            min_opportunities=min_opportunities,
            profile_year=profile_year if profile_year is not None else 2023,
            name=name,
        )
    elif command == "hier-finetune":
        hier_finetune.remote(
            perspective=perspective,
            base_model=base_model,
            seasons=seasons,
            val_seasons=val_seasons,
            epochs=epochs,
            batch_size=batch_size if batch_size != 64 else 32,
            identity_lr=identity_lr,
            level3_lr=level3_lr,
            head_lr=head_lr,
            n_archetypes=n_archetypes,
            archetype_model=archetype_model,
            min_opportunities=min_opportunities,
            profile_year=profile_year,
            max_seq_len=max_seq_len,
            context_window=context_window,
        )
    else:
        raise ValueError(
            f"Unknown command: {command!r}. "
            "Use 'prepare', 'pretrain', 'finetune', 'hier-finetune', 'build-identity', or 'check'."
        )
