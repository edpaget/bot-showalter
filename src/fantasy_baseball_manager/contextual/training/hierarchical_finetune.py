"""Hierarchical model fine-tuning trainer."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from fantasy_baseball_manager.contextual.persistence import ContextualModelMetadata
from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
    HierarchicalFineTuneBatch,
    HierarchicalFineTuneSample,
    collate_hierarchical_samples,
)
from fantasy_baseball_manager.contextual.training.precomputed_dataset import (
    PrecomputedBatch,
    PrecomputedDataset,
    collate_precomputed_samples,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.contextual.model.hierarchical import HierarchicalModel
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.contextual.training.config import (
        HierarchicalFineTuneConfig,
    )
    from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
        HierarchicalFineTuneDataset,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HierarchicalFineTuneMetrics:
    """Metrics from a hierarchical fine-tuning training or validation pass."""

    loss: float
    per_stat_mse: dict[str, float]
    per_stat_mae: dict[str, float]
    n_samples: int
    baseline_per_stat_mse: dict[str, float] = field(default_factory=dict)
    baseline_per_stat_mae: dict[str, float] = field(default_factory=dict)


class HierarchicalFineTuneTrainer:
    """Fine-tunes a HierarchicalModel on per-game stat prediction with identity."""

    def __init__(
        self,
        model: HierarchicalModel,
        config: HierarchicalFineTuneConfig,
        model_store: ContextualModelStore,
        target_stats: tuple[str, ...],
        device: torch.device | None = None,
        log_fn: Callable[[str], object] | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._store = model_store
        self._target_stats = target_stats
        self._device = device or torch.device("cpu")
        self._loss_weights: torch.Tensor | None = None
        self._log = log_fn or logger.info
        self._model.to(self._device)

    def train(
        self,
        train_dataset: HierarchicalFineTuneDataset | PrecomputedDataset,
        val_dataset: HierarchicalFineTuneDataset | PrecomputedDataset,
    ) -> dict[str, float]:
        """Run the hierarchical fine-tuning loop.

        Accepts either HierarchicalFineTuneDataset (standard path with backbone)
        or PrecomputedDataset (precomputed game embeddings, skips backbone).

        Returns:
            Dict with final validation metrics.
        """
        config = self._config
        self._precomputed = isinstance(train_dataset, PrecomputedDataset)

        # Parameter groups with per-group learning rates
        param_groups = [
            {"params": list(self._model.identity_module.parameters()), "lr": config.identity_learning_rate},
            {"params": list(self._model.level3.parameters()), "lr": config.level3_learning_rate},
            {"params": list(self._model.head.parameters()), "lr": config.head_learning_rate},
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

        n_batches_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
        total_steps = config.epochs * math.ceil(n_batches_per_epoch / config.accumulation_steps)
        warmup_steps = max(config.min_warmup_steps, int(total_steps * config.warmup_fraction))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0,
                ),
            ],
            milestones=[warmup_steps],
        )

        best_val_loss = float("inf")
        global_step = 0
        patience_counter = 0

        loader_kwargs: dict[str, Any] = {}
        if self._device.type == "cuda":
            loader_kwargs.update(num_workers=4, pin_memory=True, persistent_workers=True)

        collate_fn = collate_precomputed_samples if self._precomputed else collate_hierarchical_samples
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(config.seed),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

        # Compute per-stat loss weights (inverse variance, normalized)
        target_std = train_dataset.compute_target_std()
        weights = 1.0 / (target_std ** 2)
        weights = weights / weights.sum() * len(weights)
        self._loss_weights = weights.to(self._device)

        best_state_dict = None
        train_metrics = HierarchicalFineTuneMetrics(0.0, {}, {}, 0)
        epoch = 0
        training_start = time.monotonic()

        for epoch in range(config.epochs):
            epoch_start = time.monotonic()
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, global_step)
            global_step += len(train_loader)

            val_metrics = self._validate(val_loader)

            epoch_elapsed = time.monotonic() - epoch_start
            total_elapsed = time.monotonic() - training_start
            epochs_done = epoch + 1
            sec_per_epoch = total_elapsed / epochs_done
            epochs_remaining = config.epochs - epochs_done
            eta = sec_per_epoch * epochs_remaining
            self._log(
                f"Epoch {epochs_done}/{config.epochs} — "
                f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f}  "
                f"({epoch_elapsed:.0f}s this epoch, {total_elapsed:.0f}s total, ETA {eta:.0f}s)"
            )

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                self._save_checkpoint(
                    f"hierarchical_{self._config.perspective}_best", epoch, train_metrics, val_metrics,
                    optimizer, global_step, best_val_loss,
                )
            else:
                patience_counter += 1

            if (epoch + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(
                    f"hierarchical_{self._config.perspective}_epoch_{epoch + 1}", epoch, train_metrics, val_metrics,
                    optimizer, global_step, best_val_loss,
                )

            if patience_counter >= config.patience:
                self._log(f"Early stopping at epoch {epoch + 1} (patience={config.patience})")
                break

        # Save latest
        val_metrics = self._validate(val_loader)
        self._save_checkpoint(
            f"hierarchical_{self._config.perspective}_latest", epoch, train_metrics, val_metrics,
            optimizer, global_step, best_val_loss,
        )

        # Restore best model
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        return {
            "val_loss": val_metrics.loss,
            **{f"val_{stat}_mse": v for stat, v in val_metrics.per_stat_mse.items()},
            **{f"val_{stat}_mae": v for stat, v in val_metrics.per_stat_mae.items()},
        }

    def _train_epoch(
        self,
        loader: DataLoader[HierarchicalFineTuneSample],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int,
    ) -> HierarchicalFineTuneMetrics:
        self._model.train()
        total_loss = 0.0
        total_samples = 0
        per_stat_se = torch.zeros(len(self._target_stats))
        per_stat_ae = torch.zeros(len(self._target_stats))

        n_batches = len(loader)
        accum = self._config.accumulation_steps
        log_interval = self._config.log_interval
        epoch_start = time.monotonic()
        optimizer.zero_grad()
        for batch_idx, raw_batch in enumerate(loader):
            preds, targets = self._forward_batch(raw_batch)

            loss = self._compute_weighted_loss(preds, targets)
            scaled_loss = loss / accum
            scaled_loss.backward()

            if (batch_idx + 1) % accum == 0 or batch_idx == n_batches - 1:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            n = targets.shape[0]
            total_loss += loss.item() * n
            total_samples += n

            with torch.no_grad():
                diff = preds.cpu() - targets.cpu()
                per_stat_se += (diff ** 2).sum(dim=0)
                per_stat_ae += diff.abs().sum(dim=0)

            global_step += 1

            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                elapsed = time.monotonic() - epoch_start
                batches_done = batch_idx + 1
                sec_per_batch = elapsed / batches_done
                remaining = sec_per_batch * (n_batches - batches_done)
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                self._log(
                    f"  batch {batches_done}/{n_batches} — "
                    f"loss={avg_loss:.4f}  {elapsed:.1f}s elapsed  ETA {remaining:.0f}s"
                )

        if total_samples == 0:
            return HierarchicalFineTuneMetrics(0.0, {}, {}, 0)

        return HierarchicalFineTuneMetrics(
            loss=total_loss / total_samples,
            per_stat_mse={
                stat: (per_stat_se[i] / total_samples).item()
                for i, stat in enumerate(self._target_stats)
            },
            per_stat_mae={
                stat: (per_stat_ae[i] / total_samples).item()
                for i, stat in enumerate(self._target_stats)
            },
            n_samples=total_samples,
        )

    @torch.no_grad()
    def _validate(self, loader: DataLoader[HierarchicalFineTuneSample]) -> HierarchicalFineTuneMetrics:
        self._model.eval()
        total_loss = 0.0
        total_samples = 0
        per_stat_se = torch.zeros(len(self._target_stats))
        per_stat_ae = torch.zeros(len(self._target_stats))
        baseline_se = torch.zeros(len(self._target_stats))
        baseline_ae = torch.zeros(len(self._target_stats))

        for raw_batch in loader:
            preds, targets = self._forward_batch(raw_batch)
            context_mean = self._get_context_mean(raw_batch)

            loss = self._compute_weighted_loss(preds, targets)

            n = targets.shape[0]
            total_loss += loss.item() * n
            total_samples += n

            diff = preds.cpu() - targets.cpu()
            per_stat_se += (diff ** 2).sum(dim=0)
            per_stat_ae += diff.abs().sum(dim=0)

            baseline_diff = context_mean.cpu() - targets.cpu()
            baseline_se += (baseline_diff ** 2).sum(dim=0)
            baseline_ae += baseline_diff.abs().sum(dim=0)

        if total_samples == 0:
            return HierarchicalFineTuneMetrics(0.0, {}, {}, 0)

        return HierarchicalFineTuneMetrics(
            loss=total_loss / total_samples,
            per_stat_mse={
                stat: (per_stat_se[i] / total_samples).item()
                for i, stat in enumerate(self._target_stats)
            },
            per_stat_mae={
                stat: (per_stat_ae[i] / total_samples).item()
                for i, stat in enumerate(self._target_stats)
            },
            n_samples=total_samples,
            baseline_per_stat_mse={
                stat: (baseline_se[i] / total_samples).item()
                for i, stat in enumerate(self._target_stats)
            },
            baseline_per_stat_mae={
                stat: (baseline_ae[i] / total_samples).item()
                for i, stat in enumerate(self._target_stats)
            },
        )

    def _compute_weighted_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-stat inverse-variance weighted MSE loss."""
        per_stat_mse = (preds - targets).pow(2).mean(dim=0)
        assert self._loss_weights is not None
        return (per_stat_mse * self._loss_weights).mean()

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        train_metrics: HierarchicalFineTuneMetrics,
        val_metrics: HierarchicalFineTuneMetrics,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        best_val_loss: float,
    ) -> None:
        metadata = ContextualModelMetadata(
            name=name,
            epoch=epoch + 1,
            train_loss=train_metrics.loss,
            val_loss=val_metrics.loss,
            perspective=self._config.perspective,
            target_stats=self._target_stats,
            per_stat_mse=val_metrics.per_stat_mse,
            target_mode=self._config.target_mode,
            target_window=self._config.target_window,
        )
        scheduler_state = {
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "global_step": global_step,
        }
        self._store.save_hierarchical_checkpoint(
            name, self._model, metadata,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler_state,
        )

    def _forward_batch(
        self,
        raw_batch: HierarchicalFineTuneBatch | PrecomputedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch forward pass based on batch type, returning (preds, targets)."""
        if isinstance(raw_batch, PrecomputedBatch):
            batch = self._batch_to_device_precomputed(raw_batch)
            output = self._model.forward_precomputed(
                batch.game_embeddings, batch.game_mask,
                batch.identity_features, batch.archetype_ids,
            )
            return output["performance_preds"], batch.targets
        else:
            batch = self._batch_to_device(raw_batch)
            output = self._model(batch.context, batch.identity_features, batch.archetype_ids)
            return output["performance_preds"], batch.targets

    def _get_context_mean(
        self,
        raw_batch: HierarchicalFineTuneBatch | PrecomputedBatch,
    ) -> torch.Tensor:
        """Extract context_mean from a batch, moving to device if needed."""
        if isinstance(raw_batch, PrecomputedBatch):
            return raw_batch.context_mean.to(self._device)
        return raw_batch.context_mean.to(self._device)

    def _batch_to_device(self, batch: HierarchicalFineTuneBatch) -> HierarchicalFineTuneBatch:
        if self._device.type == "cpu":
            return batch
        from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedBatch

        ctx = batch.context
        return HierarchicalFineTuneBatch(
            context=TensorizedBatch(
                pitch_type_ids=ctx.pitch_type_ids.to(self._device),
                pitch_result_ids=ctx.pitch_result_ids.to(self._device),
                bb_type_ids=ctx.bb_type_ids.to(self._device),
                stand_ids=ctx.stand_ids.to(self._device),
                p_throws_ids=ctx.p_throws_ids.to(self._device),
                pa_event_ids=ctx.pa_event_ids.to(self._device),
                numeric_features=ctx.numeric_features.to(self._device),
                numeric_mask=ctx.numeric_mask.to(self._device),
                padding_mask=ctx.padding_mask.to(self._device),
                player_token_mask=ctx.player_token_mask.to(self._device),
                game_ids=ctx.game_ids.to(self._device),
                seq_lengths=ctx.seq_lengths.to(self._device),
            ),
            targets=batch.targets.to(self._device),
            context_mean=batch.context_mean.to(self._device),
            identity_features=batch.identity_features.to(self._device),
            archetype_ids=batch.archetype_ids.to(self._device),
        )

    def _batch_to_device_precomputed(self, batch: PrecomputedBatch) -> PrecomputedBatch:
        if self._device.type == "cpu":
            return batch
        return PrecomputedBatch(
            game_embeddings=batch.game_embeddings.to(self._device),
            game_mask=batch.game_mask.to(self._device),
            targets=batch.targets.to(self._device),
            context_mean=batch.context_mean.to(self._device),
            identity_features=batch.identity_features.to(self._device),
            archetype_ids=batch.archetype_ids.to(self._device),
        )
