"""Fine-tuning trainer for per-game stat prediction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fantasy_baseball_manager.contextual.persistence import ContextualModelMetadata
from fantasy_baseball_manager.contextual.training.dataset import (
    FineTuneBatch,
    FineTuneSample,
    collate_finetune_samples,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.contextual.training.config import FineTuneConfig
    from fantasy_baseball_manager.contextual.training.dataset import FineTuneDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FineTuneMetrics:
    """Metrics from a fine-tuning training or validation pass."""

    loss: float
    per_stat_mse: dict[str, float]
    per_stat_mae: dict[str, float]
    n_samples: int
    baseline_per_stat_mse: dict[str, float] = field(default_factory=dict)
    baseline_per_stat_mae: dict[str, float] = field(default_factory=dict)


class FineTuneTrainer:
    """Fine-tunes a pre-trained ContextualPerformanceModel on per-game stat prediction."""

    def __init__(
        self,
        model: ContextualPerformanceModel,
        model_config: ModelConfig,
        config: FineTuneConfig,
        model_store: ContextualModelStore,
        target_stats: tuple[str, ...],
        device: torch.device | None = None,
    ) -> None:
        self._model = model
        self._model_config = model_config
        self._config = config
        self._store = model_store
        self._target_stats = target_stats
        self._device = device or torch.device("cpu")
        self._model.to(self._device)

        if config.freeze_backbone:
            for param in self._model.embedder.parameters():
                param.requires_grad = False
            for param in self._model.transformer.parameters():
                param.requires_grad = False

    def train(
        self,
        train_dataset: FineTuneDataset,
        val_dataset: FineTuneDataset,
        resume_from: str | None = None,
    ) -> dict[str, float]:
        """Run the fine-tuning loop.

        Returns:
            Dict with final validation metrics.
        """
        config = self._config

        # Set up parameter groups with discriminative learning rates
        backbone_params = (
            list(self._model.embedder.parameters())
            + list(self._model.transformer.parameters())
        )
        head_params = list(self._model.head.parameters())

        param_groups = [
            {"params": head_params, "lr": config.head_learning_rate},
        ]
        if not config.freeze_backbone:
            param_groups.insert(
                0, {"params": backbone_params, "lr": config.backbone_learning_rate},
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

        total_steps = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
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

        start_epoch = 0
        best_val_loss = float("inf")
        global_step = 0
        patience_counter = 0

        # Resume from checkpoint if requested
        if resume_from is not None:
            state_dict, opt_state, sched_state = self._store.load_training_state(resume_from)
            self._model.load_state_dict(state_dict)
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
            if sched_state is not None:
                start_epoch = sched_state.get("epoch", 0)
                best_val_loss = sched_state.get("best_val_loss", float("inf"))
                global_step = sched_state.get("global_step", 0)
            logger.info("Resumed from checkpoint '%s' at epoch %d", resume_from, start_epoch)

        loader_kwargs: dict[str, Any] = {}
        if self._device.type == "cuda":
            loader_kwargs.update(num_workers=4, pin_memory=True, persistent_workers=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_finetune_samples,
            generator=torch.Generator().manual_seed(config.seed),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_finetune_samples,
            **loader_kwargs,
        )

        best_state_dict = None

        for epoch in range(start_epoch, config.epochs):
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, global_step)
            global_step += len(train_loader)

            val_metrics = self._validate(val_loader)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f val_loss=%.4f",
                epoch + 1, config.epochs,
                train_metrics.loss, val_metrics.loss,
            )

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                self._save_checkpoint(
                    "finetune_best", epoch, train_metrics, val_metrics,
                    optimizer, global_step, best_val_loss,
                )
            else:
                patience_counter += 1

            # Periodic checkpoint
            if (epoch + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(
                    f"finetune_epoch_{epoch + 1}", epoch, train_metrics, val_metrics,
                    optimizer, global_step, best_val_loss,
                )

            # Early stopping
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
                break

        # Save latest
        val_metrics = self._validate(val_loader)
        self._save_checkpoint(
            "finetune_latest", epoch, train_metrics, val_metrics,
            optimizer, global_step, best_val_loss,
        )

        # Restore best model
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        return {
            "val_loss": val_metrics.loss,
            **{f"val_{stat}_mse": v for stat, v in val_metrics.per_stat_mse.items()},
            **{f"val_{stat}_mae": v for stat, v in val_metrics.per_stat_mae.items()},
            **{f"baseline_{stat}_mse": v for stat, v in val_metrics.baseline_per_stat_mse.items()},
            **{f"baseline_{stat}_mae": v for stat, v in val_metrics.baseline_per_stat_mae.items()},
        }

    def _train_epoch(
        self,
        loader: DataLoader[FineTuneSample],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int,
    ) -> FineTuneMetrics:
        self._model.train()
        total_loss = 0.0
        total_samples = 0
        per_stat_se = torch.zeros(len(self._target_stats))
        per_stat_ae = torch.zeros(len(self._target_stats))

        n_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            batch = self._batch_to_device(batch)

            output = self._model(batch.context)
            preds = output["performance_preds"]  # (batch, n_player_tokens, n_targets)
            pooled_preds = preds.mean(dim=1)  # (batch, n_targets)

            loss = F.mse_loss(pooled_preds, batch.targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            n = batch.targets.shape[0]
            total_loss += loss.item() * n
            total_samples += n

            with torch.no_grad():
                diff = pooled_preds.cpu() - batch.targets.cpu()
                per_stat_se += (diff ** 2).sum(dim=0)
                per_stat_ae += diff.abs().sum(dim=0)

            if (batch_idx + 1) % self._config.log_interval == 0 or batch_idx == n_batches - 1:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                logger.info(
                    "  batch %d/%d — loss=%.4f",
                    batch_idx + 1, n_batches, avg_loss,
                )

            global_step += 1

        if total_samples == 0:
            return FineTuneMetrics(0.0, {}, {}, 0)

        return FineTuneMetrics(
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
    def _validate(self, loader: DataLoader[FineTuneSample]) -> FineTuneMetrics:
        self._model.eval()
        total_loss = 0.0
        total_samples = 0
        per_stat_se = torch.zeros(len(self._target_stats))
        per_stat_ae = torch.zeros(len(self._target_stats))
        baseline_se = torch.zeros(len(self._target_stats))
        baseline_ae = torch.zeros(len(self._target_stats))

        for batch in loader:
            batch = self._batch_to_device(batch)

            output = self._model(batch.context)
            preds = output["performance_preds"]  # (batch, n_player_tokens, n_targets)
            pooled_preds = preds.mean(dim=1)  # (batch, n_targets)

            loss = F.mse_loss(pooled_preds, batch.targets)

            n = batch.targets.shape[0]
            total_loss += loss.item() * n
            total_samples += n

            diff = pooled_preds.cpu() - batch.targets.cpu()
            per_stat_se += (diff ** 2).sum(dim=0)
            per_stat_ae += diff.abs().sum(dim=0)

            baseline_diff = batch.context_mean.cpu() - batch.targets.cpu()
            baseline_se += (baseline_diff ** 2).sum(dim=0)
            baseline_ae += baseline_diff.abs().sum(dim=0)

        if total_samples == 0:
            return FineTuneMetrics(0.0, {}, {}, 0)

        return FineTuneMetrics(
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

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        train_metrics: FineTuneMetrics,
        val_metrics: FineTuneMetrics,
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
        )
        scheduler_state = {
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "global_step": global_step,
        }
        self._store.save_checkpoint(
            name, self._model, metadata,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler_state,
        )

    def _batch_to_device(self, batch: FineTuneBatch) -> FineTuneBatch:
        if self._device.type == "cpu":
            return batch
        from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedBatch

        ctx = batch.context
        return FineTuneBatch(
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
        )
