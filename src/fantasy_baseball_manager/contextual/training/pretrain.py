"""Masked Gamestate Modeling (MGM) pre-training trainer."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from fantasy_baseball_manager.contextual.persistence import ContextualModelMetadata
from fantasy_baseball_manager.contextual.training.dataset import (
    MaskedBatch,
    MaskedSample,
    collate_masked_samples,
    masked_batch_to_tensorized_batch,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
    from fantasy_baseball_manager.contextual.training.dataset import MGMDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Metrics from a training or validation pass."""

    loss: float
    pitch_type_loss: float
    pitch_result_loss: float
    pitch_type_accuracy: float
    pitch_result_accuracy: float
    num_masked_tokens: int


class MGMTrainer:
    """Pre-trains a ContextualPerformanceModel using Masked Gamestate Modeling."""

    def __init__(
        self,
        model: ContextualPerformanceModel,
        model_config: ModelConfig,
        config: PreTrainingConfig,
        model_store: ContextualModelStore,
        device: torch.device | None = None,
    ) -> None:
        self._model = model
        self._model_config = model_config
        self._config = config
        self._store = model_store
        self._device = device or torch.device("cpu")
        self._model.to(self._device)

    def train(
        self,
        train_dataset: MGMDataset,
        val_dataset: MGMDataset,
        resume_from: str | None = None,
    ) -> dict[str, float]:
        """Run the pre-training loop.

        Returns:
            Dict with final validation metrics.
        """
        config = self._config

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._scaler = GradScaler(enabled=config.amp_enabled)

        total_steps = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
        warmup_steps = max(config.min_warmup_steps, int(total_steps * config.warmup_fraction))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0
                ),
            ],
            milestones=[warmup_steps],
        )

        start_epoch = 0
        best_val_loss = float("inf")
        global_step = 0

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
                scaler_state = sched_state.get("scaler")
                if scaler_state is not None:
                    self._scaler.load_state_dict(scaler_state)
            logger.info("Resumed from checkpoint '%s' at epoch %d", resume_from, start_epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_masked_samples,
            generator=torch.Generator().manual_seed(config.seed),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_masked_samples,
        )

        best_state_dict = None

        for epoch in range(start_epoch, config.epochs):
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, global_step)
            global_step += len(train_loader)

            # Validation
            val_metrics = self._validate(val_loader)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f val_loss=%.4f pt_acc=%.4f pr_acc=%.4f",
                epoch + 1, config.epochs,
                train_metrics.loss, val_metrics.loss,
                val_metrics.pitch_type_accuracy, val_metrics.pitch_result_accuracy,
            )

            # Track best
            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                self._save_checkpoint(
                    "pretrain_best", epoch, train_metrics, val_metrics, optimizer, global_step, best_val_loss
                )

            # Periodic checkpoint
            if (epoch + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(
                    f"pretrain_epoch_{epoch + 1}", epoch, train_metrics, val_metrics,
                    optimizer, global_step, best_val_loss,
                )

        # Save latest
        val_metrics = self._validate(val_loader)
        self._save_checkpoint(
            "pretrain_latest", config.epochs - 1, train_metrics, val_metrics,
            optimizer, global_step, best_val_loss,
        )

        # Restore best model
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        return {
            "val_loss": val_metrics.loss,
            "val_pitch_type_loss": val_metrics.pitch_type_loss,
            "val_pitch_result_loss": val_metrics.pitch_result_loss,
            "val_pitch_type_accuracy": val_metrics.pitch_type_accuracy,
            "val_pitch_result_accuracy": val_metrics.pitch_result_accuracy,
        }

    def _train_epoch(
        self,
        loader: DataLoader[MaskedSample],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int,
    ) -> TrainingMetrics:
        self._model.train()
        total_loss = 0.0
        total_pt_loss = 0.0
        total_pr_loss = 0.0
        total_pt_correct = 0
        total_pr_correct = 0
        total_masked = 0

        n_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            batch = self._batch_to_device(batch)
            tb = masked_batch_to_tensorized_batch(batch)

            with torch.amp.autocast(self._device.type, enabled=self._config.amp_enabled):
                output = self._model(tb)
                pt_logits = output["pitch_type_logits"]  # (B, S, vocab)
                pr_logits = output["pitch_result_logits"]  # (B, S, vocab)

                pt_loss, pr_loss = self._compute_loss(
                    pt_logits, pr_logits,
                    batch.target_pitch_type_ids,
                    batch.target_pitch_result_ids,
                )

                loss = (
                    self._config.pitch_type_loss_weight * pt_loss
                    + self._config.pitch_result_loss_weight * pr_loss
                )

            optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            scheduler.step()

            n_masked = int(batch.mask_positions.sum().item())
            total_loss += loss.item() * n_masked
            total_pt_loss += pt_loss.item() * n_masked
            total_pr_loss += pr_loss.item() * n_masked
            total_masked += n_masked

            # Accuracy on masked positions
            if n_masked > 0:
                mask = batch.mask_positions
                pt_preds = pt_logits[mask].argmax(dim=-1)
                pr_preds = pr_logits[mask].argmax(dim=-1)
                total_pt_correct += int((pt_preds == batch.target_pitch_type_ids[mask]).sum().item())
                total_pr_correct += int((pr_preds == batch.target_pitch_result_ids[mask]).sum().item())

            if (batch_idx + 1) % self._config.log_interval == 0 or batch_idx == n_batches - 1:
                avg_loss = total_loss / total_masked if total_masked > 0 else 0.0
                logger.info(
                    "  batch %d/%d — loss=%.4f",
                    batch_idx + 1, n_batches, avg_loss,
                )

            global_step += 1

        if total_masked == 0:
            return TrainingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        return TrainingMetrics(
            loss=total_loss / total_masked,
            pitch_type_loss=total_pt_loss / total_masked,
            pitch_result_loss=total_pr_loss / total_masked,
            pitch_type_accuracy=total_pt_correct / total_masked,
            pitch_result_accuracy=total_pr_correct / total_masked,
            num_masked_tokens=total_masked,
        )

    @torch.no_grad()
    def _validate(self, loader: DataLoader[MaskedSample]) -> TrainingMetrics:
        self._model.eval()
        total_loss = 0.0
        total_pt_loss = 0.0
        total_pr_loss = 0.0
        total_pt_correct = 0
        total_pr_correct = 0
        total_masked = 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            tb = masked_batch_to_tensorized_batch(batch)

            with torch.amp.autocast(self._device.type, enabled=self._config.amp_enabled):
                output = self._model(tb)
                pt_logits = output["pitch_type_logits"]
                pr_logits = output["pitch_result_logits"]

                pt_loss, pr_loss = self._compute_loss(
                    pt_logits, pr_logits,
                    batch.target_pitch_type_ids,
                    batch.target_pitch_result_ids,
                )

                loss = (
                    self._config.pitch_type_loss_weight * pt_loss
                    + self._config.pitch_result_loss_weight * pr_loss
                )

            n_masked = int(batch.mask_positions.sum().item())
            total_loss += loss.item() * n_masked
            total_pt_loss += pt_loss.item() * n_masked
            total_pr_loss += pr_loss.item() * n_masked
            total_masked += n_masked

            if n_masked > 0:
                mask = batch.mask_positions
                pt_preds = pt_logits[mask].argmax(dim=-1)
                pr_preds = pr_logits[mask].argmax(dim=-1)
                total_pt_correct += int((pt_preds == batch.target_pitch_type_ids[mask]).sum().item())
                total_pr_correct += int((pr_preds == batch.target_pitch_result_ids[mask]).sum().item())

        if total_masked == 0:
            return TrainingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        return TrainingMetrics(
            loss=total_loss / total_masked,
            pitch_type_loss=total_pt_loss / total_masked,
            pitch_result_loss=total_pr_loss / total_masked,
            pitch_type_accuracy=total_pt_correct / total_masked,
            pitch_result_accuracy=total_pr_correct / total_masked,
            num_masked_tokens=total_masked,
        )

    def _compute_loss(
        self,
        pt_logits: torch.Tensor,
        pr_logits: torch.Tensor,
        target_pt: torch.Tensor,
        target_pr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-entropy loss for pitch type and result at masked positions.

        Targets are 0 at non-masked positions, so ignore_index=0 skips them.
        """
        # Reshape to (N, vocab_size) and (N,)
        pt_loss = F.cross_entropy(
            pt_logits.view(-1, pt_logits.size(-1)),
            target_pt.view(-1),
            ignore_index=0,
        )
        pr_loss = F.cross_entropy(
            pr_logits.view(-1, pr_logits.size(-1)),
            target_pr.view(-1),
            ignore_index=0,
        )
        return pt_loss, pr_loss

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        train_metrics: TrainingMetrics,
        val_metrics: TrainingMetrics,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        best_val_loss: float,
    ) -> None:
        metadata = ContextualModelMetadata(
            name=name,
            epoch=epoch + 1,
            train_loss=train_metrics.loss,
            val_loss=val_metrics.loss,
            pitch_type_accuracy=val_metrics.pitch_type_accuracy,
            pitch_result_accuracy=val_metrics.pitch_result_accuracy,
        )
        scheduler_state = {
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "scaler": self._scaler.state_dict(),
        }
        self._store.save_checkpoint(
            name, self._model, metadata,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler_state,
        )

    def _batch_to_device(self, batch: MaskedBatch) -> MaskedBatch:
        if self._device.type == "cpu":
            return batch
        return MaskedBatch(
            pitch_type_ids=batch.pitch_type_ids.to(self._device),
            pitch_result_ids=batch.pitch_result_ids.to(self._device),
            bb_type_ids=batch.bb_type_ids.to(self._device),
            stand_ids=batch.stand_ids.to(self._device),
            p_throws_ids=batch.p_throws_ids.to(self._device),
            pa_event_ids=batch.pa_event_ids.to(self._device),
            numeric_features=batch.numeric_features.to(self._device),
            numeric_mask=batch.numeric_mask.to(self._device),
            padding_mask=batch.padding_mask.to(self._device),
            player_token_mask=batch.player_token_mask.to(self._device),
            game_ids=batch.game_ids.to(self._device),
            target_pitch_type_ids=batch.target_pitch_type_ids.to(self._device),
            target_pitch_result_ids=batch.target_pitch_result_ids.to(self._device),
            mask_positions=batch.mask_positions.to(self._device),
            seq_lengths=batch.seq_lengths.to(self._device),
        )
