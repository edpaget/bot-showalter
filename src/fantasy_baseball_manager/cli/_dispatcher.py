from __future__ import annotations

import logging
import time

from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    AblationResult,
    Evaluable,
    FineTunable,
    ModelConfig,
    Predictable,
    PredictResult,
    Preparable,
    PrepareResult,
    Model,
    Trainable,
    TrainResult,
    Tunable,
    TuneResult,
)
from fantasy_baseball_manager.models.run_manager import RunManager

logger = logging.getLogger(__name__)


class UnsupportedOperation(Exception):
    pass


_OPERATION_MAP: dict[str, tuple[type, str]] = {
    "prepare": (Preparable, "prepare"),
    "train": (Trainable, "train"),
    "evaluate": (Evaluable, "evaluate"),
    "predict": (Predictable, "predict"),
    "finetune": (FineTunable, "finetune"),
    "ablate": (Ablatable, "ablate"),
    "tune": (Tunable, "tune"),
}

type _AnyResult = PrepareResult | TrainResult | SystemMetrics | PredictResult | AblationResult | TuneResult


def dispatch(
    operation: str,
    model: Model,
    config: ModelConfig,
    run_manager: RunManager | None = None,
) -> _AnyResult:
    """Check capability and invoke the operation on a pre-built model instance."""
    if operation not in _OPERATION_MAP:
        raise UnsupportedOperation(f"Model '{model.name}' does not support '{operation}'")

    protocol, method_name = _OPERATION_MAP[operation]

    if not isinstance(model, protocol):
        raise UnsupportedOperation(f"Model '{model.name}' does not support '{operation}'")

    t0 = time.perf_counter()
    context = None
    if run_manager is not None and operation in {"train", "predict"}:
        context = run_manager.begin_run(model, config, operation=operation)

    method = getattr(model, method_name)
    result = method(config)
    logger.info("Completed '%s' on '%s' in %.1fs", operation, model.name, time.perf_counter() - t0)

    if context is not None and isinstance(result, TrainResult):
        for key, value in result.metrics.items():
            context.log_metric(key, value)

    if context is not None and run_manager is not None:
        run_manager.finalize_run(context, config)

    return result
