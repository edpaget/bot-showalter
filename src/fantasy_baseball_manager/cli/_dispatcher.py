from __future__ import annotations

from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    AblationResult,
    Evaluable,
    EvalResult,
    FineTunable,
    ModelConfig,
    Predictable,
    PredictResult,
    Preparable,
    PrepareResult,
    Model,
    Trainable,
    TrainResult,
)
from fantasy_baseball_manager.models.run_manager import RunManager


class UnsupportedOperation(Exception):
    pass


_OPERATION_MAP: dict[str, tuple[type, str]] = {
    "prepare": (Preparable, "prepare"),
    "train": (Trainable, "train"),
    "evaluate": (Evaluable, "evaluate"),
    "predict": (Predictable, "predict"),
    "finetune": (FineTunable, "finetune"),
    "ablate": (Ablatable, "ablate"),
}

type _AnyResult = PrepareResult | TrainResult | EvalResult | PredictResult | AblationResult


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

    context = None
    if run_manager is not None and operation in {"train", "predict"}:
        context = run_manager.begin_run(model, config, operation=operation)

    method = getattr(model, method_name)
    result = method(config)

    if context is not None and isinstance(result, TrainResult):
        for key, value in result.metrics.items():
            context.log_metric(key, value)

    if context is not None and run_manager is not None:
        run_manager.finalize_run(context, config)

    return result
