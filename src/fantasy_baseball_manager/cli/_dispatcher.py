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
    Trainable,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import get


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


def dispatch(operation: str, model_name: str, config: ModelConfig) -> _AnyResult:
    """Resolve a model from the registry, check capability, and invoke the operation."""
    model = get(model_name)

    if operation not in _OPERATION_MAP:
        raise UnsupportedOperation(f"Model '{model_name}' does not support '{operation}'")

    protocol, method_name = _OPERATION_MAP[operation]

    if not isinstance(model, protocol):
        raise UnsupportedOperation(f"Model '{model_name}' does not support '{operation}'")

    method = getattr(model, method_name)
    return method(config)
