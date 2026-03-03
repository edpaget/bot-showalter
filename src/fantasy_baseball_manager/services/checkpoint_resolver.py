from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.exceptions import FbmException

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import FeatureCheckpoint
    from fantasy_baseball_manager.repos import CheckpointRepo


def is_checkpoint_spec(value: str) -> bool:
    """Return True if value matches the ``checkpoint:<name>`` pattern."""
    return value.startswith("checkpoint:") and len(value) > len("checkpoint:")


def resolve_checkpoint(repo: CheckpointRepo, spec: str, model: str) -> FeatureCheckpoint:
    """Parse ``checkpoint:<name>`` and look up the checkpoint.

    Raises ``FbmException`` if not found.
    """
    name = spec.removeprefix("checkpoint:")
    checkpoint = repo.get(name, model)
    if checkpoint is None:
        raise FbmException(f"Checkpoint '{name}' not found for model '{model}'.")
    return checkpoint
