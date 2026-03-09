from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import typer

from fantasy_baseball_manager.config_toml import load_toml
from fantasy_baseball_manager.domain import current_season

if TYPE_CHECKING:
    from pathlib import Path

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@dataclass(frozen=True)
class CliDefaults:
    system: str
    version: str
    season: int


def load_cli_defaults(config_dir: Path | None = None) -> CliDefaults:
    """Load CLI defaults from ``fbm.toml`` / ``fbm.local.toml``.

    Falls back to hardcoded defaults when keys are missing from the config.
    """
    toml_data = load_toml(config_dir)
    common = toml_data.get("common", {})
    return CliDefaults(
        system=common.get("system", "zar"),
        version=common.get("version", "production"),
        season=current_season(),
    )
