import tomllib
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.models.protocols import ModelConfig

_CONFIG_FILENAME = "fbm.toml"


def load_config(
    model_name: str,
    config_dir: Path | None = None,
    output_dir: str | None = None,
    seasons: list[int] | None = None,
) -> ModelConfig:
    """Load config from TOML file (if present) with CLI overrides applied on top."""
    toml_data = _load_toml(config_dir)
    common: dict[str, Any] = toml_data.get("common", {})
    model_section: dict[str, Any] = toml_data.get("models", {}).get(model_name, {})
    model_params: dict[str, Any] = model_section.get("params", {})

    version: str | None = model_section.get("version")
    tags: dict[str, str] = model_section.get("tags", {})

    return ModelConfig(
        data_dir=common.get("data_dir", "./data"),
        artifacts_dir=common.get("artifacts_dir", "./artifacts"),
        seasons=seasons if seasons is not None else common.get("seasons", []),
        model_params=model_params,
        output_dir=output_dir,
        version=version,
        tags=tags,
    )


def _load_toml(config_dir: Path | None) -> dict[str, Any]:
    """Read fbm.toml from config_dir, returning empty dict if not found."""
    if config_dir is None:
        config_dir = Path.cwd()
    toml_path = config_dir / _CONFIG_FILENAME
    if not toml_path.exists():
        return {}
    with toml_path.open("rb") as f:
        return tomllib.load(f)
