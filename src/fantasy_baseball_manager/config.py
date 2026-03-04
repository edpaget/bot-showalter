from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.config_toml import deep_merge, load_toml
from fantasy_baseball_manager.models import ModelConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_config(
    model_name: str,
    config_dir: Path | None = None,
    output_dir: str | None = None,
    seasons: list[int] | None = None,
    version: str | None = None,
    tags: dict[str, str] | None = None,
    top: int | None = None,
    model_params: dict[str, Any] | None = None,
) -> ModelConfig:
    """Load config from TOML file (if present) with CLI overrides applied on top."""
    toml_data = load_toml(config_dir)
    common: dict[str, Any] = toml_data.get("common", {})
    model_section: dict[str, Any] = toml_data.get("models", {}).get(model_name, {})
    toml_params: dict[str, Any] = model_section.get("params", {})
    resolved_params = deep_merge(toml_params, model_params or {})

    toml_version: str | None = model_section.get("version")
    resolved_version = version if version is not None else toml_version

    toml_tags: dict[str, str] = model_section.get("tags", {})
    resolved_tags = {**toml_tags, **(tags or {})}

    return ModelConfig(
        data_dir=common.get("data_dir", "./data"),
        artifacts_dir=common.get("artifacts_dir", "./artifacts"),
        seasons=seasons if seasons is not None else common.get("seasons", []),
        model_params=resolved_params,
        output_dir=output_dir,
        version=resolved_version,
        tags=resolved_tags,
        top=top,
    )
