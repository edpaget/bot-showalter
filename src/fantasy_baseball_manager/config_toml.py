import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIG_FILENAME = "fbm.toml"
_LOCAL_CONFIG_FILENAME = "fbm.local.toml"


@dataclass(frozen=True)
class SystemVersion:
    """A system/version pair for filtering projections or valuations."""

    system: str
    version: str


@dataclass(frozen=True)
class WebConfig:
    """Configuration for the web UI, loaded from ``[web]`` in fbm.toml."""

    projections: list[SystemVersion] = field(default_factory=list)
    valuations: list[SystemVersion] = field(default_factory=list)


def load_web_config(config_dir: Path | None = None) -> WebConfig:
    """Load ``[web]`` section from fbm.toml into a :class:`WebConfig`.

    Returns a default (empty) config when no ``[web]`` section exists, which
    means "show everything" — no filtering applied.
    """
    toml_data = load_toml(config_dir)
    web = toml_data.get("web", {})

    projections = [SystemVersion(system=p["system"], version=p["version"]) for p in web.get("projections", [])]
    valuations = [SystemVersion(system=v["system"], version=v["version"]) for v in web.get("valuations", [])]
    return WebConfig(projections=projections, valuations=valuations)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, with override winning on conflicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_toml(config_dir: Path | None = None) -> dict[str, Any]:
    """Load fbm.toml, then deep-merge fbm.local.toml on top if present.

    Returns ``{}`` if neither file exists.
    """
    if config_dir is None:
        config_dir = Path.cwd()

    base: dict[str, Any] = {}
    base_path = config_dir / _CONFIG_FILENAME
    if base_path.exists():
        with base_path.open("rb") as f:
            base = tomllib.load(f)

    local_path = config_dir / _LOCAL_CONFIG_FILENAME
    if local_path.exists():
        with local_path.open("rb") as f:
            local = tomllib.load(f)
        base = deep_merge(base, local)

    return base
