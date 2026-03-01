"""Shared CLI parsing utilities."""

import json
from typing import Any

import typer

from fantasy_baseball_manager.cli._output import print_error


def parse_system_version(system: str) -> tuple[str, str]:
    """Parse a 'system/version' string into a (system, version) tuple.

    Raises ``typer.Exit`` with code 1 when the format is invalid.
    """
    parts = system.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid system format '{system}', expected 'system/version'")
        raise typer.Exit(code=1)
    return parts[0], parts[1]


def coerce_value(value: str) -> Any:
    """Coerce a CLI string value to bool, int, float, JSON object, or leave as str."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key like 'pitcher.learning_rate'."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def parse_params(raw_params: list[str] | None) -> dict[str, Any] | None:
    """Parse ``--param key=value`` options into a dict, coercing types."""
    if not raw_params:
        return None
    parsed: dict[str, Any] = {}
    for param in raw_params:
        key, _, value = param.partition("=")
        set_nested(parsed, key, coerce_value(value))
    return parsed


def parse_tags(raw_tags: list[str] | None) -> dict[str, str] | None:
    """Parse ``--tag key=value`` options into a dict."""
    if not raw_tags:
        return None
    parsed: dict[str, str] = {}
    for tag in raw_tags:
        key, _, value = tag.partition("=")
        parsed[key] = value
    return parsed
