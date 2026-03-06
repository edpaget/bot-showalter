import json
from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ModelRunRecord


def print_run_list(records: list[ModelRunRecord]) -> None:
    """Print a table of model runs."""
    if not records:
        console.print("No runs found.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("System")
    table.add_column("Version")
    table.add_column("Operation")
    table.add_column("Created")
    table.add_column("Tags")
    for r in records:
        tags_str = ", ".join(f"{k}={v}" for k, v in r.tags_json.items()) if r.tags_json else ""
        table.add_row(r.system, r.version, r.operation, r.created_at, tags_str)
    console.print(table)


def print_run_detail(record: ModelRunRecord) -> None:
    """Print full details of a model run."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("System", record.system)
    table.add_row("Version", record.version)
    table.add_row("Operation", record.operation)
    table.add_row("Created", record.created_at)
    table.add_row("Git Commit", record.git_commit or "N/A")
    table.add_row("Artifact Type", record.artifact_type)
    table.add_row("Artifact Path", record.artifact_path or "N/A")
    if record.config_json:
        table.add_row("Config", json.dumps(record.config_json, indent=2))
    if record.metrics_json:
        table.add_row("Metrics", json.dumps(record.metrics_json, indent=2))
    if record.tags_json:
        tags_str = ", ".join(f"{k}={v}" for k, v in record.tags_json.items())
        table.add_row("Tags", tags_str)
    console.print(table)


def print_run_inspect(record: ModelRunRecord, section: str | None = None) -> None:
    """Print structured inspection of a model run."""
    if section is None or section == "header":
        header = (
            f"[bold]{record.system}[/bold] / {record.version}\n"
            f"Operation: {record.operation}\n"
            f"Created: {record.created_at}\n"
            f"Git Commit: {record.git_commit or 'N/A'}"
        )
        console.print(Panel(header, title="Run Info"))

    if (section is None or section == "config") and record.config_json:
        console.print("[bold]Config[/bold]")
        config = record.config_json
        # Group into subsections
        if "seasons" in config:
            console.print("  [dim]Seasons:[/dim]", json.dumps(config["seasons"]))
        if "model_params" in config:
            params_table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
            params_table.add_column("Param", style="bold")
            params_table.add_column("Value")
            for k, v in config["model_params"].items():
                params_table.add_row(f"  {k}", str(v))
            console.print("  [dim]Model Params:[/dim]")
            console.print(params_table)
        other = {k: v for k in config if k not in ("seasons", "model_params") for v in [config[k]]}
        if other:
            other_table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
            other_table.add_column("Key", style="bold")
            other_table.add_column("Value")
            for k, v in other.items():
                other_table.add_row(f"  {k}", str(v))
            console.print("  [dim]Other:[/dim]")
            console.print(other_table)

    if (section is None or section == "metrics") and record.metrics_json:
        console.print("[bold]Metrics[/bold]")
        metrics_table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")
        for k, v in record.metrics_json.items():
            display = f"{v:.4f}" if isinstance(v, float) else str(v)
            metrics_table.add_row(f"  {k}", display)
        console.print(metrics_table)

    if (section is None or section == "tags") and record.tags_json:
        console.print("[bold]Tags[/bold]")
        for k, v in record.tags_json.items():
            console.print(f"  {k} = {v}")


def diff_records(a: ModelRunRecord, b: ModelRunRecord) -> dict[str, dict[str, Any]]:
    """Compute structured diff between two model run records.

    Returns a dict with 'config' and 'metrics' keys, each containing
    'added', 'removed', and 'changed' sub-dicts.
    """
    result: dict[str, dict[str, Any]] = {}
    for section, get_data in [
        ("config", lambda r: r.config_json or {}),
        ("metrics", lambda r: r.metrics_json or {}),
    ]:
        data_a = get_data(a)
        data_b = get_data(b)
        keys_a = set(data_a.keys())
        keys_b = set(data_b.keys())

        added = {k: data_b[k] for k in sorted(keys_b - keys_a)}
        removed = {k: data_a[k] for k in sorted(keys_a - keys_b)}
        changed: dict[str, Any] = {}
        for k in sorted(keys_a & keys_b):
            if data_a[k] != data_b[k]:
                entry: dict[str, Any] = {"old": data_a[k], "new": data_b[k]}
                if section == "metrics" and isinstance(data_a[k], int | float) and isinstance(data_b[k], int | float):
                    entry["delta"] = data_b[k] - data_a[k]
                changed[k] = entry

        result[section] = {"added": added, "removed": removed, "changed": changed}
    return result


def print_run_diff(a: ModelRunRecord, b: ModelRunRecord) -> None:
    """Print a colored diff between two model run records."""
    console.print(f"[bold]Diff:[/bold] {a.system}/{a.version} vs {b.system}/{b.version}")
    diff = diff_records(a, b)

    for section_name in ("config", "metrics"):
        section = diff[section_name]
        has_changes = any(section[k] for k in ("added", "removed", "changed"))
        if not has_changes:
            continue

        console.print(f"\n[bold]{section_name.title()}[/bold]")
        table = Table(show_header=True, box=None, pad_edge=False, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Change")
        table.add_column(a.version)
        table.add_column(b.version)
        if section_name == "metrics":
            table.add_column("Delta", justify="right")

        for k, v in section["added"].items():
            row = [f"  {k}", "[green]added[/green]", "", str(v)]
            if section_name == "metrics":
                row.append("")
            table.add_row(*row)

        for k, v in section["removed"].items():
            row = [f"  {k}", "[red]removed[/red]", str(v), ""]
            if section_name == "metrics":
                row.append("")
            table.add_row(*row)

        for k, v in section["changed"].items():
            old_str = str(v["old"])
            new_str = str(v["new"])
            row = [f"  {k}", "[yellow]changed[/yellow]", old_str, new_str]
            if section_name == "metrics":
                delta = v.get("delta")
                if delta is not None:
                    sign = "+" if delta > 0 else ""
                    row.append(f"{sign}{delta:.4f}")
                else:
                    row.append("")
            table.add_row(*row)

        console.print(table)
