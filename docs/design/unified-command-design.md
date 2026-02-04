# Unified Command Definition Design

## Problem

Agent tools (`src/fantasy_baseball_manager/agent/tools.py`) and CLI commands (e.g., `ros/cli.py`, `draft/cli.py`) share structural similarities but are defined separately:

| Aspect | Agent Tools | CLI Commands |
|--------|-------------|--------------|
| Decorator | `@tool` (LangChain) | `@app.command()` (Typer) |
| Parameters | Python function args | `Annotated[type, typer.Option]` |
| Return type | `str` (always) | `None` (prints to stdout) |
| Error handling | Return error strings | `typer.Exit(code=1)` |

Both use the same underlying logic (e.g., `build_projections_and_positions()`) but duplicate parameter definitions and output formatting.

## Proposed Solution

A **Command abstraction** that can be adapted to both LangChain tools and Typer CLI commands.

### Core Types

```python
# src/fantasy_baseball_manager/commands/core.py
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

class ParamKind(Enum):
    ARGUMENT = "argument"  # positional
    OPTION = "option"      # --flag style

@dataclass
class Param:
    name: str
    type: type
    default: Any = None
    description: str = ""
    kind: ParamKind = ParamKind.OPTION
    cli_only: bool = False  # If True, excluded from agent tools

@dataclass
class Command:
    name: str
    description: str
    handler: Callable[..., str]  # Always returns formatted string
    params: list[Param] = field(default_factory=list)

# Registry
COMMANDS: dict[str, Command] = {}

def register(cmd: Command) -> Command:
    """Register a command in the global registry."""
    COMMANDS[cmd.name] = cmd
    return cmd
```

### Example Command Definition

```python
# src/fantasy_baseball_manager/commands/projections.py
from fantasy_baseball_manager.commands.core import Command, Param, register
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, SUPPORTED_ENGINES

def _project_batters(year: int | None, engine: str, top_n: int) -> str:
    """Core implementation that returns formatted string."""
    if year is None:
        year = datetime.now().year
    if engine not in SUPPORTED_ENGINES:
        return f"Invalid engine '{engine}'. Supported: {', '.join(SUPPORTED_ENGINES)}"

    all_values, _ = build_projections_and_positions(engine, year)
    batter_values = [pv for pv in all_values if pv.position_type == "B"]
    return _format_batter_table(batter_values, top_n)

project_batters = register(Command(
    name="project_batters",
    description="Get projected batter statistics and z-score valuations.",
    handler=_project_batters,
    params=[
        Param("year", int | None, None, "Projection year (default: current)"),
        Param("engine", str, DEFAULT_ENGINE, f"Projection engine: {', '.join(SUPPORTED_ENGINES)}"),
        Param("top_n", int, 25, "Number of top batters to return"),
    ],
))
```

### LangChain Adapter

```python
# src/fantasy_baseball_manager/commands/adapters/langchain.py
from langchain_core.tools import StructuredTool
from fantasy_baseball_manager.commands.core import Command, COMMANDS

def to_langchain_tool(cmd: Command) -> StructuredTool:
    """Convert a Command to a LangChain StructuredTool."""
    # Filter out CLI-only params for agent use
    agent_params = {p.name: p for p in cmd.params if not p.cli_only}

    def wrapper(**kwargs):
        # Only pass through non-CLI params
        filtered = {k: v for k, v in kwargs.items() if k in agent_params}
        return cmd.handler(**filtered)

    return StructuredTool.from_function(
        func=wrapper,
        name=cmd.name,
        description=cmd.description,
    )

def all_tools() -> list[StructuredTool]:
    """Get all registered commands as LangChain tools."""
    return [to_langchain_tool(cmd) for cmd in COMMANDS.values()]
```

### Typer Adapter

```python
# src/fantasy_baseball_manager/commands/adapters/typer.py
import typer
from typing import Annotated, get_type_hints
from fantasy_baseball_manager.commands.core import Command, Param, ParamKind, COMMANDS

def _build_annotation(param: Param):
    """Build a Typer-compatible Annotated type from a Param."""
    if param.kind == ParamKind.ARGUMENT:
        return Annotated[param.type, typer.Argument(help=param.description)]
    else:
        return Annotated[param.type, typer.Option(help=param.description)]

def to_typer_command(cmd: Command, app: typer.Typer) -> None:
    """Register a Command as a Typer command on the given app."""

    def make_wrapper(command: Command):
        def wrapper(**kwargs):
            result = command.handler(**kwargs)
            typer.echo(result)

        # Set annotations for Typer to inspect
        wrapper.__annotations__ = {
            p.name: _build_annotation(p) for p in command.params
        }
        wrapper.__doc__ = command.description

        # Set defaults
        for p in command.params:
            if p.default is not None:
                wrapper.__kwdefaults__ = wrapper.__kwdefaults__ or {}
                wrapper.__kwdefaults__[p.name] = p.default

        return wrapper

    cli_name = cmd.name.replace("_", "-")
    app.command(name=cli_name)(make_wrapper(cmd))

def register_all(app: typer.Typer) -> None:
    """Register all commands on a Typer app."""
    for cmd in COMMANDS.values():
        to_typer_command(cmd, app)
```

## File Structure

```
src/fantasy_baseball_manager/
├── commands/
│   ├── __init__.py
│   ├── core.py           # Command, Param, registry
│   ├── projections.py    # project_batters, project_pitchers, etc.
│   ├── keepers.py        # rank_keepers, etc.
│   ├── players.py        # lookup_player, compare_players
│   └── adapters/
│       ├── __init__.py
│       ├── langchain.py  # to_langchain_tool, all_tools
│       └── typer.py      # to_typer_command, register_all
```

## Usage

### In Agent

```python
# src/fantasy_baseball_manager/agent/core.py
from fantasy_baseball_manager.commands.adapters.langchain import all_tools

def create_agent(model):
    tools = all_tools()
    return create_react_agent(model, tools)
```

### In CLI

```python
# src/fantasy_baseball_manager/cli.py
from fantasy_baseball_manager.commands.adapters.typer import register_all

app = typer.Typer()
register_all(app)
```

## Benefits

1. **Single source of truth** - Define command logic and params once
2. **Consistent behavior** - Same validation and output across interfaces
3. **Easy testing** - Test `Command.handler` directly without mocking CLI/agent infrastructure
4. **CLI-specific options** - Use `cli_only=True` for params like `--no-cache`, file paths
5. **Extensible** - Add adapters for REST API, Discord bot, etc.

## Migration Path

1. Create `commands/core.py` with base types
2. Migrate one simple command (e.g., `get_league_settings`) as proof-of-concept
3. Verify both CLI and agent work correctly
4. Migrate remaining commands incrementally
5. Remove old `agent/tools.py` definitions

## Open Questions

- Should commands support async handlers for I/O-bound operations?
- How to handle CLI-specific error codes (e.g., `typer.Exit(1)`) vs agent error strings?
- Should the registry support command categories/groups for CLI subcommands?
