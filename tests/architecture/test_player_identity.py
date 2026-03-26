"""Ensure fantasy-relevant code uses (player_id, player_type) composite keys.

Bare ``set[int]`` or ``dict[int, ...]`` patterns in keeper/valuation/draft
contexts can silently break two-way player handling (e.g., keeping Ohtani
as a batter should not remove Ohtani-pitcher from the draft pool).
"""

import re
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "fantasy_baseball_manager"

# Files where bare player_id sets are expected/acceptable (non-fantasy contexts)
_ALLOWED_FILES: frozenset[str] = frozenset(
    {
        # Repos deal with raw DB IDs, not fantasy identity
        "repos/",
        # Player resolution maps names to IDs before identity is established
        "name_utils.py",
        "services/player_name_resolver.py",
        "services/alias_seeder.py",
        # Ingest/import boundaries resolve identity at the boundary
        "ingest/",
        "yahoo/",
        # Domain models define the types themselves
        "domain/",
        # Web layer serializes to/from external formats
        "web/",
        # CLI commands are composition roots
        "cli/",
        # Agent/tools are thin adapters
        "agent/",
        "tools/",
        # Config is not player-related
        "config",
        # DB layer
        "db/",
    }
)

# Pattern: function parameters named *keeper_ids* typed as set[int]
_KEEPER_IDS_BARE_RE = re.compile(r"league_keeper_ids:\s*set\[int\]")


def _service_py_files() -> list[Path]:
    """Service and model Python files where composite keys should be enforced."""
    services_dir = _SRC_ROOT / "services"
    models_dir = _SRC_ROOT / "models"
    files: list[Path] = []
    for d in [services_dir, models_dir]:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(files)


class TestNoBarKeeperIdSets:
    """Keeper ID sets must use (player_id, player_type) composite keys."""

    def test_no_bare_keeper_id_sets_in_services(self) -> None:
        violations: list[str] = []
        for path in _service_py_files():
            if path.name == "__init__.py":
                continue
            text = path.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if _KEEPER_IDS_BARE_RE.search(line):
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"  {rel}:{i} — {line.strip()}")

        assert not violations, (
            "Found bare `set[int]` keeper ID parameters that should use"
            " `set[tuple[int, str]]` for two-way player safety:\n" + "\n".join(violations)
        )
