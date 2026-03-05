"""Architecture tests for package re-exports.

Ensures every public package defines ``__all__``, re-exports all public
symbols from its submodules, and that cross-package consumers use
package-level imports rather than reaching into submodules.
"""

import ast
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"
PKG_ROOT = SRC_ROOT / "fantasy_baseball_manager"

AUDITED_PACKAGES: list[str] = [
    "domain",
    "repos",
    "services",
    "models",
    "ingest",
    "features",
]

# Names that are never re-exported — they appear in many submodules and
# are not part of any package's public API.
_GLOBALLY_IGNORED_NAMES: frozenset[str] = frozenset({"logger"})

# Symbols that are intentionally NOT re-exported from each package.
# Keys are "<package>" and values list the excluded names.
_EXCLUDED: dict[str, set[str]] = {
    # Model-internal details — not part of the public models API
    "models": {
        # ablation.py
        "PlayerTypeConfig",
        "evaluate_projections",
        "single_holdout_importance",
        "multi_holdout_importance",
        "run_ablation",
        # distributions.py
        "samples_to_distribution",
        "samples_to_distributions",
        # gbm_training.py
        "CVFold",
        "CorrelationGroup",
        "FeatureImportance",
        "GridSearchResult",
        "GroupedImportanceResult",
        "MinValueFilter",
        "RowFilter",
        "TargetVector",
        "build_cv_folds",
        "compute_cv_permutation_importance",
        "compute_grouped_permutation_importance",
        "compute_permutation_importance",
        "extract_features",
        "extract_sample_weights",
        "extract_targets",
        "fit_models",
        "grid_search_cv",
        "identify_prune_candidates",
        "score_predictions",
        "sweep_cv",
        "validate_pruning",
        # sample_weight_transforms.py
        "REGISTRY",
        "WeightTransform",
        "clamp",
        "get_transform",
        "log1p",
        "raw",
        "sqrt",
        # sampling.py
        "holdout_metrics",
        "season_kfold",
        "temporal_expanding_cv",
        "temporal_holdout_split",
        # stat_utils.py
        "best_rows_per_player",
        "compute_batter_rates",
        "compute_pitcher_rates",
    },
    # Services-internal details — not cross-package API
    "services": {
        # draft_session.py — command classes and helpers
        "PickCommand",
        "UndoCommand",
        "BestCommand",
        "NeedCommand",
        "NeedsCommand",
        "BalanceCommand",
        "RosterCommand",
        "PoolCommand",
        "StatusCommand",
        "SaveCommand",
        "HelpCommand",
        "ReportCommand",
        "QuitCommand",
        "Command",
        "parse_command",
        "save_draft",
        "auto_detect_position",
        "RecommendFn",
        "ReportFn",
        # draft_recommender.py — internal protocol
        "CategoryBalanceFn",
    },
    # Features-internal details — not cross-package API
    "features": {
        # assembler.py — implementation detail
        # (SqliteDatasetAssembler IS re-exported)
        # consensus_pt.py — internal constants/helpers
        "steamer_pa",
        "zips_pa",
        "steamer_ip",
        "zips_ip",
        "make_consensus_transform",
        "CONSENSUS_PA",
        "CONSENSUS_IP",
        # group_library.py has only the three make_*_lags functions re-exported
        # library.py — feature constant sets, not cross-package API
        "STANDARD_BATTING_COUNTING",
        "STANDARD_BATTING_RATES",
        "PLAYER_METADATA",
        "STATCAST_PITCH_MIX",
        "STATCAST_BATTED_BALL",
        "STATCAST_PLATE_DISCIPLINE",
        "STATCAST_EXPECTED_STATS",
        "STATCAST_SPIN_PROFILE",
        # sql.py — internal SQL generation, not cross-package API
        "JoinSpec",
        "generate_sql",
    },
    # Ingest internal helpers — underscore-prefixed modules' contents
    "ingest": {
        "nullify_empty_strings",
        "strip_bom",
    },
}

_BYPASS_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


# ---------------------------------------------------------------------------
# Test 1: __all__ is defined and every symbol is importable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg_name", AUDITED_PACKAGES)
def test_all_defined_and_importable(pkg_name: str) -> None:
    """Each audited package must define __all__ and every name must resolve."""
    mod = importlib.import_module(f"fantasy_baseball_manager.{pkg_name}")
    assert hasattr(mod, "__all__"), f"{pkg_name}/__init__.py is missing __all__"
    all_names: Sequence[str] = mod.__all__
    assert len(all_names) > 0, f"{pkg_name}.__all__ is empty"

    missing = [name for name in all_names if not hasattr(mod, name)]
    assert not missing, f"{pkg_name}.__all__ lists names that are not importable: {missing}"


# ---------------------------------------------------------------------------
# Test 2: Regression — new public symbols must appear in __all__ or _EXCLUDED
# ---------------------------------------------------------------------------


def _public_names_from_file(filepath: Path) -> list[str]:
    """Return top-level public names (classes, functions, assignments) from a .py file."""
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    names: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif (
            isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and not node.target.id.startswith("_")
        ):
            names.append(node.target.id)
    return names


def _submodule_files(pkg_name: str) -> list[tuple[str, Path]]:
    """Return (submodule_key, filepath) pairs for a package's top-level .py files."""
    pkg_dir = PKG_ROOT / pkg_name
    results: list[tuple[str, Path]] = []
    for filepath in sorted(pkg_dir.glob("*.py")):
        if filepath.name == "__init__.py":
            continue
        submod = filepath.stem
        results.append((f"{pkg_name}.{submod}", filepath))
    return results


@pytest.mark.parametrize("pkg_name", AUDITED_PACKAGES)
def test_new_symbols_covered(pkg_name: str) -> None:
    """Every public symbol in submodules is either in __all__ or _EXCLUDED."""
    mod = importlib.import_module(f"fantasy_baseball_manager.{pkg_name}")
    all_set = set(mod.__all__)
    excluded = _EXCLUDED.get(pkg_name, set())

    uncovered: list[str] = []
    for submod_key, filepath in _submodule_files(pkg_name):
        for name in _public_names_from_file(filepath):
            if name in _GLOBALLY_IGNORED_NAMES:
                continue
            if name in all_set:
                continue
            if name in excluded:
                continue
            uncovered.append(f"{submod_key}.{name}")

    assert not uncovered, f"Public symbols in {pkg_name}/ not in __all__ or _EXCLUDED:\n" + "\n".join(
        f"  - {s}" for s in uncovered
    )


# ---------------------------------------------------------------------------
# Test 3: Bypass detection — no submodule imports for __all__ symbols
# ---------------------------------------------------------------------------


def _collect_bypass_violations() -> list[str]:
    """Find cross-package submodule imports that should use package-level imports."""
    # Load __all__ sets
    all_sets: dict[str, set[str]] = {}
    for pkg_name in AUDITED_PACKAGES:
        mod = importlib.import_module(f"fantasy_baseball_manager.{pkg_name}")
        all_sets[pkg_name] = set(mod.__all__)

    violations: list[str] = []
    prefix = "fantasy_baseball_manager."

    for pyfile in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in str(pyfile):
            continue
        # Determine which package this file belongs to
        rel = pyfile.relative_to(PKG_ROOT)
        parts = rel.parts
        file_pkg = parts[0] if len(parts) > 1 else None

        source = pyfile.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(pyfile))
        except SyntaxError:
            continue

        rel_str = str(rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod_path = node.module
            if not mod_path.startswith(prefix):
                continue
            rest = mod_path[len(prefix) :]
            dotparts = rest.split(".")
            if len(dotparts) < 2:
                continue  # Already a package-level import
            pkg = dotparts[0]
            if pkg not in all_sets:
                continue
            if file_pkg == pkg:
                continue  # Intra-package import — allowed

            for alias in node.names:
                name = alias.name
                if name in all_sets[pkg]:
                    if (rel_str, name) in _BYPASS_ALLOWLIST:
                        continue
                    violations.append(
                        f"{rel_str}:{node.lineno}: "
                        f"`from {mod_path} import {name}` "
                        f"should be `from fantasy_baseball_manager.{pkg} import {name}`"
                    )

    return violations


def test_no_bypass_imports() -> None:
    """Cross-package consumers must not reach into submodules for __all__ symbols."""
    violations = _collect_bypass_violations()
    assert not violations, "Found cross-package submodule imports that should use package-level imports:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
