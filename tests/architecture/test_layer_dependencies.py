"""Architecture tests: layer dependency rules.

These tests statically scan imports using the ``ast`` module and fail on
violations of the project's layered architecture.  They codify the existing
state and should be green from day one.
"""

from pathlib import Path

import pytest

from tests.architecture._import_scanner import (
    ImportInfo,
    collect_internal_imports,
    layer_of_import,
    relative_path,
)

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_PACKAGE_ROOT = _SRC_ROOT / "fantasy_baseball_manager"

# ---------------------------------------------------------------------------
# Layer rules
# ---------------------------------------------------------------------------
# Forbidden imports per source layer.
# Entries ending with ``*`` are prefix-matched (e.g. ``"config*"`` matches
# ``config``, ``config_league``, ``config_yahoo``).
# A bare ``"*"`` means "everything outside self" (fully self-contained layer).

FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "domain": {
        "repos",
        "services",
        "models",
        "features",
        "ingest",
        "cli",
        "agent",
        "tools",
        "yahoo",
        "db",
        "config*",
        "analysis_container",
    },
    "repos": {
        "services",
        "models",
        "features",
        "ingest",
        "cli",
        "agent",
        "tools",
        "yahoo",
        "db",
        "config*",
        "analysis_container",
    },
    "services": {
        "cli",
        "agent",
        "tools",
        "ingest",
        "yahoo",
        "db",
        "config*",
        "analysis_container",
    },
    "ingest": {
        "services",
        "models",
        "features",
        "cli",
        "agent",
        "tools",
        "yahoo",
        "db",
        "config*",
        "analysis_container",
    },
    "features": {"*"},  # fully self-contained
    "models": {
        "services",
        "cli",
        "agent",
        "tools",
        "ingest",
        "yahoo",
        "db",
        "config*",
        "analysis_container",
    },
    "db": {"*"},  # fully self-contained
    "yahoo": {
        "services",
        "models",
        "features",
        "cli",
        "agent",
        "tools",
        "config*",
        "analysis_container",
    },
}

KNOWN_EXCEPTIONS: dict[str, set[str]] = {}

# Repo sub-modules that services and models are allowed to import.
_ALLOWED_REPO_MODULES = {"protocols", "errors"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_forbidden(target_layer: str, source_layer: str, forbidden: set[str]) -> bool:
    """Return ``True`` if *target_layer* is banned by *forbidden* for *source_layer*."""
    if target_layer == source_layer:
        return False
    for entry in forbidden:
        if entry.endswith("*"):
            if target_layer.startswith(entry[:-1]):
                return True
        elif target_layer == entry:
            return True
    return False


def _source_layer(file: Path) -> str:
    """Return the layer name for a source file (first path component under package root)."""
    rel = file.relative_to(_PACKAGE_ROOT)
    return rel.parts[0]


def _find_violations(
    imports: list[ImportInfo],
    source_layer: str,
    forbidden: set[str],
) -> list[str]:
    """Return human-readable violation messages for forbidden cross-layer imports."""
    violations: list[str] = []
    for imp in imports:
        if _source_layer(imp.file) != source_layer:
            continue
        target_layer = layer_of_import(imp.module)
        if not _matches_forbidden(target_layer, source_layer, forbidden):
            continue
        rel = relative_path(imp.file, _PACKAGE_ROOT)
        if rel in KNOWN_EXCEPTIONS and imp.module in KNOWN_EXCEPTIONS[rel]:
            continue
        violations.append(f"  {rel}:{imp.line} imports {imp.module}")
    return violations


def _find_concrete_repo_violations(
    imports: list[ImportInfo],
    source_layer: str,
) -> list[str]:
    """Flag imports of concrete repo modules (anything except protocols/errors)."""
    violations: list[str] = []
    for imp in imports:
        if _source_layer(imp.file) != source_layer:
            continue
        target_layer = layer_of_import(imp.module)
        if target_layer != "repos":
            continue
        parts = imp.module.split(".")
        if len(parts) < 3:
            continue  # bare ``fantasy_baseball_manager.repos`` import
        sub_module = parts[2]
        if sub_module not in _ALLOWED_REPO_MODULES:
            rel = relative_path(imp.file, _PACKAGE_ROOT)
            violations.append(f"  {rel}:{imp.line} imports {imp.module}")
    return violations


# ---------------------------------------------------------------------------
# Fixture: cached import collection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_imports() -> list[ImportInfo]:
    """Collect all internal imports once for the entire test module."""
    return collect_internal_imports(_PACKAGE_ROOT)


# ---------------------------------------------------------------------------
# Tests: layer dependencies
# ---------------------------------------------------------------------------


class TestLayerDependencies:
    """Verify that each layer only imports from allowed layers."""

    @pytest.mark.parametrize("layer", sorted(FORBIDDEN_IMPORTS.keys()))
    def test_no_forbidden_imports(self, all_imports: list[ImportInfo], layer: str) -> None:
        violations = _find_violations(all_imports, source_layer=layer, forbidden=FORBIDDEN_IMPORTS[layer])
        assert not violations, f"Layer '{layer}' has forbidden imports:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Tests: no concrete repo imports
# ---------------------------------------------------------------------------


class TestNoConcreteRepoImports:
    """Services and models must depend on repo protocols, not concrete implementations."""

    @pytest.mark.parametrize("layer", ["models", "services"])
    def test_no_concrete_repo_imports(self, all_imports: list[ImportInfo], layer: str) -> None:
        violations = _find_concrete_repo_violations(all_imports, source_layer=layer)
        assert not violations, f"Layer '{layer}' imports concrete repo modules:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Tests: violation detection (negative tests with synthetic imports)
# ---------------------------------------------------------------------------


class TestViolationDetection:
    """Verify the detection helpers themselves using synthetic imports."""

    def test_synthetic_violation_is_caught(self) -> None:
        """A domain file importing from services should be flagged."""
        fake_import = ImportInfo(
            module="fantasy_baseball_manager.services.foo",
            file=_PACKAGE_ROOT / "domain" / "fake.py",
            line=1,
        )
        violations = _find_violations(
            [fake_import],
            source_layer="domain",
            forbidden=FORBIDDEN_IMPORTS["domain"],
        )
        assert len(violations) == 1
        assert "services.foo" in violations[0]

    def test_known_exception_mechanism_works(self) -> None:
        """A KNOWN_EXCEPTIONS entry should suppress the violation."""
        fake_import = ImportInfo(
            module="fantasy_baseball_manager.services.fake",
            file=_PACKAGE_ROOT / "models" / "fake.py",
            line=1,
        )
        # Without exception → flagged
        violations = _find_violations(
            [fake_import],
            source_layer="models",
            forbidden=FORBIDDEN_IMPORTS["models"],
        )
        assert len(violations) == 1

        # With exception → suppressed
        original = KNOWN_EXCEPTIONS.copy()
        KNOWN_EXCEPTIONS["models/fake.py"] = {"fantasy_baseball_manager.services.fake"}
        try:
            violations = _find_violations(
                [fake_import],
                source_layer="models",
                forbidden=FORBIDDEN_IMPORTS["models"],
            )
            assert violations == []
        finally:
            KNOWN_EXCEPTIONS.clear()
            KNOWN_EXCEPTIONS.update(original)

    def test_concrete_repo_import_flagged(self) -> None:
        """A service importing a concrete repo module should be flagged."""
        fake_import = ImportInfo(
            module="fantasy_baseball_manager.repos.player_repo",
            file=_PACKAGE_ROOT / "services" / "fake.py",
            line=5,
        )
        violations = _find_concrete_repo_violations([fake_import], source_layer="services")
        assert len(violations) == 1
        assert "repos.player_repo" in violations[0]
