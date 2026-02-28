"""Architecture tests: domain purity rules.

These tests statically scan domain model files using the ``ast`` module to
enforce that domain classes remain pure data (frozen dataclasses with no
custom methods) and only import from a strict stdlib whitelist plus other
domain modules.
"""

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_DOMAIN_ROOT = _SRC_ROOT / "fantasy_baseball_manager" / "domain"

PACKAGE_NAME = "fantasy_baseball_manager"

# Stdlib modules that domain files are allowed to import.
_ALLOWED_STDLIB = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "math",
        "re",
        "scipy",
        "statistics",
        "typing",
        "warnings",
    }
)

# Dunder methods that dataclass machinery generates or that are commonly
# overridden on pure-data types.  Anything outside this set is a custom method.
_ALLOWED_METHODS = frozenset(
    {
        "__class_getitem__",
        "__eq__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__post_init__",
        "__repr__",
        "__str__",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _domain_py_files() -> list[Path]:
    """Return all ``.py`` files under the domain package, sorted for determinism."""
    return sorted(p for p in _DOMAIN_ROOT.rglob("*.py") if p.name != "__init__.py")


def _parse_file(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def _relative(path: Path) -> str:
    return str(path.relative_to(_DOMAIN_ROOT))


def _has_frozen_true(decorator: ast.expr) -> bool:
    """Return True if the decorator is ``@dataclass(frozen=True)``."""
    if not isinstance(decorator, ast.Call):
        return False
    for kw in decorator.keywords:
        if kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _is_dataclass_decorator(decorator: ast.expr) -> bool:
    """Return True if *decorator* refers to ``dataclass`` (name or attribute)."""
    if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
        return True
    if isinstance(decorator, ast.Call):
        return _is_dataclass_decorator(decorator.func)
    return isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    """Every dataclass in the domain layer must be frozen."""

    def test_all_dataclasses_are_frozen(self) -> None:
        violations: list[str] = []
        for path in _domain_py_files():
            tree = _parse_file(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for dec in node.decorator_list:
                    if _is_dataclass_decorator(dec) and not _has_frozen_true(dec):
                        violations.append(f"  {_relative(path)}:{node.lineno} class {node.name}")
        assert not violations, "Domain dataclasses must be frozen (missing frozen=True):\n" + "\n".join(violations)


class TestNoCustomMethods:
    """Domain classes must not define custom methods beyond the allowed dunder set."""

    def test_no_custom_methods(self) -> None:
        violations: list[str] = []
        for path in _domain_py_files():
            tree = _parse_file(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name not in _ALLOWED_METHODS:
                        violations.append(f"  {_relative(path)}:{item.lineno} {node.name}.{item.name}")
        assert not violations, "Domain classes must not define custom methods:\n" + "\n".join(violations)


class TestDomainImportWhitelist:
    """Domain modules may only import from allowed stdlib modules and other domain modules."""

    def test_only_allowed_imports(self) -> None:
        violations: list[str] = []
        for path in _domain_py_files():
            tree = _parse_file(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in _ALLOWED_STDLIB:
                            violations.append(f"  {_relative(path)}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        # Relative import — always domain-internal.
                        continue
                    if node.module is None:
                        continue
                    top = node.module.split(".")[0]
                    if top == PACKAGE_NAME:
                        # Must be a domain sub-import.
                        parts = node.module.split(".")
                        if len(parts) >= 2 and parts[1] != "domain":
                            violations.append(f"  {_relative(path)}:{node.lineno} from {node.module}")
                    elif top not in _ALLOWED_STDLIB:
                        violations.append(f"  {_relative(path)}:{node.lineno} from {node.module}")
        assert not violations, "Domain modules have disallowed imports:\n" + "\n".join(violations)
