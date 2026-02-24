"""Reusable import-scanning helpers for architecture tests."""

import ast
import dataclasses
from pathlib import Path

PACKAGE_NAME = "fantasy_baseball_manager"


@dataclasses.dataclass(frozen=True)
class ImportInfo:
    """A single internal import found in a source file."""

    module: str
    file: Path
    line: int


def collect_internal_imports(directory: Path) -> list[ImportInfo]:
    """Walk ``.py`` files under *directory* and return all ``fantasy_baseball_manager.*`` imports."""
    imports: list[ImportInfo] = []
    for py_file in sorted(directory.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(f"{PACKAGE_NAME}."):
                        imports.append(ImportInfo(module=alias.name, file=py_file, line=node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(f"{PACKAGE_NAME}."):
                    imports.append(ImportInfo(module=node.module, file=py_file, line=node.lineno))
    return imports


def layer_of_import(module: str) -> str:
    """Extract the layer name from a dotted module path.

    Example: ``"fantasy_baseball_manager.domain.player"`` → ``"domain"``
    """
    parts = module.split(".")
    if len(parts) < 2:
        return ""
    return parts[1]


def relative_path(file: Path, package_root: Path) -> str:
    """Return *file* relative to *package_root* as a string for readable messages."""
    return str(file.relative_to(package_root))
