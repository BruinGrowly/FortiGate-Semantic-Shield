"""
Compatibility layer for the vendored Semantic Substrate Database package.

This exposes the upstream `semantic_substrate_database` module when the
Semantic-Substrate-Database repository is included inside this project.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional


_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _PACKAGE_ROOT / "Semantic-Substrate-Database-main" / "src"
_REPO_ROOT = _PACKAGE_ROOT.parent

__path__ = [str(_PACKAGE_ROOT)]
if _SRC_ROOT.exists():
    __path__.append(str(_SRC_ROOT))

# Ensure the repository root is importable so cross-package imports work.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_internal_module: Optional[ModuleType] = None
__all__: List[str] = []


def _load_internal_module() -> ModuleType:
    """Load the upstream database package module."""
    if not _SRC_ROOT.exists():
        raise FileNotFoundError(f"Semantic Substrate Database src not found at {_SRC_ROOT}")

    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

    spec = importlib.util.spec_from_file_location(
        "semantic_substrate_database._internal",
        _SRC_ROOT / "__init__.py",
        submodule_search_locations=[str(_SRC_ROOT)],
    )

    if spec is None or spec.loader is None:
        raise ImportError("Unable to load internal Semantic Substrate Database module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


try:
    _internal_module = _load_internal_module()

    __all__ = list(getattr(_internal_module, "__all__", []))
    for name in __all__:
        globals()[name] = getattr(_internal_module, name)

    __version__ = getattr(_internal_module, "__version__", "0.0.0")
    __author__ = getattr(_internal_module, "__author__", "")
    __license__ = getattr(_internal_module, "__license__", "")
    __description__ = getattr(_internal_module, "__description__", "")

except Exception as exc:  # pragma: no cover
    _internal_module = None
    __all__ = []
    print(f"[SEMANTIC DB] Bridge error: {exc}")

    def __getattr__(name: str):
        raise ImportError(
            "Semantic Substrate Database resources are unavailable. "
            "Ensure the Semantic-Substrate-Database repository is present under "
            "'semantic_substrate_database/Semantic-Substrate-Database-main'."
        ) from exc
