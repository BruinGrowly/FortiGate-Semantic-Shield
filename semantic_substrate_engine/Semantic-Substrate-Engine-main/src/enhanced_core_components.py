"""
Vendor bridge module that re-exports the enhanced core components bundled with
FortiGate Semantic Shield.

The upstream Semantic Substrate Engine expects a top-level
``enhanced_core_components`` module to be available.  Inside this repository
those implementations live under the Semantic Substrate Database project.  To
keep the original imports working we lazily load that implementation from the
vendored sources and expose its public surface here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "SemanticUnit",
    "SacredNumber",
    "BridgeFunction",
    "UniversalAnchor",
    "UniversalAnchorPoint",
    "SevenUniversalPrinciples",
    "UniversalPrinciple",
    "ContextualResonance",
]


def _load_external_module() -> ModuleType:
    """
    Load the enhanced core components implementation from the vendored database.

    The Semantic-Substrate-Database repository carries the canonical reference
    implementation.  We load it dynamically to avoid maintaining two divergent
    copies while still keeping legacy absolute imports functional.
    """

    repo_root = Path(__file__).resolve().parents[3]
    fallback_path = (
        repo_root
        / "semantic_substrate_database"
        / "Semantic-Substrate-Database-main"
        / "src"
        / "enhanced_core_components.py"
    )

    if not fallback_path.exists():
        raise ImportError(
            "Enhanced core components not found. Expected file at "
            f"{fallback_path}"
        )

    module_name = "semantic_substrate_engine._enhanced_core_components_impl"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing

    spec = importlib.util.spec_from_file_location(module_name, fallback_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {fallback_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


_external = sys.modules.get("enhanced_core_components")
if not isinstance(_external, ModuleType) or getattr(_external, "__file__", None) == __file__:
    _external = _load_external_module()
    sys.modules.setdefault("enhanced_core_components", _external)


def __getattr__(name: str) -> Any:
    return getattr(_external, name)


for _name in __all__:
    globals()[_name] = getattr(_external, _name)
