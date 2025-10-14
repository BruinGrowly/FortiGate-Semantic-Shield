"""
Vendor bridge module that re-exports the project-level enhanced core components.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_external = import_module("enhanced_core_components")

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

for _name in __all__:
    globals()[_name] = getattr(_external, _name)


def __getattr__(name: str) -> Any:
    return getattr(_external, name)
