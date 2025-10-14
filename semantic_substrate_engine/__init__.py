"""
Lightweight wrapper around the vendored Semantic Substrate Engine package.

We execute the upstream `__init__.py` in this module's namespace so that
relative imports (e.g. `.ice_semantic_substrate_engine`) continue to work
exactly as expected, while exposing the full public surface to the rest of
the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import sys


_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _PACKAGE_ROOT / "Semantic-Substrate-Engine-main" / "src"

# Ensure this package behaves like the upstream one when executing relative imports.
__path__ = [str(_PACKAGE_ROOT)]
if _SRC_ROOT.exists():
    __path__.append(str(_SRC_ROOT))
else:
    raise ImportError(
        "Semantic Substrate Engine sources not found. "
        "Expected directory: semantic_substrate_engine/Semantic-Substrate-Engine-main/src"
    )

# Keep repository root importable so sibling packages (e.g. semantic_substrate_database)
# can locate this shim without manual path juggling.
_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Execute upstream initializer in the current module's namespace.
_upstream_init = _SRC_ROOT / "__init__.py"
_globals: Dict[str, Any] = globals()
exec(
    compile(_upstream_init.read_text(encoding="utf-8"), str(_upstream_init), "exec"),
    _globals,
)

# Some downstream code expects a SacredNumber implementation exposed even when the
# upstream package omits it. Provide a graceful fallback.
if "SacredNumber" not in _globals:
    try:
        from enhanced_core_components import SacredNumber as _ProjectSacredNumber  # type: ignore
    except Exception:
        class SacredNumber:  # type: ignore
            """Minimal SacredNumber shim used when advanced components are unavailable."""

            def __init__(self, value, sacred_context: str = "biblical"):
                self.value = value
                self.sacred_context = sacred_context
                self.is_sacred = True
                self.divine_attributes = {
                    "love": 1.0,
                    "power": 1.0,
                    "wisdom": 1.0,
                    "justice": 1.0,
                }
                self.biblical_significance = 1.0
                self.sacred_resonance = 1.0
                self.mystical_properties = {"value": value}

            def __repr__(self) -> str:
                return f"SacredNumber(value={self.value!r}, context={self.sacred_context!r})"
    else:
        SacredNumber = _ProjectSacredNumber  # type: ignore

    _globals["SacredNumber"] = SacredNumber  # type: ignore
    if "__all__" in _globals:
        _globals["__all__"].append("SacredNumber")
