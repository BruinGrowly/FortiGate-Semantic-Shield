"""
Simple Test for Cardinal Semantic Axioms
========================================

Validates that the 4 cardinal axioms (LOVE, POWER, WISDOM, JUSTICE) are preserved.
"""

import importlib
import sys
from pathlib import Path

ENGINE_SRC = Path(__file__).resolve().parent / "semantic_substrate_engine" / "Semantic-Substrate-Engine-main" / "src"

def _ensure_module(short_name: str, full_name: str) -> None:
    try:
        module = importlib.import_module(full_name)
    except ImportError:
        if str(ENGINE_SRC) not in sys.path:
            sys.path.append(str(ENGINE_SRC))
        module = importlib.import_module(short_name)
    sys.modules.setdefault(short_name, module)


_ensure_module("cardinal_semantic_axioms", "semantic_substrate_engine.cardinal_semantic_axioms")
_ensure_module("advanced_semantic_mathematics", "semantic_substrate_engine.advanced_semantic_mathematics")

def test_cardinal_preservation() -> None:
    """Validate anchor, axis dominance, business mapping, ICE alignment, and math alignment."""
    from cardinal_semantic_axioms import (
        JEHOVAH_ANCHOR,
        CardinalAxiom,
        BusinessSemanticMapping,
        ICEFramework,
        SemanticVector,
        create_divine_anchor_vector,
    )
    from advanced_semantic_mathematics import create_semantic_vector, compute_semantic_alignment

    anchor = create_divine_anchor_vector()
    assert anchor.to_tuple() == (1.0, 1.0, 1.0, 1.0)
    assert JEHOVAH_ANCHOR == (1.0, 1.0, 1.0, 1.0)

    test_vec = SemanticVector(love=0.9, power=0.3, wisdom=0.4, justice=0.2)
    assert test_vec.to_tuple() == (0.9, 0.3, 0.4, 0.2)
    assert test_vec.dominant_axiom() == CardinalAxiom.LOVE

    business_values = {'integrity': 0.8, 'strength': 0.7, 'wisdom': 0.9, 'justice': 0.6}
    semantic_vec = BusinessSemanticMapping.map_business_to_cardinal(business_values)
    assert semantic_vec.love == 0.8
    assert semantic_vec.power == 0.7
    assert semantic_vec.wisdom == 0.9
    assert semantic_vec.justice == 0.6

    ice_vec = ICEFramework.compute_ice_vector(semantic_vec)
    for axis in ('love', 'power', 'wisdom', 'justice'):
        assert hasattr(ice_vec, axis)

    math_vec = create_semantic_vector(0.8, 0.7, 0.9, 0.6)
    alignment = compute_semantic_alignment(math_vec)
    assert 0.0 <= alignment <= 1.0


def main() -> bool:
    print("=" * 60)
    print("CARDINAL SEMANTIC AXIOMS - SIMPLE VALIDATION")
    print("=" * 60)
    try:
        test_cardinal_preservation()
    except AssertionError as exc:
        print("\n" + "=" * 60)
        print("FAILURE: Cardinal axioms may be compromised!")
        print(str(exc))
        print("=" * 60)
        return False
    else:
        print("\n" + "=" * 60)
        print("SUCCESS: All cardinal axioms preserved!")
        print("LOVE, POWER, WISDOM, JUSTICE remain as the foundation!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
