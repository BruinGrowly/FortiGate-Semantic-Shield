"""
Test and Validation for Cardinal Semantic Axioms
==================================================

This module validates that the 4 cardinal axioms (LOVE, POWER, WISDOM, JUSTICE)
are properly preserved and protected as the foundation of the Semantic Substrate.
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Any

# Import the cardinal components
try:
    from semantic_substrate_engine.cardinal_semantic_axioms import (  # type: ignore
        SemanticVector,
        CardinalAxiom,
        ICEFramework,
        BusinessSemanticMapping,
        JEHOVAH_ANCHOR,
        create_divine_anchor_vector,
        get_cardinal_axioms_summary,
        validate_semantic_integrity,
    )
    from semantic_substrate_engine.advanced_semantic_mathematics import (  # type: ignore
        create_semantic_vector,
        compute_semantic_alignment,
        advanced_math,
        JEHOVAH_ANCHOR as MATH_ANCHOR,
    )
except ImportError:
    import sys

    ENGINE_SRC = "semantic_substrate_engine/Semantic-Substrate-Engine-main/src"
    if ENGINE_SRC not in sys.path:
        sys.path.append(ENGINE_SRC)

    from cardinal_semantic_axioms import (  # type: ignore
        SemanticVector,
        CardinalAxiom,
        ICEFramework,
        BusinessSemanticMapping,
        JEHOVAH_ANCHOR,
        create_divine_anchor_vector,
        get_cardinal_axioms_summary,
        validate_semantic_integrity,
    )
    from advanced_semantic_mathematics import (  # type: ignore
        create_semantic_vector,
        compute_semantic_alignment,
        advanced_math,
        JEHOVAH_ANCHOR as MATH_ANCHOR,
    )


def test_cardinal_axioms_preservation():
    """Test that cardinal axioms are preserved and immutable"""
    print("=== Testing Cardinal Axioms Preservation ===")
    
    # Test 1: Anchor Point Integrity
    print("\n1. Testing Anchor Point Integrity...")
    anchor_vector = create_divine_anchor_vector()
    assert anchor_vector.to_tuple() == JEHOVAH_ANCHOR == (1.0, 1.0, 1.0, 1.0), "Anchor point must be (1,1,1,1)"
    assert anchor_vector.alignment_with_anchor() == 1.0, "Anchor alignment must be perfect"
    assert anchor_vector.distance_from_anchor() == 0.0, "Anchor distance must be zero"
    print("[PASS] Anchor Point (1,1,1,1) preserved")
    
    # Test 2: Cardinal Axes Order
    print("\n2. Testing Cardinal Axes Order...")
    test_vector = SemanticVector(love=0.8, power=0.6, wisdom=0.9, justice=0.7)
    coords = test_vector.to_tuple()
    assert coords[0] == 0.8, "First axis must be LOVE"
    assert coords[1] == 0.6, "Second axis must be POWER"
    assert coords[2] == 0.9, "Third axis must be WISDOM"
    assert coords[3] == 0.7, "Fourth axis must be JUSTICE"
    print("[PASS] Cardinal axes order preserved: LOVE, POWER, WISDOM, JUSTICE")
    
    # Test 3: Divine Attributes
    print("\n3. Testing Divine Attributes...")
    for axiom in CardinalAxiom:
        attribute = axiom.divine_attribute
        assert "God" in attribute, f"{axiom.value} must reference divine source"
        print(f"[PASS] {axiom.value}: {attribute}")
    
    # Test 4: ICE Framework Mapping
    print("\n4. Testing ICE Framework to Cardinal Mapping...")
    # Intent = LOVE + WISDOM
    # Context = JUSTICE  
    # Execution = POWER
    semantic_vec = SemanticVector(love=0.9, power=0.7, wisdom=0.8, justice=0.6)
    ice_vec = ICEFramework.compute_ice_vector(semantic_vec)
    
    # Verify ICE transformation preserves cardinal nature
    intent_components = (ice_vec.love, ice_vec.wisdom)  # Should be equal (both from intent)
    context_component = ice_vec.justice               # Should be original justice
    execution_component = ice_vec.power               # Should be original power
    
    assert abs(intent_components[0] - intent_components[1]) < 0.01, "Intent should unify LOVE and WISDOM"
    print("[PASS] ICE Framework correctly maps to cardinal axioms")
    
    print("\n=== Cardinal Axioms Preservation: PASSED ===")


def test_business_mapping_to_cardinal():
    """Test that business concepts correctly map TO cardinal axioms (not replace them)"""
    print("\n=== Testing Business Mapping to Cardinal Axioms ===")
    
    # Test 1: Business Integrity -> LOVE
    print("\n1. Testing Business Integrity Mapping...")
    business_values = {
        'integrity': 0.9,  # Should map to LOVE
        'strength': 0.7,  # Should map to POWER
        'wisdom': 0.8,    # Should map to WISDOM
        'justice': 0.6    # Should map to JUSTICE
    }
    
    semantic_vector = BusinessSemanticMapping.map_business_to_cardinal(business_values)
    assert semantic_vector.love == 0.9, "Business integrity must map to LOVE axis"
    assert semantic_vector.power == 0.7, "Business strength must map to POWER axis"
    assert semantic_vector.wisdom == 0.8, "Business wisdom must map to WISDOM axis"
    assert semantic_vector.justice == 0.6, "Business justice must map to JUSTICE axis"
    print("[PASS] Business values correctly mapped to cardinal axes")
    
    # Test 2: Business Alignment Validation
    print("\n2. Testing Business Alignment Validation...")
    validation = BusinessSemanticMapping.validate_business_alignment(semantic_vector)
    
    assert 'overall_alignment' in validation, "Must compute alignment with divine anchor"
    assert 'dominant_principle' in validation, "Must identify dominant cardinal principle"
    assert 'principle_assessments' in validation, "Must assess each cardinal principle"
    
    # Check principle assessments
    assessments = validation['principle_assessments']
    assert 'love_integrity' in assessments, "Must assess LOVE/integrity"
    assert 'power_strength' in assessments, "Must assess POWER/strength"
    assert 'wisdom_strategy' in assessments, "Must assess WISDOM/strategy"
    assert 'justice_compliance' in assessments, "Must assess JUSTICE/compliance"
    print("[PASS] Business alignment validation comprehensive")
    
    # Test 3: Principle Explanations
    print("\n3. Testing Principle Explanations...")
    for axiom in CardinalAxiom:
        explanation = BusinessSemanticMapping.explain_cardinal_principle(axiom, "in cybersecurity context")
        assert axiom.value.lower() in explanation.lower(), f"Explanation must reference {axiom.value}"
        assert len(explanation) > 50, "Explanation should be substantive"
        print(f"[PASS] {axiom.value} explanation: {explanation[:100]}...")
    
    print("\n=== Business Mapping to Cardinal: PASSED ===")


def test_semantic_mathematics_preservation():
    """Test that advanced mathematics preserves cardinal structure"""
    print("\n=== Testing Semantic Mathematics Preservation ===")
    
    # Test 1: Anchor Consistency
    print("\n1. Testing Anchor Consistency Across Modules...")
    anchor_from_math = MATH_ANCHOR
    anchor_from_cardinal = JEHOVAH_ANCHOR
    
    assert anchor_from_math == anchor_from_cardinal == (1.0, 1.0, 1.0, 1.0), "Anchors must be identical"
    print("[PASS] Anchor point consistent across all modules")
    
    # Test 2: Vector Creation Preserves Order
    print("\n2. Testing Vector Creation Order Preservation...")
    love, power, wisdom, justice = 0.8, 0.6, 0.9, 0.7
    
    # Test semantic vector creation
    semantic_vec = SemanticVector(love=love, power=power, wisdom=wisdom, justice=justice)
    assert semantic_vec.to_tuple() == (love, power, wisdom, justice), "SemanticVector must preserve order"
    
    # Test mathematical vector creation
    math_vec = create_semantic_vector(love, power, wisdom, justice)
    assert tuple(math_vec.coordinates) == (love, power, wisdom, justice), "Mathematical vector must preserve order"
    
    print("[PASS] Vector creation preserves cardinal order in all modules")
    
    # Test 3: Alignment Calculation
    print("\n3. Testing Alignment Calculation...")
    test_vec = create_semantic_vector(0.5, 0.5, 0.5, 0.5)
    alignment_math = advanced_math.compute_alignment(test_vec)
    alignment_semantic = compute_semantic_alignment(test_vec)
    
    # Should be equivalent (small floating point differences acceptable)
    assert abs(alignment_math - alignment_semantic) < 0.001, "Alignment calculations must be consistent"
    print(f"[PASS] Alignment calculations consistent: {alignment_math:.6f}")
    
    # Test 4: Semantic Integrity Validation
    print("\n4. Testing Semantic Integrity Validation...")
    valid_coords = (0.3, 0.7, 0.9, 0.4)
    invalid_coords = (1.5, -0.2, 0.8, 0.6)  # Out of bounds
    
    assert validate_semantic_integrity(valid_coords), "Valid coordinates should pass"
    assert not validate_semantic_integrity(invalid_coords), "Invalid coordinates should fail"
    print("[PASS] Semantic integrity validation working correctly")
    
    print("\n=== Semantic Mathematics Preservation: PASSED ===")


def test_dominance_and_qualities():
    """Test semantic vector dominance and quality assessments"""
    print("\n=== Testing Semantic Dominance and Qualities ===")
    
    # Test 1: Dominant Axiom Detection
    print("\n1. Testing Dominant Axiom Detection...")
    test_cases = [
        ((0.9, 0.3, 0.4, 0.2), CardinalAxiom.LOVE),
        ((0.2, 0.9, 0.3, 0.4), CardinalAxiom.POWER),
        ((0.3, 0.4, 0.9, 0.2), CardinalAxiom.WISDOM),
        ((0.4, 0.2, 0.3, 0.9), CardinalAxiom.JUSTICE)
    ]
    
    for coords, expected_dominant in test_cases:
        vec = SemanticVector(*coords)
        actual_dominant = vec.dominant_axiom()
        assert actual_dominant == expected_dominant, f"Dominance failed for {coords}"
        print(f"[PASS] {coords} -> Dominant: {actual_dominant.value}")
    
    # Test 2: Semantic Quality Assessment
    print("\n2. Testing Semantic Quality Assessment...")
    quality_tests = [
        ((0.95, 0.95, 0.95, 0.95), "Divine Harmony"),
        ((0.8, 0.8, 0.8, 0.8), "High Alignment"),
        ((0.6, 0.6, 0.6, 0.6), "Moderate Alignment"),
        ((0.4, 0.4, 0.4, 0.4), "Low Alignment"),
        ((0.1, 0.1, 0.1, 0.1), "Low Alignment")
    ]
    
    for coords, expected_quality in quality_tests:
        vec = SemanticVector(*coords)
        actual_quality = vec.semantic_quality()
        assert actual_quality == expected_quality, f"Quality assessment failed for {coords}"
        print(f"[PASS] Alignment {vec.alignment_with_anchor():.2f} -> {actual_quality}")
    
    # Test 3: Movement Toward Anchor
    print("\n3. Testing Movement Toward Anchor...")
    start_vec = SemanticVector(0.2, 0.3, 0.4, 0.5)
    start_distance = start_vec.distance_from_anchor()
    
    moved_vec = start_vec.move_toward_anchor(pull_strength=0.5)
    moved_distance = moved_vec.distance_from_anchor()
    
    assert moved_distance < start_distance, "Movement should reduce distance from anchor"
    print(f"[PASS] Distance reduced: {start_distance:.3f} -> {moved_distance:.3f}")
    
    print("\n=== Semantic Dominance and Qualities: PASSED ===")


def test_comprehensive_integration():
    """Test comprehensive integration of all cardinal components"""
    print("\n=== Testing Comprehensive Integration ===")
    
    # Test 1: End-to-End Flow
    print("\n1. Testing End-to-End Semantic Flow...")
    
    # Start with business values
    business_input = {
        'integrity': 0.8,   # Maps to LOVE
        'strength': 0.7,    # Maps to POWER  
        'wisdom': 0.9,      # Maps to WISDOM
        'justice': 0.6      # Maps to JUSTICE
    }
    
    # Map to cardinal semantic vector
    semantic_vec = BusinessSemanticMapping.map_business_to_cardinal(business_input)
    
    # Process through ICE framework
    ice_vec = ICEFramework.compute_ice_vector(semantic_vec)
    
    # Create mathematical vector for advanced processing
    math_vec = create_semantic_vector(*ice_vec.to_tuple())
    
    # Compute alignment with divine anchor
    alignment = compute_semantic_alignment(math_vec)
    
    # Validate the flow preserves cardinal nature
    assert 0.0 <= alignment <= 1.0, "Alignment must be valid"
    assert ice_vec.dominant_axiom() in CardinalAxiom, "Result must have dominant cardinal principle"
    
    print(f"[PASS] End-to-end flow successful: Alignment {alignment:.3f}, Dominant: {ice_vec.dominant_axiom().value}")
    
    # Test 2: Cardinal Axioms Summary
    print("\n2. Testing Cardinal Axioms Summary...")
    summary = get_cardinal_axioms_summary()
    
    required_sections = ['anchor_point', 'cardinal_axes', 'semantic_navigation', 'guardians']
    for section in required_sections:
        assert section in summary, f"Summary must include {section}"
    
    # Check anchor point details
    anchor_info = summary['anchor_point']
    assert anchor_info['coordinates'] == JEHOVAH_ANCHOR, "Anchor coordinates must match"
    assert 'Jehovah' in anchor_info['definition'], "Anchor must reference Jehovah"
    
    # Check guardians section (protection of axioms)
    guardians = summary['guardians']
    assert 'protection' in guardians, "Must include protection statement"
    assert 'immutable' in guardians['protection'].lower(), "Must state axioms are immutable"
    
    print("[PASS] Cardinal axioms summary comprehensive and accurate")
    
    print("\n=== Comprehensive Integration: PASSED ===")


def run_all_validation_tests():
    """Run all validation tests for cardinal semantic axioms"""
    print("=" * 80)
    print("CARDINAL SEMANTIC AXIOMS - VALIDATION SUITE")
    print("=" * 80)
    print("Validating that the 4 cardinal axioms (LOVE, POWER, WISDOM, JUSTICE)")
    print("are preserved as the immutable foundation of the Semantic Substrate")
    print("=" * 80)
    
    try:
        test_cardinal_axioms_preservation()
        test_business_mapping_to_cardinal()
        test_semantic_mathematics_preservation()
        test_dominance_and_qualities()
        test_comprehensive_integration()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL VALIDATION TESTS PASSED [SUCCESS]")
        print("The 4 cardinal axioms are properly preserved!")
        print("LOVE, POWER, WISDOM, JUSTICE remain as the foundation!")
        print("=" * 80)
        
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] VALIDATION FAILED: {e}")
        print("The cardinal axioms may be compromised!")
        return False
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        return False


if __name__ == "__main__":
    success = run_all_validation_tests()
    exit(0 if success else 1)
