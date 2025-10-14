"""
Simple Test for Cardinal Semantic Axioms
========================================

Validates that the 4 cardinal axioms (LOVE, POWER, WISDOM, JUSTICE) are preserved.
"""

import sys
sys.path.append('semantic_substrate_engine/Semantic-Substrate-Engine-main/src')

def test_cardinal_preservation():
    """Simple test of cardinal axioms preservation"""
    print("Testing Cardinal Semantic Axioms Preservation...")
    
    # Test 1: Anchor point
    try:
        from cardinal_semantic_axioms import JEHOVAH_ANCHOR, create_divine_anchor_vector
        
        anchor = create_divine_anchor_vector()
        assert anchor.to_tuple() == (1.0, 1.0, 1.0, 1.0)
        assert JEHOVAH_ANCHOR == (1.0, 1.0, 1.0, 1.0)
        print("[PASS] Anchor Point (1,1,1,1) preserved")
        
    except Exception as e:
        print(f"[FAIL] Anchor test failed: {e}")
        return False
    
    # Test 2: Cardinal axes
    try:
        from cardinal_semantic_axioms import SemanticVector, CardinalAxiom
        
        # Test vector with clear dominance
        test_vec = SemanticVector(love=0.9, power=0.3, wisdom=0.4, justice=0.2)
        coords = test_vec.to_tuple()
        dominant = test_vec.dominant_axiom()
        
        assert coords == (0.9, 0.3, 0.4, 0.2), "Coordinates must preserve order"
        assert dominant == CardinalAxiom.LOVE, "LOVE should be dominant"
        print("[PASS] Cardinal axes order and dominance working")
        
    except Exception as e:
        print(f"[FAIL] Cardinal axes test failed: {e}")
        return False
    
    # Test 3: Business mapping
    try:
        from cardinal_semantic_axioms import BusinessSemanticMapping
        
        business_values = {'integrity': 0.8, 'strength': 0.7, 'wisdom': 0.9, 'justice': 0.6}
        semantic_vec = BusinessSemanticMapping.map_business_to_cardinal(business_values)
        
        assert semantic_vec.love == 0.8, "Integrity maps to LOVE"
        assert semantic_vec.power == 0.7, "Strength maps to POWER"
        assert semantic_vec.wisdom == 0.9, "Wisdom maps to WISDOM"
        assert semantic_vec.justice == 0.6, "Justice maps to JUSTICE"
        print("[PASS] Business mapping to cardinal axioms working")
        
    except Exception as e:
        print(f"[FAIL] Business mapping test failed: {e}")
        return False
    
    # Test 4: ICE framework
    try:
        from cardinal_semantic_axioms import ICEFramework
        
        semantic_vec = SemanticVector(love=0.8, power=0.7, wisdom=0.9, justice=0.6)
        ice_vec = ICEFramework.compute_ice_vector(semantic_vec)
        
        # ICE should preserve cardinal nature
        assert hasattr(ice_vec, 'love'), "ICE vector must have LOVE"
        assert hasattr(ice_vec, 'power'), "ICE vector must have POWER"
        assert hasattr(ice_vec, 'wisdom'), "ICE vector must have WISDOM"
        assert hasattr(ice_vec, 'justice'), "ICE vector must have JUSTICE"
        print("[PASS] ICE framework preserves cardinal structure")
        
    except Exception as e:
        print(f"[FAIL] ICE framework test failed: {e}")
        return False
    
    # Test 5: Mathematical alignment
    try:
        from advanced_semantic_mathematics import create_semantic_vector, compute_semantic_alignment
        
        math_vec = create_semantic_vector(0.8, 0.7, 0.9, 0.6)
        alignment = compute_semantic_alignment(math_vec)
        
        assert 0.0 <= alignment <= 1.0, "Alignment must be valid"
        print(f"[PASS] Mathematical alignment working: {alignment:.3f}")
        
    except Exception as e:
        print(f"[FAIL] Mathematical alignment test failed: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("CARDINAL SEMANTIC AXIOMS - SIMPLE VALIDATION")
    print("=" * 60)
    
    if test_cardinal_preservation():
        print("\n" + "=" * 60)
        print("SUCCESS: All cardinal axioms preserved!")
        print("LOVE, POWER, WISDOM, JUSTICE remain as the foundation!")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("FAILURE: Cardinal axioms may be compromised!")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)