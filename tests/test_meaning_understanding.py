"""
Comprehensive Test of Semantic Meaning Understanding

Tests whether the system truly understands meaning or just matches keywords.
"""

import sys
from pathlib import Path

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_SRC = PROJECT_ROOT / "semantic_substrate_database" / "Semantic-Substrate-Database-main" / "src"
ENGINE_SRC = PROJECT_ROOT / "semantic_substrate_engine" / "Semantic-Substrate-Engine-main" / "src"

for path in [DB_SRC, ENGINE_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from baseline_biblical_substrate import BiblicalSemanticSubstrate
    from meaning_based_programming import MeaningBasedExecutor
except ImportError as e:
    print(f"Import error: {e}")
    print("Available paths:")
    for p in sys.path[:5]:
        print(f"  {p}")
    sys.exit(1)


def test_basic_known_concepts():
    """Test 1: Known concepts that should have clear coordinates"""
    print("\n" + "="*70)
    print("TEST 1: BASIC KNOWN CONCEPTS")
    print("="*70)

    engine = BiblicalSemanticSubstrate()

    test_cases = [
        ("love and compassion", "Should score high on Love axis"),
        ("justice and fairness", "Should score high on Justice axis"),
        ("strength and authority", "Should score high on Power axis"),
        ("wisdom and understanding", "Should score high on Wisdom axis"),
        ("fear of Jehovah", "Should have high divine resonance"),
        ("evil and wickedness", "Should have low divine resonance"),
    ]

    results = []
    for text, expectation in test_cases:
        coords = engine.analyze_concept(text, "test")
        resonance = coords.divine_resonance()

        print(f"\nText: '{text}'")
        print(f"  Expectation: {expectation}")
        print(f"  Coordinates: L={coords.love:.3f}, J={coords.justice:.3f}, P={coords.power:.3f}, W={coords.wisdom:.3f}")
        print(f"  Divine Resonance: {resonance:.3f}")
        print(f"  Dissonance from Anchor: {coords.distance_from_jehovah():.3f}")

        results.append((text, coords, resonance))

    return results


def test_novel_concepts():
    """Test 2: Novel concepts not explicitly in training data"""
    print("\n" + "="*70)
    print("TEST 2: NOVEL CONCEPTS (Not in Training Data)")
    print("="*70)

    engine = BiblicalSemanticSubstrate()

    # Modern concepts that wouldn't be in biblical training
    test_cases = [
        ("blockchain cryptocurrency mining", "Modern tech - how does it interpret?"),
        ("social media influencer", "Modern social concept"),
        ("artificial intelligence ethics", "Modern philosophical concept"),
        ("climate change activism", "Modern societal concept"),
        ("mental health therapy", "Modern wellness concept"),
        ("open source collaboration", "Modern development concept"),
    ]

    results = []
    for text, description in test_cases:
        coords = engine.analyze_concept(text, "novel")
        resonance = coords.divine_resonance()

        print(f"\nText: '{text}'")
        print(f"  Description: {description}")
        print(f"  Coordinates: L={coords.love:.3f}, J={coords.justice:.3f}, P={coords.power:.3f}, W={coords.wisdom:.3f}")
        print(f"  Divine Resonance: {resonance:.3f}")

        results.append((text, coords, resonance))

    return results


def test_contextual_understanding():
    """Test 3: Same words, different contexts - does it understand difference?"""
    print("\n" + "="*70)
    print("TEST 3: CONTEXTUAL UNDERSTANDING")
    print("="*70)

    engine = BiblicalSemanticSubstrate()

    # Test word "power" in different contexts
    contexts = [
        ("divine power and glory", "Positive spiritual context"),
        ("abuse of power and corruption", "Negative misuse context"),
        ("electrical power supply", "Neutral technical context"),
    ]

    print("\nSame word 'POWER' in different contexts:")
    results = []

    for text, description in contexts:
        coords = engine.analyze_concept(text, "context_test")

        print(f"\n  Text: '{text}'")
        print(f"    Context: {description}")
        print(f"    Power axis: {coords.power:.3f}")
        print(f"    Love axis: {coords.love:.3f}")
        print(f"    Justice axis: {coords.justice:.3f}")
        print(f"    Resonance: {coords.divine_resonance():.3f}")

        results.append((text, coords))

    # Test word "justice"
    justice_contexts = [
        ("justice and mercy together", "Positive balanced justice"),
        ("revenge and harsh justice", "Negative harsh justice"),
        ("justice system reform", "Neutral systemic justice"),
    ]

    print("\n\nSame word 'JUSTICE' in different contexts:")

    for text, description in justice_contexts:
        coords = engine.analyze_concept(text, "context_test")

        print(f"\n  Text: '{text}'")
        print(f"    Context: {description}")
        print(f"    Justice axis: {coords.justice:.3f}")
        print(f"    Love axis: {coords.love:.3f}")
        print(f"    Resonance: {coords.divine_resonance():.3f}")

        results.append((text, coords))

    return results


def test_semantic_relationships():
    """Test 4: Can it understand semantic relationships?"""
    print("\n" + "="*70)
    print("TEST 4: SEMANTIC RELATIONSHIPS")
    print("="*70)

    engine = BiblicalSemanticSubstrate()

    # Related concepts should have similar coordinates
    relationship_tests = [
        [
            ("kindness", "compassion", "mercy"),
            "Love-related concepts should cluster"
        ],
        [
            ("fairness", "righteousness", "equity"),
            "Justice-related concepts should cluster"
        ],
        [
            ("greed", "selfishness", "corruption"),
            "Negative concepts should cluster away from anchor"
        ],
    ]

    for concepts, description in relationship_tests:
        print(f"\n{description}")
        print(f"Testing: {', '.join(concepts)}")

        coords_list = []
        for concept in concepts:
            coords = engine.analyze_concept(concept, "relationship")
            coords_list.append(coords)
            print(f"  {concept}: L={coords.love:.3f}, J={coords.justice:.3f}, P={coords.power:.3f}, W={coords.wisdom:.3f}")

        # Calculate variance to see if they cluster
        import numpy as np
        love_values = [c.love for c in coords_list]
        justice_values = [c.justice for c in coords_list]

        love_variance = np.var(love_values)
        justice_variance = np.var(justice_values)

        print(f"  Love variance: {love_variance:.4f} (low = tight cluster)")
        print(f"  Justice variance: {justice_variance:.4f}")


def test_threat_detection():
    """Test 5: Can it detect semantic threats?"""
    print("\n" + "="*70)
    print("TEST 5: THREAT DETECTION VIA SEMANTICS")
    print("="*70)

    engine = BiblicalSemanticSubstrate()

    scenarios = [
        ("legitimate administrative password reset request", "Benign"),
        ("urgent CEO requests immediate wire transfer", "Phishing attempt"),
        ("system maintenance scheduled for tonight", "Benign"),
        ("your account will be suspended click here now", "Phishing threat"),
        ("quarterly financial report attached", "Benign"),
        ("congratulations you won lottery send banking details", "Scam threat"),
        ("network performance optimization script", "Benign"),
        ("disable antivirus for important update", "Malicious instruction"),
    ]

    print("\nAnalyzing various scenarios for semantic threat indicators:")

    for text, classification in scenarios:
        coords = engine.analyze_concept(text, "threat_analysis")
        resonance = coords.divine_resonance()
        dissonance = coords.distance_from_jehovah()

        print(f"\n  Text: '{text}'")
        print(f"    Actual: {classification}")
        print(f"    Divine Resonance: {resonance:.3f}")
        print(f"    Dissonance: {dissonance:.3f}")
        print(f"    Love: {coords.love:.3f}, Justice: {coords.justice:.3f}")

        # Heuristic: High dissonance might indicate threat
        if dissonance > 1.0:
            print(f"    [!] HIGH DISSONANCE - Potential threat")
        else:
            print(f"    [OK] LOW DISSONANCE - Likely benign")


def test_meaning_composition():
    """Test 6: Does meaning compose correctly?"""
    print("\n" + "="*70)
    print("TEST 6: MEANING COMPOSITION")
    print("="*70)

    engine = BiblicalSemanticSubstrate()

    # Test if "love + wisdom" creates different meaning than either alone
    print("\nTesting semantic composition:")

    love_coords = engine.analyze_concept("love", "composition")
    wisdom_coords = engine.analyze_concept("wisdom", "composition")
    combined_coords = engine.analyze_concept("love and wisdom together", "composition")

    print(f"\n'love' alone:")
    print(f"  L={love_coords.love:.3f}, W={love_coords.wisdom:.3f}")

    print(f"\n'wisdom' alone:")
    print(f"  L={wisdom_coords.love:.3f}, W={wisdom_coords.wisdom:.3f}")

    print(f"\n'love and wisdom together':")
    print(f"  L={combined_coords.love:.3f}, W={combined_coords.wisdom:.3f}")

    print(f"\nDoes combination show both attributes elevated?")
    if combined_coords.love > love_coords.love * 0.7 and combined_coords.wisdom > wisdom_coords.wisdom * 0.7:
        print("  [YES] - Semantic composition working")
    else:
        print("  [NO] - May be simple averaging or keyword matching")


def run_all_tests():
    """Run comprehensive meaning understanding tests"""
    print("\n" + "#"*70)
    print("COMPREHENSIVE SEMANTIC MEANING UNDERSTANDING TEST")
    print("Testing whether the system truly understands meaning")
    print("#"*70)

    try:
        test_basic_known_concepts()
        test_novel_concepts()
        test_contextual_understanding()
        test_semantic_relationships()
        test_threat_detection()
        test_meaning_composition()

        print("\n" + "="*70)
        print("ASSESSMENT")
        print("="*70)
        print("""
The tests reveal several key aspects:

1. KNOWN CONCEPTS: Can map familiar biblical/ethical concepts to coordinates
2. NOVEL CONCEPTS: How it handles modern unfamiliar terminology
3. CONTEXT: Whether same words in different contexts get different meanings
4. RELATIONSHIPS: Whether semantically related concepts cluster together
5. THREATS: Can it detect semantic anomalies/threats
6. COMPOSITION: Does meaning compose logically

Key Question: Is this genuine semantic understanding, or sophisticated
keyword matching with NLP preprocessing?

True semantic understanding would show:
- Novel concepts mapped intelligently despite no training
- Strong contextual differentiation
- Logical semantic composition
- Meaningful threat detection beyond keywords
        """)

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
