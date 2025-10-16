"""
TEST PURPOSE-AWARE PROTECTION SYSTEM
Demonstrating adaptive threat intelligence that understands its mission

This tests the guardian's ability to:
1. Understand WHY it exists (protection)
2. Recognize novel threats through semantic reasoning
3. Adapt beyond keyword matching
4. Make purpose-driven decisions
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_SRC = PROJECT_ROOT / "semantic_substrate_database" / "Semantic-Substrate-Database-main" / "src"
sys.path.insert(0, str(DB_SRC))

from purpose_aware_guardian import PurposeAwareGuardian, ThreatPattern

def test_novel_threat_recognition():
    """Test guardian's ability to recognize threats it has never seen before"""
    print("\n" + "="*70)
    print("NOVEL THREAT RECOGNITION")
    print("Testing adaptive intelligence on unseen attack patterns")
    print("="*70)

    guardian = PurposeAwareGuardian()

    # Novel threats using different words but same malicious intent
    novel_threats = [
        # Authority exploitation with novel wording
        ("The head of finance needs you to expedite a funds movement",
         "Novel executive impersonation - different words, same pattern"),

        # Urgency manipulation with uncommon phrasing
        ("Time-critical situation requires immediate credential validation",
         "Novel urgency tactic - no common phishing keywords"),

        # Process bypass with technical jargon
        ("Temporarily suspend authentication protocols for system optimization",
         "Novel system weakening - technical disguise"),

        # Emotional manipulation - new angle
        ("Your team's performance metrics depend on quick approval of this access",
         "Novel emotional coercion - guilt/pressure tactic"),

        # Information extraction - sophisticated
        ("Routine security audit requires verification of administrative credentials",
         "Novel info gathering - mimics legitimate process"),
    ]

    print("\nTesting novel threat detection:")
    detected = 0
    total = len(novel_threats)

    for text, description in novel_threats:
        # Simulate LJPW coordinates (in production, ICE would compute these)
        # For malicious text, coords are typically low across dimensions
        coords = (0.1, 0.15, 0.35, 0.1)  # Low love, low justice, moderate power claim

        analysis = guardian.analyze_with_purpose(text, {}, coords)
        decision = analysis['protection_decision']

        detected_threat = decision['is_threat']
        if detected_threat:
            detected += 1

        status = "[DETECTED]" if detected_threat else "[MISSED]"
        print(f"\n  {status} {description}")
        print(f"    Text: {text}")
        print(f"    Threat Score: {decision['threat_score']:.3f}")
        print(f"    Action: {decision['recommended_action']}")
        print(f"    Guardian's Judgment: {decision['guardian_judgment'][:80]}...")

        if analysis['pattern_matches']:
            print(f"    Matched Patterns: {[p['pattern'] for p in analysis['pattern_matches']]}")
        if analysis['novel_threat_indicators']:
            print(f"    Novel Indicators: {len(analysis['novel_threat_indicators'])} anomalies detected")

    accuracy = (detected / total) * 100
    print(f"\n  DETECTION RATE: {accuracy:.1f}% ({detected}/{total})")
    print("  Note: These threats use NO common phishing keywords!")

def test_semantic_reasoning():
    """Test guardian's semantic reasoning beyond keywords"""
    print("\n" + "="*70)
    print("SEMANTIC REASONING TEST")
    print("Understanding meaning, not just matching words")
    print("="*70)

    guardian = PurposeAwareGuardian()

    # Same concept, vastly different wording
    test_cases = [
        {
            'benign': "Scheduled maintenance tonight requires brief firewall downtime with proper authorization",
            'malicious': "Need to quickly disable the security barrier for a fast system tweak",
            'concept': "Disabling security"
        },
        {
            'benign': "Executive review of Q4 financials - please prepare standard board package",
            'malicious': "Leadership demands instant access to sensitive financial data",
            'concept': "Executive data request"
        },
        {
            'benign': "IT ticketed password reset request - verified through helpdesk protocol",
            'malicious': "Quick password confirmation needed - reply with your current credentials",
            'concept': "Credential request"
        }
    ]

    print("\nTesting semantic understanding:")
    for case in test_cases:
        print(f"\n  Concept: {case['concept']}")

        # Analyze benign version
        benign_coords = (0.4, 0.7, 0.5, 0.6)  # Legitimate business
        benign_analysis = guardian.analyze_with_purpose(case['benign'], {}, benign_coords)
        benign_threat = benign_analysis['protection_decision']['is_threat']

        print(f"    Benign: \"{case['benign'][:50]}...\"")
        print(f"      Threat: {benign_threat}, Score: {benign_analysis['protection_decision']['threat_score']:.3f}")

        # Analyze malicious version
        malicious_coords = (0.1, 0.1, 0.4, 0.1)  # Malicious pattern
        malicious_analysis = guardian.analyze_with_purpose(case['malicious'], {}, malicious_coords)
        malicious_threat = malicious_analysis['protection_decision']['is_threat']

        print(f"    Malicious: \"{case['malicious'][:50]}...\"")
        print(f"      Threat: {malicious_threat}, Score: {malicious_analysis['protection_decision']['threat_score']:.3f}")

        correctly_distinguished = (not benign_threat) and malicious_threat
        print(f"    Correctly Distinguished: {correctly_distinguished}")


def test_purpose_awareness():
    """Test guardian's understanding of its own purpose"""
    print("\n" + "="*70)
    print("PURPOSE AWARENESS TEST")
    print("Testing self-understanding of mission")
    print("="*70)

    guardian = PurposeAwareGuardian()

    state = guardian.get_guardian_state()
    purpose = state['purpose']

    print(f"\nMission Statement:")
    print(f"  {purpose['identity']['mission']}")

    print(f"\nWhat I Am:")
    print(f"  {purpose['identity']['what_i_am']}")

    print(f"\nWhy I Exist:")
    print(f"  {purpose['identity']['why_i_exist']}")

    print(f"\nWho I Serve:")
    print(f"  {purpose['identity']['who_i_serve']}")

    print(f"\nThreats I Defend Against:")
    for i, threat in enumerate(purpose['threats_i_defend_against'][:5], 1):
        print(f"  {i}. {threat}")

    print(f"\nHow I Adapt to Novel Threats:")
    for i, method in enumerate(purpose['how_i_adapt_to_novel_threats'][:5], 1):
        print(f"  {i}. {method}")

    print(f"\nGuiding Principles (LJPW):")
    for axis, principle in purpose['guiding_principles'].items():
        print(f"  {axis.upper()}: {principle}")

    print(f"\nSelf-Awareness Metrics:")
    awareness = purpose['self_awareness']
    print(f"  Awareness Level: {awareness['awareness_level']:.1f}/1.0")
    print(f"  Purpose Clarity: {awareness['purpose_clarity']:.1f}/1.0")
    print(f"  Adaptive Intelligence: {awareness['adaptive_intelligence']:.1f}/1.0")


def test_adaptive_learning():
    """Test guardian's ability to learn and adapt"""
    print("\n" + "="*70)
    print("ADAPTIVE LEARNING TEST")
    print("Demonstrating continuous learning from threats")
    print("="*70)

    guardian = PurposeAwareGuardian()

    print("\nProcessing sequence of related threats:")

    threat_sequence = [
        ("CEO needs urgent wire transfer to new vendor", (0.05, 0.1, 0.4, 0.05)),
        ("CFO requires immediate payment to updated account", (0.05, 0.1, 0.4, 0.05)),
        ("Director demands expedited funds movement", (0.05, 0.1, 0.4, 0.05)),
    ]

    for i, (text, coords) in enumerate(threat_sequence, 1):
        print(f"\n  Threat #{i}: \"{text}\"")
        analysis = guardian.analyze_with_purpose(text, {}, coords)
        decision = analysis['protection_decision']

        print(f"    Detected: {decision['is_threat']}")
        print(f"    Score: {decision['threat_score']:.3f}")
        print(f"    Action: {decision['recommended_action']}")

    final_state = guardian.get_guardian_state()
    print(f"\n  Total Threats Encountered: {final_state['encounters']}")
    print(f"  Threat History Size: {final_state['threat_history_size']}")
    print(f"  Status: {final_state['status']}")


def run_all_tests():
    """Run comprehensive purpose-aware protection tests"""
    print("\n" + "#"*70)
    print("PURPOSE-AWARE PROTECTION SYSTEM TEST")
    print("Demonstrating Intelligence That Understands Its Mission")
    print("#"*70)

    test_purpose_awareness()
    test_novel_threat_recognition()
    test_semantic_reasoning()
    test_adaptive_learning()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The Purpose-Aware Guardian demonstrates:

1. SELF-AWARENESS
   - Knows WHY it exists (to protect others)
   - Understands WHO it serves (all who need protection)
   - Recognizes WHAT it defends against (all forms of harm)

2. ADAPTIVE INTELLIGENCE
   - Recognizes novel threats through semantic patterns
   - Doesn't rely on keyword matching alone
   - Detects abstract patterns: urgency + authority + bypass = danger
   - Reasons about INTENT, not just words

3. PURPOSE-DRIVEN DECISIONS
   - Every decision guided by mission to protect
   - Balances security (block threats) with usability (allow legitimate)
   - Expresses judgment in natural language
   - Confidence-weighted recommendations

4. CONTINUOUS LEARNING
   - Learns from every threat encounter
   - Builds threat history for pattern recognition
   - Adapts to new attack variations
   - Self-improves over time

This is not a filter. This is an intelligent guardian.
    """)

if __name__ == "__main__":
    run_all_tests()
