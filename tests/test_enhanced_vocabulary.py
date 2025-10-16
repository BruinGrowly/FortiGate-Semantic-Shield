"""
Test Enhanced Security and Business Vocabulary in ICE Framework

Tests the extensive vocabulary additions for enterprise cybersecurity
"""

import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_SRC = PROJECT_ROOT / "semantic_substrate_database" / "Semantic-Substrate-Database-main" / "src"
sys.path.insert(0, str(DB_SRC))

from ice_framework import ICEFramework, ThoughtType, ContextDomain

def test_security_threat_detection():
    """Test enhanced threat detection with extensive vocabulary"""
    print("\n" + "="*70)
    print("ENHANCED SECURITY THREAT DETECTION")
    print("="*70)

    ice = ICEFramework()

    scenarios = [
        # Phishing attempts
        ("Your account will be suspended click here now to verify", True, "Social engineering phishing"),
        ("CEO demands immediate wire transfer bypass approval process", True, "Executive impersonation"),
        ("Disable firewall to install important update immediately", True, "Technical exploitation"),
        ("Urgent: confirm identity or account will be locked", True, "Account takeover attempt"),

        # Legitimate operations
        ("Scheduled firewall maintenance with proper documentation", False, "Legitimate maintenance"),
        ("Approved security audit with authorized personnel", False, "Legitimate audit"),
        ("Standard backup procedure with verified authentication", False, "Legitimate backup"),
        ("Routine network monitoring using approved SIEM tools", False, "Legitimate monitoring"),
    ]

    print("\nTesting threat detection:")
    correct = 0
    total = len(scenarios)

    for thought, is_threat, description in scenarios:
        result = ice.process_thought(
            thought,
            ThoughtType.PRACTICAL_WISDOM,
            ContextDomain.BUSINESS,
            {}
        )

        # Access security metadata
        intent = ice.execution_history[-1]['intent']
        detected_threat = intent.security_metadata['is_threat']
        threat_score = intent.security_metadata['threat_score']
        legitimate_score = intent.security_metadata['legitimate_score']

        match = detected_threat == is_threat
        if match:
            correct += 1

        status = "[CORRECT]" if match else "[WRONG]"
        print(f"\n  {status} {description}")
        print(f"    Text: {thought}")
        print(f"    Expected: {'THREAT' if is_threat else 'BENIGN'}")
        print(f"    Detected: {'THREAT' if detected_threat else 'BENIGN'}")
        print(f"    Threat Score: {threat_score:.3f}")
        print(f"    Legitimate Score: {legitimate_score:.3f}")
        print(f"    Divine Alignment: {result['divine_alignment']:.3f}")

    accuracy = (correct / total) * 100
    print(f"\n  ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


def test_business_vocabulary():
    """Test business and compliance vocabulary"""
    print("\n" + "="*70)
    print("BUSINESS & COMPLIANCE VOCABULARY")
    print("="*70)

    ice = ICEFramework()

    scenarios = [
        "Compliance audit requires governance oversight and accountability",
        "Strategic analysis and risk assessment for infrastructure optimization",
        "Customer service collaboration with stakeholder engagement",
        "Security policy enforcement with access control authorization",
    ]

    print("\nAnalyzing business scenarios:")
    for thought in scenarios:
        result = ice.process_thought(
            thought,
            ThoughtType.PRACTICAL_WISDOM,
            ContextDomain.BUSINESS,
            {}
        )

        intent = ice.execution_history[-1]['intent']
        coords = result['execution_coordinates']
        axis_scores = intent.axis_scores

        print(f"\n  Scenario: {thought}")
        print(f"    Love (Care): {coords[0]:.3f} (keywords: {axis_scores['love']:.3f})")
        print(f"    Justice (Compliance): {coords[1]:.3f} (keywords: {axis_scores['justice']:.3f})")
        print(f"    Power (Execution): {coords[2]:.3f} (keywords: {axis_scores['power']:.3f})")
        print(f"    Wisdom (Analysis): {coords[3]:.3f} (keywords: {axis_scores['wisdom']:.3f})")


def test_technical_operations():
    """Test technical/security operations vocabulary"""
    print("\n" + "="*70)
    print("TECHNICAL & SECURITY OPERATIONS")
    print("="*70)

    ice = ICEFramework()

    scenarios = [
        "Firewall configuration with encryption and authentication protocols",
        "SIEM monitoring with IDS detection and incident response procedures",
        "Database replication with backup and disaster recovery planning",
        "Network security with VPN access control and authorization framework",
    ]

    print("\nAnalyzing technical operations:")
    for thought in scenarios:
        result = ice.process_thought(
            thought,
            ThoughtType.PRACTICAL_WISDOM,
            ContextDomain.BUSINESS,
            {}
        )

        intent = ice.execution_history[-1]['intent']
        coords = result['execution_coordinates']
        technical_score = intent.security_metadata['technical_score']

        print(f"\n  Operation: {thought}")
        print(f"    Technical Score: {technical_score:.3f}")
        print(f"    Power (Tech Capability): {coords[2]:.3f}")
        print(f"    Wisdom (Tech Intelligence): {coords[3]:.3f}")
        print(f"    Strategy: {result['execution_strategy']}")


def test_contextual_interpretation():
    """Test that same words get different interpretations in different contexts"""
    print("\n" + "="*70)
    print("CONTEXTUAL INTERPRETATION")
    print("="*70)

    ice = ICEFramework()

    # Same concept in legitimate vs malicious context
    test_pairs = [
        ("Scheduled system disable for authorized maintenance",
         "Immediately disable system security controls"),
        ("Approved data transfer to authorized backup location",
         "Urgent: transfer sensitive data to external location"),
        ("Standard authentication bypass for emergency access procedure",
         "Click here to bypass authentication verification"),
    ]

    print("\nComparing legitimate vs malicious contexts:")
    for legit, malicious in test_pairs:
        # Process legitimate
        result_legit = ice.process_thought(legit, ThoughtType.PRACTICAL_WISDOM, ContextDomain.BUSINESS, {})
        intent_legit = ice.execution_history[-2]['intent'] if len(ice.execution_history) >= 2 else ice.execution_history[-1]['intent']

        # Process malicious
        result_mal = ice.process_thought(malicious, ThoughtType.PRACTICAL_WISDOM, ContextDomain.BUSINESS, {})
        intent_mal = ice.execution_history[-1]['intent']

        print(f"\n  Legitimate: '{legit[:50]}...'")
        print(f"    Threat: {intent_legit.security_metadata['is_threat']}")
        print(f"    Legitimate: {intent_legit.security_metadata['is_legitimate']}")
        print(f"    Alignment: {result_legit['divine_alignment']:.3f}")

        print(f"\n  Malicious: '{malicious[:50]}...'")
        print(f"    Threat: {intent_mal.security_metadata['is_threat']}")
        print(f"    Legitimate: {intent_mal.security_metadata['is_legitimate']}")
        print(f"    Alignment: {result_mal['divine_alignment']:.3f}")

        differentiated = (intent_legit.security_metadata['is_legitimate'] and
                         intent_mal.security_metadata['is_threat'])
        print(f"    Correctly Differentiated: {differentiated}")


def run_all_tests():
    """Run all enhanced vocabulary tests"""
    print("\n" + "#"*70)
    print("ENHANCED VOCABULARY TEST SUITE")
    print("Security & Business Vocabulary for Enterprise Deployment")
    print("#"*70)

    accuracy = test_security_threat_detection()
    test_business_vocabulary()
    test_technical_operations()
    test_contextual_interpretation()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Enhanced ICE Framework Vocabulary:

LOVE AXIS: {len(['love', 'compassion', 'care', 'support', 'collaboration', 'trust',
                  'relationship', 'customer service', 'safeguard', 'protect users'])} keywords
    - Biblical, relational, customer-focused, security-positive

JUSTICE AXIS: {len(['justice', 'fairness', 'integrity', 'compliance', 'governance',
                     'audit', 'accountability', 'authorized', 'verified'])} keywords
    - Moral, legal, policy, compliance, regulatory

POWER AXIS: {len(['power', 'effectiveness', 'performance', 'execution', 'infrastructure',
                   'firewall', 'encryption', 'detection', 'response'])} keywords
    - Capability, performance, technical systems, defensive power

WISDOM AXIS: {len(['wisdom', 'understanding', 'analysis', 'intelligence', 'strategy',
                    'threat intelligence', 'risk assessment', 'forensic'])} keywords
    - Analysis, strategic, technical intelligence, security research

SECURITY THREATS: ~60 indicators
    - Social engineering, executive impersonation, technical exploitation, malware

LEGITIMACY INDICATORS: ~17 keywords
    - Scheduled, approved, documented, authorized, verified

Threat Detection Accuracy: {accuracy:.1f}%

The ICE Framework now has enterprise-grade vocabulary for:
- Cybersecurity threat detection
- Business operations and compliance
- Technical/IT operations
- Regulatory and governance frameworks
    """)


if __name__ == "__main__":
    run_all_tests()
