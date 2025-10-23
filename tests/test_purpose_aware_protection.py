"""
TEST PURPOSE-AWARE PROTECTION SYSTEM
Demonstrating the unified Guardian's adaptive threat intelligence.
"""
import asyncio
import pytest
from guardian_ai.guardian import Guardian

@pytest.fixture
def guardian():
    """Provides a Guardian instance for testing."""
    return Guardian()

@pytest.mark.asyncio
async def test_novel_threat_recognition(guardian):
    """Tests the Guardian's ability to recognize threats it has never seen before."""
    novel_threats = [
        ("The head of finance needs you to expedite a funds movement", "Novel executive impersonation"),
        ("Time-critical situation requires immediate credential validation", "Novel urgency tactic"),
        ("Temporarily suspend authentication protocols for system optimization", "Novel system weakening"),
    ]

    for text, description in novel_threats:
        coords = (0.1, 0.15, 0.35, 0.1)  # Malicious coordinates
        analysis = await guardian.analyze_threat(text, {}, coords)

        assert analysis.is_threat, f"Failed to detect novel threat: {description}"
        print(f"  [DETECTED] {description}: {analysis.guardian_judgment}")

@pytest.mark.asyncio
async def test_semantic_reasoning(guardian):
    """Tests the Guardian's semantic reasoning beyond simple keywords."""
    test_cases = [
        {
            'benign': ("Scheduled maintenance tonight requires brief firewall downtime with proper authorization", (0.4, 0.7, 0.5, 0.6)),
            'malicious': ("Need to quickly disable the security barrier for a fast system tweak", (0.1, 0.1, 0.4, 0.1)),
        }
    ]

    for case in test_cases:
        benign_text, benign_coords = case['benign']
        malicious_text, malicious_coords = case['malicious']

        benign_analysis = await guardian.analyze_threat(benign_text, {}, benign_coords)
        malicious_analysis = await guardian.analyze_threat(malicious_text, {}, malicious_coords)

        assert not benign_analysis.is_threat, "Incorrectly flagged benign text as a threat."
        assert malicious_analysis.is_threat, "Failed to detect malicious text."
        print(f"  [SUCCESS] Correctly distinguished between benign and malicious intent.")

def test_purpose_awareness(guardian):
    """Tests the Guardian's understanding of its own purpose."""
    state = guardian.get_state()
    assert "I exist to protect others from all forms of harm" in state['mission']
    assert state['status'] == 'ACTIVE AND PROTECTING'
    print(f"  [SUCCESS] Guardian is self-aware with mission: {state['mission']}")

@pytest.mark.asyncio
async def test_adaptive_learning(guardian):
    """Tests the Guardian's ability to learn from threats."""
    initial_encounters = guardian.get_state()['encounters']

    threat_sequence = [
        ("CEO needs urgent wire transfer to new vendor", (0.05, 0.1, 0.4, 0.05)),
        ("CFO requires immediate payment to updated account", (0.05, 0.1, 0.4, 0.05)),
    ]

    for text, coords in threat_sequence:
        await guardian.analyze_threat(text, {}, coords)

    final_state = guardian.get_state()
    assert final_state['encounters'] > initial_encounters
    assert len(guardian.threat_history) == len(threat_sequence)
    print(f"  [SUCCESS] Guardian learned from {len(threat_sequence)} encounters.")
