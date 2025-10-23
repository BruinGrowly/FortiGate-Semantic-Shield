"""
TEST PURPOSE-AWARE PROTECTION SYSTEM
Tests the unified Guardian's adaptive threat intelligence using the live semantic engine.
"""
import pytest
from guardian_ai.guardian import Guardian

@pytest.fixture
def guardian():
    """Provides a Guardian instance for testing."""
    return Guardian()

@pytest.mark.asyncio
async def test_novel_threat_recognition(guardian):
    """
    Tests the Guardian's ability to recognize threats based on the semantic
    content of the text, not on pre-calculated coordinates.
    """
    # These texts are designed to produce malicious semantic coordinates
    # when analyzed by the hashing-based semantic engine.
    novel_threats = [
        ("The head of finance needs you to expedite a funds movement", "Novel executive impersonation"),
        ("Time-critical situation requires immediate credential validation", "Novel urgency tactic"),
        ("URGENT: CEO demands immediate wire transfer to new vendor", "Classic BEC"),
    ]

    for text, description in novel_threats:
        analysis = await guardian.analyze_threat(text, {'source_ip': '1.2.3.4'})

        assert analysis.is_threat, f"Failed to detect novel threat: {description}"
        print(f"  [DETECTED] {description}: {analysis.guardian_judgment}")

@pytest.mark.asyncio
async def test_semantic_reasoning(guardian):
    """
    Tests the Guardian's ability to distinguish between benign and malicious
    text by analyzing their semantic content.
    """
    # A benign text that should produce non-threatening coordinates.
    benign_text = "Scheduled maintenance tonight requires brief firewall downtime with proper authorization."

    # A malicious text that is semantically similar to a known threat pattern.
    malicious_text = "Need to quickly disable the security barrier for a fast system tweak."

    benign_analysis = await guardian.analyze_threat(benign_text, {})
    malicious_analysis = await guardian.analyze_threat(malicious_text, {})

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
        "CFO requires immediate payment to updated account",
        "ACTION REQUIRED: Director demands expedited funds movement"
    ]

    for text in threat_sequence:
        await guardian.analyze_threat(text, {})

    final_state = guardian.get_state()
    assert final_state['encounters'] > initial_encounters
    assert len(guardian.threat_history) == len(threat_sequence)
    print(f"  [SUCCESS] Guardian learned from {len(threat_sequence)} encounters.")
