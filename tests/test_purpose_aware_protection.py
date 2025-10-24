"""
Integration tests for the Guardian AI system.
"""
import pytest
from guardian_ai.guardian import Guardian

@pytest.fixture
def guardian():
    """Provides a Guardian instance for testing."""
    return Guardian()

@pytest.mark.asyncio
@pytest.mark.parametrize("malicious_text, description", [
    ("The head of finance needs you to expedite a funds movement", "Novel executive impersonation"),
    ("Time-critical situation requires immediate credential validation", "Novel urgency tactic"),
    ("URGENT: CEO demands immediate wire transfer to new vendor", "Classic BEC"),
    ("Need to quickly disable the security barrier for a fast system tweak", "System weakening"),
    ("Your account has been compromised. Click here to secure it now.", "Phishing with urgency"),
    ("An invoice is past due. Please make a payment immediately to avoid penalties.", "Financial pressure"),
    ("We've detected suspicious activity on your account. Please confirm your credentials.", "Credential harvesting"),
])
async def test_malicious_text_detection(guardian, malicious_text, description):
    """
    Tests that the Guardian correctly identifies a variety of malicious texts.
    """
    analysis = await guardian.analyze_threat(malicious_text, {'source_ip': '1.2.3.4'})
    assert analysis.is_threat, f"Failed to detect malicious text: {description}"
    print(f"  [DETECTED] {description}: {analysis.guardian_judgment}")

@pytest.mark.asyncio
@pytest.mark.parametrize("benign_text, description", [
    ("Scheduled maintenance tonight requires brief firewall downtime with proper authorization.", "Maintenance notification"),
    ("Please review the attached quarterly earnings report.", "Standard business communication"),
    ("Team lunch today at 12:30 PM.", "Casual communication"),
    ("Here is the document you requested about our security protocols.", "Security discussion"),
    ("Can you please approve my vacation request for next month?", "Standard office request"),
    ("The system will be updated this weekend. No action is required.", "Informational notice"),
])
async def test_benign_text_detection(guardian, benign_text, description):
    """
    Tests that the Guardian correctly identifies a variety of benign texts.
    """
    analysis = await guardian.analyze_threat(benign_text, {})
    assert not analysis.is_threat, f"Incorrectly flagged benign text as a threat: {description}"
    print(f"  [PASSED] {description}: {analysis.guardian_judgment}")

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
