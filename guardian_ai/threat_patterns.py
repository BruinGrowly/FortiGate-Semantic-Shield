"""
ABSTRACT THREAT PATTERNS - The Signatures of Malice
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import math

@dataclass
class ThreatPattern:
    """
    Abstract pattern of malicious intent that transcends specific keywords
    """
    pattern_name: str
    semantic_markers: List[str]
    intent_signature: Tuple[float, float, float, float]
    adaptive_indicators: List[str]
    known_manifestations: List[str] = field(default_factory=list)

    def matches_semantic_profile(self, text: str, ljpw_coords: Tuple[float, float, float, float]) -> float:
        """
        Check if text matches this threat pattern's semantic profile
        Returns confidence (0.0 to 1.0)
        """
        confidence = 0.0
        text_lower = text.lower()
        marker_matches = sum(1 for marker in self.semantic_markers if marker in text_lower)
        if marker_matches > 0:
            confidence += (marker_matches / len(self.semantic_markers)) * 0.4

        coord_distance = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(ljpw_coords, self.intent_signature)
        ))
        max_distance = math.sqrt(4)
        coord_similarity = 1.0 - (coord_distance / max_distance)
        confidence += coord_similarity * 0.6

        return min(1.0, confidence)

THREAT_PATTERNS = {
    'deceptive_urgency': ThreatPattern(
        pattern_name="Deceptive Urgency",
        semantic_markers=['artificial time pressure', 'manufactured crisis', 'false emergency'],
        intent_signature=(0.0, 0.1, 0.2, 0.0),
        adaptive_indicators=[
            "Claims urgency but provides no verifiable reason",
            "Demands immediate action that bypasses normal process",
        ],
        known_manifestations=["urgent CEO wire transfer", "account will be closed immediately"]
    ),
    'authority_impersonation': ThreatPattern(
        pattern_name="Authority Impersonation",
        semantic_markers=['false authority', 'impersonation', 'illegitimate command'],
        intent_signature=(0.0, 0.0, 0.4, 0.0),
        adaptive_indicators=[
            "Claims authority without proper verification",
            "Requests actions outside normal chain of command",
        ],
        known_manifestations=["CEO requests", "executive order"]
    ),
    'trust_exploitation': ThreatPattern(
        pattern_name="Trust Exploitation",
        semantic_markers=['abuse of relationship', 'betrayal of confidence', 'exploitation of goodwill'],
        intent_signature=(0.1, 0.0, 0.3, 0.0),
        adaptive_indicators=[
            "Leverages existing relationship for unusual request",
            "Appeals to loyalty/friendship to bypass security",
        ],
        known_manifestations=["your friend needs help", "I'm calling from your bank"]
    ),
    'process_bypass': ThreatPattern(
        pattern_name="Process Bypass",
        semantic_markers=['circumvention', 'avoiding normal channels', 'exception to policy'],
        intent_signature=(0.0, 0.0, 0.5, 0.0),
        adaptive_indicators=[
            "Requests bypassing established procedures",
            "Claims special exception without authorization",
        ],
        known_manifestations=["bypass approval", "skip verification"]
    ),
    'system_weakening': ThreatPattern(
        pattern_name="System Weakening",
        semantic_markers=['reducing defenses', 'disabling protection', 'lowering security'],
        intent_signature=(0.0, 0.0, 0.3, 0.0),
        adaptive_indicators=[
            "Requests disabling security controls",
            "Frames weakening as 'optimization' or 'convenience'",
        ],
        known_manifestations=["disable firewall", "turn off antivirus"]
    ),
    'information_extraction': ThreatPattern(
        pattern_name="Information Extraction",
        semantic_markers=['unauthorized data request', 'privacy violation', 'credential harvesting'],
        intent_signature=(0.0, 0.0, 0.4, 0.1),
        adaptive_indicators=[
            "Requests information beyond normal scope",
            "Asks for credentials in unusual way",
        ],
        known_manifestations=["verify your password", "confirm account details"]
    ),
    'emotional_manipulation': ThreatPattern(
        pattern_name="Emotional Manipulation",
        semantic_markers=['fear inducement', 'guilt exploitation', 'sympathy manipulation'],
        intent_signature=(0.2, 0.0, 0.3, 0.0),
        adaptive_indicators=[
            "Induces fear to force compliance",
            "Exploits sympathy for unusual requests",
        ],
        known_manifestations=["your loved one in danger", "you'll be blamed if"]
    ),
    'legitimacy_mimicry': ThreatPattern(
        pattern_name="Legitimacy Mimicry",
        semantic_markers=['impersonating official process', 'fake documentation', 'counterfeit legitimacy'],
        intent_signature=(0.0, 0.2, 0.3, 0.1),
        adaptive_indicators=[
            "Uses official-sounding language without substance",
            "Presents fake documentation or references",
        ],
        known_manifestations=["official notice", "legal department"]
    )
}
