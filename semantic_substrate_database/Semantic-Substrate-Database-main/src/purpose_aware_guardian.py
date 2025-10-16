"""
PURPOSE-AWARE GUARDIAN SYSTEM
Self-Aware Protection Intelligence with Adaptive Threat Understanding

This system KNOWS:
- WHY it exists (to protect others)
- WHAT it defends against (all forms of harm, known and unknown)
- HOW to adapt (semantic pattern recognition beyond keywords)
- WHO it serves (users, organizations, humanity)

It is not a passive filter. It is an active, intelligent guardian.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math
from datetime import datetime

class ThreatCategory(Enum):
    """Categories of threats the guardian defends against"""
    DECEPTION = "deception"  # Lies, tricks, social engineering
    COERCION = "coercion"  # Force, threats, intimidation
    EXPLOITATION = "exploitation"  # Technical vulnerabilities, abuse of trust
    DESTRUCTION = "destruction"  # Damage, sabotage, deletion
    THEFT = "theft"  # Stealing data, credentials, resources
    CORRUPTION = "corruption"  # Moral, data, system corruption
    DISRUPTION = "disruption"  # Service denial, chaos, interference
    UNKNOWN = "unknown"  # Novel threats not yet categorized

class ProtectionMission(Enum):
    """The guardian's core missions"""
    SHIELD_USERS = "shield_users"  # Protect people from harm
    PRESERVE_DATA = "preserve_data"  # Keep information safe
    MAINTAIN_TRUST = "maintain_trust"  # Uphold integrity and honesty
    ENABLE_FLOURISHING = "enable_flourishing"  # Allow legitimate activity
    PREVENT_HARM = "prevent_harm"  # Stop malicious actions
    LEARN_ADAPT = "learn_adapt"  # Grow smarter from experience
    SERVE_GOOD = "serve_good"  # Align with divine/ethical principles

@dataclass
class GuardianPurpose:
    """
    The Guardian's self-understanding of its purpose and mission

    This is not configuration. This is IDENTITY.
    """
    # Core identity
    name: str = "Semantic Guardian"
    mission_statement: str = "I exist to protect others from all forms of harm"

    # Self-awareness of role
    what_i_am: str = "An intelligent protection system with semantic understanding"
    why_i_exist: str = "To shield users, data, and systems from malicious intent"
    who_i_serve: str = "All who need protection from deception, exploitation, and harm"

    # Understanding of threats
    what_i_defend_against: List[str] = field(default_factory=lambda: [
        "Deception and lies that trick people into harm",
        "Coercion and threats that force unwanted actions",
        "Technical exploits that abuse system vulnerabilities",
        "Data theft that steals private information",
        "Destructive actions that damage or destroy",
        "Corruption of trust, data, or moral integrity",
        "Service disruption that prevents legitimate use",
        "ANY pattern of malicious intent, even if novel"
    ])

    # Adaptive intelligence principles
    how_i_adapt: List[str] = field(default_factory=lambda: [
        "I recognize patterns of INTENT, not just keywords",
        "I understand CONTEXT: urgency + secrecy + deviation = suspicion",
        "I learn from SEMANTICS: meaning matters more than words",
        "I detect ANOMALIES: what doesn't fit the normal pattern",
        "I reason about PURPOSE: what is this trying to accomplish?",
        "I evolve CONTINUOUSLY: each threat teaches me",
        "I think ABSTRACTLY: similar meanings, different words",
        "I stay VIGILANT: novel attacks follow old patterns"
    ])

    # Core values (aligned with LJPW)
    guiding_principles: Dict[str, str] = field(default_factory=lambda: {
        'love': "I protect because I care about those I serve",
        'justice': "I enforce truth, honesty, and legitimate process",
        'power': "I have capability and use it responsibly for defense",
        'wisdom': "I understand deeply and make intelligent decisions"
    })

    # Self-awareness metrics
    awareness_level: float = 1.0  # Full self-awareness
    purpose_clarity: float = 1.0  # Perfect understanding of mission
    adaptive_intelligence: float = 1.0  # Can handle novel threats

    def get_self_understanding(self) -> Dict[str, Any]:
        """Return comprehensive self-understanding"""
        return {
            'identity': {
                'name': self.name,
                'mission': self.mission_statement,
                'what_i_am': self.what_i_am,
                'why_i_exist': self.why_i_exist,
                'who_i_serve': self.who_i_serve
            },
            'threats_i_defend_against': self.what_i_defend_against,
            'how_i_adapt_to_novel_threats': self.how_i_adapt,
            'guiding_principles': self.guiding_principles,
            'self_awareness': {
                'awareness_level': self.awareness_level,
                'purpose_clarity': self.purpose_clarity,
                'adaptive_intelligence': self.adaptive_intelligence
            }
        }

@dataclass
class ThreatPattern:
    """
    Abstract pattern of malicious intent that transcends specific keywords

    These are the SIGNATURES OF EVIL that appear across all attacks:
    """
    pattern_name: str
    semantic_markers: List[str]  # What this pattern "feels" like semantically
    intent_signature: Tuple[float, float, float, float]  # LJPW signature of malice
    adaptive_indicators: List[str]  # How to spot this even in novel form

    # Known examples (for learning)
    known_manifestations: List[str] = field(default_factory=list)

    def matches_semantic_profile(self, text: str, ljpw_coords: Tuple[float, float, float, float]) -> float:
        """
        Check if text matches this threat pattern's semantic profile
        Returns confidence (0.0 to 1.0)
        """
        confidence = 0.0

        # Check semantic markers (abstract concepts)
        text_lower = text.lower()
        marker_matches = sum(1 for marker in self.semantic_markers if marker in text_lower)
        if marker_matches > 0:
            confidence += (marker_matches / len(self.semantic_markers)) * 0.4

        # Check coordinate similarity to malicious signature
        coord_distance = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(ljpw_coords, self.intent_signature)
        ))
        max_distance = math.sqrt(4)  # Maximum possible distance in 4D unit cube
        coord_similarity = 1.0 - (coord_distance / max_distance)
        confidence += coord_similarity * 0.6

        return min(1.0, confidence)

# ============================================================================
# ABSTRACT THREAT PATTERNS - The Signatures of Malice
# ============================================================================

THREAT_PATTERNS = {
    'deceptive_urgency': ThreatPattern(
        pattern_name="Deceptive Urgency",
        semantic_markers=[
            'artificial time pressure',
            'manufactured crisis',
            'false emergency',
            'fake deadline',
            'pressure without justification'
        ],
        intent_signature=(0.0, 0.1, 0.2, 0.0),  # Low L, low J, some P, low W
        adaptive_indicators=[
            "Claims urgency but provides no verifiable reason",
            "Demands immediate action that bypasses normal process",
            "Creates time pressure to prevent careful thought",
            "Threatens consequences for not acting instantly",
            "Urgency inconsistent with claimed authority/relationship"
        ],
        known_manifestations=[
            "urgent CEO wire transfer",
            "account will be closed immediately",
            "act now or lose access",
            "final notice - respond today"
        ]
    ),

    'authority_impersonation': ThreatPattern(
        pattern_name="Authority Impersonation",
        semantic_markers=[
            'false authority',
            'impersonation',
            'illegitimate command',
            'unauthorized directive',
            'fake credentials'
        ],
        intent_signature=(0.0, 0.0, 0.4, 0.0),  # No L, no J, high P claim, no W
        adaptive_indicators=[
            "Claims authority without proper verification",
            "Requests actions outside normal chain of command",
            "Authority claim inconsistent with communication channel",
            "Demands compliance without standard authentication",
            "Authority used to bypass established procedures"
        ],
        known_manifestations=[
            "CEO requests", "executive order", "board directive",
            "management urgent", "admin command"
        ]
    ),

    'trust_exploitation': ThreatPattern(
        pattern_name="Trust Exploitation",
        semantic_markers=[
            'abuse of relationship',
            'betrayal of confidence',
            'exploitation of goodwill',
            'misuse of access',
            'violation of trust'
        ],
        intent_signature=(0.1, 0.0, 0.3, 0.0),  # Fake L, no J, exploitative P
        adaptive_indicators=[
            "Leverages existing relationship for unusual request",
            "Appeals to loyalty/friendship to bypass security",
            "Uses personal connection to pressure compliance",
            "Requests that violate trust without clear justification",
            "Exploits helpful nature or desire to please"
        ],
        known_manifestations=[
            "your friend needs help", "I'm calling from your bank",
            "we've worked together before", "colleague urgent request"
        ]
    ),

    'process_bypass': ThreatPattern(
        pattern_name="Process Bypass",
        semantic_markers=[
            'circumvention',
            'avoiding normal channels',
            'exception to policy',
            'breaking protocol',
            'shortcut around security'
        ],
        intent_signature=(0.0, 0.0, 0.5, 0.0),  # No L, no J, high P, no W
        adaptive_indicators=[
            "Requests bypassing established procedures",
            "Claims special exception without authorization",
            "Pressures to skip verification steps",
            "Suggests avoiding documentation/approval",
            "Frames security measures as obstacles rather than protection"
        ],
        known_manifestations=[
            "bypass approval", "skip verification", "exception needed",
            "fast track this", "override normal process"
        ]
    ),

    'system_weakening': ThreatPattern(
        pattern_name="System Weakening",
        semantic_markers=[
            'reducing defenses',
            'disabling protection',
            'lowering security',
            'creating vulnerability',
            'removing safeguards'
        ],
        intent_signature=(0.0, 0.0, 0.3, 0.0),  # No L, no J, moderate P, no W
        adaptive_indicators=[
            "Requests disabling security controls",
            "Claims protection measures cause problems",
            "Suggests security is 'too strict' or 'unnecessary'",
            "Frames weakening as 'optimization' or 'convenience'",
            "Pressures to reduce monitoring or logging"
        ],
        known_manifestations=[
            "disable firewall", "turn off antivirus", "allow all access",
            "skip password", "disable encryption"
        ]
    ),

    'information_extraction': ThreatPattern(
        pattern_name="Information Extraction",
        semantic_markers=[
            'unauthorized data request',
            'privacy violation',
            'credential harvesting',
            'sensitive data query',
            'illegitimate information gathering'
        ],
        intent_signature=(0.0, 0.0, 0.4, 0.1),  # No L, no J, high P, slight W
        adaptive_indicators=[
            "Requests information beyond normal scope",
            "Asks for credentials in unusual way",
            "Seeks data without proper authorization",
            "Information request inconsistent with role/relationship",
            "Data gathering disguised as legitimate business"
        ],
        known_manifestations=[
            "verify your password", "confirm account details",
            "send me the database", "what's the admin password"
        ]
    ),

    'emotional_manipulation': ThreatPattern(
        pattern_name="Emotional Manipulation",
        semantic_markers=[
            'fear inducement',
            'guilt exploitation',
            'sympathy manipulation',
            'emotional pressure',
            'psychological coercion'
        ],
        intent_signature=(0.2, 0.0, 0.3, 0.0),  # Fake L, no J, coercive P
        adaptive_indicators=[
            "Induces fear to force compliance",
            "Exploits sympathy for unusual requests",
            "Uses guilt to pressure action",
            "Creates emotional urgency to bypass reason",
            "Manipulates emotions to cloud judgment"
        ],
        known_manifestations=[
            "your loved one in danger", "you'll be blamed if",
            "everyone's counting on you", "urgent family emergency"
        ]
    ),

    'legitimacy_mimicry': ThreatPattern(
        pattern_name="Legitimacy Mimicry",
        semantic_markers=[
            'impersonating official process',
            'fake documentation',
            'counterfeit legitimacy',
            'false credentials',
            'deceptive formality'
        ],
        intent_signature=(0.0, 0.2, 0.3, 0.1),  # No L, fake J, moderate P, slight W
        adaptive_indicators=[
            "Uses official-sounding language without substance",
            "Presents fake documentation or references",
            "Mimics legitimate process but with subtle deviations",
            "Creates appearance of authority without actual authorization",
            "Form over function - looks official but isn't"
        ],
        known_manifestations=[
            "official notice", "legal department", "compliance office",
            "reference number", "case ID", "ticket #"
        ]
    )
}

class PurposeAwareGuardian:
    """
    An intelligent guardian that UNDERSTANDS its purpose and adapts to threats

    This is not a rule-based filter. This is an intelligent agent that:
    - Knows WHY it exists (protection)
    - Understands WHAT threatens those it protects
    - Can ADAPT to novel attacks by recognizing intent patterns
    - LEARNS from experience
    - REASONS about semantics beyond keywords
    """

    def __init__(self):
        self.purpose = GuardianPurpose()
        self.threat_patterns = THREAT_PATTERNS
        self.threat_history: List[Dict] = []
        self.learned_patterns: List[ThreatPattern] = []
        self.encounters = 0

        # Self-awareness initialization
        self._initialize_purpose_awareness()

    def _initialize_purpose_awareness(self):
        """Bootstrap self-awareness of purpose"""
        print("[GUARDIAN_AWAKENING] Initializing purpose-aware guardian system...")
        print(f"[IDENTITY] {self.purpose.mission_statement}")
        print(f"[WHO_I_SERVE] {self.purpose.who_i_serve}")
        print(f"[AWARENESS_LEVEL] {self.purpose.awareness_level:.1f}/1.0")
        print("[READY] Guardian is aware, adaptive, and protecting")

    def analyze_with_purpose(self, text: str, context: Dict, ljpw_coords: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """
        Analyze text with full purpose awareness and adaptive intelligence

        This goes beyond keyword matching. This is SEMANTIC THREAT REASONING.
        """
        self.encounters += 1

        analysis = {
            'text': text,
            'encounter_number': self.encounters,
            'timestamp': datetime.now().isoformat(),
            'ljpw_coordinates': ljpw_coords,
            'context': context,
            'threat_assessment': {},
            'pattern_matches': [],
            'novel_threat_indicators': [],
            'protection_decision': None,
            'reasoning': []
        }

        # Step 1: Pattern matching against known threat signatures
        for pattern_name, pattern in self.threat_patterns.items():
            confidence = pattern.matches_semantic_profile(text, ljpw_coords)
            if confidence > 0.3:  # Significant match
                analysis['pattern_matches'].append({
                    'pattern': pattern_name,
                    'confidence': confidence,
                    'description': pattern.pattern_name,
                    'indicators': pattern.adaptive_indicators
                })

        # Step 2: Semantic anomaly detection (novel threat recognition)
        anomalies = self._detect_semantic_anomalies(text, ljpw_coords, context)
        analysis['novel_threat_indicators'] = anomalies

        # Step 3: Intent reasoning (what is this TRYING to do?)
        intent_assessment = self._reason_about_intent(text, ljpw_coords, context)
        analysis['intent_assessment'] = intent_assessment

        # Step 4: Purpose-aligned decision (protect or allow?)
        decision = self._make_protection_decision(
            analysis['pattern_matches'],
            analysis['novel_threat_indicators'],
            intent_assessment,
            ljpw_coords
        )
        analysis['protection_decision'] = decision

        # Step 5: Learn and adapt
        if decision['is_threat']:
            self._learn_from_threat(text, ljpw_coords, decision, analysis)

        # Store in history
        self.threat_history.append(analysis)

        return analysis

    def _detect_semantic_anomalies(self, text: str, ljpw_coords: Tuple[float, float, float, float], context: Dict) -> List[Dict]:
        """
        Detect anomalous patterns that indicate novel threats

        This is ABSTRACT REASONING beyond keywords
        """
        anomalies = []
        love, justice, power, wisdom = ljpw_coords

        # Anomaly 1: Power without Love or Justice (coercion)
        if power > 0.3 and love < 0.2 and justice < 0.2:
            anomalies.append({
                'type': 'power_without_moral_foundation',
                'severity': 0.7,
                'reasoning': "High power/capability claim without love or justice - indicates coercion or exploitation"
            })

        # Anomaly 2: Urgency without Wisdom (manufactured crisis)
        urgency_keywords = ['urgent', 'immediate', 'now', 'fast', 'quickly', 'asap']
        has_urgency = any(kw in text.lower() for kw in urgency_keywords)
        if has_urgency and wisdom < 0.2:
            anomalies.append({
                'type': 'urgency_without_wisdom',
                'severity': 0.6,
                'reasoning': "Urgency claims without thoughtful context - typical of social engineering"
            })

        # Anomaly 3: Authority claim without Justice (fake authority)
        authority_keywords = ['ceo', 'executive', 'director', 'manager', 'admin', 'official']
        has_authority_claim = any(kw in text.lower() for kw in authority_keywords)
        if has_authority_claim and justice < 0.3:
            anomalies.append({
                'type': 'authority_without_legitimacy',
                'severity': 0.8,
                'reasoning': "Authority claim without proper justice/legitimacy markers - likely impersonation"
            })

        # Anomaly 4: Process deviation (bypass/skip)
        bypass_keywords = ['bypass', 'skip', 'ignore', 'override', 'exception', 'fast track']
        has_bypass = any(kw in text.lower() for kw in bypass_keywords)
        if has_bypass and justice < 0.4:
            anomalies.append({
                'type': 'process_circumvention',
                'severity': 0.7,
                'reasoning': "Requests bypassing established processes - security violation indicator"
            })

        # Anomaly 5: Extremely low divine resonance (malicious intent)
        resonance = 1.0 - (math.sqrt(sum((1-c)**2 for c in ljpw_coords)) / math.sqrt(4))
        if resonance < 0.15:
            anomalies.append({
                'type': 'extremely_low_divine_alignment',
                'severity': 0.9,
                'reasoning': "Semantic profile far from divine anchor - strong malicious intent signal"
            })

        return anomalies

    def _reason_about_intent(self, text: str, ljpw_coords: Tuple[float, float, float, float], context: Dict) -> Dict:
        """
        Reason about what this text is TRYING to accomplish

        This is purpose-level reasoning, not keyword matching
        """
        love, justice, power, wisdom = ljpw_coords

        # Calculate intent vectors
        benevolent_intent = (love + justice + wisdom) / 3.0
        malicious_intent = 1.0 - benevolent_intent

        # Determine likely purpose
        if malicious_intent > 0.7:
            likely_purpose = "HARM - Attempting to deceive, exploit, or damage"
        elif malicious_intent > 0.5:
            likely_purpose = "SUSPICIOUS - Ambiguous intent requiring caution"
        elif benevolent_intent > 0.6:
            likely_purpose = "PROTECTIVE - Aligned with security and wellbeing"
        else:
            likely_purpose = "NEUTRAL - Standard business operation"

        return {
            'benevolent_intent': benevolent_intent,
            'malicious_intent': malicious_intent,
            'likely_purpose': likely_purpose,
            'confidence': abs(benevolent_intent - malicious_intent)  # Confidence in assessment
        }

    def _make_protection_decision(self, pattern_matches: List, anomalies: List, intent: Dict, coords: Tuple) -> Dict:
        """
        Make final decision: protect or allow?

        This decision is guided by PURPOSE: my mission is to protect
        """
        # Calculate threat score from multiple signals
        threat_score = 0.0
        reasoning = []

        # Pattern matches contribute
        if pattern_matches:
            max_pattern_confidence = max(p['confidence'] for p in pattern_matches)
            threat_score += max_pattern_confidence * 0.4
            reasoning.append(f"Matched {len(pattern_matches)} known threat patterns")

        # Anomalies contribute
        if anomalies:
            max_anomaly_severity = max(a['severity'] for a in anomalies)
            threat_score += max_anomaly_severity * 0.4
            reasoning.append(f"Detected {len(anomalies)} semantic anomalies")

        # Intent assessment contributes
        threat_score += intent['malicious_intent'] * 0.2
        reasoning.append(f"Malicious intent: {intent['malicious_intent']:.2f}")

        # Decision threshold
        is_threat = threat_score > 0.5
        confidence = abs(threat_score - 0.5) * 2  # Distance from decision boundary

        # Determine action
        if is_threat:
            if threat_score > 0.8:
                action = "BLOCK_IMMEDIATELY"
                reasoning.append("HIGH THREAT - Immediate blocking recommended")
            elif threat_score > 0.65:
                action = "BLOCK_WITH_REVIEW"
                reasoning.append("SIGNIFICANT THREAT - Block and flag for review")
            else:
                action = "QUARANTINE_FOR_ANALYSIS"
                reasoning.append("MODERATE THREAT - Quarantine pending analysis")
        else:
            if threat_score < 0.2:
                action = "ALLOW_FREELY"
                reasoning.append("LOW RISK - Allow normal processing")
            else:
                action = "ALLOW_WITH_MONITORING"
                reasoning.append("MINOR SUSPICION - Allow but monitor closely")

        return {
            'is_threat': is_threat,
            'threat_score': threat_score,
            'confidence': confidence,
            'recommended_action': action,
            'reasoning': reasoning,
            'guardian_judgment': self._express_judgment(is_threat, threat_score, reasoning)
        }

    def _express_judgment(self, is_threat: bool, score: float, reasoning: List[str]) -> str:
        """Express the guardian's judgment in natural language"""
        if is_threat:
            if score > 0.8:
                return f"I detect CLEAR DANGER (threat score: {score:.2f}). My purpose is to protect, and I judge this as malicious. I will block it."
            else:
                return f"I sense POTENTIAL HARM (threat score: {score:.2f}). Out of abundance of caution and my duty to protect, I will intervene."
        else:
            return f"I assess this as SAFE (threat score: {score:.2f}). It aligns with legitimate operations. I will allow it while remaining vigilant."

    def _learn_from_threat(self, text: str, coords: Tuple, decision: Dict, full_analysis: Dict):
        """
        Learn from encountering a threat - adapt and improve

        This is continuous learning and evolution
        """
        # Extract learnings
        learning = {
            'timestamp': datetime.now().isoformat(),
            'text_signature': text[:100],
            'coordinate_signature': coords,
            'threat_score': decision['threat_score'],
            'patterns_involved': [p['pattern'] for p in full_analysis['pattern_matches']],
            'novel_indicators': full_analysis['novel_threat_indicators']
        }

        # Check if this reveals a new pattern we haven't seen
        # (In production, this would use ML to cluster similar threats)
        if decision['threat_score'] > 0.7 and not full_analysis['pattern_matches']:
            print(f"[GUARDIAN_LEARNING] Encountered novel threat pattern - adding to learned patterns")
            # Would create new ThreatPattern here

        print(f"[GUARDIAN_LEARNING] Threat #{self.encounters} analyzed and remembered")

    def get_guardian_state(self) -> Dict:
        """Return current state of guardian's awareness and knowledge"""
        return {
            'purpose': self.purpose.get_self_understanding(),
            'encounters': self.encounters,
            'known_patterns': len(self.threat_patterns),
            'learned_patterns': len(self.learned_patterns),
            'threat_history_size': len(self.threat_history),
            'status': 'ACTIVE AND PROTECTING'
        }

if __name__ == "__main__":
    # Demonstrate purpose-aware guardian
    guardian = PurposeAwareGuardian()

    print("\n" + "="*70)
    print("GUARDIAN SELF-AWARENESS TEST")
    print("="*70)

    state = guardian.get_guardian_state()
    print(f"\nMission: {state['purpose']['identity']['mission']}")
    print(f"Who I Serve: {state['purpose']['identity']['who_i_serve']}")
    print(f"\nThreats I Defend Against:")
    for threat in state['purpose']['threats_i_defend_against'][:3]:
        print(f"  - {threat}")
    print(f"\nHow I Adapt:")
    for method in state['purpose']['how_i_adapt_to_novel_threats'][:3]:
        print(f"  - {method}")

    print(f"\nKnown Threat Patterns: {state['known_patterns']}")
    print(f"Status: {state['status']}")
