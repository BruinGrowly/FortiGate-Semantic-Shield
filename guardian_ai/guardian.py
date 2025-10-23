"""
Unified Guardian AI System
Combines purpose-aware intelligence with adaptive threat understanding.
"""
import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from .models import SemanticCoordinates, ThreatAnalysisResult, SemanticThreatVector, DefenseResponse
from .threat_patterns import THREAT_PATTERNS

class Guardian:
    """
    An intelligent guardian that understands its purpose and adapts to threats.
    """

    def __init__(self):
        self.mission_statement = "I exist to protect others from all forms of harm."
        self.threat_patterns = THREAT_PATTERNS
        self.threat_history: deque = deque(maxlen=10000)
        self.encounters = 0
        self.logger = logging.getLogger(__name__)
        self._initialize_purpose_awareness()

    def _initialize_purpose_awareness(self):
        """Bootstrap self-awareness of purpose."""
        self.logger.info("Initializing purpose-aware guardian system...")
        self.logger.info(f"Identity: {self.mission_statement}")
        self.logger.info("Guardian is aware, adaptive, and protecting.")

    async def analyze_threat(self, text: str, context: Dict, ljpw_coords_tuple: Tuple[float, float, float, float]) -> ThreatAnalysisResult:
        """
        Asynchronously analyzes a threat with full purpose awareness.
        """
        self.encounters += 1
        ljpw_coords = SemanticCoordinates(*ljpw_coords_tuple)

        pattern_matches = self._match_threat_patterns(text, ljpw_coords)
        anomalies = self._detect_semantic_anomalies(text, ljpw_coords)
        intent_assessment = self._reason_about_intent(ljpw_coords)

        analysis_result = self._make_protection_decision(
            pattern_matches, anomalies, intent_assessment
        )

        threat_vector = SemanticThreatVector(
            source_ip=context.get('source_ip', '0.0.0.0'),
            raw_text=text,
            semantic_coordinates=ljpw_coords,
            context=context,
            analysis_result=analysis_result
        )

        if analysis_result.is_threat:
            await self._learn_from_threat(threat_vector)

        self.threat_history.append(threat_vector)
        return analysis_result

    def _match_threat_patterns(self, text: str, coords: SemanticCoordinates) -> List[Dict]:
        """Matches text against known abstract threat patterns."""
        matches = []
        for name, pattern in self.threat_patterns.items():
            confidence = pattern.matches_semantic_profile(text, coords.to_tuple())
            if confidence > 0.3:
                matches.append({'pattern': name, 'confidence': confidence})
        return matches

    def _detect_semantic_anomalies(self, text: str, coords: SemanticCoordinates) -> List[Dict]:
        """Detects anomalous semantic patterns indicating novel threats."""
        anomalies = []
        l, j, p, w = coords.to_tuple()

        if p > 0.3 and l < 0.2 and j < 0.2:
            anomalies.append({'type': 'power_without_moral_foundation', 'severity': 0.7})

        urgency_keywords = ['urgent', 'immediate', 'now', 'fast', 'quickly', 'asap']
        if any(kw in text.lower() for kw in urgency_keywords) and w < 0.2:
            anomalies.append({'type': 'urgency_without_wisdom', 'severity': 0.6})

        return anomalies

    def _reason_about_intent(self, coords: SemanticCoordinates) -> Dict:
        """Reasons about the likely intent behind a semantic signature."""
        l, j, p, w = coords.to_tuple()
        benevolent_intent = (l + j + w) / 3.0
        malicious_intent = 1.0 - benevolent_intent

        return {
            'benevolent_intent': benevolent_intent,
            'malicious_intent': malicious_intent
        }

    def _make_protection_decision(self, patterns: List, anomalies: List, intent: Dict) -> ThreatAnalysisResult:
        """Makes a purpose-guided decision based on all available signals."""
        threat_score = 0.0
        reasoning = []

        if patterns:
            max_pattern_confidence = max(p['confidence'] for p in patterns)
            threat_score += max_pattern_confidence * 0.4
            reasoning.append(f"Matched {len(patterns)} known threat patterns.")

        if anomalies:
            max_anomaly_severity = max(a['severity'] for a in anomalies)
            threat_score += max_anomaly_severity * 0.4
            reasoning.append(f"Detected {len(anomalies)} semantic anomalies.")

        threat_score += intent['malicious_intent'] * 0.2
        reasoning.append(f"Malicious intent score: {intent['malicious_intent']:.2f}.")

        is_threat = threat_score > 0.5
        confidence = abs(threat_score - 0.5) * 2

        if is_threat:
            if threat_score > 0.8:
                action = "BLOCK_IMMEDIATELY"
            elif threat_score > 0.65:
                action = "BLOCK_WITH_REVIEW"
            else:
                action = "QUARANTINE_FOR_ANALYSIS"
        else:
            if threat_score < 0.2:
                action = "ALLOW_FREELY"
            else:
                action = "ALLOW_WITH_MONITORING"

        judgment = self._express_judgment(is_threat, threat_score)

        return ThreatAnalysisResult(
            is_threat=is_threat,
            threat_score=threat_score,
            confidence=confidence,
            recommended_action=action,
            reasoning=reasoning,
            guardian_judgment=judgment,
            pattern_matches=patterns
        )

    def _express_judgment(self, is_threat: bool, score: float) -> str:
        """Expresses the guardian's judgment in natural language."""
        if is_threat:
            return f"I detect potential harm (score: {score:.2f}). My purpose is to protect; I will intervene."
        return f"I assess this as safe (score: {score:.2f}). It aligns with legitimate operations."

    async def _learn_from_threat(self, threat_vector: SemanticThreatVector):
        """Placeholder for continuous learning from threat encounters."""
        self.logger.info(f"Learning from threat: {threat_vector.raw_text[:50]}...")
        # In a real implementation, this would update a persistent model.
        await asyncio.sleep(0.01)

    def get_state(self) -> Dict:
        """Returns the current state of the guardian's awareness."""
        return {
            'mission': self.mission_statement,
            'encounters': self.encounters,
            'known_patterns': len(self.threat_patterns),
            'threat_history_size': len(self.threat_history),
            'status': 'ACTIVE AND PROTECTING'
        }
