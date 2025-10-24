"""
Unified data models for the Guardian AI system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Tuple

@dataclass
class SemanticCoordinates:
    """4D semantic coordinate system (Love, Justice, Power, Wisdom)."""
    love: float = 0.0
    justice: float = 0.0
    power: float = 0.0
    wisdom: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.love, self.justice, self.power, self.wisdom)

    def distance_from_anchor(self, anchor: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> float:
        """Calculate Euclidean distance from the divine anchor point (1,1,1,1)."""
        import math
        coords = self.to_tuple()
        return math.sqrt(sum((c - a)**2 for c, a in zip(coords, anchor)))

    def divine_alignment(self, anchor: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> float:
        """Calculate alignment score (0-1) based on distance from the anchor."""
        distance = self.distance_from_anchor(anchor)
        return max(0.0, 1.0 - (distance / 2.0))

@dataclass
class ThreatAnalysisResult:
    """Result of a threat analysis by the Guardian."""
    is_threat: bool
    threat_score: float
    confidence: float
    recommended_action: str
    reasoning: List[str]
    guardian_judgment: str
    pattern_matches: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SemanticThreatVector:
    """A threat vector enriched with semantic analysis."""
    source_ip: str
    raw_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    semantic_coordinates: SemanticCoordinates = field(default_factory=SemanticCoordinates)
    context: Dict[str, Any] = field(default_factory=dict)
    analysis_result: ThreatAnalysisResult = None

@dataclass
class DefenseResponse:
    """A defense response guided by the Guardian's analysis."""
    response_id: str
    threat_vector: SemanticThreatVector
    defense_strategy: str
    execution_mode: str
    confidence: float
    justification: str
    blocking_rules: List[Dict[str, Any]] = field(default_factory=list)
    routing_modifications: Dict[str, Any] = field(default_factory=dict)
    quarantine_actions: List[str] = field(default_factory=list)
