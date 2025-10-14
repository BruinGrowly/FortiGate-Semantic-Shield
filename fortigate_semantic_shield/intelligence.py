"""
FortiGate Semantic Shield (SSE + SSD Intelligence Layer)
Integrates the Semantic Substrate Database for learning intelligence with biblical wisdom.

This revolutionary system combines:
- Semantic Substrate Engine (SSE) - Real-time semantic processing
- Semantic Substrate Database (SSD) - Persistent learning and memory
- FortiGate Network Security - Advanced threat defense
- Biblical Wisdom - Divine principles and guidance

Result: True AI intelligence that learns and adapts while maintaining biblical values.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .device_interface import (
    FortiGatePolicyApplier,
    FortiGateTelemetryCollector,
    LearningPersistenceManager,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# Add paths for SSE and SSD
sys.path.append(
    str(
        REPO_ROOT
        / "semantic_substrate_engine"
        / "Semantic-Substrate-Engine-main"
        / "src"
    )
)
sys.path.append(
    str(
        REPO_ROOT
        / "semantic_substrate_database"
        / "Semantic-Substrate-Database-main"
        / "src"
    )
)

# Import SSE components
try:
    from ice_semantic_substrate_engine import (
        ICESemanticSubstrateEngine,
        SemanticCoordinates,
        ThoughtType,
        ContextDomain,
        ExecutionStrategy
    )
    SSE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] SSE not available: {e}")
    SSE_AVAILABLE = False

# Import SSD components
try:
    from semantic_substrate_database import SemanticSubstrateDatabase
    SSD_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] SSD not available: {e}")
    SSD_AVAILABLE = False

# Import FortiGate components
from .semantic_components import (
    NetworkIntent,
    NetworkContext,
    NetworkExecutionMode,
    SemanticThreatVector,
    SemanticDefenseResponse
)

@dataclass
class LearnedPattern:
    """Represents a learned threat pattern"""
    
    pattern_id: str
    threat_type: str
    semantic_signature: str
    success_rate: float
    total_occurrences: int
    successful_strategies: List[str]
    context_domains: List[str]
    biblical_effectiveness: Dict[str, float]
    confidence_level: float
    last_updated: datetime
    wisdom_insights: List[str]

@dataclass
class IntelligenceInsight:
    """Intelligence derived from learning"""
    
    similar_threats: int
    pattern_confidence: float
    recommended_strategy: str
    biblical_application: str
    learning_confidence: float
    historical_success: float
    predictive_accuracy: float
    wisdom_summary: str

class FortiGateSemanticShield:
    """
    SSE + SSD Enhanced FortiGate Engine V6.0
    
    This system combines real-time semantic processing (SSE) with
    persistent learning memory (SSD) to create true AI intelligence.
    """
    
    def __init__(
        self,
        database_path: str = "fortigate_intelligence.db",
        device_interface: Optional[FortiGatePolicyApplier] = None,
        telemetry_collector: Optional[FortiGateTelemetryCollector] = None,
        persistence_manager: Optional[LearningPersistenceManager] = None,
    ):
        """Initialize the enhanced intelligence system"""
        
        # Core SSE engine
        if SSE_AVAILABLE:
            self.sse_engine = ICESemanticSubstrateEngine()
        else:
            self.sse_engine = None
            
        # Learning database (SSD)
        if SSD_AVAILABLE:
            self.ssd_database = SemanticSubstrateDatabase(database_path)
        else:
            self.ssd_database = None
            
        # Learning components
        self.pattern_memory = {}
        self.learning_history = deque(maxlen=10000)
        self.biblical_wisdom_repository = {}
        self.confidence_tracker = defaultdict(list)
        
        # Network security mappings
        self.context_mappings = self._create_context_mappings()
        self.strategy_mappings = self._create_strategy_mappings()
        
        # Performance metrics
        self.metrics = {
            'threats_processed': 0,
            'patterns_learned': 0,
            'accuracy_improvement': 0.0,
            'confidence_growth': 0.0,
            'biblical_wisdom_applied': 0,
            'learning_corrections': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("FortiGate Semantic Shield initialized with SSE + SSD")
        
        self.device_interface = device_interface
        self.telemetry_collector = telemetry_collector
        self.persistence_manager = persistence_manager
        self._snapshot_interval = 50
        self._persisted_updates = 0
        
        self._load_persisted_learning()
        
    def _create_context_mappings(self) -> Dict[NetworkContext, ContextDomain]:
        """Map network contexts to SSE domains"""
        
        return {
            NetworkContext.CRITICAL_INFRASTRUCTURE: ContextDomain.TECHNICAL,
            NetworkContext.FINANCIAL_SYSTEMS: ContextDomain.BUSINESS,
            NetworkContext.HEALTHCARE_NETWORKS: ContextDomain.SPIRITUAL,
            NetworkContext.EDUCATIONAL_PLATFORMS: ContextDomain.EDUCATIONAL,
            NetworkContext.GOVERNMENT_SYSTEMS: ContextDomain.ETHICAL,
            NetworkContext.CORPORATE_ENTERPRISE: ContextDomain.BUSINESS,
            NetworkContext.SPIRITUAL_ORGANIZATIONS: ContextDomain.SPIRITUAL
        }
    
    def _create_strategy_mappings(self) -> Dict[ExecutionStrategy, NetworkIntent]:
        """Map SSE strategies to network intents"""
        
        return {
            ExecutionStrategy.COMPASSIONATE_ACTION: NetworkIntent.LOVE_SHIELD,
            ExecutionStrategy.AUTHORITATIVE_COMMAND: NetworkIntent.HOLINESS_SEPARATION,
            ExecutionStrategy.INSTRUCTIVE_GUIDANCE: NetworkIntent.WISDOM_DEFENSE,
            ExecutionStrategy.CORRECTIVE_JUDGMENT: NetworkIntent.JUSTICE_ENFORCEMENT,
            ExecutionStrategy.BALANCED_RESPONSE: NetworkIntent.DIVINE_PROTECTION
        }
    
    async def process_threat_with_intelligence(self, threat_data: Dict[str, Any]) -> SemanticDefenseResponse:
        """
        Process threat with enhanced SSE + SSD intelligence
        
        This method combines real-time semantic analysis with learned patterns
        to provide increasingly intelligent responses over time.
        """
        
        start_time = time.time()
        
        try:
            if self.telemetry_collector and self.telemetry_collector.should_pause_learning():
                self.logger.warning("Resource guardrails triggered - issuing fallback response")
                fallback = self._create_fallback_response(threat_data)
                self._apply_to_device(fallback)
                return fallback

            # Extract threat information
            threat_type = threat_data.get('threat_type', 'unknown')
            source_ip = threat_data.get('source_ip', '0.0.0.0')
            destination_ip = threat_data.get('destination_ip', '0.0.0.0')
            network_context = NetworkContext(threat_data.get('context', 'corporate_enterprise'))
            
            self.logger.info(f"Processing {threat_type} threat from {source_ip}")
            
            # 1. Real-time SSE Analysis
            sse_result = await self._analyze_with_sse(threat_data)
            
            # 2. Query Learned Intelligence (SSD)
            intelligence = await self._query_intelligence(threat_data, sse_result)
            
            # 3. Create Enhanced Response
            enhanced_response = self._create_intelligent_response(threat_data, sse_result, intelligence)
            
            # 4. Store Learning for Future
            await self._store_learning(threat_data, sse_result, enhanced_response)
            
            # 5. Update Metrics
            processing_time = time.time() - start_time
            self._update_metrics(enhanced_response, processing_time)
            self._apply_to_device(enhanced_response)
            
            self.metrics['threats_processed'] += 1
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error processing threat with intelligence: {e}")
            return self._create_fallback_response(threat_data)
    
    async def _analyze_with_sse(self, threat_data: Dict[str, Any]) -> Any:
        """Analyze threat using SSE engine"""
        
        if not self.sse_engine:
            return self._mock_sse_analysis(threat_data)
        
        # Create threat description
        threat_type = threat_data.get('threat_type', 'unknown')
        source_ip = threat_data.get('source_ip', '0.0.0.0')
        description = f"{threat_type} attack from {source_ip}"
        
        # Map to SSE types
        thought_map = {
            'phishing': ThoughtType.EMOTIONAL_EXPRESSION,
            'malware': ThoughtType.MORAL_JUDGMENT,
            'ddos': ThoughtType.EMOTIONAL_EXPRESSION,
            'data_breach': ThoughtType.MORAL_JUDGMENT,
            'insider_threat': ThoughtType.MORAL_JUDGMENT
        }
        
        thought_type = thought_map.get(threat_type, ThoughtType.MORAL_JUDGMENT)
        context_domain = self.context_mappings.get(
            NetworkContext(threat_data.get('context', 'corporate_enterprise')),
            ContextDomain.GENERAL
        )
        
        # Process through SSE
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.sse_engine.transform(description, thought_type, context_domain)
        )
        
        return result
    
    async def _query_intelligence(self, threat_data: Dict[str, Any], sse_result: Any) -> IntelligenceInsight:
        """Query learned intelligence from SSD"""
        
        if not self.ssd_database:
            return self._mock_intelligence_query(threat_data, sse_result)
        
        try:
            # Create semantic signature for querying
            semantic_signature = self._create_semantic_signature(threat_data, sse_result)
            
            # Query similar patterns in database
            similar_patterns = await self._find_similar_patterns(semantic_signature, threat_data)
            
            # Analyze patterns for insights
            intelligence = self._analyze_pattern_intelligence(similar_patterns, threat_data)
            
            return intelligence
            
        except Exception as e:
            self.logger.error(f"Error querying intelligence: {e}")
            return self._mock_intelligence_query(threat_data, sse_result)
    
    def _create_semantic_signature(self, threat_data: Dict[str, Any], sse_result: Any) -> str:
        """Create semantic signature for pattern matching"""
        
        threat_type = threat_data.get('threat_type', 'unknown')
        context = threat_data.get('context', 'unknown')
        
        if hasattr(sse_result, 'context_adjusted_coordinates'):
            coords = sse_result.context_adjusted_coordinates
            coord_str = f"{coords[0]:.2f}_{coords[1]:.2f}_{coords[2]:.2f}_{coords[3]:.2f}"
        else:
            coord_str = "0.50_0.50_0.50_0.50"
        
        signature_data = f"{threat_type}_{context}_{coord_str}"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:16]
    
    async def _find_similar_patterns(self, semantic_signature: str, threat_data: Dict[str, Any]) -> List[LearnedPattern]:
        """Find similar threat patterns in memory"""
        
        similar_patterns = []
        
        # Query database for similar patterns (mock implementation for demo)
        for i in range(3):  # Find up to 3 similar patterns
            pattern = LearnedPattern(
                pattern_id=f"pattern_{i}",
                threat_type=threat_data.get('threat_type', 'unknown'),
                semantic_signature=semantic_signature,
                success_rate=70 + (i * 10),  # Mock success rates
                total_occurrences=5 + i,
                successful_strategies=['justice_enforcement', 'balanced_response'],
                context_domains=[threat_data.get('context', 'unknown')],
                biblical_effectiveness={
                    'mercy': 0.7 + (i * 0.1),
                    'justice': 0.8 - (i * 0.05),
                    'wisdom': 0.6 + (i * 0.15)
                },
                confidence_level=0.6 + (i * 0.1),
                last_updated=datetime.now() - timedelta(days=i+1),
                wisdom_insights=[f"Learned insight {i}", f"Biblical wisdom {i}"]
            )
            similar_patterns.append(pattern)
        
        return similar_patterns
    
    def _analyze_pattern_intelligence(self, patterns: List[LearnedPattern], threat_data: Dict[str, Any]) -> IntelligenceInsight:
        """Analyze patterns to derive intelligence insights"""
        
        if not patterns:
            return IntelligenceInsight(
                similar_threats=0,
                pattern_confidence=0.5,
                recommended_strategy='balanced_response',
                biblical_application='Standard wisdom approach',
                learning_confidence=0.5,
                historical_success=50.0,
                predictive_accuracy=50.0,
                wisdom_summary='No historical patterns available'
            )
        
        # Calculate weighted intelligence from patterns
        total_occurrences = sum(p.total_occurrences for p in patterns)
        avg_success_rate = sum(p.success_rate * p.total_occurrences for p in patterns) / total_occurrences
        avg_confidence = sum(p.confidence_level * p.total_occurrences for p in patterns) / total_occurrences
        
        # Determine best strategy
        strategy_counts = defaultdict(int)
        for pattern in patterns:
            for strategy in pattern.successful_strategies:
                strategy_counts[strategy] += pattern.total_occurrences
        
        best_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate biblical effectiveness
        biblical_effectiveness = {}
        for aspect in ['mercy', 'justice', 'wisdom']:
            effectiveness = sum(
                p.biblical_effectiveness.get(aspect, 0.5) * p.total_occurrences 
                for p in patterns
            ) / total_occurrences
            biblical_effectiveness[aspect] = effectiveness
        
        # Determine best biblical application
        if biblical_effectiveness['justice'] > 0.7:
            biblical_app = "Strong justice emphasis with righteous judgment"
        elif biblical_effectiveness['mercy'] > 0.7:
            biblical_app = "Compassionate approach with abundant mercy"
        elif biblical_effectiveness['wisdom'] > 0.7:
            biblical_app = "Wisdom-led strategy with discerning action"
        else:
            biblical_app = "Balanced biblical approach with harmony"
        
        # Generate wisdom summary
        dominant_themes = []
        if avg_success_rate > 80:
            dominant_themes.append("High success patterns detected")
        if avg_confidence > 0.7:
            dominant_themes.append("Strong confidence in approach")
        if total_occurrences > 20:
            dominant_themes.append("Extensive experience base")
        
        wisdom_summary = "; ".join(dominant_themes) if dominant_themes else "Learning patterns developing"
        
        return IntelligenceInsight(
            similar_threats=len(patterns),
            pattern_confidence=avg_confidence,
            recommended_strategy=best_strategy,
            biblical_application=biblical_app,
            learning_confidence=avg_confidence,
            historical_success=avg_success_rate,
            predictive_accuracy=min(95.0, avg_success_rate + len(patterns) * 2),
            wisdom_summary=wisdom_summary
        )
    
    def _create_intelligent_response(self, threat_data: Dict[str, Any], sse_result: Any, intelligence: IntelligenceInsight) -> SemanticDefenseResponse:
        """Create response enhanced with learned intelligence"""
        
        # Get base strategy from SSE
        if hasattr(sse_result, 'execution_strategy'):
            sse_strategy = sse_result.execution_strategy.value
        else:
            sse_strategy = 'balanced_response'
        
        # Enhance with intelligence
        historical_success = intelligence.historical_success
        
        if historical_success > 85:
            # High historical success - use learned best strategy
            final_strategy = intelligence.recommended_strategy
            confidence = 0.9
            learning_applied = "Applied high-success learned strategy"
        elif historical_success > 70:
            # Moderate success - blend SSE with learning
            final_strategy = self._blend_strategies(sse_strategy, intelligence.recommended_strategy)
            confidence = 0.8
            learning_applied = "Blended SSE analysis with learned patterns"
        elif historical_success < 50:
            # Low success - adjust based on learning
            final_strategy = self._adjust_strategy_for_improvement(sse_strategy, intelligence)
            confidence = 0.6
            learning_applied = "Strategy adjusted based on learning"
        else:
            # Use SSE with learning validation
            final_strategy = sse_strategy
            confidence = 0.75
            learning_applied = "SSE strategy validated by learning"
        
        # Create enhanced biblical justification
        biblical_justification = self._create_enhanced_biblical_justification(
            final_strategy, intelligence.biblical_application
        )
        
        # Generate network actions based on enhanced strategy
        actions = self._generate_enhanced_actions(final_strategy, threat_data, intelligence)
        
        # Create threat vector
        threat_vector = SemanticThreatVector(
            source_ip=threat_data.get('source_ip', '0.0.0.0'),
            destination_ip=threat_data.get('destination_ip', '0.0.0.0'),
            protocol=threat_data.get('protocol', 'TCP'),
            port=threat_data.get('port', 80),
            payload_size=threat_data.get('payload_size', 0),
            timestamp=datetime.now(),
            semantic_coordinates=self._convert_sse_coordinates(sse_result),
            threat_intent=NetworkIntent.DIVINE_PROTECTION,
            threat_context=NetworkContext(threat_data.get('context', 'corporate_enterprise')),
            divine_threat_level=0.7,  # Based on intelligence
            semantic_signature=self._create_semantic_signature(threat_data, sse_result),
            biblical_threat_type=self._map_threat_type_biblical(threat_data.get('threat_type', 'unknown')),
            justice_requirement=self._calculate_justice_requirement(intelligence),
            wisdom_response=self._calculate_wisdom_response(intelligence),
            intent_alignment=confidence,
            context_resonance=intelligence.pattern_confidence,
            execution_priority=intelligence.predictive_accuracy / 100
        )
        
        # Create defense response
        response = SemanticDefenseResponse(
            response_id=f"intelligent_{datetime.now().timestamp()}",
            threat_vector=threat_vector,
            defense_strategy=NetworkIntent[final_strategy.upper()],
            execution_mode=NetworkExecutionMode.ACTIVE_PROTECTION,
            
            # Enhanced parameters
            divine_protection_level=confidence,
            wisdom_accuracy=intelligence.learning_confidence,
            justice_enforcement=self._get_justice_level(final_strategy, intelligence),
            love_mercy_factor=self._get_mercy_level(final_strategy, intelligence),
            
            # Learning-enhanced metadata
            response_signature=f"{final_strategy}_{confidence:.2f}_{intelligence.similar_threats}",
            biblical_justification=biblical_justification,
            psalms_reference=self._get_psalms_reference(final_strategy),
            
            # Network operations
            blocking_rules=actions['blocking'],
            routing_modifications=actions['routing'],
            quarantine_actions=actions['quarantine'],
            healing_protocols=actions['healing']
        )
        
        # Store learning metadata
        response.learning_metadata = {
            'intelligence_applied': learning_applied,
            'historical_success': historical_success,
            'pattern_confidence': intelligence.pattern_confidence,
            'similar_threats': intelligence.similar_threats,
            'wisdom_summary': intelligence.wisdom_summary
        }
        
        return response
    
    def _blend_strategies(self, sse_strategy: str, learned_strategy: str) -> str:
        """Blend SSE strategy with learned best strategy"""
        
        # Simple blending logic - in production would use more sophisticated approach
        if learned_strategy in ['justice_enforcement', 'corrective_judgment']:
            return learned_strategy  # Prioritize strong learned strategies
        elif sse_strategy == learned_strategy:
            return sse_strategy
        else:
            return 'balanced_response'  # Default to balanced when strategies differ
    
    def _adjust_strategy_for_improvement(self, sse_strategy: str, intelligence: IntelligenceInsight) -> str:
        """Adjust strategy based on learning insights"""
        
        # If current approach has low success, try alternative
        if sse_strategy == 'compassionate_action':
            return 'balanced_response'
        elif sse_strategy == 'authoritative_command':
            return 'corrective_judgment'
        else:
            return intelligence.recommended_strategy
    
    def _create_enhanced_biblical_justification(self, strategy: str, learning_application: str) -> str:
        """Create biblical justification enhanced with learning"""
        
        base_justifications = {
            'justice_enforcement': "Execute justice and righteousness, for the LORD your God will judge everyone (Psalm 106:3)",
            'compassionate_action': "Be merciful, even as your Father is merciful (Luke 6:36)",
            'corrective_judgment': "Correct the wise and they will love you (Proverbs 9:8)",
            'instructive_guidance': "Teach me knowledge and good judgment (Psalm 119:66)",
            'balanced_response': "There is a time for everything under heaven (Ecclesiastes 3:1)",
            'love_shield': "Above all, love each other deeply (1 Peter 4:8)",
            'wisdom_defense': "The wisdom of the prudent is to give thought to their ways (Proverbs 14:8)"
        }
        
        base = base_justifications.get(strategy, "Trust in the Lord with all your heart (Proverbs 3:5)")
        
        # Add learning insight
        enhanced = f"{base} | {learning_application}"
        
        return enhanced
    
    def _generate_enhanced_actions(self, strategy: str, threat_data: Dict[str, Any], intelligence: IntelligenceInsight) -> Dict[str, List]:
        """Generate network actions enhanced with intelligence"""
        
        actions = {
            'blocking': [],
            'routing': {},
            'quarantine': [],
            'healing': []
        }
        
        source_ip = threat_data.get('source_ip', 'unknown')
        
        # Base actions by strategy
        if strategy in ['justice_enforcement', 'corrective_judgment', 'holiness_separation']:
            # Strong enforcement actions
            actions['blocking'].append({
                'action': 'block',
                'source_ip': source_ip,
                'justification': f"High-threat {threat_data.get('threat_type', 'attack')} based on {intelligence.similar_threats} similar patterns"
            })
            
            actions['quarantine'].extend([
                f"Quarantine {source_ip} for investigation",
                "Immediate forensic analysis"
            ])
            
            if intelligence.historical_success > 80:
                actions['blocking'][0]['confidence'] = "high"
            
        elif strategy in ['compassionate_action', 'love_shield']:
            # Compassionate approach
            actions['routing'] = {
                'action': 'monitor_enhanced',
                'source_ip': source_ip,
                'monitoring_level': 'high'
            }
            
            actions['quarantine'].append(f"Enhanced monitoring of {source_ip}")
            actions['healing'].extend([
                "Educational resources deployment",
                "Security awareness enhancement"
            ])
            
        else:  # balanced_response, instructive_guidance, wisdom_defense
            # Balanced approach
            if intelligence.pattern_confidence > 0.7:
                actions['blocking'].append({
                    'action': 'conditional_block',
                    'source_ip': source_ip,
                    'conditions': ['threshold_exceeded', 'pattern_match'],
                    'justification': "Pattern-based conditional blocking"
                })
            else:
                actions['routing'] = {
                    'action': 'adaptive_filter',
                    'source_ip': source_ip,
                    'filter_level': 'medium'
                }
            
            actions['quarantine'].append(f"Adaptive monitoring of {source_ip}")
            actions['healing'].append("Contextual security training")
        
        # Add intelligence-enhanced actions
        if intelligence.similar_threats > 5:
            actions['quarantine'].append("Pattern analysis review")
        
        if intelligence.predictive_accuracy > 80:
            actions['routing']['proactive_defense'] = True
        
        return actions
    
    async def _store_learning(self, threat_data: Dict[str, Any], sse_result: Any, response: SemanticDefenseResponse):
        """Store learning for future intelligence"""
        
        if not self.ssd_database:
            self._mock_store_learning(threat_data, response)
            return
        
        try:
            # Create learning record
            learning_record = {
                'timestamp': datetime.now(),
                'threat_data': threat_data,
                'sse_analysis': {
                    'strategy': getattr(sse_result, 'execution_strategy', {}).value if hasattr(sse_result, 'execution_strategy') else 'unknown',
                    'divine_alignment': getattr(sse_result, 'divine_alignment', 0.5),
                    'coordinates': getattr(sse_result, 'context_adjusted_coordinates', (0.5, 0.5, 0.5, 0.5))
                },
                'response_intelligence': {
                    'strategy': response.defense_strategy.value,
                    'confidence': response.divine_protection_level,
                    'biblical_justification': response.biblical_justification,
                    'learning_applied': getattr(response, 'learning_metadata', {}).get('intelligence_applied', 'Unknown')
                },
                'success_prediction': getattr(response, 'learning_metadata', {}).get('historical_success', 50.0)
            }
            
            semantic_signature = self._create_semantic_signature(threat_data, sse_result)
            if semantic_signature not in self.pattern_memory:
                self.pattern_memory[semantic_signature] = []

            pattern_entry = {
                'timestamp': learning_record['timestamp'].isoformat(),
                'strategy': response.defense_strategy.value,
                'confidence': response.divine_protection_level,
                'wisdom': response.wisdom_accuracy,
                'success': response.divine_protection_level >= 0.6
            }
            self.pattern_memory[semantic_signature].append(pattern_entry)
            if len(self.pattern_memory[semantic_signature]) > 100:
                self.pattern_memory[semantic_signature] = self.pattern_memory[semantic_signature][-100:]

            self.confidence_tracker[semantic_signature].append(response.divine_protection_level)
            if len(self.confidence_tracker[semantic_signature]) > 100:
                self.confidence_tracker[semantic_signature] = self.confidence_tracker[semantic_signature][-100:]

            self.learning_history.append(learning_record)
            self.metrics['patterns_learned'] = len(self.pattern_memory)

            self._persist_learning_record(semantic_signature, threat_data, learning_record, response)
            
            self.logger.info(f"Learning stored: {semantic_signature}")
            
        except Exception as e:
            self.logger.error(f"Error storing learning: {e}")
    
    def _convert_sse_coordinates(self, sse_result: Any) -> SemanticCoordinates:
        """Convert SSE coordinates to biblical coordinates"""
        
        if hasattr(sse_result, 'context_adjusted_coordinates'):
            coords = sse_result.context_adjusted_coordinates
            return SemanticCoordinates(
                love=coords[0],
                power=coords[1],
                wisdom=coords[2],
                justice=coords[3]
            )
        else:
            return SemanticCoordinates(0.5, 0.5, 0.5, 0.5)
    
    def _map_threat_type_biblical(self, threat_type: str) -> str:
        """Map threat types to biblical categories"""
        
        biblical_mappings = {
            'phishing': 'deception_false_witness',
            'malware': 'corruption_spiritual_defilement',
            'ddos': 'overwhelming_force_intimidation',
            'data_breach': 'theft_stolen_treasure',
            'insider_threat': 'betrayal_judas_threat',
            'unknown': 'spiritual_attack'
        }
        
        return biblical_mappings.get(threat_type, 'spiritual_attack')
    
    def _calculate_justice_requirement(self, intelligence: IntelligenceInsight) -> float:
        """Calculate justice requirement based on intelligence"""
        
        base_justice = 0.7
        intelligence_boost = (intelligence.historical_success / 100) * 0.2
        pattern_boost = intelligence.pattern_confidence * 0.1
        
        return min(1.0, base_justice + intelligence_boost + pattern_boost)
    
    def _calculate_wisdom_response(self, intelligence: IntelligenceInsight) -> float:
        """Calculate wisdom response based on intelligence"""
        
        base_wisdom = 0.6
        learning_wisdom = intelligence.learning_confidence * 0.3
        predictive_wisdom = (intelligence.predictive_accuracy / 100) * 0.1
        
        return min(1.0, base_wisdom + learning_wisdom + predictive_wisdom)
    
    def _get_justice_level(self, strategy: str, intelligence: IntelligenceInsight) -> float:
        """Get justice level for strategy"""
        
        justice_levels = {
            'justice_enforcement': 0.9,
            'corrective_judgment': 0.85,
            'holiness_separation': 0.95,
            'balanced_response': 0.7,
            'instructive_guidance': 0.6,
            'wisdom_defense': 0.5,
            'love_shield': 0.4,
            'compassionate_action': 0.3
        }
        
        base_justice = justice_levels.get(strategy, 0.7)
        
        # Adjust based on intelligence
        if intelligence.historical_success > 80:
            base_justice += 0.05
        
        return min(1.0, base_justice)
    
    def _get_mercy_level(self, strategy: str, intelligence: IntelligenceInsight) -> float:
        """Get mercy level for strategy"""
        
        mercy_levels = {
            'compassionate_action': 0.9,
            'love_shield': 0.85,
            'balanced_response': 0.7,
            'instructive_guidance': 0.6,
            'wisdom_defense': 0.5,
            'justice_enforcement': 0.3,
            'corrective_judgment': 0.4,
            'holiness_separation': 0.2
        }
        
        base_mercy = mercy_levels.get(strategy, 0.7)
        
        # Adjust based on learning
        if 'mercy' in intelligence.biblical_application.lower():
            base_mercy += 0.1
        
        return min(1.0, base_mercy)
    
    def _get_psalms_reference(self, strategy: str) -> str:
        """Get Psalms reference for strategy"""
        
        psalms_references = {
            'justice_enforcement': "Psalm 75:7 - It is God who judges",
            'compassionate_action': "Psalm 86:5 - You are forgiving and good",
            'corrective_judgment': "Psalm 25:9 - He guides the humble in what is right",
            'instructive_guidance': "Psalm 119:105 - Your word is a lamp for my feet",
            'balanced_response': "Psalm 23:4 - Your rod and staff comfort me",
            'love_shield': "Psalm 91:4 - He will cover you with his feathers",
            'wisdom_defense': "Psalm 111:10 - The fear of the Lord is the beginning of wisdom",
            'holiness_separation': "Psalm 4:8 - In peace I will lie down and sleep"
        }
        
        return psalms_references.get(strategy, "Psalm 46:1 - God is our refuge and strength")
    
    def _update_metrics(self, response: SemanticDefenseResponse, processing_time: float):
        """Update performance metrics"""
        
        # Track confidence growth
        self.confidence_tracker['confidence_levels'].append(response.divine_protection_level)
        
        # Track biblical applications
        if response.biblical_justification:
            self.metrics['biblical_wisdom_applied'] += 1
        
        # Calculate accuracy improvement (mock for demo)
        if hasattr(response, 'learning_metadata'):
            historical_success = response.learning_metadata.get('historical_success', 50.0)
            self.metrics['accuracy_improvement'] = max(0, historical_success - 50.0)
        
        # Calculate confidence growth
        if len(self.confidence_tracker['confidence_levels']) > 1:
            recent_confidence = np.mean(list(self.confidence_tracker['confidence_levels'])[-10:])
            early_confidence = np.mean(list(self.confidence_tracker['confidence_levels'])[:10])
            self.metrics['confidence_growth'] = recent_confidence - early_confidence

        self._persist_defense_history(response, processing_time)

    def _load_persisted_learning(self) -> None:
        """Load persisted learning records and patterns from the SSD database."""
        if not self.ssd_database:
            return

        try:
            cursor = self.ssd_database.conn.cursor()
            tables = [
                row["name"]
                for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]

            if "threat_patterns" in tables:
                cursor.execute("SELECT semantic_signature, pattern_json FROM threat_patterns")
                for row in cursor.fetchall():
                    signature = row["semantic_signature"]
                    pattern_json = row["pattern_json"] or "{}"
                    try:
                        payload = json.loads(pattern_json)
                    except json.JSONDecodeError:
                        payload = {}
                    entries = payload.get("entries", [])
                    self.pattern_memory[signature] = entries
                    for entry in entries:
                        confidence = entry.get("confidence")
                        if confidence is not None:
                            self.confidence_tracker[signature].append(confidence)
                            self.confidence_tracker["confidence_levels"].append(confidence)

            if "learning_history" in tables:
                cursor.execute(
                    "SELECT * FROM learning_history ORDER BY id DESC LIMIT ?",
                    (self.learning_history.maxlen or 10000,)
                )
                rows = cursor.fetchall()
                for row in reversed(rows):
                    record_json = row["learning_json"] or "{}"
                    try:
                        record = json.loads(record_json)
                    except json.JSONDecodeError:
                        record = {}
                    timestamp = row["timestamp"]
                    if timestamp:
                        try:
                            record["timestamp"] = datetime.fromisoformat(timestamp)
                        except ValueError:
                            record["timestamp"] = timestamp
                    self.learning_history.append(record)

            if "defense_history" in tables:
                cursor.execute("SELECT COUNT(*) AS total FROM defense_history")
                self.metrics['threats_processed'] = cursor.fetchone()["total"]

            if "teaching_sessions" in tables:
                cursor.execute(
                    "SELECT SUM(CASE WHEN outcome='failure' THEN 1 ELSE 0 END) AS failures FROM teaching_sessions"
                )
                row = cursor.fetchone()
                failures = row["failures"] if row else None
                if failures is not None:
                    self.metrics['learning_corrections'] = failures

            self.metrics['patterns_learned'] = len(self.pattern_memory)

            confidence_values = list(self.confidence_tracker["confidence_levels"])
            if len(confidence_values) > 1:
                recent_confidence = np.mean(confidence_values[-10:])
                early_confidence = np.mean(confidence_values[:10])
                self.metrics['confidence_growth'] = recent_confidence - early_confidence

        except Exception as exc:
            self.logger.error(f"Error loading persisted learning: {exc}")

    def _persist_learning_record(
        self,
        semantic_signature: str,
        threat_data: Dict[str, Any],
        learning_record: Dict[str, Any],
        response: SemanticDefenseResponse
    ) -> None:
        """Persist learning history and threat pattern updates to the SSD."""
        if not self.ssd_database:
            return

        try:
            cursor = self.ssd_database.conn.cursor()
            timestamp_iso = learning_record['timestamp'].isoformat()
            threat_type = threat_data.get('threat_type', 'unknown')
            severity = str(threat_data.get('severity', 'unknown'))
            context = threat_data.get('context', 'unknown')
            learning_json = json.dumps(
                {**learning_record, 'timestamp': timestamp_iso},
                default=str
            )

            cursor.execute(
                """
                INSERT INTO learning_history (
                    timestamp,
                    semantic_signature,
                    threat_type,
                    severity,
                    context,
                    strategy,
                    success_confidence,
                    divine_alignment,
                    success_prediction,
                    learning_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_iso,
                    semantic_signature,
                    threat_type,
                    severity,
                    context,
                    response.defense_strategy.value,
                    response.divine_protection_level,
                    learning_record['sse_analysis'].get('divine_alignment', 0.5),
                    learning_record.get('success_prediction', 50.0),
                    learning_json
                )
            )

            self._update_threat_pattern(cursor, semantic_signature, threat_type, response)
            self.ssd_database.conn.commit()
            self._persisted_updates += 1
            if self.persistence_manager and self._persisted_updates >= self._snapshot_interval:
                try:
                    self.persistence_manager.export_snapshot()
                finally:
                    self._persisted_updates = 0

        except Exception as exc:
            self.logger.error(f"Error persisting learning record: {exc}")

    def _update_threat_pattern(
        self,
        cursor,
        semantic_signature: str,
        threat_type: str,
        response: SemanticDefenseResponse
    ) -> None:
        """Insert or update the persisted threat pattern."""
        entries = self.pattern_memory.get(semantic_signature, [])
        occurrences = len(entries)
        success_rate = 0.0
        if occurrences:
            success_rate = sum(1 for entry in entries if entry.get('success')) / occurrences

        pattern_payload = {
            'threat_type': threat_type,
            'entries': entries[-50:],
            'last_response': {
                'strategy': response.defense_strategy.value,
                'confidence': response.divine_protection_level,
                'wisdom': response.wisdom_accuracy
            }
        }

        cursor.execute(
            "SELECT id FROM threat_patterns WHERE semantic_signature = ?",
            (semantic_signature,)
        )
        row = cursor.fetchone()
        pattern_json = json.dumps(pattern_payload, default=str)
        timestamp_iso = datetime.now().isoformat()

        if row:
            cursor.execute(
                """
                UPDATE threat_patterns
                SET threat_type = ?,
                    occurrences = ?,
                    success_rate = ?,
                    last_updated = ?,
                    pattern_json = ?
                WHERE semantic_signature = ?
                """,
                (
                    threat_type,
                    occurrences,
                    success_rate,
                    timestamp_iso,
                    pattern_json,
                    semantic_signature
                )
            )
        else:
            cursor.execute(
                """
                INSERT INTO threat_patterns (
                    semantic_signature,
                    threat_type,
                    occurrences,
                    success_rate,
                    last_updated,
                    pattern_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    semantic_signature,
                    threat_type,
                    occurrences,
                    success_rate,
                    timestamp_iso,
                    pattern_json
                )
            )

    def _persist_defense_history(self, response: SemanticDefenseResponse, processing_time: float) -> None:
        """Persist defense execution metrics to the SSD."""
        if not self.ssd_database:
            return

        try:
            cursor = self.ssd_database.conn.cursor()
            threat_vector = getattr(response, 'threat_vector', None)
            semantic_signature = getattr(threat_vector, 'semantic_signature', None)
            threat_type = getattr(threat_vector, 'biblical_threat_type', None)

            cursor.execute(
                """
                INSERT INTO defense_history (
                    timestamp,
                    semantic_signature,
                    threat_type,
                    strategy,
                    success,
                    processing_time,
                    divine_protection,
                    wisdom_accuracy,
                    justice_enforcement,
                    mercy_factor,
                    response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    semantic_signature,
                    threat_type,
                    response.defense_strategy.value,
                    1 if response.divine_protection_level >= 0.6 else 0,
                    processing_time,
                    response.divine_protection_level,
                    response.wisdom_accuracy,
                    getattr(response, 'justice_enforcement', None),
                    getattr(response, 'love_mercy_factor', None),
                    json.dumps(self._serialize_response(response), default=str)
                )
            )

            self.ssd_database.conn.commit()
        except Exception as exc:
            self.logger.error(f"Error persisting defense history: {exc}")

    async def record_teaching_session(self, session: Dict[str, Any]) -> None:
        """Persist a teaching session for long-term learning analysis."""
        if not self.ssd_database:
            return

        try:
            cursor = self.ssd_database.conn.cursor()
            session_copy = dict(session)
            timestamp = session_copy.get('timestamp', datetime.now())
            if isinstance(timestamp, datetime):
                timestamp_iso = timestamp.isoformat()
            else:
                timestamp_iso = str(timestamp)
            session_copy['timestamp'] = timestamp_iso

            teachings_json = json.dumps(session_copy, default=str)

            cursor.execute(
                """
                INSERT INTO teaching_sessions (
                    timestamp,
                    attack_type,
                    severity,
                    outcome,
                    response_strategy,
                    teachings_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_iso,
                    session_copy.get('attack_type'),
                    session_copy.get('severity'),
                    session_copy.get('outcome'),
                    session_copy.get('response_strategy'),
                    teachings_json
                )
            )

            if session_copy.get('outcome') == 'failure':
                self.metrics['learning_corrections'] += 1

            self.ssd_database.conn.commit()
        except Exception as exc:
            self.logger.error(f"Error persisting teaching session: {exc}")

    def _serialize_response(self, response: SemanticDefenseResponse) -> Dict[str, Any]:
        """Create a JSON-serializable snapshot of a defense response."""
        data: Dict[str, Any] = {
            'response_id': getattr(response, 'response_id', None),
            'defense_strategy': getattr(response.defense_strategy, 'value', response.defense_strategy),
            'execution_mode': getattr(response.execution_mode, 'value', response.execution_mode),
            'divine_protection_level': getattr(response, 'divine_protection_level', None),
            'wisdom_accuracy': getattr(response, 'wisdom_accuracy', None),
            'justice_enforcement': getattr(response, 'justice_enforcement', None),
            'love_mercy_factor': getattr(response, 'love_mercy_factor', None),
            'biblical_justification': getattr(response, 'biblical_justification', None),
            'psalms_reference': getattr(response, 'psalms_reference', None),
            'blocking_rules': getattr(response, 'blocking_rules', []),
            'routing_modifications': getattr(response, 'routing_modifications', {}),
            'quarantine_actions': getattr(response, 'quarantine_actions', []),
            'healing_protocols': getattr(response, 'healing_protocols', []),
            'learning_metadata': getattr(response, 'learning_metadata', {})
        }

        threat_vector = getattr(response, 'threat_vector', None)
        if threat_vector:
            data['threat_vector'] = {
                'source_ip': getattr(threat_vector, 'source_ip', None),
                'destination_ip': getattr(threat_vector, 'destination_ip', None),
                'protocol': getattr(threat_vector, 'protocol', None),
                'port': getattr(threat_vector, 'port', None),
                'payload_size': getattr(threat_vector, 'payload_size', None),
                'timestamp': getattr(threat_vector, 'timestamp', None).isoformat()
                if isinstance(getattr(threat_vector, 'timestamp', None), datetime) else str(getattr(threat_vector, 'timestamp', None)),
                'semantic_signature': getattr(threat_vector, 'semantic_signature', None),
                'biblical_threat_type': getattr(threat_vector, 'biblical_threat_type', None),
                'threat_intent': getattr(threat_vector.threat_intent, 'value', getattr(threat_vector, 'threat_intent', None)),
                'threat_context': getattr(threat_vector.threat_context, 'value', getattr(threat_vector, 'threat_context', None)),
                'divine_threat_level': getattr(threat_vector, 'divine_threat_level', None),
                'justice_requirement': getattr(threat_vector, 'justice_requirement', None),
                'wisdom_response': getattr(threat_vector, 'wisdom_response', None),
                'semantic_coordinates': getattr(threat_vector.semantic_coordinates, 'to_tuple', lambda: None)()
                if getattr(threat_vector, 'semantic_coordinates', None) else None
            }

        return data

    # ------------------------------------------------------------------ #
    # Integration helpers                                                #
    # ------------------------------------------------------------------ #

    def set_device_interface(self, interface: Optional[FortiGatePolicyApplier]) -> None:
        self.device_interface = interface

    def set_telemetry_collector(self, collector: Optional[FortiGateTelemetryCollector]) -> None:
        self.telemetry_collector = collector

    def set_persistence_manager(self, manager: Optional[LearningPersistenceManager]) -> None:
        self.persistence_manager = manager

    def set_snapshot_interval(self, interval: int) -> None:
        if interval > 0:
            self._snapshot_interval = interval

    def update_telemetry(self, metrics: Dict[str, Any]) -> None:
        if self.telemetry_collector:
            self.telemetry_collector.update_metrics(metrics)

    def export_learning_snapshot(self) -> Optional[str]:
        if not self.persistence_manager:
            self.logger.warning("No persistence manager configured for snapshots")
            return None
        return self.persistence_manager.export_snapshot()

    def _apply_to_device(self, response: SemanticDefenseResponse) -> None:
        if not self.device_interface:
            return
        try:
            self.device_interface.apply_response(response)
        except Exception as exc:
            self.logger.error("Error applying response to FortiGate: %s", exc)
    
    def _mock_sse_analysis(self, threat_data: Dict[str, Any]) -> Any:
        """Mock SSE analysis when engine not available"""
        
        class MockSSEResult:
            def __init__(self, threat_data):
                self.execution_strategy = ExecutionStrategy.BALANCED_RESPONSE
                self.divine_alignment = 0.5 + (hash(threat_data.get('source_ip', '')) % 50) / 100
                self.semantic_integrity = 0.95
                self.context_adjusted_coordinates = (
                    0.4 + (hash(threat_data.get('threat_type', '')) % 30) / 100,
                    0.5 + (hash(threat_data.get('context', '')) % 40) / 100,
                    0.6 + (hash(threat_data.get('protocol', '')) % 30) / 100,
                    0.7 + (hash(threat_data.get('source_ip', '')) % 20) / 100
                )
        
        return MockSSEResult(threat_data)
    
    def _mock_intelligence_query(self, threat_data: Dict[str, Any], sse_result: Any) -> IntelligenceInsight:
        """Mock intelligence query when database not available"""
        
        import random
        
        similar_threats = random.randint(0, 10)
        pattern_confidence = 0.5 + random.random() * 0.4
        historical_success = 60 + random.randint(-20, 30)
        
        if historical_success > 80:
            recommended_strategy = random.choice(['justice_enforcement', 'balanced_response'])
            biblical_app = "High-success pattern with strong biblical alignment"
        elif historical_success > 60:
            recommended_strategy = 'balanced_response'
            biblical_app = "Moderate success with balanced approach"
        else:
            recommended_strategy = 'compassionate_action'
            biblical_app = "Learning phase with merciful approach"
        
        return IntelligenceInsight(
            similar_threats=similar_threats,
            pattern_confidence=pattern_confidence,
            recommended_strategy=recommended_strategy,
            biblical_application=biblical_app,
            learning_confidence=pattern_confidence,
            historical_success=historical_success,
            predictive_accuracy=min(95.0, historical_success + random.randint(0, 10)),
            wisdom_summary=f"Mock intelligence with {similar_threats} similar patterns"
        )
    
    def _mock_store_learning(self, threat_data: Dict[str, Any], response: SemanticDefenseResponse):
        """Mock learning storage when database not available"""
        
        learning_id = f"mock_learning_{datetime.now().timestamp()}"
        self.logger.info(f"Mock learning stored: {learning_id}")
    
    def _create_fallback_response(self, threat_data: Dict[str, Any]) -> SemanticDefenseResponse:
        """Create fallback response when processing fails"""
        
        return SemanticDefenseResponse(
            response_id="fallback_intelligent_response",
            threat_vector=SemanticThreatVector(
                source_ip=threat_data.get('source_ip', '0.0.0.0'),
                destination_ip=threat_data.get('destination_ip', '0.0.0.0'),
                protocol=threat_data.get('protocol', 'TCP'),
                port=threat_data.get('port', 80),
                payload_size=threat_data.get('payload_size', 0),
                timestamp=datetime.now(),
                semantic_coordinates=SemanticCoordinates(0.5, 0.5, 0.5, 0.5),
                threat_intent=NetworkIntent.DIVINE_PROTECTION,
                threat_context=NetworkContext.CORPORATE_ENTERPRISE,
                divine_threat_level=0.5,
                semantic_signature="fallback_signature",
                biblical_threat_type="unknown",
                justice_requirement=0.5,
                wisdom_response=0.5,
                intent_alignment=0.5,
                context_resonance=0.5,
                execution_priority=0.5
            ),
            defense_strategy=NetworkIntent.DIVINE_PROTECTION,
            execution_mode=NetworkExecutionMode.ACTIVE_PROTECTION,
            divine_protection_level=0.5,
            wisdom_accuracy=0.5,
            justice_enforcement=0.5,
            love_mercy_factor=0.5,
            response_signature="fallback_intelligent",
            biblical_justification="The Lord is my rock and my fortress (Psalm 18:2) | Fallback protection",
            psalms_reference="Psalm 46:1 - God is our refuge and strength",
            blocking_rules=[{'action': 'monitor', 'source_ip': threat_data.get('source_ip', 'unknown')}],
            routing_modifications={'action': 'fallback_routing'},
            quarantine_actions=['Monitor threat'],
            healing_protocols=['Basic security protocol']
        )
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive intelligence metrics"""
        
        base_metrics = {
            'threats_processed': self.metrics['threats_processed'],
            'patterns_learned': self.metrics['patterns_learned'],
            'accuracy_improvement': self.metrics['accuracy_improvement'],
            'confidence_growth': self.metrics['confidence_growth'],
            'biblical_wisdom_applied': self.metrics['biblical_wisdom_applied'],
            'learning_corrections': self.metrics['learning_corrections']
        }
        
        # Calculate intelligence growth
        if len(self.learning_history) > 0:
            recent_success = np.mean([h['success_prediction'] for h in list(self.learning_history)[-10:]])
            base_metrics['recent_success_rate'] = recent_success
            
            # Learning velocity (threats per hour in last 24h)
            recent_time = datetime.now() - timedelta(hours=24)
            recent_learning = [h for h in self.learning_history if h['timestamp'] > recent_time]
            base_metrics['learning_velocity'] = len(recent_learning)
        
        # Pattern effectiveness
        if self.pattern_memory:
            total_patterns = len(self.pattern_memory)
            active_patterns = sum(1 for patterns in self.pattern_memory.values() 
                                if patterns and (datetime.now() - patterns[-1]['timestamp']).days < 7)
            base_metrics['pattern_effectiveness'] = active_patterns / total_patterns if total_patterns > 0 else 0
        
        return base_metrics

# Export primary symbols
__all__ = ['FortiGateSemanticShield', 'IntelligenceInsight', 'LearnedPattern']
