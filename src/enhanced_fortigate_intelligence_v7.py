"""
FortiGate Semantic Shield v7.0 - Enterprise Intelligence Integration
====================================================================

Advanced cybersecurity intelligence system with enhanced semantic reasoning,
predictive analytics, and business-aligned decision making.

Author: FortiGate Semantic Shield Team
License: Enterprise License
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
SUBSTRATE_SRC = PACKAGE_ROOT / "semantic_substrate_engine" / "Semantic-Substrate-Engine-main" / "src"
DATABASE_SRC = PACKAGE_ROOT / "semantic_substrate_database" / "Semantic-Substrate-Database-main" / "src"

for candidate in (SUBSTRATE_SRC, DATABASE_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

# Import enhanced semantic components
from enhanced_ice_framework import (
    BusinessIntent,
    EnhancedICEFramework,
    ExecutionStrategy,
    ICEProcessingResult,
    enhanced_ice,
    initialize_enhanced_ice,
)
from enterprise_semantic_database import (
    AsyncSemanticDatabase,
    BusinessContext,
    LearningMode,
    LearningPattern,
    SemanticSignature,
    initialize_semantic_database,
    semantic_db,
)
from cardinal_semantic_axioms import (
    BusinessSemanticMapping,
    CardinalAxiom,
    JEHOVAH_ANCHOR,
    SemanticVector,
    create_divine_anchor_vector,
)
from advanced_semantic_mathematics import (
    Vector4D,
    advanced_math,
    compute_business_maturity,
    compute_semantic_alignment,
    create_business_vector,
)

# Import existing FortiGate components
from fortigate_semantic_shield.device_interface import FortiGateAPIConfig, FortiGateTelemetryCollector
from fortigate_semantic_shield.semantic_components import (
    NetworkIntent, NetworkContext, ThreatLevel, ResponseAction,
    NetworkPacket, ThreatSignature, SemanticCoordinates
)


class BusinessThreatContext(Enum):
    """Business-aligned threat contexts"""
    FINANCIAL_FRAUD = "financial_fraud"
    DATA_BREACH = "data_breach"
    OPERATIONAL_DISRUPTION = "operational_disruption"
    REPUTATIONAL_DAMAGE = "reputational_damage"
    REGULATORY_VIOLATION = "regulatory_violation"
    INTELLECTUAL_PROPERTY_THEFT = "intellectual_property_theft"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    CUSTOMER_DATA_COMPROMISE = "customer_data_compromise"


@dataclass
class BusinessThreatIntelligence:
    """Enhanced threat intelligence with business context"""
    threat_signature: ThreatSignature
    business_context: BusinessThreatContext
    business_impact_score: float
    compliance_risk: float
    customer_impact: float
    financial_exposure: float
    reputation_risk: float
    regulatory_requirements: List[str]
    mitigation_strategies: List[str]
    business_justification: str
    semantic_signature_id: Optional[str] = None
    processing_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SemanticThreatResponse:
    """Semantic intelligence-enhanced threat response"""
    threat_intelligence: BusinessThreatIntelligence
    ice_processing_result: ICEProcessingResult
    recommended_actions: List[ResponseAction]
    business_rationale: str
    confidence_score: float
    execution_strategy: ExecutionStrategy
    resource_requirements: Dict[str, Any]
    monitoring_requirements: List[str]
    success_metrics: List[str]
    compliance_notes: List[str]


class PredictiveThreatAnalytics:
    """Predictive analytics for threat anticipation"""
    
    def __init__(self, lookback_window: int = 1000):
        self.lookback_window = lookback_window
        self.threat_history = deque(maxlen=lookback_window)
        self.pattern_detector = defaultdict(list)
        self.predictive_models = {}
        self._lock = threading.RLock()
    
    def add_threat_event(self, threat_intel: BusinessThreatIntelligence, 
                        response: SemanticThreatResponse):
        """Add threat event to history"""
        with self._lock:
            event = {
                'timestamp': threat_intel.processing_timestamp,
                'threat_type': threat_intel.threat_signature.threat_type,
                'business_context': threat_intel.business_context,
                'impact_score': threat_intel.business_impact_score,
                'response_time': response.ice_processing_result.processing_time,
                'success': response.confidence_score > 0.7
            }
            self.threat_history.append(event)
            
            # Update pattern detector
            pattern_key = (threat_intel.business_context, threat_intel.threat_signature.threat_type)
            self.pattern_detector[pattern_key].append(event)
    
    def predict_threat_trends(self, time_horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict threat trends for specified time horizon"""
        with self._lock:
            if len(self.threat_history) < 50:
                return {'prediction': 'insufficient_data', 'confidence': 0.0}
            
            # Analyze recent patterns
            recent_events = list(self.threat_history)[-100:]
            
            # Calculate threat frequencies by context
            context_counts = defaultdict(int)
            for event in recent_events:
                context_counts[event['business_context']] += 1
            
            # Identify trending threats
            trends = {}
            for context, count in context_counts.items():
                if count > 5:  # Significant frequency
                    # Calculate trend (simple linear regression on event frequency)
                    trend_score = count / len(recent_events)
                    trends[context.value] = {
                        'frequency': trend_score,
                        'trend': 'increasing' if trend_score > 0.1 else 'stable',
                        'predicted_volume': int(trend_score * time_horizon_hours)
                    }
            
            return {
                'prediction': 'trend_analysis_available',
                'confidence': min(len(recent_events) / 100.0, 1.0),
                'trends': trends,
                'time_horizon_hours': time_horizon_hours,
                'recommendations': self._generate_predictive_recommendations(trends)
            }
    
    def _generate_predictive_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on threat trends"""
        recommendations = []
        
        for context, data in trends.items():
            if data['trend'] == 'increasing' and data['frequency'] > 0.15:
                recommendations.append(f"Increase monitoring for {context}")
                recommendations.append(f"Prepare additional resources for {context}")
        
        if not recommendations:
            recommendations.append("Maintain current threat posture")
        
        return recommendations


class BusinessImpactAnalyzer:
    """Business impact analysis for threat events"""
    
    def __init__(self):
        self.impact_factors = {
            BusinessThreatContext.FINANCIAL_FRAUD: {
                'financial_weight': 0.9,
                'reputation_weight': 0.7,
                'compliance_weight': 0.8,
                'operational_weight': 0.4
            },
            BusinessThreatContext.DATA_BREACH: {
                'financial_weight': 0.6,
                'reputation_weight': 0.9,
                'compliance_weight': 0.8,
                'operational_weight': 0.7
            },
            BusinessThreatContext.OPERATIONAL_DISRUPTION: {
                'financial_weight': 0.8,
                'reputation_weight': 0.5,
                'compliance_weight': 0.3,
                'operational_weight': 0.9
            },
            BusinessThreatContext.REPUTATIONAL_DAMAGE: {
                'financial_weight': 0.5,
                'reputation_weight': 0.9,
                'compliance_weight': 0.4,
                'operational_weight': 0.3
            },
            BusinessThreatContext.REGULATORY_VIOLATION: {
                'financial_weight': 0.7,
                'reputation_weight': 0.6,
                'compliance_weight': 0.9,
                'operational_weight': 0.4
            }
        }
    
    def calculate_business_impact(self, threat_intel: BusinessThreatIntelligence) -> Dict[str, float]:
        """Calculate comprehensive business impact scores"""
        context_factors = self.impact_factors.get(
            threat_intel.business_context,
            {
                'financial_weight': 0.5,
                'reputation_weight': 0.5,
                'compliance_weight': 0.5,
                'operational_weight': 0.5
            }
        )
        
        # Calculate weighted impact scores
        financial_impact = (
            threat_intel.financial_exposure * context_factors['financial_weight'] +
            threat_intel.business_impact_score * 0.3
        )
        
        reputation_impact = (
            threat_intel.reputation_risk * context_factors['reputation_weight'] +
            threat_intel.customer_impact * context_factors['reputation_weight'] * 0.5
        )
        
        compliance_impact = (
            threat_intel.compliance_risk * context_factors['compliance_weight'] +
            len(threat_intel.regulatory_requirements) * 0.1
        )
        
        operational_impact = (
            threat_intel.business_impact_score * context_factors['operational_weight'] +
            threat_intel.customer_impact * context_factors['operational_weight'] * 0.3
        )
        
        # Calculate overall business impact
        overall_impact = (
            financial_impact * 0.3 +
            reputation_impact * 0.25 +
            compliance_impact * 0.25 +
            operational_impact * 0.2
        )
        
        return {
            'financial_impact': min(financial_impact, 1.0),
            'reputation_impact': min(reputation_impact, 1.0),
            'compliance_impact': min(compliance_impact, 1.0),
            'operational_impact': min(operational_impact, 1.0),
            'overall_impact': min(overall_impact, 1.0)
        }


class FortiGateSemanticIntelligence:
    """Enhanced FortiGate intelligence with semantic reasoning"""
    
    def __init__(self, api_config: FortiGateAPIConfig,
                 learning_mode: LearningMode = LearningMode.BALANCED):
        self.api_config = api_config
        self.learning_mode = learning_mode
        
        # Initialize components
        self.predictive_analytics = PredictiveThreatAnalytics()
        self.impact_analyzer = BusinessImpactAnalyzer()
        
        # Performance tracking
        self.performance_metrics = {
            'threats_processed': 0,
            'avg_processing_time': 0.0,
            'success_rate': 0.0,
            'false_positive_rate': 0.0,
            'business_impact_prevented': 0.0
        }
        
        self._lock = threading.RLock()
        self._session = None
    
    async def initialize(self):
        """Initialize the intelligence system"""
        # Initialize semantic components
        await initialize_semantic_database("fortigate_semantic_enterprise.db", self.learning_mode)
        await initialize_enhanced_ice(self.learning_mode)
        
        # Initialize HTTP session for FortiGate API
        self._session = aiohttp.ClientSession()
        
        logging.info("FortiGate Semantic Intelligence initialized successfully")
    
    async def process_threat_intelligently(self, threat_data: Dict[str, Any]) -> SemanticThreatResponse:
        """Process threat with enhanced semantic intelligence"""
        start_time = time.time()
        
        try:
            # 1. Create threat intelligence object
            threat_intel = self._create_threat_intelligence(threat_data)
            
            # 2. Analyze business impact
            impact_analysis = self.impact_analyzer.calculate_business_impact(threat_intel)
            
            # 3. Map to business intent for ICE processing
            business_intent = self._map_threat_to_intent(threat_intel, impact_analysis)
            
            # 4. Create context data for ICE framework
            context_data = {
                'threat_data': threat_data,
                'impact_analysis': impact_analysis,
                'urgency': threat_intel.business_impact_score,
                'risk_tolerance': self._calculate_risk_tolerance(threat_intel),
                'compliance_requirements': threat_intel.regulatory_requirements,
                'business_context': threat_intel.business_context.value
            }
            
            # 5. Process through enhanced ICE framework
            execution_strategy = self._determine_execution_strategy(threat_intel, impact_analysis)
            ice_result = await enhanced_ice.process_intent(
                business_intent, context_data, execution_strategy)
            
            # 6. Generate semantic threat response
            response = self._create_semantic_response(
                threat_intel, ice_result, impact_analysis, execution_strategy)
            
            # 7. Store learning data
            await self._store_learning_data(threat_intel, response)
            
            # 8. Update predictive analytics
            self.predictive_analytics.add_threat_event(threat_intel, response)
            
            # 9. Update performance metrics
            self._update_performance_metrics(response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing threat intelligently: {e}")
            return self._create_fallback_response(threat_data, str(e))
    
    def _create_threat_intelligence(self, threat_data: Dict[str, Any]) -> BusinessThreatIntelligence:
        """Create business threat intelligence from raw threat data"""
        # Extract threat signature
        threat_signature = ThreatSignature(
            threat_type=threat_data.get('threat_type', 'unknown'),
            source_ip=threat_data.get('source_ip', '0.0.0.0'),
            destination_ip=threat_data.get('destination_ip', '0.0.0.0'),
            port=threat_data.get('port', 0),
            protocol=threat_data.get('protocol', 'unknown'),
            payload_hash=threat_data.get('payload_hash', ''),
            severity=threat_data.get('severity', ThreatLevel.MEDIUM),
            confidence=threat_data.get('confidence', 0.5)
        )
        
        # Determine business context
        business_context = self._determine_business_context(threat_data)
        
        # Calculate business impacts
        business_impact_score = self._calculate_business_impact_score(threat_data)
        compliance_risk = self._calculate_compliance_risk(threat_data, business_context)
        customer_impact = self._calculate_customer_impact(threat_data)
        financial_exposure = self._calculate_financial_exposure(threat_data)
        reputation_risk = self._calculate_reputation_risk(threat_data)
        
        # Determine regulatory requirements
        regulatory_requirements = self._get_regulatory_requirements(business_context)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(threat_data, business_context)
        
        # Create business justification
        business_justification = self._create_business_justification(
            threat_data, business_context, business_impact_score)
        
        return BusinessThreatIntelligence(
            threat_signature=threat_signature,
            business_context=business_context,
            business_impact_score=business_impact_score,
            compliance_risk=compliance_risk,
            customer_impact=customer_impact,
            financial_exposure=financial_exposure,
            reputation_risk=reputation_risk,
            regulatory_requirements=regulatory_requirements,
            mitigation_strategies=mitigation_strategies,
            business_justification=business_justification
        )
    
    def _determine_business_context(self, threat_data: Dict[str, Any]) -> BusinessThreatContext:
        """Determine business threat context from threat data"""
        # Rule-based context determination
        if 'financial' in threat_data.get('category', '').lower():
            return BusinessThreatContext.FINANCIAL_FRAUD
        
        if 'data_breach' in threat_data.get('category', '').lower():
            return BusinessThreatContext.DATA_BREACH
        
        if 'customer_data' in threat_data.get('affected_systems', []):
            return BusinessThreatContext.CUSTOMER_DATA_COMPROMISE
        
        if 'critical_infrastructure' in threat_data.get('target_type', '').lower():
            return BusinessThreatContext.CRITICAL_INFRASTRUCTURE
        
        if 'compliance' in threat_data.get('risk_factors', []):
            return BusinessThreatContext.REGULATORY_VIOLATION
        
        # Default based on severity and impact
        if threat_data.get('severity') == ThreatLevel.CRITICAL:
            return BusinessThreatContext.OPERATIONAL_DISRUPTION
        
        return BusinessThreatContext.DATA_BREACH  # Default
    
    def _calculate_business_impact_score(self, threat_data: Dict[str, Any]) -> float:
        """Calculate business impact score (0.0 - 1.0)"""
        factors = {
            'severity': {
                ThreatLevel.LOW: 0.2,
                ThreatLevel.MEDIUM: 0.5,
                ThreatLevel.HIGH: 0.8,
                ThreatLevel.CRITICAL: 1.0
            },
            'confidence': threat_data.get('confidence', 0.5),
            'scope': len(threat_data.get('affected_systems', [])) / 10.0,  # Normalize by 10 systems
            'data_sensitivity': threat_data.get('data_sensitivity', 0.5)
        }
        
        # Weighted calculation
        impact = (
            factors['severity'][threat_data.get('severity', ThreatLevel.MEDIUM)] * 0.4 +
            factors['confidence'] * 0.2 +
            min(factors['scope'], 1.0) * 0.2 +
            factors['data_sensitivity'] * 0.2
        )
        
        return min(impact, 1.0)
    
    def _calculate_compliance_risk(self, threat_data: Dict[str, Any], 
                                  context: BusinessThreatContext) -> float:
        """Calculate compliance risk score"""
        base_risk = {
            BusinessThreatContext.FINANCIAL_FRAUD: 0.9,
            BusinessThreatContext.DATA_BREACH: 0.8,
            BusinessThreatContext.CUSTOMER_DATA_COMPROMISE: 0.9,
            BusinessThreatContext.REGULATORY_VIOLATION: 1.0,
            BusinessThreatContext.CRITICAL_INFRASTRUCTURE: 0.7
        }
        
        return base_risk.get(context, 0.5)
    
    def _calculate_customer_impact(self, threat_data: Dict[str, Any]) -> float:
        """Calculate customer impact score"""
        customer_systems = threat_data.get('affected_systems', [])
        customer_impact_indicators = ['customer_portal', 'api_gateway', 'database', 'auth_service']
        
        impact_score = 0.0
        for system in customer_systems:
            if any(indicator in system.lower() for indicator in customer_impact_indicators):
                impact_score += 0.3
        
        return min(impact_score, 1.0)
    
    def _calculate_financial_exposure(self, threat_data: Dict[str, Any]) -> float:
        """Calculate financial exposure score"""
        # Simplified calculation based on threat type and scope
        base_exposure = {
            'financial_fraud': 0.9,
            'data_breach': 0.7,
            'operational_disruption': 0.8,
            'reputation_damage': 0.6
        }
        
        threat_type = threat_data.get('category', '').lower()
        exposure = base_exposure.get(threat_type, 0.5)
        
        # Adjust for scope
        scope_multiplier = min(len(threat_data.get('affected_systems', [])) / 5.0, 2.0)
        
        return min(exposure * scope_multiplier, 1.0)
    
    def _calculate_reputation_risk(self, threat_data: Dict[str, Any]) -> float:
        """Calculate reputation risk score"""
        public_facing_indicators = ['customer_portal', 'website', 'api_public', 'mobile_app']
        affected_systems = threat_data.get('affected_systems', [])
        
        risk_score = 0.0
        for system in affected_systems:
            if any(indicator in system.lower() for indicator in public_facing_indicators):
                risk_score += 0.4
        
        # Adjust for data sensitivity
        data_sensitivity = threat_data.get('data_sensitivity', 0.5)
        risk_score *= (0.5 + data_sensitivity)
        
        return min(risk_score, 1.0)
    
    def _get_regulatory_requirements(self, context: BusinessThreatContext) -> List[str]:
        """Get applicable regulatory requirements"""
        requirements = {
            BusinessThreatContext.FINANCIAL_FRAUD: ['SOX', 'PCI-DSS', 'GLBA'],
            BusinessThreatContext.DATA_BREACH: ['GDPR', 'CCPA', 'HIPAA', 'PCI-DSS'],
            BusinessThreatContext.CUSTOMER_DATA_COMPROMISE: ['GDPR', 'CCPA', 'HIPAA'],
            BusinessThreatContext.REGULATORY_VIOLATION: ['SOX', 'GDPR', 'industry_specific'],
            BusinessThreatContext.CRITICAL_INFRASTRUCTURE: ['NERC', 'NERC-CIP', 'industry_specific']
        }
        
        return requirements.get(context, ['general_compliance'])
    
    def _generate_mitigation_strategies(self, threat_data: Dict[str, Any], 
                                      context: BusinessThreatContext) -> List[str]:
        """Generate mitigation strategies"""
        strategies = {
            BusinessThreatContext.FINANCIAL_FRAUD: [
                'implement_transaction_monitoring',
                'enhance_authentication_controls',
                'conduct_forensic_analysis',
                'notify_compliance_team'
            ],
            BusinessThreatContext.DATA_BREACH: [
                'isolate_affected_systems',
                'initiate_incident_response',
                'notify_data_protection_officer',
                'assess_regulatory_notification_requirements'
            ],
            BusinessThreatContext.OPERATIONAL_DISRUPTION: [
                'activate_disaster_recovery',
                'engage_incident_response_team',
                'communicate_with_stakeholders',
                'implement_continuity_plan'
            ]
        }
        
        base_strategies = strategies.get(context, ['contain_threat', 'investigate_source'])
        
        # Add threat-specific strategies
        if threat_data.get('threat_type') == 'malware':
            base_strategies.append('quarantine_affected_systems')
            base_strategies.append('conduct_malware_analysis')
        
        return base_strategies
    
    def _create_business_justification(self, threat_data: Dict[str, Any], 
                                     context: BusinessThreatContext, 
                                     impact_score: float) -> str:
        """Create business justification for response actions"""
        justifications = {
            BusinessThreatContext.FINANCIAL_FRAUD: f"Financial fraud threat detected with {impact_score:.1%} business impact. Immediate action required to prevent financial losses and maintain regulatory compliance.",
            BusinessThreatContext.DATA_BREACH: f"Potential data breach with {impact_score:.1%} impact score. Swift response necessary to protect customer data and comply with privacy regulations.",
            BusinessThreatContext.OPERATIONAL_DISRUPTION: f"Operational disruption threat at {impact_score:.1%} impact level. Response critical to maintain business continuity and service delivery.",
            BusinessThreatContext.REGULATORY_VIOLATION: f"Regulatory compliance risk at {impact_score:.1%} impact. Immediate action required to maintain compliance and avoid penalties."
        }
        
        return justifications.get(
            context, 
            f"Security threat detected with {impact_score:.1%} business impact. Response justified by risk assessment and business protection requirements."
        )
    
    def _map_threat_to_intent(self, threat_intel: BusinessThreatIntelligence, 
                            impact_analysis: Dict[str, float]) -> BusinessIntent:
        """Map threat context to business intent"""
        # Map based on primary impact area
        max_impact = max(impact_analysis.items(), key=lambda x: x[1])
        
        impact_to_intent = {
            'financial_impact': BusinessIntent.FINANCIAL_STABILITY,
            'reputation_impact': BusinessIntent.CUSTOMER_SUCCESS,
            'compliance_impact': BusinessIntent.REGULATORY_COMPLIANCE,
            'operational_impact': BusinessIntent.OPERATIONAL_EXCELLENCE
        }
        
        return impact_to_intent.get(max_impact[0], BusinessIntent.RISK_MITIGATION)
    
    def _calculate_risk_tolerance(self, threat_intel: BusinessThreatIntelligence) -> float:
        """Calculate risk tolerance for this threat"""
        # Lower risk tolerance for high-impact threats
        base_tolerance = 0.7
        impact_adjustment = threat_intel.business_impact_score * 0.5
        
        return max(base_tolerance - impact_adjustment, 0.1)
    
    def _determine_execution_strategy(self, threat_intel: BusinessThreatIntelligence,
                                    impact_analysis: Dict[str, float]) -> ExecutionStrategy:
        """Determine optimal execution strategy"""
        overall_impact = impact_analysis['overall_impact']
        
        if overall_impact > 0.8:
            return ExecutionStrategy.AGGRESSIVE
        elif overall_impact > 0.6:
            return ExecutionStrategy.BALANCED
        elif overall_impact > 0.4:
            return ExecutionStrategy.CONSERVATIVE
        else:
            return ExecutionStrategy.ADVISORY
    
    def _create_semantic_response(self, threat_intel: BusinessThreatIntelligence,
                               ice_result: ICEProcessingResult,
                               impact_analysis: Dict[str, float],
                               strategy: ExecutionStrategy) -> SemanticThreatResponse:
        """Create semantic threat response"""
        # Map ICE recommendations to FortiGate actions
        recommended_actions = self._map_to_fortigate_actions(ice_result.recommendation)
        
        # Generate business rationale
        business_rationale = f"{threat_intel.business_justification} {ice_result.business_principle}"
        
        # Determine resource requirements
        resource_requirements = {
            'processing_power': 'high' if impact_analysis['overall_impact'] > 0.7 else 'medium',
            'storage': 'temporary' if strategy != ExecutionStrategy.AGGRESSIVE else 'extended',
            'network_bandwidth': 'high' if threat_intel.business_context == BusinessThreatContext.DATA_BREACH else 'medium',
            'human_intervention': strategy in [ExecutionStrategy.COLLABORATIVE, ExecutionStrategy.AGGRESSIVE]
        }
        
        # Define monitoring requirements
        monitoring_requirements = [
            'threat_activity_monitoring',
            'system_performance_monitoring',
            'compliance_status_tracking',
            'business_impact_assessment'
        ]
        
        # Define success metrics
        success_metrics = [
            'threat_containment_time',
            'business_impact_minimized',
            'compliance_maintained',
            'customer_protection_ensured'
        ]
        
        # Generate compliance notes
        compliance_notes = [
            f"Applicable regulations: {', '.join(threat_intel.regulatory_requirements)}",
            f"Compliance risk level: {threat_intel.compliance_risk:.1%}",
            "Document all response actions for audit purposes"
        ]
        
        return SemanticThreatResponse(
            threat_intelligence=threat_intel,
            ice_processing_result=ice_result,
            recommended_actions=recommended_actions,
            business_rationale=business_rationale,
            confidence_score=ice_result.confidence,
            execution_strategy=strategy,
            resource_requirements=resource_requirements,
            monitoring_requirements=monitoring_requirements,
            success_metrics=success_metrics,
            compliance_notes=compliance_notes
        )
    
    def _map_to_fortigate_actions(self, ice_recommendation: Dict[str, Any]) -> List[ResponseAction]:
        """Map ICE recommendations to FortiGate actions"""
        actions = []
        
        recommendation = ice_recommendation.get('recommendation', '')
        action_items = ice_recommendation.get('action_items', [])
        
        if 'block' in recommendation.lower() or 'isolate' in recommendation.lower():
            actions.append(ResponseAction.BLOCK)
        
        if 'quarantine' in recommendation.lower():
            actions.append(ResponseAction.QUARANTINE)
        
        if 'monitor' in recommendation.lower():
            actions.append(ResponseAction.MONITOR)
        
        if 'allow' in recommendation.lower():
            actions.append(ResponseAction.ALLOW)
        
        if 'route' in recommendation.lower():
            actions.append(ResponseAction.ROUTE)
        
        # Default action if no specific mapping
        if not actions:
            actions.append(ResponseAction.MONITOR)
        
        return actions
    
    async def _store_learning_data(self, threat_intel: BusinessThreatIntelligence,
                                 response: SemanticThreatResponse):
        """Store learning data for continuous improvement"""
        try:
            # Create semantic signature for learning
            signature = SemanticSignature(
                signature_id=f"threat_{threat_intel.threat_signature.payload_hash}_{int(time.time())}",
                coordinates=response.ice_processing_result.ice_state.to_coordinates(),
                confidence=response.confidence_score,
                context=self._map_business_context(response.threat_intelligence.business_context),
                timestamp=datetime.now(timezone.utc),
                business_impact=response.threat_intelligence.business_impact_score,
                learning_source="fortigate_semantic_shield",
                metadata={
                    'threat_type': threat_intel.threat_signature.threat_type,
                    'business_context': threat_intel.business_context.value,
                    'recommended_actions': [action.value for action in response.recommended_actions],
                    'execution_strategy': response.execution_strategy.value,
                    'resource_requirements': response.resource_requirements
                }
            )
            
            await semantic_db.store_signature(signature)
            
        except Exception as e:
            logging.error(f"Error storing learning data: {e}")
    
    def _map_business_context(self, threat_context: BusinessThreatContext) -> BusinessContext:
        """Map threat context to semantic database context"""
        mapping = {
            BusinessThreatContext.FINANCIAL_FRAUD: BusinessContext.JUSTICE,
            BusinessThreatContext.DATA_BREACH: BusinessContext.INTEGRITY,
            BusinessThreatContext.OPERATIONAL_DISRUPTION: BusinessContext.STRENGTH,
            BusinessThreatContext.REPUTATIONAL_DAMAGE: BusinessContext.CUSTOMER_TRUST,
            BusinessThreatContext.REGULATORY_VIOLATION: BusinessContext.JUSTICE,
            BusinessThreatContext.INTELLECTUAL_PROPERTY_THEFT: BusinessContext.INTEGRITY,
            BusinessThreatContext.CRITICAL_INFRASTRUCTURE: BusinessContext.STRENGTH,
            BusinessThreatContext.CUSTOMER_DATA_COMPROMISE: BusinessContext.CUSTOMER_TRUST
        }
        return mapping.get(threat_context, BusinessContext.INTEGRITY)
    
    def _update_performance_metrics(self, response: SemanticThreatResponse, 
                                  processing_time: float):
        """Update performance metrics"""
        with self._lock:
            self.performance_metrics['threats_processed'] += 1
            
            # Update average processing time
            total = self.performance_metrics['threats_processed']
            current_avg = self.performance_metrics['avg_processing_time']
            self.performance_metrics['avg_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
            
            # Update success rate
            if response.confidence_score > 0.7:
                current_success_rate = self.performance_metrics['success_rate']
                self.performance_metrics['success_rate'] = (
                    (current_success_rate * (total - 1) + 1.0) / total
                )
            
            # Update business impact prevented
            impact_prevented = response.threat_intelligence.business_impact_score * response.confidence_score
            current_impact = self.performance_metrics['business_impact_prevented']
            self.performance_metrics['business_impact_prevented'] = (
                (current_impact * (total - 1) + impact_prevented) / total
            )
    
    def _create_fallback_response(self, threat_data: Dict[str, Any], 
                                error_message: str) -> SemanticThreatResponse:
        """Create fallback response when processing fails"""
        threat_intel = BusinessThreatIntelligence(
            threat_signature=ThreatSignature(
                threat_type=threat_data.get('threat_type', 'unknown'),
                source_ip=threat_data.get('source_ip', '0.0.0.0'),
                destination_ip=threat_data.get('destination_ip', '0.0.0.0'),
                port=threat_data.get('port', 0),
                protocol=threat_data.get('protocol', 'unknown'),
                payload_hash=threat_data.get('payload_hash', ''),
                severity=threat_data.get('severity', ThreatLevel.MEDIUM),
                confidence=0.1
            ),
            business_context=BusinessThreatContext.DATA_BREACH,
            business_impact_score=0.5,
            compliance_risk=0.5,
            customer_impact=0.5,
            financial_exposure=0.5,
            reputation_risk=0.5,
            regulatory_requirements=['general_compliance'],
            mitigation_strategies=['monitor_threat'],
            business_justification=f"Error in processing: {error_message}"
        )
        
        fallback_ice_result = ICEProcessingResult(
            ice_state=None,  # Will be created with default values
            recommendation={'recommendation': 'error_fallback', 'action_items': ['monitor']},
            confidence=0.1,
            business_principle="Handle errors with minimal disruption",
            action_items=['monitor'],
            expected_outcome=0.1,
            risk_assessment={'processing_error': 1.0}
        )
        
        return SemanticThreatResponse(
            threat_intelligence=threat_intel,
            ice_processing_result=fallback_ice_result,
            recommended_actions=[ResponseAction.MONITOR],
            business_rationale=f"Fallback response due to processing error: {error_message}",
            confidence_score=0.1,
            execution_strategy=ExecutionStrategy.CONSERVATIVE,
            resource_requirements={'processing_power': 'low'},
            monitoring_requirements=['basic_monitoring'],
            success_metrics=['error_resolution'],
            compliance_notes=['Document error for technical review']
        )
    
    async def apply_response_to_fortigate(self, response: SemanticThreatResponse) -> bool:
        """Apply semantic response to FortiGate device"""
        try:
            if not self._session:
                logging.error("HTTP session not initialized")
                return False
            
            # Convert recommended actions to FortiGate API calls
            for action in response.recommended_actions:
                success = await self._execute_fortigate_action(action, response)
                if not success:
                    logging.warning(f"Failed to execute action: {action}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error applying response to FortiGate: {e}")
            return False
    
    async def _execute_fortigate_action(self, action: ResponseAction, 
                                      response: SemanticThreatResponse) -> bool:
        """Execute specific action on FortiGate device"""
        # Implementation would depend on specific FortiGate API
        # This is a placeholder for the actual API integration
        action_configs = {
            ResponseAction.BLOCK: {
                'endpoint': '/api/v2/monitor/firewall/policy',
                'method': 'POST',
                'data': {
                    'action': 'deny',
                    'srcaddr': response.threat_intelligence.threat_signature.source_ip,
                    'dstaddr': response.threat_intelligence.threat_signature.destination_ip
                }
            },
            ResponseAction.QUARANTINE: {
                'endpoint': '/api/v2/monitor/quarantine',
                'method': 'POST',
                'data': {
                    'ip': response.threat_intelligence.threat_signature.source_ip
                }
            }
        }
        
        config = action_configs.get(action)
        if not config:
            logging.warning(f"No configuration for action: {action}")
            return True  # No-op for unsupported actions
        
        try:
            async with self._session.request(
                config['method'],
                f"{self.api_config.base_url}{config['endpoint']}",
                headers={'Authorization': f'Bearer {self.api_config.token}'},
                json=config['data']
            ) as resp:
                return resp.status == 200
                
        except Exception as e:
            logging.error(f"FortiGate API error: {e}")
            return False
    
    async def get_predictive_intelligence(self) -> Dict[str, Any]:
        """Get predictive threat intelligence"""
        return self.predictive_analytics.predict_threat_trends()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self._lock:
            return self.performance_metrics.copy()
    
    async def create_intelligence_snapshot(self, snapshot_name: str) -> str:
        """Create intelligence snapshot"""
        return await semantic_db.create_snapshot(snapshot_name)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session:
            await self._session.close()


# Global intelligence instance
fortigate_intelligence = None


async def initialize_fortigate_intelligence(api_config: FortiGateAPIConfig,
                                          learning_mode: LearningMode = LearningMode.BALANCED) -> FortiGateSemanticIntelligence:
    """Initialize global FortiGate intelligence instance"""
    global fortigate_intelligence
    fortigate_intelligence = FortiGateSemanticIntelligence(api_config, learning_mode)
    await fortigate_intelligence.initialize()
    return fortigate_intelligence
