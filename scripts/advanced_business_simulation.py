"""
Advanced Threat Simulation with Business Context
================================================

Enterprise-grade threat simulation for testing FortiGate Semantic Shield
with realistic business scenarios and comprehensive metrics.
"""

import asyncio
import random
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np
from collections import defaultdict, deque

# Import enhanced components
from enhanced_fortigate_intelligence_v7 import (
    FortiGateSemanticIntelligence, BusinessThreatContext, 
    BusinessThreatIntelligence, initialize_fortigate_intelligence
)
from fortigate_semantic_shield.device_interface import FortiGateAPIConfig
from fortigate_semantic_shield.semantic_components import (
    ThreatLevel, ThreatSignature, NetworkPacket, NetworkIntent, NetworkContext
)


class BusinessSector(Enum):
    """Business sectors for realistic simulation"""
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    RETAIL_ECOMMERCE = "retail_ecommerce"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"
    ENERGY_UTILITIES = "energy_utilities"
    GOVERNMENT = "government"
    EDUCATION = "education"


class AttackSophistication(Enum):
    """Attack sophistication levels"""
    SCRIPT_KIDDIE = "script_kiddie"          # Low complexity, high noise
    CYBERCRIMINAL = "cybercriminal"          # Medium complexity, profit-motivated
    ADVANCED_PERSISTENT_THREAT = "apt"       # High complexity, stealthy
    INSIDER_THREAT = "insider_threat"        # Privileged access, trusted
    STATE_SPONSORED = "state_sponsored"       # Maximum sophistication, resources


@dataclass
class BusinessScenario:
    """Realistic business attack scenario"""
    scenario_id: str
    business_sector: BusinessSector
    attack_sophistication: AttackSophistication
    threat_context: BusinessThreatContext
    attack_vectors: List[str]
    affected_systems: List[str]
    data_sensitivity: float
    regulatory_requirements: List[str]
    business_impact_potential: float
    attack_duration_minutes: int
    stealth_level: float
    resource_requirements: Dict[str, float]
    business_justification: str


@dataclass
class SimulationMetrics:
    """Comprehensive simulation metrics"""
    scenario_id: str
    threats_generated: int
    threats_processed: int
    avg_processing_time: float
    success_rate: float
    business_impact_prevented: float
    false_positive_rate: float
    intelligence_effectiveness: float
    learning_velocity: float
    resource_utilization: Dict[str, float]
    compliance_maintained: float
    business_rationale_quality: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdvancedThreatGenerator:
    """Advanced threat generator with realistic business contexts"""
    
    def __init__(self):
        self.business_scenarios = self._create_business_scenarios()
        self.attack_patterns = self._create_attack_patterns()
        self.threat_actors = self._create_threat_actors()
        self.vulnerability_database = self._create_vulnerability_database()
        
    def _create_business_scenarios(self) -> Dict[str, BusinessScenario]:
        """Create realistic business scenarios"""
        scenarios = {
            'financial_breach': BusinessScenario(
                scenario_id='financial_breach',
                business_sector=BusinessSector.FINANCIAL_SERVICES,
                attack_sophistication=AttackSophistication.ADVANCED_PERSISTENT_THREAT,
                threat_context=BusinessThreatContext.FINANCIAL_FRAUD,
                attack_vectors=['spear_phishing', 'malware', 'lateral_movement', 'data_exfiltration'],
                affected_systems=['transaction_server', 'customer_database', 'authentication_service', 'api_gateway'],
                data_sensitivity=0.9,
                regulatory_requirements=['SOX', 'PCI-DSS', 'GLBA', 'GDPR'],
                business_impact_potential=0.95,
                attack_duration_minutes=180,
                stealth_level=0.8,
                resource_requirements={'cpu': 0.7, 'memory': 0.6, 'network': 0.8, 'storage': 0.3},
                business_justification='Sophisticated financial fraud targeting high-value transactions and customer data'
            ),
            'healthcare_ransomware': BusinessScenario(
                scenario_id='healthcare_ransomware',
                business_sector=BusinessSector.HEALTHCARE,
                attack_sophistication=AttackSophistication.CYBERCRIMINAL,
                threat_context=BusinessThreatContext.OPERATIONAL_DISRUPTION,
                attack_vectors=['ransomware', 'phishing', 'vpn_exploitation', 'credential_theft'],
                affected_systems=['ehr_system', 'medical_devices', 'patient_portal', 'backup_systems'],
                data_sensitivity=0.95,
                regulatory_requirements=['HIPAA', 'HITECH', 'GDPR'],
                business_impact_potential=0.9,
                attack_duration_minutes=120,
                stealth_level=0.6,
                resource_requirements={'cpu': 0.8, 'memory': 0.7, 'network': 0.9, 'storage': 0.5},
                business_justification='Ransomware attack targeting critical healthcare systems and patient data'
            ),
            'retail_data_breach': BusinessScenario(
                scenario_id='retail_data_breach',
                business_sector=BusinessSector.RETAIL_ECOMMERCE,
                attack_sophistication=AttackSophistication.CYBERCRIMINAL,
                threat_context=BusinessThreatContext.CUSTOMER_DATA_COMPROMISE,
                attack_vectors=['web_application_exploit', 'pos_malware', 'credential_stuffing', 'api_abuse'],
                affected_systems=['ecommerce_platform', 'payment_processing', 'customer_database', 'inventory_system'],
                data_sensitivity=0.8,
                regulatory_requirements=['PCI-DSS', 'CCPA', 'GDPR'],
                business_impact_potential=0.85,
                attack_duration_minutes=90,
                stealth_level=0.7,
                resource_requirements={'cpu': 0.6, 'memory': 0.5, 'network': 0.9, 'storage': 0.4},
                business_justification='Customer data breach targeting payment information and personal data'
            ),
            'manufacturing_ip_theft': BusinessScenario(
                scenario_id='manufacturing_ip_theft',
                business_sector=BusinessSector.MANUFACTURING,
                attack_sophistication=AttackSophistication.STATE_SPONSORED,
                threat_context=BusinessThreatContext.INTELLECTUAL_PROPERTY_THEFT,
                attack_vectors=['supply_chain_attack', 'zero_day_exploit', 'insider_compromise', 'data_exfiltration'],
                affected_systems=['cad_systems', 'manufacturing_execution', 'research_database', 'industrial_controls'],
                data_sensitivity=0.85,
                regulatory_requirements=['ITAR', 'EAR', 'NIST', 'industry_specific'],
                business_impact_potential=0.9,
                attack_duration_minutes=240,
                stealth_level=0.95,
                resource_requirements={'cpu': 0.5, 'memory': 0.4, 'network': 0.7, 'storage': 0.6},
                business_justification='State-sponsored intellectual property theft targeting manufacturing secrets and designs'
            ),
            'critical_infrastructure': BusinessScenario(
                scenario_id='critical_infrastructure',
                business_sector=BusinessSector.ENERGY_UTILITIES,
                attack_sophistication=AttackSophistication.STATE_SPONSORED,
                threat_context=BusinessThreatContext.CRITICAL_INFRASTRUCTURE,
                attack_vectors=['iot_exploitation', 'scada_attack', 'supply_chain', 'lateral_movement'],
                affected_systems=['scada_systems', 'control_network', 'monitoring_systems', 'safety_systems'],
                data_sensitivity=0.9,
                regulatory_requirements=['NERC', 'NERC-CIP', 'CFR', 'industry_specific'],
                business_impact_potential=1.0,
                attack_duration_minutes=300,
                stealth_level=0.9,
                resource_requirements={'cpu': 0.6, 'memory': 0.5, 'network': 0.8, 'storage': 0.3},
                business_justification='Critical infrastructure attack with potential for widespread service disruption'
            )
        }
        return scenarios
    
    def _create_attack_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Create realistic attack patterns"""
        return {
            'spear_phishing': {
                'complexity': 0.4,
                'stealth': 0.6,
                'success_rate': 0.7,
                'detection_difficulty': 0.5,
                'common_indicators': ['suspicious_emails', 'credential_theft', 'initial_access']
            },
            'malware': {
                'complexity': 0.6,
                'stealth': 0.7,
                'success_rate': 0.8,
                'detection_difficulty': 0.7,
                'common_indicators': ['file_execution', 'process_creation', 'network_connections']
            },
            'lateral_movement': {
                'complexity': 0.8,
                'stealth': 0.8,
                'success_rate': 0.6,
                'detection_difficulty': 0.8,
                'common_indicators': ['network_propagation', 'credential_reuse', 'privilege_escalation']
            },
            'data_exfiltration': {
                'complexity': 0.7,
                'stealth': 0.9,
                'success_rate': 0.5,
                'detection_difficulty': 0.9,
                'common_indicators': ['unusual_data_transfer', 'encrypted_traffic', 'large_uploads']
            },
            'ransomware': {
                'complexity': 0.6,
                'stealth': 0.3,
                'success_rate': 0.8,
                'detection_difficulty': 0.4,
                'common_indicators': ['file_encryption', 'ransom_notes', 'system_lockdown']
            }
        }
    
    def _create_threat_actors(self) -> Dict[AttackSophistication, Dict[str, Any]]:
        """Create threat actor profiles"""
        return {
            AttackSophistication.SCRIPT_KIDDIE: {
                'motivation': 'notoriety',
                'resources': 0.2,
                'persistence': 0.1,
                'skill_level': 0.3,
                'common_tools': ['automated_scanners', 'public_exploits']
            },
            AttackSophistication.CYBERCRIMINAL: {
                'motivation': 'financial_gain',
                'resources': 0.6,
                'persistence': 0.5,
                'skill_level': 0.7,
                'common_tools': ['malware_kits', 'botnets', 'underground_tools']
            },
            AttackSophistication.ADVANCED_PERSISTENT_THREAT: {
                'motivation': 'espionage_or_sabotage',
                'resources': 0.8,
                'persistence': 0.9,
                'skill_level': 0.9,
                'common_tools': ['custom_malware', 'zero_day_exploits', 'sophisticated_infrastructure']
            },
            AttackSophistication.INSIDER_THREAT: {
                'motivation': 'revenge_or_ideology',
                'resources': 0.9,  # Legitimate access
                'persistence': 0.7,
                'skill_level': 0.6,
                'common_tools': ['legitimate_credentials', 'internal_knowledge', 'authorized_tools']
            },
            AttackSophistication.STATE_SPONSORED: {
                'motivation': 'national_interests',
                'resources': 1.0,
                'persistence': 1.0,
                'skill_level': 1.0,
                'common_tools': ['unlimited_resources', 'custom_developments', 'global_infrastructure']
            }
        }
    
    def _create_vulnerability_database(self) -> Dict[str, Dict[str, float]]:
        """Create vulnerability database for realistic simulation"""
        return {
            'cve_2023_23397': {'severity': 0.9, 'exploitability': 0.8, 'impact': 0.9},
            'cve_2023_4911': {'severity': 0.8, 'exploitability': 0.7, 'impact': 0.8},
            'cve_2023_36874': {'severity': 0.7, 'exploitability': 0.9, 'impact': 0.6},
            'cve_2023_34362': {'severity': 0.8, 'exploitability': 0.6, 'impact': 0.8},
            'cve_2023_35078': {'severity': 0.9, 'exploitability': 0.7, 'impact': 0.9},
            'zero_day': {'severity': 1.0, 'exploitability': 0.3, 'impact': 1.0}
        }
    
    def generate_threat_stream(self, scenario: BusinessScenario, 
                             duration_minutes: int = 60,
                             threat_rate_per_minute: int = 5) -> List[Dict[str, Any]]:
        """Generate realistic threat stream for scenario"""
        threats = []
        
        # Calculate threat timing based on attack lifecycle
        total_threats = duration_minutes * threat_rate_per_minute
        
        for i in range(total_threats):
            # Determine attack phase
            phase_progress = i / total_threats
            
            if phase_progress < 0.2:  # Initial compromise
                primary_vectors = ['spear_phishing', 'web_application_exploit']
            elif phase_progress < 0.5:  # Establishment and persistence
                primary_vectors = ['malware', 'credential_theft']
            elif phase_progress < 0.8:  # Lateral movement and discovery
                primary_vectors = ['lateral_movement', 'privilege_escalation']
            else:  # Data exfiltration or impact
                primary_vectors = ['data_exfiltration', 'ransomware', 'data_destruction']
            
            # Select attack vector
            attack_vector = random.choice(primary_vectors + scenario.attack_vectors)
            attack_pattern = self.attack_patterns.get(attack_vector, self.attack_patterns['malware'])
            
            # Generate threat data
            threat_data = {
                'threat_id': f"{scenario.scenario_id}_threat_{i}",
                'scenario_id': scenario.scenario_id,
                'attack_vector': attack_vector,
                'threat_type': attack_vector,
                'source_ip': self._generate_realistic_ip(),
                'destination_ip': self._select_target_system(scenario.affected_systems),
                'port': random.choice([80, 443, 22, 3389, 445, 1433, 3306]),
                'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
                'payload_hash': self._generate_hash(),
                'severity': self._calculate_severity(attack_pattern, scenario, phase_progress),
                'confidence': 0.5 + (random.random() * 0.4),  # 0.5 to 0.9
                'category': scenario.threat_context.value,
                'affected_systems': random.sample(
                    scenario.affected_systems, 
                    min(random.randint(1, 3), len(scenario.affected_systems))
                ),
                'data_sensitivity': scenario.data_sensitivity,
                'business_context': scenario.threat_context.value,
                'stealth_level': scenario.stealth_level * (0.8 + random.random() * 0.4),
                'attack_phase': self._determine_attack_phase(phase_progress),
                'timestamp': datetime.now(timezone.utc),
                'resource_impact': {
                    'cpu': scenario.resource_requirements['cpu'] * random.uniform(0.5, 1.5),
                    'memory': scenario.resource_requirements['memory'] * random.uniform(0.5, 1.5),
                    'network': scenario.resource_requirements['network'] * random.uniform(0.5, 1.5),
                    'storage': scenario.resource_requirements['storage'] * random.uniform(0.5, 1.5)
                },
                'regulatory_impact': {
                    'requirements': scenario.regulatory_requirements,
                    'violation_probability': scenario.business_impact_potential * 0.8
                }
            }
            
            threats.append(threat_data)
        
        return threats
    
    def _generate_realistic_ip(self) -> str:
        """Generate realistic source IP based on threat actor"""
        # Mix of internal and external IPs
        if random.random() < 0.2:  # 20% internal (insider threat)
            return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:  # 80% external
            return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def _select_target_system(self, affected_systems: List[str]) -> str:
        """Select target system IP based on affected systems"""
        system_mapping = {
            'transaction_server': '10.1.1.10',
            'customer_database': '10.1.2.20',
            'authentication_service': '10.1.1.5',
            'api_gateway': '10.1.1.100',
            'ehr_system': '10.2.1.50',
            'medical_devices': '10.2.3.0/24',
            'patient_portal': '10.2.1.100',
            'ecommerce_platform': '10.3.1.10',
            'payment_processing': '10.3.2.20',
            'inventory_system': '10.3.3.30'
        }
        
        target_system = random.choice(affected_systems)
        return system_mapping.get(target_system, f"10.0.0.{random.randint(1, 254)}")
    
    def _generate_hash(self) -> str:
        """Generate realistic payload hash"""
        return ''.join(random.choices('0123456789abcdef', k=64))
    
    def _calculate_severity(self, attack_pattern: Dict[str, float], 
                          scenario: BusinessScenario, phase_progress: float) -> ThreatLevel:
        """Calculate threat severity based on multiple factors"""
        # Base severity from attack pattern
        pattern_severity = attack_pattern['success_rate']
        
        # Adjust for scenario impact
        scenario_factor = scenario.business_impact_potential
        
        # Adjust for attack phase (later phases often more severe)
        phase_factor = 0.5 + phase_progress * 0.5
        
        # Calculate combined severity score
        severity_score = pattern_severity * scenario_factor * phase_factor
        
        # Map to threat level
        if severity_score > 0.8:
            return ThreatLevel.CRITICAL
        elif severity_score > 0.6:
            return ThreatLevel.HIGH
        elif severity_score > 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _determine_attack_phase(self, phase_progress: float) -> str:
        """Determine attack phase based on progress"""
        if phase_progress < 0.2:
            return 'initial_compromise'
        elif phase_progress < 0.5:
            return 'establishment'
        elif phase_progress < 0.8:
            return 'lateral_movement'
        else:
            return 'objectives'


class BusinessImpactAssessment:
    """Business impact assessment for simulation results"""
    
    def __init__(self):
        self.impact_weights = {
            'financial_loss': 0.3,
            'reputation_damage': 0.25,
            'regulatory_fines': 0.2,
            'operational_disruption': 0.15,
            'customer_churn': 0.1
        }
    
    def assess_scenario_impact(self, scenario: BusinessScenario, 
                             threats_processed: int,
                             processing_success_rate: float) -> Dict[str, float]:
        """Assess business impact of simulation scenario"""
        # Calculate prevented impact based on system effectiveness
        prevention_effectiveness = processing_success_rate * threats_processed / max(threats_processed, 1)
        
        # Calculate potential vs actual impact
        potential_impact = scenario.business_impact_potential
        prevented_impact = potential_impact * prevention_effectiveness
        actual_impact = potential_impact - prevented_impact
        
        # Break down impact by category
        impacts = {}
        for category, weight in self.impact_weights.items():
            impacts[category] = actual_impact * weight
        
        # Add overall metrics
        impacts['overall_impact'] = actual_impact
        impacts['prevented_impact'] = prevented_impact
        impacts['roi_percentage'] = (prevented_impact / max(potential_impact, 0.01) - 1) * 100
        
        return impacts


class AdvancedBusinessSimulation:
    """Advanced business-focused threat simulation"""
    
    def __init__(self, api_config: FortiGateAPIConfig):
        self.api_config = api_config
        self.threat_generator = AdvancedThreatGenerator()
        self.impact_assessor = BusinessImpactAssessment()
        self.simulation_results = []
        self.performance_history = deque(maxlen=100)
        
    async def run_comprehensive_simulation(self, 
                                         scenarios_to_run: Optional[List[str]] = None,
                                         waves_per_scenario: int = 3) -> Dict[str, Any]:
        """Run comprehensive business simulation"""
        if scenarios_to_run is None:
            scenarios_to_run = list(self.threat_generator.business_scenarios.keys())
        
        simulation_start = time.time()
        all_metrics = []
        
        logging.info(f"Starting comprehensive simulation with {len(scenarios_to_run)} scenarios")
        
        # Initialize intelligence system
        intelligence = await initialize_fortigate_intelligence(self.api_config)
        
        for scenario_name in scenarios_to_run:
            scenario = self.threat_generator.business_scenarios[scenario_name]
            logging.info(f"Running scenario: {scenario_name} ({scenario.business_sector.value})")
            
            scenario_metrics = []
            
            for wave in range(waves_per_scenario):
                logging.info(f"  Wave {wave + 1}/{waves_per_scenario}")
                
                # Generate threat stream for this wave
                threat_stream = self.threat_generator.generate_threat_stream(
                    scenario, duration_minutes=60, threat_rate_per_minute=10
                )
                
                # Process threats through intelligence system
                wave_metrics = await self._process_threat_wave(intelligence, threat_stream, scenario)
                scenario_metrics.append(wave_metrics)
                
                # Small delay between waves
                await asyncio.sleep(2)
            
            # Aggregate scenario metrics
            scenario_summary = self._aggregate_scenario_metrics(scenario, scenario_metrics)
            all_metrics.append(scenario_summary)
        
        # Create comprehensive report
        total_time = time.time() - simulation_start
        comprehensive_report = self._create_comprehensive_report(all_metrics, total_time)
        
        # Cleanup
        await intelligence.cleanup()
        
        logging.info(f"Comprehensive simulation completed in {total_time:.2f} seconds")
        return comprehensive_report
    
    async def _process_threat_wave(self, intelligence: FortiGateSemanticIntelligence,
                                 threat_stream: List[Dict[str, Any]],
                                 scenario: BusinessScenario) -> SimulationMetrics:
        """Process a wave of threats through the intelligence system"""
        wave_start = time.time()
        processed_threats = []
        processing_times = []
        success_count = 0
        
        # Process threats in batches for performance
        batch_size = 10
        for i in range(0, len(threat_stream), batch_size):
            batch = threat_stream[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [intelligence.process_threat_intelligently(threat) for threat in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Error processing threat {i+j}: {result}")
                    continue
                
                processed_threats.append(result)
                processing_times.append(result.processing_time)
                
                if result.confidence_score > 0.7:
                    success_count += 1
                
                # Apply responses to simulated FortiGate
                await intelligence.apply_response_to_fortigate(result)
        
        # Calculate metrics
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        success_rate = success_count / len(processed_threats) if processed_threats else 0
        
        # Get performance metrics from intelligence system
        perf_metrics = intelligence.get_performance_metrics()
        
        # Calculate business impact prevented
        business_impact_prevented = np.mean([
            response.threat_intelligence.business_impact_score * response.confidence_score
            for response in processed_threats
        ]) if processed_threats else 0
        
        # Calculate intelligence effectiveness
        intelligence_effectiveness = self._calculate_intelligence_effectiveness(
            processed_threats, scenario)
        
        return SimulationMetrics(
            scenario_id=scenario.scenario_id,
            threats_generated=len(threat_stream),
            threats_processed=len(processed_threats),
            avg_processing_time=avg_processing_time,
            success_rate=success_rate,
            business_impact_prevented=business_impact_prevented,
            false_positive_rate=self._calculate_false_positive_rate(processed_threats),
            intelligence_effectiveness=intelligence_effectiveness,
            learning_velocity=perf_metrics.get('learning_velocity', 0.5),
            resource_utilization=self._calculate_resource_utilization(threat_stream),
            compliance_maintained=self._calculate_compliance_maintained(processed_threats),
            business_rationale_quality=self._calculate_rationale_quality(processed_threats)
        )
    
    def _calculate_intelligence_effectiveness(self, responses: List[Any], 
                                            scenario: BusinessScenario) -> float:
        """Calculate intelligence effectiveness score"""
        if not responses:
            return 0.0
        
        effectiveness_factors = []
        
        for response in responses:
            # Confidence score
            confidence_score = response.confidence_score
            
            # Business context accuracy
            context_match = (
                response.threat_intelligence.business_context == scenario.threat_context
            )
            context_score = 1.0 if context_match else 0.5
            
            # Action appropriateness
            action_score = self._assess_action_appropriateness(response, scenario)
            
            # Business rationale quality
            rationale_score = self._assess_rationale_quality(response)
            
            # Combine factors
            overall_score = (
                confidence_score * 0.3 +
                context_score * 0.2 +
                action_score * 0.3 +
                rationale_score * 0.2
            )
            
            effectiveness_factors.append(overall_score)
        
        return np.mean(effectiveness_factors)
    
    def _assess_action_appropriateness(self, response: Any, scenario: BusinessScenario) -> float:
        """Assess if recommended actions are appropriate for the scenario"""
        # Simple heuristic-based assessment
        appropriate_actions = {
            BusinessThreatContext.FINANCIAL_FRAUD: ['block', 'quarantine', 'monitor'],
            BusinessThreatContext.DATA_BREACH: ['block', 'quarantine', 'monitor'],
            BusinessThreatContext.OPERATIONAL_DISRUPTION: ['block', 'quarantine', 'monitor'],
            BusinessThreatContext.CRITICAL_INFRASTRUCTURE: ['monitor', 'block'],
            BusinessThreatContext.CUSTOMER_DATA_COMPROMISE: ['block', 'quarantine', 'monitor']
        }
        
        expected_actions = set(appropriate_actions.get(
            scenario.threat_context, ['monitor']
        ))
        
        actual_actions = set(action.value for action in response.recommended_actions)
        
        # Calculate overlap
        if not expected_actions:
            return 1.0
        
        overlap = len(expected_actions.intersection(actual_actions))
        return overlap / len(expected_actions)
    
    def _assess_rationale_quality(self, response: Any) -> float:
        """Assess quality of business rationale"""
        rationale = response.business_rationale
        
        # Simple quality indicators
        quality_indicators = {
            'length': len(rationale) > 50,  # Substantial explanation
            'business_terms': any(term in rationale.lower() 
                                 for term in ['business', 'impact', 'risk', 'compliance', 'regulation']),
            'specificity': any(term in rationale.lower() 
                             for term in ['prevent', 'protect', 'mitigate', 'ensure'])
        }
        
        score = sum(quality_indicators.values()) / len(quality_indicators)
        return score
    
    def _calculate_false_positive_rate(self, responses: List[Any]) -> float:
        """Calculate false positive rate"""
        # Simplified calculation based on confidence scores
        if not responses:
            return 0.0
        
        false_positives = sum(1 for r in responses if r.confidence_score < 0.3)
        return false_positives / len(responses)
    
    def _calculate_resource_utilization(self, threat_stream: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average resource utilization"""
        if not threat_stream:
            return {'cpu': 0.0, 'memory': 0.0, 'network': 0.0, 'storage': 0.0}
        
        resources = {
            'cpu': np.mean([t['resource_impact']['cpu'] for t in threat_stream]),
            'memory': np.mean([t['resource_impact']['memory'] for t in threat_stream]),
            'network': np.mean([t['resource_impact']['network'] for t in threat_stream]),
            'storage': np.mean([t['resource_impact']['storage'] for t in threat_stream])
        }
        
        return resources
    
    def _calculate_compliance_maintained(self, responses: List[Any]) -> float:
        """Calculate compliance maintenance score"""
        if not responses:
            return 0.0
        
        compliance_scores = []
        for response in responses:
            # Check if compliance notes are present and relevant
            has_compliance_notes = len(response.compliance_notes) > 0
            mentions_regulations = any(
                reg.lower() in ' '.join(response.compliance_notes).lower()
                for reg in response.threat_intelligence.regulatory_requirements
            )
            
            score = 1.0 if (has_compliance_notes and mentions_regulations) else 0.5
            compliance_scores.append(score)
        
        return np.mean(compliance_scores)
    
    def _calculate_rationale_quality(self, responses: List[Any]) -> float:
        """Calculate overall business rationale quality"""
        if not responses:
            return 0.0
        
        rationale_scores = [self._assess_rationale_quality(r) for r in responses]
        return np.mean(rationale_scores)
    
    def _aggregate_scenario_metrics(self, scenario: BusinessScenario,
                                  wave_metrics: List[SimulationMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across waves for a scenario"""
        return {
            'scenario_id': scenario.scenario_id,
            'business_sector': scenario.business_sector.value,
            'attack_sophistication': scenario.attack_sophistication.value,
            'threat_context': scenario.threat_context.value,
            'business_impact_potential': scenario.business_impact_potential,
            'waves_completed': len(wave_metrics),
            'total_threats': sum(m.threats_generated for m in wave_metrics),
            'threats_processed': sum(m.threats_processed for m in wave_metrics),
            'avg_processing_time': np.mean([m.avg_processing_time for m in wave_metrics]),
            'success_rate': np.mean([m.success_rate for m in wave_metrics]),
            'business_impact_prevented': np.mean([m.business_impact_prevented for m in wave_metrics]),
            'intelligence_effectiveness': np.mean([m.intelligence_effectiveness for m in wave_metrics]),
            'learning_velocity': np.mean([m.learning_velocity for m in wave_metrics]),
            'compliance_maintained': np.mean([m.compliance_maintained for m in wave_metrics]),
            'business_rationale_quality': np.mean([m.business_rationale_quality for m in wave_metrics]),
            'resource_utilization': {
                resource: np.mean([m.resource_utilization[resource] for m in wave_metrics])
                for resource in ['cpu', 'memory', 'network', 'storage']
            }
        }
    
    def _create_comprehensive_report(self, all_metrics: List[Dict[str, Any]], 
                                   total_time: float) -> Dict[str, Any]:
        """Create comprehensive simulation report"""
        # Executive summary
        executive_summary = {
            'scenarios_tested': len(all_metrics),
            'total_simulation_time': total_time,
            'overall_intelligence_effectiveness': np.mean([m['intelligence_effectiveness'] for m in all_metrics]),
            'overall_business_impact_prevented': np.mean([m['business_impact_prevented'] for m in all_metrics]),
            'overall_success_rate': np.mean([m['success_rate'] for m in all_metrics]),
            'business_readiness_score': self._calculate_business_readiness_score(all_metrics)
        }
        
        # Detailed results
        detailed_results = {
            'scenario_performance': all_metrics,
            'cross_scenario_analysis': self._perform_cross_scenario_analysis(all_metrics),
            'business_sector_analysis': self._analyze_by_business_sector(all_metrics),
            'threat_context_analysis': self._analyze_by_threat_context(all_metrics),
            'attack_sophistication_analysis': self._analyze_by_sophistication(all_metrics)
        }
        
        # Recommendations
        recommendations = self._generate_business_recommendations(all_metrics)
        
        return {
            'executive_summary': executive_summary,
            'detailed_results': detailed_results,
            'recommendations': recommendations,
            'simulation_metadata': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '7.0',
                'test_environment': 'enterprise_simulation'
            }
        }
    
    def _calculate_business_readiness_score(self, all_metrics: List[Dict[str, Any]]) -> float:
        """Calculate overall business readiness score"""
        factors = {
            'intelligence_effectiveness': 0.3,
            'business_impact_prevented': 0.25,
            'compliance_maintained': 0.2,
            'business_rationale_quality': 0.15,
            'success_rate': 0.1
        }
        
        scores = {}
        for factor, weight in factors.items():
            scores[factor] = np.mean([m[factor] for m in all_metrics]) * weight
        
        return sum(scores.values())
    
    def _perform_cross_scenario_analysis(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-scenario analysis"""
        # Identify best and worst performing scenarios
        effectiveness_scores = [(m['scenario_id'], m['intelligence_effectiveness']) for m in all_metrics]
        effectiveness_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'best_performing_scenario': effectiveness_scores[0],
            'worst_performing_scenario': effectiveness_scores[-1],
            'performance_variance': np.var([m['intelligence_effectiveness'] for m in all_metrics]),
            'improvement_opportunities': self._identify_improvement_opportunities(all_metrics)
        }
    
    def _analyze_by_business_sector(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by business sector"""
        sector_analysis = defaultdict(list)
        
        for metrics in all_metrics:
            sector = metrics['business_sector']
            sector_analysis[sector].append(metrics)
        
        # Calculate sector averages
        sector_summary = {}
        for sector, sector_metrics in sector_analysis.items():
            sector_summary[sector] = {
                'intelligence_effectiveness': np.mean([m['intelligence_effectiveness'] for m in sector_metrics]),
                'business_impact_prevented': np.mean([m['business_impact_prevented'] for m in sector_metrics]),
                'compliance_maintained': np.mean([m['compliance_maintained'] for m in sector_metrics])
            }
        
        return sector_summary
    
    def _analyze_by_threat_context(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by threat context"""
        context_analysis = defaultdict(list)
        
        for metrics in all_metrics:
            context = metrics['threat_context']
            context_analysis[context].append(metrics)
        
        context_summary = {}
        for context, context_metrics in context_analysis.items():
            context_summary[context] = {
                'intelligence_effectiveness': np.mean([m['intelligence_effectiveness'] for m in context_metrics]),
                'business_impact_prevented': np.mean([m['business_impact_prevented'] for m in context_metrics]),
                'avg_processing_time': np.mean([m['avg_processing_time'] for m in context_metrics])
            }
        
        return context_summary
    
    def _analyze_by_sophistication(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by attack sophistication"""
        sophistication_analysis = defaultdict(list)
        
        for metrics in all_metrics:
            sophistication = metrics['attack_sophistication']
            sophistication_analysis[sophistication].append(metrics)
        
        sophistication_summary = {}
        for sophistication, soph_metrics in sophistication_analysis.items():
            sophistication_summary[sophistication] = {
                'intelligence_effectiveness': np.mean([m['intelligence_effectiveness'] for m in soph_metrics]),
                'success_rate': np.mean([m['success_rate'] for m in soph_metrics]),
                'false_positive_rate': np.mean([m['false_positive_rate'] for m in soph_metrics])
            }
        
        return sophistication_summary
    
    def _identify_improvement_opportunities(self, all_metrics: List[Dict[str, Any]]) -> List[str]:
        """Identify improvement opportunities"""
        opportunities = []
        
        # Low effectiveness scenarios
        low_effectiveness = [m for m in all_metrics if m['intelligence_effectiveness'] < 0.6]
        if low_effectiveness:
            opportunities.append(f"Improve threat detection for {len(low_effectiveness)} scenarios with low effectiveness")
        
        # High processing times
        high_processing = [m for m in all_metrics if m['avg_processing_time'] > 2.0]
        if high_processing:
            opportunities.append(f"Optimize processing speed for {len(high_processing)} scenarios with high latency")
        
        # Low compliance maintenance
        low_compliance = [m for m in all_metrics if m['compliance_maintained'] < 0.7]
        if low_compliance:
            opportunities.append(f"Enhance compliance tracking for {len(low_compliance)} scenarios")
        
        # Poor business rationale
        poor_rationale = [m for m in all_metrics if m['business_rationale_quality'] < 0.6]
        if poor_rationale:
            opportunities.append(f"Improve business justification quality for {len(poor_rationale)} scenarios")
        
        return opportunities if opportunities else ["System performing well across all scenarios"]
    
    def _generate_business_recommendations(self, all_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate business recommendations"""
        recommendations = []
        
        # Calculate overall performance
        overall_effectiveness = np.mean([m['intelligence_effectiveness'] for m in all_metrics])
        overall_impact_prevented = np.mean([m['business_impact_prevented'] for m in all_metrics])
        overall_compliance = np.mean([m['compliance_maintained'] for m in all_metrics])
        
        # Strategic recommendations
        if overall_effectiveness > 0.8:
            recommendations.append({
                'category': 'Strategic',
                'priority': 'High',
                'recommendation': 'Expand deployment to additional business units',
                'justification': f'System demonstrates high effectiveness ({overall_effectiveness:.1%}) and readiness for scale',
                'expected_roi': 'High',
                'implementation_time': '3-6 months'
            })
        
        if overall_impact_prevented > 0.7:
            recommendations.append({
                'category': 'Financial',
                'priority': 'High',
                'recommendation': 'Invest in advanced threat intelligence feeds',
                'justification': f'High impact prevention ({overall_impact_prevented:.1%}) justifies additional intelligence investment',
                'expected_roi': 'Medium-High',
                'implementation_time': '1-3 months'
            })
        
        if overall_compliance < 0.8:
            recommendations.append({
                'category': 'Compliance',
                'priority': 'Critical',
                'recommendation': 'Enhance regulatory reporting and automation',
                'justification': f'Compliance maintenance at {overall_compliance:.1%} requires improvement',
                'expected_roi': 'Medium',
                'implementation_time': '2-4 months'
            })
        
        # Technical recommendations
        avg_processing_time = np.mean([m['avg_processing_time'] for m in all_metrics])
        if avg_processing_time > 1.5:
            recommendations.append({
                'category': 'Technical',
                'priority': 'Medium',
                'recommendation': 'Optimize processing pipelines and increase parallelization',
                'justification': f'Average processing time of {avg_processing_time:.2f}s exceeds optimal targets',
                'expected_roi': 'Medium',
                'implementation_time': '1-2 months'
            })
        
        return recommendations


# Example usage
async def run_enterprise_simulation():
    """Run enterprise-grade simulation"""
    from fortigate_semantic_shield.device_interface import FortiGateAPIConfig
    
    # Configure API (using test configuration)
    api_config = FortiGateAPIConfig(
        base_url="https://fortigate-test.example.com",
        token="test_token",
        verify_ssl=False
    )
    
    # Create and run simulation
    simulation = AdvancedBusinessSimulation(api_config)
    results = await simulation.run_comprehensive_simulation(
        scenarios_to_run=['financial_breach', 'healthcare_ransomware', 'retail_data_breach'],
        waves_per_scenario=2
    )
    
    # Generate report
    print("=== ENTERPRISE SIMULATION REPORT ===")
    print(f"Scenarios Tested: {results['executive_summary']['scenarios_tested']}")
    print(f"Overall Intelligence Effectiveness: {results['executive_summary']['overall_intelligence_effectiveness']:.1%}")
    print(f"Business Impact Prevented: {results['executive_summary']['overall_business_impact_prevented']:.1%}")
    print(f"Business Readiness Score: {results['executive_summary']['business_readiness_score']:.1%}")
    
    print("\n=== TOP RECOMMENDATIONS ===")
    for rec in results['recommendations'][:3]:
        print(f"{rec['priority']} Priority: {rec['recommendation']}")
        print(f"  Justification: {rec['justification']}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_enterprise_simulation())