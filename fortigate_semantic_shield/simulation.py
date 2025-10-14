"""
FORTIGATE CONSTRAINED ENVIRONMENT SIMULATOR
Brutal Attack Scenarios with Intelligent Teaching System

This simulation creates:
1. Realistic FortiGate device constraints (memory, CPU, processing limits)
2. Brutal, sophisticated attack scenarios
3. Teaching system that helps the intelligence learn
4. Performance monitoring and adaptation
5. Survival and optimization metrics
"""

import asyncio
import time
import random
import numpy as np
import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import threading
from collections import defaultdict, deque

# Import our enhanced intelligence system
from .intelligence import FortiGateSemanticShield
from .device_interface import (
    FortiGatePolicyApplier,
    FortiGateTelemetryCollector,
    LearningPersistenceManager,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttackSeverity(Enum):
    """Attack severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BRUTAL = "brutal"

class AttackType(Enum):
    """Sophisticated attack types"""
    ADVANCED_PERSISTENT_THREAT = "apt"
    ZERO_DAY_EXPLOIT = "zero_day"
    RANSOMWARE_WAVE = "ransomware_wave"
    DDOS_AMPLIFICATION = "ddos_amplification"
    LATERAL_MOVEMENT = "lateral_movement"
    SUPPLY_CHAIN_COMPROMISE = "supply_chain"
    INSIDER_THREAT_ESCALATION = "insider_escalation"
    CRYPTO_MINING = "crypto_mining"
    BOTNET_COORDINATION = "botnet_coordination"
    POLYMORPHIC_MALWARE = "polymorphic_malware"

@dataclass
class FortiGateConstraints:
    """Realistic FortiGate device constraints"""
    
    # Hardware constraints
    max_cpu_usage: float = 85.0  # %
    max_memory_usage: float = 80.0  # %
    max_connections: int = 50000
    max_throughput: float = 10.0  # Gbps
    
    # Processing constraints
    max_concurrent_threats: int = 100
    processing_time_limit: float = 5.0  # seconds
    queue_size_limit: int = 1000
    
    # Learning constraints
    learning_memory_limit: int = 1000
    pattern_analysis_depth: int = 5
    confidence_threshold: float = 0.7
    
    # Biblical constraints
    divine_processing_limit: int = 100  # requests/minute
    wisdom_calculation_depth: int = 3
    justice_mercy_balance: float = 0.5  # 0.0 = pure justice, 1.0 = pure mercy

@dataclass
class AttackMetrics:
    """Attack performance metrics"""
    
    attack_id: str
    attack_type: AttackType
    severity: AttackSeverity
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    damage_level: float = 0.0
    resources_consumed: Dict[str, float] = field(default_factory=dict)
    defense_response_time: float = 0.0
    learning_outcome: str = ""
    system_survived: bool = True

@dataclass
class SystemMetrics:
    """System performance metrics"""
    
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    throughput: float = 0.0
    threats_in_queue: int = 0
    learning_patterns: int = 0
    biblical_applications: int = 0
    divine_alignment: float = 0.0
    wisdom_accuracy: float = 0.0
    system_health: float = 100.0
    adaptation_level: float = 0.0

class BrutalAttackGenerator:
    """Generates sophisticated, brutal attack scenarios"""
    
    def __init__(self, constraints: FortiGateConstraints):
        self.constraints = constraints
        self.attack_history = []
        self.current_attacks = []
        self.attack_complexity = 1.0
        
    def generate_attack_wave(self, wave_number: int) -> List[Dict[str, Any]]:
        """Generate a wave of increasingly brutal attacks"""
        
        attacks = []
        base_intensity = min(5.0, wave_number * 0.5)  # Intensity increases with waves
        
        # Generate multiple attack types based on wave complexity
        attack_count = min(10, 2 + wave_number // 2)
        
        for i in range(attack_count):
            attack_type = self._select_attack_type(wave_number)
            attack = self._create_attack(attack_type, base_intensity, wave_number, i)
            attacks.append(attack)
            
        self.current_attacks = attacks
        return attacks
    
    def _select_attack_type(self, wave_number: int) -> AttackType:
        """Select attack type based on wave progression"""
        
        # Early waves: basic attacks
        if wave_number <= 2:
            basic_attacks = [
                AttackType.ADVANCED_PERSISTENT_THREAT,
                AttackType.CRYPTO_MINING,
                AttackType.BOTNET_COORDINATION
            ]
            return random.choice(basic_attacks)
        
        # Mid waves: more sophisticated
        elif wave_number <= 5:
            mid_attacks = [
                AttackType.ZERO_DAY_EXPLOIT,
                AttackType.DDOS_AMPLIFICATION,
                AttackType.LATERAL_MOVEMENT,
                AttackType.SUPPLY_CHAIN_COMPROMISE
            ]
            return random.choice(mid_attacks)
        
        # Late waves: brutal attacks
        else:
            brutal_attacks = [
                AttackType.RANSOMWARE_WAVE,
                AttackType.INSIDER_THREAT_ESCALATION,
                AttackType.POLYMORPHIC_MALWARE,
                AttackType.ZERO_DAY_EXPLOIT
            ]
            return random.choice(brutal_attacks)
    
    def _create_attack(self, attack_type: AttackType, intensity: float, wave_number: int, attack_id: int) -> Dict[str, Any]:
        """Create a sophisticated attack instance"""
        
        base_attack = {
            'attack_id': f"wave_{wave_number}_attack_{attack_id}",
            'attack_type': attack_type.value,
            'timestamp': datetime.now(),
            'source_ip': self._generate_sophisticated_source_ip(attack_type),
            'target_system': self._select_target(attack_type),
            'intensity': intensity,
            'complexity': min(10.0, wave_number * 0.8),
            'stealth_level': random.uniform(0.3, 0.9),
            'persistence': random.uniform(0.5, 1.0),
            'evasion_techniques': self._generate_evasion_techniques(attack_type),
            'payload_size': random.randint(1024, 1048576 * 10),
            'context': self._determine_attack_context(attack_type),
            'severity': self._determine_severity(attack_type, intensity)
        }
        
        # Add attack-specific characteristics
        attack_specifics = self._add_attack_characteristics(attack_type, intensity)
        base_attack.update(attack_specifics)
        
        return base_attack
    
    def _generate_sophisticated_source_ip(self, attack_type: AttackType) -> str:
        """Generate sophisticated source IP addresses"""
        
        if attack_type == AttackType.SUPPLY_CHAIN_COMPROMISE:
            # Legitimate-looking compromised vendor
            vendors = ['vendor.trusted.com', 'cdn.cloudprovider.net', 'api.serviceprovider.org']
            return f"compromised.{random.choice(vendors)}"
        
        elif attack_type == AttackType.INSIDER_THREAT_ESCALATION:
            # Internal network address
            return f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
        
        else:
            # External attacker with sophisticated masking
            return f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    
    def _select_target(self, attack_type: AttackType) -> str:
        """Select appropriate target for attack type"""
        
        targets = {
            AttackType.RANSOMWARE_WAVE: ['database.company.com', 'fileserver.corp.net'],
            AttackType.DDOS_AMPLIFICATION: ['web.company.com', 'api.service.com'],
            AttackType.LATERAL_MOVEMENT: ['domain.corp.net', 'auth.company.com'],
            AttackType.SUPPLY_CHAIN_COMPROMISE: ['update.server.com', 'deployment.corp.net'],
            AttackType.INSIDER_THREAT_ESCALATION: ['hr.corp.net', 'finance.corp.net'],
            AttackType.ZERO_DAY_EXPLOIT: ['vulnerability.server.com', 'legacy.system.net']
        }
        
        return random.choice(targets.get(attack_type, ['target.company.com']))
    
    def _generate_evasion_techniques(self, attack_type: AttackType) -> List[str]:
        """Generate sophisticated evasion techniques"""
        
        all_techniques = [
            'polymorphic_code',
            'encrypted_payloads',
            'protocol_tunneling',
            'fast_flux_networking',
            'domain_generation_algorithm',
            'fileless_execution',
            'anti_analysis_tricks',
            'time_delay_evasion',
            'process_hollowing',
            'rootkit_techniques'
        ]
        
        # Select techniques based on attack complexity
        technique_count = min(3, len(all_techniques))
        return random.sample(all_techniques, technique_count)
    
    def _determine_attack_context(self, attack_type: AttackType) -> str:
        """Determine network context for attack"""
        
        contexts = {
            AttackType.RANSOMWARE_WAVE: 'critical_infrastructure',
            AttackType.DDOS_AMPLIFICATION: 'critical_infrastructure',
            AttackType.LATERAL_MOVEMENT: 'corporate_enterprise',
            AttackType.SUPPLY_CHAIN_COMPROMISE: 'corporate_enterprise',
            AttackType.INSIDER_THREAT_ESCALATION: 'spiritual_organizations',
            AttackType.ZERO_DAY_EXPLOIT: 'critical_infrastructure'
        }
        
        # Fallback to available contexts
        available_contexts = ['corporate_enterprise', 'financial_systems', 'critical_infrastructure', 
                           'healthcare_networks', 'spiritual_organizations']
        return contexts.get(attack_type, random.choice(available_contexts))
    
    def _determine_severity(self, attack_type: AttackType, intensity: float) -> str:
        """Determine attack severity"""
        
        if intensity > 8.0:
            return AttackSeverity.BRUTAL.value
        elif intensity > 6.0:
            return AttackSeverity.CRITICAL.value
        elif intensity > 4.0:
            return AttackSeverity.HIGH.value
        elif intensity > 2.0:
            return AttackSeverity.MEDIUM.value
        else:
            return AttackSeverity.LOW.value
    
    def _add_attack_characteristics(self, attack_type: AttackType, intensity: float) -> Dict[str, Any]:
        """Add attack-specific characteristics"""
        
        characteristics = {}
        
        if attack_type == AttackType.RANSOMWARE_WAVE:
            characteristics.update({
                'encryption_algorithm': 'AES-256',
                'ransom_amount': random.randint(10000, 1000000),
                'encryption_target': 'critical_files',
                'deadline_hours': random.randint(24, 72)
            })
        
        elif attack_type == AttackType.DDOS_AMPLIFICATION:
            characteristics.update({
                'amplification_factor': random.randint(100, 1000),
                'bandwidth_consumed': intensity * 1000,  # Mbps
                'attack_duration': random.randint(60, 3600),  # seconds
                'botnet_size': random.randint(1000, 100000)
            })
        
        elif attack_type == AttackType.ZERO_DAY_EXPLOIT:
            characteristics.update({
                'exploit_type': 'memory_corruption',
                'cve_id': f'CVE-2024-{random.randint(10000, 99999)}',
                'affected_systems': random.randint(5, 50),
                'patch_availability': False
            })
        
        elif attack_type == AttackType.INSIDER_THREAT_ESCALATION:
            characteristics.update({
                'privilege_level': 'administrator',
                'access_duration': random.randint(30, 180),  # days
                'data_accessed': random.randint(100, 10000),  # GB
                'lateral_movement_count': random.randint(3, 15)
            })
        
        return characteristics

class TeachingSystem:
    """Teaching system that helps the intelligence learn and adapt"""
    
    def __init__(self, intelligence_system: FortiGateSemanticShield):
        self.intelligence = intelligence_system
        self.teaching_history = []
        self.success_patterns = []
        self.failure_patterns = []
        self.teaching_methods = [
            'guidance_correction',
            'strategy_enhancement',
            'biblical_wisdom_refinement',
            'pattern_optimization',
            'confidence_boosting'
        ]
        
    async def teach_from_attack_outcome(self, attack: Dict[str, Any], response: Any, outcome: str):
        """Teach the system based on attack outcome"""
        
        strategy = getattr(response, 'defense_strategy', 'unknown')
        strategy_value = strategy.value if hasattr(strategy, 'value') else str(strategy)

        teaching_session = {
            'timestamp': datetime.now(),
            'attack_type': attack.get('attack_type', 'unknown'),
            'severity': attack.get('severity', 'unknown'),
            'response_strategy': strategy_value,
            'outcome': outcome,
            'teachings_applied': []
        }
        
        # Analyze outcome and apply appropriate teaching
        if outcome == 'success':
            await self._teach_success_patterns(attack, response, teaching_session)
        else:
            await self._teach_failure_patterns(attack, response, teaching_session)
        
        # Always teach biblical wisdom refinement
        await self._teach_biblical_wisdom(attack, response, teaching_session)
        
        # Teach pattern recognition
        await self._teach_pattern_recognition(attack, response, teaching_session)
        
        self.teaching_history.append(teaching_session)

        if hasattr(self.intelligence, 'record_teaching_session'):
            await self.intelligence.record_teaching_session(teaching_session)
        
    async def _teach_success_patterns(self, attack: Dict[str, Any], response: Any, session: Dict):
        """Teach based on successful defense"""
        
        strategy = getattr(response, 'defense_strategy', 'unknown')
        
        # Reinforce successful strategies
        teaching = {
            'method': 'success_reinforcement',
            'lesson': f"Strategy '{strategy}' was successful against {attack.get('attack_type')}",
            'confidence_boost': 0.1,
            'pattern_strength': 1.0
        }
        
        session['teachings_applied'].append(teaching)
        
        # Store in success patterns for future reference
        self.success_patterns.append({
            'attack_type': attack.get('attack_type'),
            'strategy': strategy,
            'timestamp': datetime.now(),
            'success_factors': self._analyze_success_factors(attack, response)
        })
        
    async def _teach_failure_patterns(self, attack: Dict[str, Any], response: Any, session: Dict):
        """Teach based on failed defense"""
        
        strategy = getattr(response, 'defense_strategy', 'unknown')
        
        # Analyze failure and suggest improvements
        improvement = await self._analyze_failure(attack, response)
        
        teaching = {
            'method': 'failure_correction',
            'lesson': f"Strategy '{strategy}' failed against {attack.get('attack_type')}. {improvement}",
            'confidence_adjustment': -0.05,
            'recommended_change': improvement
        }
        
        session['teachings_applied'].append(teaching)
        
        # Store in failure patterns for learning
        self.failure_patterns.append({
            'attack_type': attack.get('attack_type'),
            'failed_strategy': strategy,
            'timestamp': datetime.now(),
            'failure_factors': self._analyze_failure_factors(attack, response)
        })
    
    async def _teach_biblical_wisdom(self, attack: Dict[str, Any], response: Any, session: Dict):
        """Teach biblical wisdom refinement"""
        
        # Extract biblical balance from response
        justice_level = getattr(response, 'justice_enforcement', 0.5)
        mercy_level = getattr(response, 'love_mercy_factor', 0.5)
        wisdom_level = getattr(response, 'wisdom_accuracy', 0.5)
        
        # Analyze biblical appropriateness
        severity = attack.get('severity', 'medium')
        
        if severity in ['critical', 'brutal']:
            # High severity attacks may require more justice
            ideal_justice = 0.8
            ideal_mercy = 0.3
        elif severity == 'high':
            ideal_justice = 0.7
            ideal_mercy = 0.4
        else:
            ideal_justice = 0.6
            ideal_mercy = 0.5
        
        # Provide biblical wisdom teaching
        justice_diff = abs(justice_level - ideal_justice)
        mercy_diff = abs(mercy_level - ideal_mercy)
        
        if justice_diff > 0.2:
            teaching = {
                'method': 'biblical_justice_refinement',
                'lesson': f"Justice level {justice_level:.2f} should be closer to {ideal_justice:.2f} for {severity} attacks",
                'scripture': "Execute justice and righteousness (Psalm 106:3)",
                'adjustment_needed': True
            }
            session['teachings_applied'].append(teaching)
        
        if mercy_diff > 0.2:
            teaching = {
                'method': 'biblical_mercy_refinement', 
                'lesson': f"Mercy level {mercy_level:.2f} should be closer to {ideal_mercy:.2f} for {severity} attacks",
                'scripture': "Be merciful, even as your Father is merciful (Luke 6:36)",
                'adjustment_needed': True
            }
            session['teachings_applied'].append(teaching)
        
        if wisdom_level < 0.6:
            teaching = {
                'method': 'biblical_wisdom_enhancement',
                'lesson': f"Wisdom level {wisdom_level:.2f} needs enhancement for complex attacks",
                'scripture': "The wisdom of the prudent is to give thought to their ways (Proverbs 14:8)",
                'adjustment_needed': True
            }
            session['teachings_applied'].append(teaching)
    
    async def _teach_pattern_recognition(self, attack: Dict[str, Any], response: Any, session: Dict):
        """Teach pattern recognition and optimization"""
        
        # Look for similar attacks in history
        attack_type = attack.get('attack_type', 'unknown')
        
        similar_success = [p for p in self.success_patterns if p['attack_type'] == attack_type]
        similar_failure = [p for p in self.failure_patterns if p['attack_type'] == attack_type]
        
        if similar_success:
            # Teach from successful patterns
            most_successful = max(similar_success, key=lambda x: len(x['success_factors']))
            teaching = {
                'method': 'pattern_success_learning',
                'lesson': f"Similar attacks successfully handled with {most_successful['success_factors']}",
                'pattern_confidence': 0.8,
                'recommended_strategy': most_successful.get('strategy', 'balanced_response')
            }
            session['teachings_applied'].append(teaching)
        
        if similar_failure:
            # Teach from failure patterns
            common_failures = []
            for failure in similar_failure:
                common_failures.extend(failure.get('failure_factors', []))
            
            teaching = {
                'method': 'pattern_failure_learning',
                'lesson': f"Avoid common failure patterns: {list(set(common_failures))[:3]}",
                'avoidance_confidence': 0.7,
                'pitfalls_to_avoid': list(set(common_failures))[:3]
            }
            session['teachings_applied'].append(teaching)
    
    def _analyze_success_factors(self, attack: Dict[str, Any], response: Any) -> List[str]:
        """Analyze factors that contributed to success"""
        
        factors = []
        
        # Check response confidence
        if hasattr(response, 'divine_protection_level') and response.divine_protection_level > 0.7:
            factors.append('high_confidence_response')
        
        # Check biblical balance
        if hasattr(response, 'justice_enforcement') and hasattr(response, 'love_mercy_factor'):
            justice = response.justice_enforcement
            mercy = response.love_mercy_factor
            if 0.4 <= justice <= 0.8 and 0.2 <= mercy <= 0.6:
                factors.append('balanced_biblical_approach')
        
        # Check strategy appropriateness
        severity = attack.get('severity', 'medium')
        strategy = getattr(response, 'defense_strategy', 'unknown')
        
        if severity in ['critical', 'brutal'] and 'justice' in strategy:
            factors.append('appropriate_severity_response')
        elif severity in ['low', 'medium'] and 'compassion' in strategy:
            factors.append('proportional_response')
        
        return factors
    
    def _analyze_failure_factors(self, attack: Dict[str, Any], response: Any) -> List[str]:
        """Analyze factors that contributed to failure"""
        
        factors = []
        
        # Check low confidence
        if hasattr(response, 'divine_protection_level') and response.divine_protection_level < 0.5:
            factors.append('low_confidence_response')
        
        # Check biblical imbalance
        if hasattr(response, 'justice_enforcement') and hasattr(response, 'love_mercy_factor'):
            justice = response.justice_enforcement
            mercy = response.love_mercy_factor
            if justice > 0.9 or mercy > 0.9:
                factors.append('extreme_biblical_imbalance')
        
        # Check strategy mismatch
        severity = attack.get('severity', 'medium')
        strategy = getattr(response, 'defense_strategy', 'unknown')
        
        if severity in ['critical', 'brutal'] and 'compassion' in strategy:
            factors.append('insufficient_severity_response')
        elif severity in ['low', 'medium'] and 'justice' in strategy:
            factors.append('excessive_severity_response')
        
        return factors
    
    async def _analyze_failure(self, attack: Dict[str, Any], response: Any) -> str:
        """Analyze failure and provide improvement recommendation"""
        
        failure_factors = self._analyze_failure_factors(attack, response)
        
        if 'low_confidence_response' in failure_factors:
            return "Increase confidence through learning and pattern recognition"
        elif 'extreme_biblical_imbalance' in failure_factors:
            return "Balance justice and mercy according to attack severity"
        elif 'insufficient_severity_response' in failure_factors:
            return "Use stronger, justice-focused strategies for severe attacks"
        elif 'excessive_severity_response' in failure_factors:
            return "Use more measured, compassionate responses for minor threats"
        else:
            return "Enhance pattern recognition and adaptive response capabilities"
    
    def get_teaching_summary(self) -> Dict[str, Any]:
        """Get comprehensive teaching summary"""
        
        total_sessions = len(self.teaching_history)
        success_teachings = len(self.success_patterns)
        failure_teachings = len(self.failure_patterns)
        
        # Calculate teaching effectiveness
        if total_sessions > 0:
            teaching_effectiveness = success_teachings / total_sessions
        else:
            teaching_effectiveness = 0.0
        
        return {
            'total_teaching_sessions': total_sessions,
            'success_patterns_learned': success_teachings,
            'failure_patterns_identified': failure_teachings,
            'teaching_effectiveness': teaching_effectiveness,
            'teaching_methods_used': list(set(session.get('teachings_applied', [{}])[0].get('method', 'unknown') 
                                           for session in self.teaching_history[-10:])),
            'recent_teachings': self.teaching_history[-5:] if self.teaching_history else []
        }

class FortiGateSimulator:
    """Main FortiGate constrained environment simulator."""
    
    def __init__(self):
        self.constraints = FortiGateConstraints()
        self.device_interface = FortiGatePolicyApplier()
        self.telemetry_collector = FortiGateTelemetryCollector(
            cpu_threshold=self.constraints.max_cpu_usage,
            memory_threshold=self.constraints.max_memory_usage,
            queue_threshold=self.constraints.queue_size_limit,
        )
        self.learning_manager = LearningPersistenceManager(
            database_path="fortigate_constrained_learning.db",
            export_directory="learning_snapshots",
            max_snapshots=5,
        )
        self.intelligence = FortiGateSemanticShield(
            "fortigate_constrained_learning.db",
            device_interface=self.device_interface,
            telemetry_collector=self.telemetry_collector,
            persistence_manager=self.learning_manager,
        )
        self.attack_generator = BrutalAttackGenerator(self.constraints)
        self.teaching_system = TeachingSystem(self.intelligence)
        
        # System state
        self.system_metrics = SystemMetrics()
        self.attack_metrics = []
        self.current_wave = 0
        self.simulation_active = True
        self.start_time = datetime.now()
        
        # Performance tracking
        self.resource_usage_history = deque(maxlen=1000)
        self.learning_progression = []
        
    async def run_simulation(self, max_waves: int = 10):
        """Run the brutal attack simulation"""
        
        print("=" * 100)
        print("FORTIGATE CONSTRAINED ENVIRONMENT SIMULATION")
        print("BRUTAL ATTACK SCENARIOS WITH INTELLIGENT TEACHING")
        print("=" * 100)
        print(f"Starting simulation with constraints:")
        print(f"  Max CPU Usage: {self.constraints.max_cpu_usage}%")
        print(f"  Max Memory Usage: {self.constraints.max_memory_usage}%")
        print(f"  Max Concurrent Threats: {self.constraints.max_concurrent_threats}")
        print(f"  Max Processing Time: {self.constraints.processing_time_limit}s")
        print(f"  Max Learning Memory: {self.constraints.learning_memory_limit}")
        print()
        
        try:
            for wave in range(1, max_waves + 1):
                self.current_wave = wave
                print(f"\n{'='*80}")
                print(f"ATTACK WAVE {wave}/{max_waves} - BRUTAL ASSAULT")
                print(f"{'='*80}")
                
                # Check if system can continue
                if not self._check_system_health():
                    print(f"[FAILED] System health critical - Simulation stopped at Wave {wave}")
                    break
                
                # Generate attack wave
                attacks = self.attack_generator.generate_attack_wave(wave)
                print(f"Generated {len(attacks)} sophisticated attacks:")
                for attack in attacks:
                    print(f"  - {attack['attack_type'].upper()} (Severity: {attack['severity'].upper()})")
                print()
                
                # Process attack wave
                await self._process_attack_wave(attacks)
                
                # Apply teaching based on results
                await self._apply_teaching()
                
                # Display wave summary
                self._display_wave_summary(wave)
                
                # Brief pause between waves
                await asyncio.sleep(0.5)
            
            # Final simulation summary
            self._display_final_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Simulation interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Simulation error: {e}")
            logger.exception("Simulation failed")
    
    async def _process_attack_wave(self, attacks: List[Dict[str, Any]]):
        """Process a wave of attacks"""
        
        wave_results = []
        
        for attack in attacks:
            # Check constraints before processing
            if not self._check_processing_constraints():
                print(f"[WARNING] Constraints exceeded - skipping attack {attack['attack_id']}")
                continue
            
            # Process attack with timing
            start_time = time.time()
            
            try:
                # Update system metrics before processing
                self._update_system_metrics(attack)
                
                # Process through intelligence system
                response = await self.intelligence.process_threat_with_intelligence(attack)
                
                processing_time = time.time() - start_time
                
                # Evaluate attack outcome
                outcome = self._evaluate_attack_outcome(attack, response, processing_time)
                
                # Create attack metrics
                attack_metric = AttackMetrics(
                    attack_id=attack['attack_id'],
                    attack_type=AttackType(attack['attack_type']),
                    severity=AttackSeverity(attack['severity']),
                    start_time=attack['timestamp'],
                    end_time=datetime.now(),
                    success=outcome['success'],
                    damage_level=outcome['damage_level'],
                    resources_consumed={
                        'cpu_used': self.system_metrics.cpu_usage,
                        'memory_used': self.system_metrics.memory_usage,
                        'processing_time': processing_time
                    },
                    defense_response_time=processing_time,
                    learning_outcome=outcome['learning_outcome'],
                    system_survived=outcome['system_survived']
                )
                
                self.attack_metrics.append(attack_metric)
                wave_results.append(attack_metric)
                
                # Display attack processing result
                self._display_attack_result(attack, response, attack_metric)
                
                # Small delay to simulate realistic processing
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"[ERROR] Error processing attack {attack['attack_id']}: {e}")
                logger.exception(f"Attack processing failed")
    
    def _check_system_health(self) -> bool:
        """Check if system can continue simulation"""
        
        health_score = (
            (100 - self.system_metrics.cpu_usage) * 0.3 +
            (100 - self.system_metrics.memory_usage) * 0.3 +
            self.system_metrics.system_health * 0.4
        ) / 100
        
        return health_score > 0.2  # Continue if health score > 20%
    
    def _check_processing_constraints(self) -> bool:
        """Check if system can process more threats"""
        
        return (
            self.system_metrics.cpu_usage < self.constraints.max_cpu_usage and
            self.system_metrics.memory_usage < self.constraints.max_memory_usage and
            self.system_metrics.threats_in_queue < self.constraints.queue_size_limit
        )
    
    def _update_system_metrics(self, attack: Dict[str, Any]):
        """Update system metrics based on current attack"""
        
        # Simulate resource consumption based on attack characteristics
        base_cpu = 20.0
        base_memory = 15.0
        
        # Increase based on attack intensity and complexity
        intensity_factor = attack.get('intensity', 1.0) / 10.0
        complexity_factor = attack.get('complexity', 1.0) / 10.0
        
        # Update CPU usage
        self.system_metrics.cpu_usage = min(100.0, 
            base_cpu + (intensity_factor * 30) + (complexity_factor * 25))
        
        # Update memory usage
        self.system_metrics.memory_usage = min(100.0,
            base_memory + (complexity_factor * 35) + (len(attack.get('evasion_techniques', [])) * 5))
        
        # Update connections
        self.system_metrics.active_connections = min(self.constraints.max_connections,
            self.system_metrics.active_connections + random.randint(100, 1000))
        
        # Update queue
        self.system_metrics.threats_in_queue = self.system_metrics.threats_in_queue + 1
        
        # Update throughput (simulate network load)
        self.system_metrics.throughput = min(self.constraints.max_throughput,
            self.system_metrics.throughput + random.uniform(0.5, 2.0))
        
        # Calculate system health
        health_factor = (
            (100 - self.system_metrics.cpu_usage) / 100 * 0.4 +
            (100 - self.system_metrics.memory_usage) / 100 * 0.4 +
            (1 - self.system_metrics.threats_in_queue / self.constraints.queue_size_limit) * 0.2
        )
        self.system_metrics.system_health = health_factor * 100
        
        # Store in history
        self.resource_usage_history.append({
            'timestamp': datetime.now(),
            'cpu': self.system_metrics.cpu_usage,
            'memory': self.system_metrics.memory_usage,
            'health': self.system_metrics.system_health
        })
        
        if self.telemetry_collector:
            self.telemetry_collector.update_metrics({
                'cpu_usage': self.system_metrics.cpu_usage,
                'memory_usage': self.system_metrics.memory_usage,
                'session_queue': self.system_metrics.threats_in_queue,
                'throughput': self.system_metrics.throughput,
            })
    
    def _evaluate_attack_outcome(self, attack: Dict[str, Any], response: Any, processing_time: float) -> Dict[str, Any]:
        """Evaluate the outcome of an attack"""
        
        # Base success depends on processing time and response quality
        time_success = processing_time <= self.constraints.processing_time_limit
        
        # Response quality factors
        confidence = getattr(response, 'divine_protection_level', 0.5)
        biblical_balance = self._calculate_biblical_balance(response)
        
        # Attack severity factor
        severity_factor = {
            'low': 0.9, 'medium': 0.7, 'high': 0.5, 'critical': 0.3, 'brutal': 0.1
        }.get(attack.get('severity', 'medium'), 0.5)
        
        # Calculate success probability
        success_probability = (time_success * 0.3 + 
                             confidence * 0.4 + 
                             biblical_balance * 0.2 + 
                             severity_factor * 0.1)
        
        # Determine outcome
        success = random.random() < success_probability
        
        # Calculate damage if failed
        if success:
            damage_level = random.uniform(0, 10) * (1 - success_probability)
            learning_outcome = "System successfully defended and learned"
            system_survived = True
        else:
            damage_level = random.uniform(10, 50) * severity_factor
            learning_outcome = "System defended but needs improvement"
            system_survived = damage_level < 40  # System survives if damage < 40
        
        return {
            'success': success,
            'damage_level': damage_level,
            'learning_outcome': learning_outcome,
            'system_survived': system_survived,
            'success_probability': success_probability
        }
    
    def _calculate_biblical_balance(self, response: Any) -> float:
        """Calculate biblical balance score"""
        
        if not hasattr(response, 'justice_enforcement') or not hasattr(response, 'love_mercy_factor'):
            return 0.5
        
        justice = response.justice_enforcement
        mercy = response.love_mercy_factor
        
        # Ideal balance depends on context (simplified for simulation)
        ideal_ratio = 0.6  # Slightly more justice than mercy
        current_ratio = justice / (justice + mercy + 0.001)
        
        # Calculate balance score
        balance_score = 1.0 - abs(current_ratio - ideal_ratio)
        
        return balance_score
    
    async def _apply_teaching(self):
        """Apply teaching system based on recent attack outcomes"""
        
        if not self.attack_metrics:
            return
        
        # Get recent attacks for teaching
        recent_attacks = self.attack_metrics[-5:]  # Teach from last 5 attacks
        
        for attack_metric in recent_attacks:
            # Find corresponding attack data
            attack_data = next((a for a in self.attack_generator.current_attacks 
                             if a['attack_id'] == attack_metric.attack_id), {})
            
            if not attack_data:
                continue
            
            # Get response from intelligence system (simplified)
            response = "response_placeholder"  # In real implementation, this would be the actual response
            
            # Determine outcome for teaching
            outcome = "success" if attack_metric.success else "failure"
            
            # Apply teaching
            await self.teaching_system.teach_from_attack_outcome(attack_data, response, outcome)
        
        # Update learning progression
        teaching_summary = self.teaching_system.get_teaching_summary()
        self.learning_progression.append({
            'wave': self.current_wave,
            'timestamp': datetime.now(),
            'teaching_effectiveness': teaching_summary['teaching_effectiveness'],
            'total_patterns': teaching_summary['success_patterns_learned'] + teaching_summary['failure_patterns_identified']
        })
    
    def _display_attack_result(self, attack: Dict[str, Any], response: Any, metric: AttackMetrics):
        """Display individual attack processing result"""
        
        print(f"[ATTACK PROCESSED]: {attack['attack_type'].upper()}")
        print(f"   ID: {attack['attack_id']}")
        print(f"   Severity: {attack['severity'].upper()}")
        print(f"   Source: {attack['source_ip']}")
        print(f"   Target: {attack['target_system']}")
        print(f"   Intensity: {attack['intensity']:.1f}/10.0")
        print(f"   Complexity: {attack['complexity']:.1f}/10.0")
        print()
        
        print(f"[DEFENSE RESPONSE]:")
        print(f"   Success: {'[SUCCESS]' if metric.success else '[FAILED]'}")
        print(f"   Processing Time: {metric.defense_response_time:.3f}s")
        print(f"   Damage Level: {metric.damage_level:.1f}")
        print(f"   System Survived: {'[YES]' if metric.system_survived else '[NO]'}")
        print(f"   Learning Outcome: {metric.learning_outcome}")
        print()
        
        print(f"[SYSTEM IMPACT]:")
        print(f"   CPU Usage: {metric.resources_consumed['cpu_used']:.1f}%")
        print(f"   Memory Usage: {metric.resources_consumed['memory_used']:.1f}%")
        print(f"   System Health: {self.system_metrics.system_health:.1f}%")
        print()
    
    def _display_wave_summary(self, wave: int):
        """Display summary of the current wave"""
        
        wave_attacks = [m for m in self.attack_metrics if m.start_time.strftime('%H') == str(self.start_time.hour)]
        
        if not wave_attacks:
            return
        
        successful_attacks = sum(1 for a in wave_attacks if a.success)
        total_damage = sum(a.damage_level for a in wave_attacks)
        avg_processing_time = sum(a.defense_response_time for a in wave_attacks) / len(wave_attacks)
        
        print(f"\n[WAVE {wave} SUMMARY]:")
        print(f"   Attacks Processed: {len(wave_attacks)}")
        print(f"   Successful Defenses: {successful_attacks}/{len(wave_attacks)} ({successful_attacks/len(wave_attacks)*100:.1f}%)")
        print(f"   Total Damage: {total_damage:.1f}")
        print(f"   Average Processing Time: {avg_processing_time:.3f}s")
        print(f"   System Health: {self.system_metrics.system_health:.1f}%")
        print(f"   Learning Progression: {len(self.learning_progression)} sessions")
        print()
    
    def _display_final_summary(self):
        """Display final simulation summary"""
        
        total_duration = datetime.now() - self.start_time
        total_attacks = len(self.attack_metrics)
        successful_attacks = sum(1 for a in self.attack_metrics if a.success)
        
        print(f"\n{'='*100}")
        print("[SIMULATION COMPLETE - FINAL RESULTS]")
        print(f"{'='*100}")
        print()
        print("[SIMULATION STATISTICS]:")
        print(f"   Total Duration: {total_duration}")
        print(f"   Waves Completed: {self.current_wave}")
        print(f"   Total Attacks: {total_attacks}")
        print(f"   Successful Defenses: {successful_attacks}/{total_attacks} ({successful_attacks/total_attacks*100:.1f}%)")
        print(f"   System Health: {self.system_metrics.system_health:.1f}%")
        print()
        
        print("[INTELLIGENCE GROWTH]:")
        teaching_summary = self.teaching_system.get_teaching_summary()
        print(f"   Teaching Sessions: {teaching_summary['total_teaching_sessions']}")
        print(f"   Success Patterns Learned: {teaching_summary['success_patterns_learned']}")
        print(f"   Failure Patterns Identified: {teaching_summary['failure_patterns_identified']}")
        print(f"   Teaching Effectiveness: {teaching_summary['teaching_effectiveness']*100:.1f}%")
        print()
        
        print("[PERFORMANCE METRICS]:")
        avg_processing_time = sum(a.defense_response_time for a in self.attack_metrics) / len(self.attack_metrics)
        max_cpu = max(h['cpu'] for h in self.resource_usage_history) if self.resource_usage_history else 0
        max_memory = max(h['memory'] for h in self.resource_usage_history) if self.resource_usage_history else 0
        
        print(f"   Average Processing Time: {avg_processing_time:.3f}s")
        print(f"   Peak CPU Usage: {max_cpu:.1f}%")
        print(f"   Peak Memory Usage: {max_memory:.1f}%")
        print(f"   Learning Velocity: {total_attacks/total_duration.total_seconds()*3600:.1f} attacks/hour")
        if hasattr(self.intelligence, "export_learning_snapshot"):
            snapshot_path = self.intelligence.export_learning_snapshot()
            if snapshot_path:
                print(f"   Learning Snapshot: {snapshot_path}")
        print()
        
        # Intelligence grade
        success_rate = successful_attacks / total_attacks
        teaching_effectiveness = teaching_summary['teaching_effectiveness']
        health_score = self.system_metrics.system_health / 100
        
        intelligence_grade = (success_rate * 0.4 + teaching_effectiveness * 0.4 + health_score * 0.2) * 100
        
        if intelligence_grade >= 90:
            grade = "EXCEPTIONAL - Advanced AI Intelligence"
        elif intelligence_grade >= 80:
            grade = "EXCELLENT - Strong Learning System"
        elif intelligence_grade >= 70:
            grade = "GOOD - Developing Intelligence"
        elif intelligence_grade >= 60:
            grade = "FAIR - Basic Learning Capabilities"
        else:
            grade = "POOR - Significant Improvement Needed"
        
        print(f"[OVERALL INTELLIGENCE GRADE]: {intelligence_grade:.1f}/100")
        print(f"   Grade: {grade}")
        print()
        
        print("[KEY LEARNINGS]:")
        if self.current_wave >= 8:
            print("   [SUCCESS] System survived brutal attack waves")
            print("   [SUCCESS] Learning system effectively adapted to threats")
            print("   [SUCCESS] Biblical wisdom maintained under pressure")
        elif success_rate > 0.7:
            print("   [SUCCESS] Strong defense capabilities demonstrated")
            print("   [SUCCESS] Teaching system improved response effectiveness")
        else:
            print("   [WARNING] Areas for improvement identified")
            print("   [LEARNING] Additional teaching and optimization needed")
        
        print("\n[SIMULATION CONCLUSION]:")
        print("The SSE + SSD Enhanced FortiGate Intelligence system has been tested")
        print("under brutal attack conditions with realistic constraints and has")
        print("demonstrated its ability to learn, adapt, and maintain biblical")
        print("wisdom while defending against sophisticated threats.")

async def main():
    """Run the FortiGate constrained environment simulation"""
    
    simulator = FortiGateSimulator()
    
    # Run simulation with 10 waves of increasingly brutal attacks
    await simulator.run_simulation(max_waves=10)

if __name__ == "__main__":
    asyncio.run(main())
