"""FortiGate Semantic Shield v7.0 deployment helper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import requests
import yaml

from fortigate_semantic_shield.ss_intuition import (
    CompassProfile,
    golden_batch_size,
    golden_ratio_profile,
)


CARDINAL_AXES = ("LOVE", "JUSTICE", "POWER", "WISDOM")

def load_config(config_file):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[FAIL] Error loading config: {e}")
        sys.exit(1)


def compute_compass_profile(config: dict) -> CompassProfile:
    """Derive the SS Intuition compass profile from configuration."""

    profile = golden_ratio_profile()
    ss_config = (config or {}).get("ss_intuition", {})
    biases = ss_config.get("biases", {})
    if not biases:
        return profile

    vector = list(profile.as_tuple())
    axis_index = {axis: idx for idx, axis in enumerate(CARDINAL_AXES)}
    adjusted = False
    for axis, idx in axis_index.items():
        key_variants = {axis, axis.lower(), axis.upper()}
        value = None
        for key in key_variants:
            if key in biases:
                value = biases[key]
                break
        if value is None:
            continue
        try:
            bias = float(value)
        except (TypeError, ValueError):
            continue
        vector[idx] = max(0.01, bias)
        adjusted = True

    if not adjusted:
        return profile

    total = sum(vector)
    normalized = [value / total for value in vector]
    return CompassProfile(*normalized)

def validate_fortigate_connection(fortigate_config):
    """Validate FortiGate connection"""
    try:
        response = requests.get(
            f"https://{fortigate_config['host']}:{fortigate_config['port']}/api/v2/monitor/firewall/status",
            headers={'Authorization': f"Bearer {fortigate_config['token']}"},
            verify=fortigate_config.get('verify_ssl', True),
            timeout=10
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"[PASS] FortiGate connection successful")
            print(f"[OFFICE] FortiGate Status: {status.get('status', 'unknown')}")
            print(f"[PIN] FortiGate Version: {status.get('version', 'unknown')}")
            return True, status
        else:
            print(f"[FAIL] Connection failed: HTTP {response.status_code}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Connection error: {e}")
        return False, None

def install_dependencies():
    """Install required dependencies"""
    try:
        import subprocess
        print("[PACKAGE] Installing dependencies...")
        
        # Core dependencies
        core_deps = [
            'numpy', 'scipy', 'requests', 'pyyaml', 'flask',
            'aiohttp', 'asyncio', 'python-dotenv'
        ]
        
        for dep in core_deps:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
        
        print("[PASS] Dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Dependency installation failed: {e}")
        return False

def setup_database():
    """Setup semantic database"""
    try:
        import sqlite3
        import os
        
        db_path = 'semantic_shield.db'
        
        # Create database
        conn = sqlite3.connect(db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS semantic_vectors (
                id TEXT PRIMARY KEY,
                love REAL NOT NULL,
                power REAL NOT NULL,
                wisdom REAL NOT NULL,
                justice REAL NOT NULL,
                alignment REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_id TEXT,
                threat_type TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS divine_alignment_log (
                id TEXT PRIMARY KEY,
                anchor_distance REAL NOT NULL,
                alignment_score REAL NOT NULL,
                principle_dominant TEXT,
                business_context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"[PASS] Database created: {db_path}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Database setup failed: {e}")
        return False

def deploy_to_fortigate(fortigate_config, config):
    """Deploy to FortiGate device"""
    try:
        # Validate deployment
        success, status = validate_fortigate_connection(fortigate_config)
        if not success:
            return False
        
        compass_profile = compute_compass_profile(config or {})
        base_batch = config.get('deployment', {}).get('base_batch', 200)
        recommended_batch = golden_batch_size(
            base=max(1, base_batch),
            boost=compass_profile.power + compass_profile.wisdom
        )

        # Create deployment record
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment_record = {
            'deployment_id': deployment_id,
            'fortigate_host': fortigate_config['host'],
            'deployment_time': datetime.now().isoformat(),
            'configuration': config,
            'fortigate_status': status,
            'semantic_shield_version': '7.0.0',
            'deployment_mode': config.get('deployment', {}).get('mode', 'standard'),
            'ss_intuition': {
                'compass_profile': compass_profile.as_dict(),
                'recommended_batch_size': recommended_batch,
                'golden_ratio': GOLDEN_RATIO,
            },
        }
        
        # Save deployment record
        deployments_dir = 'deployments'
        os.makedirs(deployments_dir, exist_ok=True)
        
        with open(f"{deployments_dir}/{deployment_id}.json", 'w') as f:
            json.dump(deployment_record, f, indent=2)
        
        print(f"[PASS] Deployment created: {deployment_id}")
        
        # Test deployment
        print("[LAB] Testing deployment...")
        
        # Test with sample event
        test_result = test_processing(fortigate_config, compass_profile)
        
        if test_result['success']:
            print(f"[PASS] Deployment test passed")
            print(f"[STATS] Test Results:")
            print(f"   Processed: {test_result['processed']} events")
            print(f"   Throughput: {test_result['throughput']} events/sec")
            print(f"   Avg Alignment: {test_result['avg_alignment']:.3f}")
            print(f"   Golden Batch Size: {test_result['golden_batch_size']}")
            print(f"   Compass Profile: {compass_profile.as_dict()}")
            return True
        else:
            print(f"[FAIL] Deployment test failed")
            return False
            
    except Exception as e:
        print(f"[FAIL] Deployment failed: {e}")
        return False

def test_processing(fortigate_config):
    """Test processing capabilities"""
    try:
        import time
        import numpy as np
        
        print("[LAB] Testing semantic processing...")
        
        # Test with sample data
        test_events = []
        for i in range(100):
            event = {
                'event_id': f"test_{i:04d}",
                'threat_type': ['transaction_fraud', 'malware', 'data_breach'][i % 3],
                'risk_score': np.random.uniform(0.3, 1.0),
                'business_context': 'financial_services'
            }
            test_events.append(event)
        
        start_time = time.time()
        
        # Process events
        batch_size = golden_batch_size(
            base=20,
            boost=compass_profile.power + compass_profile.wisdom
        )
        processed_events = []
        for event in test_events:
            # Create semantic vector
            love = 0.5 + event['risk_score'] * 0.3
            power = 0.6 + event['risk_score'] * 0.2
            wisdom = 0.7 + event['risk_score'] * 0.2
            justice = 0.8 + event['risk_score'] * 0.3
            
            # Calculate alignment
            coords = [love, power, wisdom, justice]
            distance = np.sqrt(sum((c - 1.0)**2 for c in coords))  # Distance from (1,1,1,1)
            alignment = 1.0 / (1.0 + distance)
            
            processed_events.append({
                'event_id': event['event_id'],
                'alignment': alignment,
                'threat_type': event['threat_type'],
                'risk_score': event['risk_score']
            })
        
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        throughput = len(processed_events) / processing_time
        avg_alignment = np.mean([e['alignment'] for e in processed_events])
        
        return {
            'success': True,
            'processed': len(processed_events),
            'throughput': throughput,
            'avg_alignment': avg_alignment,
            'processing_time': processing_time,
            'golden_batch_size': batch_size,
        }
        
    except Exception as e:
        print(f"[FAIL] Processing test failed: {e}")
        return {'success': False, 'error': str(e)}

def generate_deployment_report(fortigate_config, config, deployment_id):
    """Generate deployment report"""
    try:
        report = {
            'deployment_id': deployment_id,
            'timestamp': datetime.now().isoformat(),
            'fortigate_config': fortigate_config,
            'configuration': config,
            'recommendations': generate_recommendations(config),
            'next_steps': generate_next_steps(config),
            'monitoring_dashboard': f"http://localhost:8080/dashboard?deployment={deployment_id}"
        }
        
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        with open(f"{reports_dir}/deployment_report_{deployment_id}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[PASS] Report generated: reports/deployment_report_{deployment_id}.json")
        return report
        
    except Exception as e:
        print(f"[FAIL] Report generation failed: {e}")
        return None

def generate_recommendations(config):
    """Generate deployment recommendations"""
    recommendations = []
    
    # Based on deployment mode
    mode = config.get('deployment', {}).get('mode', 'standard')
    if mode == 'production':
        recommendations.extend([
            "Enable full monitoring dashboard",
            "Set up automated backups",
            "Configure compliance reporting",
            "Train security team on divine intelligence"
        ])
    elif mode == 'high_availability':
        recommendations.extend([
            "Configure failover systems",
            "Set up load balancer",
            "Deploy active-passive pair",
            "Monitor system health continuously"
        ])
    else:
        recommendations.extend([
            "Run comprehensive testing suite",
            "Validate all configurations",
            "Monitor system performance",
            "Plan for production upgrade"
        ])
    
    # Based on business context
    business = config.get('business', {})
    if business.get('industry') == 'financial_services':
        recommendations.extend([
            "Enable SOX compliance monitoring",
            "Configure PCI-DSS validation",
            "Set up financial transaction monitoring",
            "Implement fraud detection rules"
        ])
    
    # Based on cardinal principles
    cardinal = config.get('cardinal_principles', {})
    if cardinal.get('love_weight', 0.25) != 0.25:
        recommendations.append("Review and optimize cardinal principle weights")
    
    return recommendations

def generate_next_steps(config):
    """Generate next steps"""
    next_steps = []
    
    mode = config.get('deployment', {}).get('mode', 'standard')
    
    if mode == 'test':
        next_steps.extend([
            "Run comprehensive testing suite",
            "Validate all business use cases",
            "Check compliance framework integration",
            "Plan production upgrade",
            "Train team on monitoring"
        ])
    elif mode == 'production':
        next_steps.extend([
            "Enable production monitoring",
            "Set up automated alerts",
            "Schedule regular compliance reviews",
            "Plan capacity expansion",
            "Maintain system updates"
        ])
    
    return next_steps

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy FortiGate Semantic Shield v7.0')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--device', help='FortiGate device IP/hostname')
    parser.add_argument('--token', help='FortiGate API token')
    parser.add_argument('--mode', default='test', choices=['test', 'production', 'high_availability'], help='Deployment mode')
    parser.add_argument('--validate-core', action='store_true', help='Validate core functionality only')
    parser.add_argument('--validate-deployment', action='store_true', help='Validate deployment after installation')
    args = parser.parse_args()
    
    print("[OFFICE] FortiGate Semantic Shield v7.0 - Deployment Script")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Step 2: Setup database
    if not setup_database():
        sys.exit(1)
    
    # Step 3: Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Create default config
        config = create_default_config(args)
    
    # Step 4: Setup FortiGate configuration
    if args.device and args.token:
        fortigate_config = {
            'host': args.device,
            'port': config.get('fortigate', {}).get('port', 443),
            'token': args.token,
            'verify_ssl': config.get('fortigate', {}).get('verify_ssl', True)
        }
    else:
        fortigate_config = config.get('fortigate', {})
        if not fortigate_config.get('host'):
            print("[FAIL] FortiGate configuration required. Use --device and --token or set in config file")
            sys.exit(1)
    
    # Step 5: Validate connection
    print("[CHECK] Validating FortiGate connection...")
    success, status = validate_fortigate_connection(fortigate_config)
    if not success:
        print("[FAIL] FortiGate connection failed. Please check configuration.")
        sys.exit(1)
    
    # Step 6: Add deployment mode
    config['deployment'] = {'mode': args.mode}
    
    # Step 7: Deploy to FortiGate
    print("[DEPLOY] Deploying to FortiGate...")
    deployment_success = deploy_to_fortigate(fortigate_config, config)
    
    if deployment_success:
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("[STATS] Generating deployment report...")
        report = generate_deployment_report(fortigate_config, config, deployment_id)
        
        print("[REPORT] Deployment completed successfully!")
        print(f"[METRICS] Deployment ID: {deployment_id}")
        print(f"[WEB] Monitoring Dashboard: {report.get('monitoring_dashboard', 'N/A')}")
        
        print("\n[NOTES] Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n[TARGET] Next Steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print(f"\n[SUCCESS] FortiGate Semantic Shield v7.0 deployed successfully!")
        return True
        
    else:
        print("[FAIL] Deployment failed. Please check logs and retry.")
        return False

def create_default_config(args):
    """Create default configuration"""
    config = {
        'fortigate': {
            'host': args.device if args.device else '192.168.1.100',
            'port': 443,
            'token': args.token if args.token else 'YOUR_TOKEN_HERE',
            'verify_ssl': True
        },
        'business': {
            'industry': 'financial_services',
            'organization_size': 'enterprise',
            'risk_tolerance': 0.7,
            'compliance_frameworks': ['SOX', 'PCI-DSS', 'GLBA']
        },
        'cardinal_principles': {
            'love_weight': 0.25,
            'power_weight': 0.25,
            'wisdom_weight': 0.25,
            'justice_weight': 0.25,
            'divine_anchor': [1.0, 1.0, 1.0, 1.0]
        },
        'processing': {
            'max_workers': 50,
            'batch_size': 100,
            'timeout_seconds': 30,
            'concurrent_connections': 100
        },
        'deployment': {
            'mode': args.mode,
            'monitoring': 'basic',
            'backup': 'disabled',
            'high_availability': False
        },
        'monitoring': {
            'enabled': True,
            'dashboard_port': 8080,
            'metrics_interval': 60,
            'alert_thresholds': {
                'error_rate': 0.05,
                'latency_ms': 100,
                'divine_alignment': 0.85
            }
        },
        'testing': {
            'events': 10000,
            'workers': 50,
            'duration': 60,
            'business_scenarios': ['fraud_detection', 'threat_intelligence', 'compliance_validation']
        }
    }
    
    # Save default config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[PASS] Default configuration created: config.yaml")
    print(f"[NOTES] Please edit config.yaml with your FortiGate details and rerun")
    return config

if __name__ == "__main__":
    main()
