# CRUSH.md - FortiGate Semantic Shield Development Guide
# =====================================================

## Project Overview
FortiGate Semantic Shield v7.0 - Enterprise cybersecurity intelligence with semantic reasoning and business-aligned decision making.

## Essential Commands

### Development Setup
```bash
# Install enhanced dependencies
pip install -r requirements_v7.txt

# Initialize semantic database
python -c "
import asyncio
from semantic_substrate_engine.Semantic-Substrate-Engine-main.src.enterprise_semantic_database import initialize_semantic_database
asyncio.run(initialize_semantic_database('fortigate_semantic_enterprise.db'))
"

# Run enhanced intelligence tests
python -m pytest tests/test_enhanced_intelligence.py -v

# Start enterprise simulation
python advanced_business_simulation.py
```

### Testing Commands
```bash
# Run comprehensive simulation (5 scenarios, 3 waves each)
python -c "
import asyncio
from advanced_business_simulation import run_enterprise_simulation
asyncio.run(run_enterprise_simulation())
"

# Test mathematical precision
python -c "
from semantic_substrate_engine.Semantic-Substrate-Engine-main.src.advanced_semantic_mathematics import advanced_math
print('Cache stats:', advanced_math.get_cache_stats())
"

# Validate semantic alignment
python -c "
from semantic_substrate_engine.Semantic-Substrate-Engine-main.src.enhanced_ice_framework import enhanced_ice
print('ICE stats:', enhanced_ice.get_processing_statistics())
"
```

### Performance Analysis
```bash
# Benchmark processing speed
python -c "
import time, asyncio
from enhanced_fortigate_intelligence_v7 import initialize_fortigate_intelligence
from fortigate_semantic_shield.device_interface import FortiGateAPIConfig

async def benchmark():
    api_config = FortiGateAPIConfig(base_url='https://test.com', token='test')
    intel = await initialize_fortigate_intelligence(api_config)
    start = time.time()
    await intel.process_threat_intelligently({
        'threat_type': 'malware', 'source_ip': '1.2.3.4', 'severity': 'high'
    })
    print(f'Processing time: {time.time() - start:.3f}s')

asyncio.run(benchmark())
"
```

### Database Operations
```bash
# Create intelligence snapshot
python -c "
import asyncio
from enhanced_fortigate_intelligence_v7 import fortigate_intelligence
asyncio.run(fortigate_intelligence.create_intelligence_snapshot('test_snapshot'))
"

# Check database performance
python -c "
import asyncio
from semantic_substrate_engine.Semantic-Substrate-Engine-main.src.enterprise_semantic_database import semantic_db
asyncio.run(semantic_db.initialize())
metrics = asyncio.run(semantic_db.get_performance_metrics())
print('Database metrics:', metrics)
"
```

## Code Style Preferences

### Business-First Approach
1. **Business Context First**: Always map technical decisions to business impact
2. **Compliance Integration**: Include regulatory considerations in all security decisions
3. **ROI Focus**: Quantify business value of security actions
4. **Industry Awareness**: Tailor solutions to specific business sectors

### Semantic Programming Patterns
```python
# Preferred: Business-aligned semantic processing
async def process_threat_with_business_intelligence(threat_data, business_context):
    """Process threat with full business impact assessment"""
    # 1. Map to business principles
    business_intent = map_threat_to_business_intent(threat_data, business_context)
    
    # 2. Apply ICE framework with business weighting
    ice_result = await enhanced_ice.process_intent(business_intent, threat_data)
    
    # 3. Generate business justification
    business_rationale = create_business_justification(ice_result, business_context)
    
    # 4. Quantify financial impact
    impact_assessment = calculate_business_impact(threat_data, ice_result)
    
    return SemanticThreatResponse(
        ice_result=ice_result,
        business_rationale=business_rationale,
        financial_impact=impact_assessment
    )
```

### Error Handling Patterns
```python
# Preferred: Business-aware error handling
try:
    response = await intelligence.process_threat_intelligently(threat_data)
except ProcessingError as e:
    # Fallback with business continuity focus
    return create_business_continuity_response(threat_data, e)
except ComplianceError as e:
    # Immediate compliance reporting
    await report_compliance_violation(threat_data, e)
    return create_compliance_focused_response(threat_data)
```

### Performance Optimization
- Use async/await for all I/O operations
- Implement connection pooling for database access
- Cache frequently used semantic patterns
- Monitor resource utilization in real-time

## Architecture Patterns

### Semantic Component Structure
```python
# Standard semantic component structure
class BusinessSemanticComponent:
    """Base class for business-aligned semantic components"""
    
    def __init__(self, business_context: BusinessContext):
        self.business_context = business_context
        self.compliance_requirements = get_compliance_requirements(business_context)
        self.risk_tolerance = calculate_risk_tolerance(business_context)
    
    async def process_with_business_intelligence(self, data: Any) -> BusinessResult:
        """Process data with full business context awareness"""
        # Implementation with business alignment
        pass
```

### Integration Patterns
```python
# FortiGate integration with semantic intelligence
async def apply_semantic_intelligence_to_fortigate(threat_response):
    """Apply semantic intelligence decisions to FortiGate device"""
    for action in threat_response.recommended_actions:
        # Map to FortiGate API calls
        success = await execute_fortigate_action(action)
        if success:
            await log_business_outcome(threat_response, success)
        else:
            await handle_action_failure(threat_response, action)
```

## Important Codebase Information

### Key Components
1. **enhanced_fortigate_intelligence_v7.py**: Main intelligence orchestrator
2. **advanced_semantic_mathematics.py**: High-performance mathematical engine
3. **enterprise_semantic_database.py**: Async database with spatial indexing
4. **enhanced_ice_framework.py**: Business-aligned ICE processing
5. **advanced_business_simulation.py**: Comprehensive testing framework

### Business Context Mappings
- **Financial Services**: SOX, PCI-DSS, GLBA focus
- **Healthcare**: HIPAA, patient privacy, operational continuity
- **Retail**: Customer data, payment security, fraud prevention
- **Manufacturing**: IP protection, supply chain security
- **Critical Infrastructure**: NERC compliance, service reliability

### Performance Targets
- **Threat Processing**: < 1 second average response time
- **Database Queries**: < 100ms for spatial queries
- **Learning Velocity**: > 80% pattern recognition after 1000 threats
- **Business Impact**: < 5% false positive rate

### Security Considerations
- All API communications encrypted
- Semantic signatures cryptographically secured
- Audit trails immutable and comprehensive
- Compliance reporting automated and tamper-proof

## Testing Strategy

### Business Simulation Testing
- Run comprehensive scenario testing before deployment
- Validate industry-specific compliance requirements
- Measure business impact prevention effectiveness
- Test learning velocity and adaptation capabilities

### Performance Testing
- Concurrent threat processing (100+ simultaneous threats)
- Database performance under load
- Memory and CPU utilization monitoring
- Resource scaling and optimization

### Compliance Validation
- Regulatory requirement mapping verification
- Audit trail completeness testing
- Business justification quality assessment
- Reporting automation validation

## Development Workflow

1. **Feature Development**: Code with business context awareness
2. **Unit Testing**: Validate individual component functionality
3. **Integration Testing**: Test semantic component interactions
4. **Business Simulation**: Run scenario-based validation
5. **Performance Testing**: Ensure enterprise-grade performance
6. **Compliance Review**: Validate regulatory requirements
7. **Documentation**: Update business impact assessments

## Monitoring and Observability

### Key Metrics
- Intelligence Effectiveness Score
- Business Impact Prevented (%)
- Compliance Maintenance Rate
- Learning Velocity
- Resource Utilization

### Business KPIs
- Threat Response Time
- False Positive Rate
- Regulatory Compliance Score
- Financial Impact Prevented
- ROI Percentage

## Troubleshooting

### Common Issues
1. **High Processing Latency**: Check database connection pooling
2. **Low Learning Velocity**: Verify training data quality
3. **Compliance Gaps**: Review regulatory requirement mappings
4. **Resource Exhaustion**: Monitor cache sizes and cleanup

### Debug Commands
```bash
# Check semantic alignment
python -c "
from semantic_substrate_engine.Semantic-Substrate-Engine-main.src.advanced_semantic_mathematics import create_business_vector, compute_business_maturity
vec = create_business_vector(0.8, 0.7, 0.9, 0.6)
print('Business maturity:', compute_business_maturity(vec))
"

# Validate database connectivity
python -c "
import asyncio
from semantic_substrate_engine.Semantic-Substrate-Engine-main.src.enterprise_semantic_database import semantic_db
asyncio.run(semantic_db.initialize())
print('Database initialized successfully')
"
```

## Deployment Notes

### Environment Configuration
- Production: PostgreSQL + Redis + FortiGate API
- Staging: Docker containers with simulated FortiGate
- Development: SQLite + local caching

### Scaling Considerations
- Horizontal scaling for threat processing
- Database read replicas for analytics
- CDN for static intelligence data
- Load balancing for API endpoints

This CRUSH.md should be updated with any new commands, patterns, or architectural decisions as the system evolves.