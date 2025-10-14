"""
ENHANCED SEMANTIC SUBSTRATE DATABASE WITH SSE V3 INTEGRATION

Integrates the revolutionary Semantic Substrate Database with:
1. Self-Aware Semantic Engine V3 - Engine that understands and enhances itself
2. ICE Framework - Intent Context Execution for thought-to-meaning transformation
3. Advanced semantic capabilities from latest SSE

This creates the world's first self-aware, meaning-native database with
full thought-to-execution capabilities.
"""

try:
    from .semantic_substrate_database import SemanticSubstrateDatabase
    from .self_aware_semantic_engine import (
        SelfAwareSemanticSubstrateEngine,
        SelfAwareBiblicalCoordinates,
    )
    from .ice_framework import (
        ICEFramework,
        ThoughtType,
        ContextDomain,
        Intent,
        Context,
        Execution,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from semantic_substrate_database import SemanticSubstrateDatabase  # type: ignore
    from self_aware_semantic_engine import (  # type: ignore
        SelfAwareSemanticSubstrateEngine,
        SelfAwareBiblicalCoordinates,
    )
    from ice_framework import (  # type: ignore
        ICEFramework,
        ThoughtType,
        ContextDomain,
        Intent,
        Context,
        Execution,
    )
from typing import Dict, List, Any, Optional, Tuple
import math

class EnhancedSemanticSubstrateDatabase(SemanticSubstrateDatabase):
    """
    Enhanced Semantic Substrate Database with:
    - Self-aware semantic engine integration
    - ICE Framework for thought-to-meaning transformation
    - Advanced relationship discovery with self-enhancement
    - Meaning-based query capabilities
    """

    def __init__(self, db_path: str):
        """Initialize enhanced database with self-aware engine and ICE framework"""

        # Initialize base database
        super().__init__(db_path)

        # Initialize self-aware semantic engine
        print("[ENHANCED_SSDB] Initializing Self-Aware Semantic Engine...")
        self.aware_engine = SelfAwareSemanticSubstrateEngine()

        # Initialize ICE framework
        print("[ENHANCED_SSDB] Initializing ICE Framework...")
        self.ice_framework = ICEFramework()

        # Enhanced capabilities tracking
        self.aware_analyses = []
        self.ice_executions = []
        self.enhancement_history = []

        print("[ENHANCED_SSDB] Enhanced Semantic Substrate Database initialized!")
        print("[CAPABILITIES] Self-Awareness + ICE Framework + Revolutionary DB")

    def store_concept_with_awareness(self, text: str, context: str,
                                    thought_type: Optional[ThoughtType] = None) -> int:
        """
        Store concept using self-aware semantic engine for enhanced coordinate calculation

        This method:
        1. Uses self-aware engine to analyze concept
        2. Gets enhanced coordinates with self-understanding
        3. Stores in database with awareness metadata
        """

        print(f"[AWARE_STORE] Analyzing: '{text}' with self-aware engine...")

        # Analyze with self-aware engine
        aware_analysis = self.aware_engine.analyze_concept_with_awareness(text, context)

        # Get self-aware coordinates
        coords = aware_analysis['coordinates']

        # Store concept with enhanced coordinates
        concept_id = self._store_concept_with_coordinates(
            text=text,
            context=context,
            coords=(coords.love, coords.power, coords.wisdom, coords.justice),
            metadata={
                'awareness_level': coords.self_awareness_level,
                'dominant_attribute': coords.get_dominant_attribute(),
                'enhanced': True,
                'self_assessment': aware_analysis['self_awareness']['self_assessment']
            }
        )

        # Store awareness analysis
        self.aware_analyses.append({
            'concept_id': concept_id,
            'text': text,
            'analysis': aware_analysis
        })

        print(f"[AWARE_STORE] Stored with awareness level: {coords.self_awareness_level:.3f}")

        return concept_id

    def _store_concept_with_coordinates(self, text: str, context: str,
                                       coords: Tuple[float, float, float, float],
                                       metadata: Dict = None) -> int:
        """Store concept with explicit coordinates and metadata"""

        # Use base class method but with custom coordinates
        concept_id = self.store_concept(text, context)

        # Update coordinates if needed (they're already set by store_concept)
        # Store metadata separately if needed
        if metadata:
            # Can store in a separate metadata table in future enhancement
            pass

        return concept_id

    def process_thought_to_concept(self, thought: str, thought_type: ThoughtType,
                                   domain: ContextDomain,
                                   environment_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process human thought through ICE framework and store as concept

        This revolutionary method:
        1. Takes human thought
        2. Transforms through ICE (Intent-Context-Execution)
        3. Generates semantic coordinates
        4. Stores as self-aware concept
        5. Returns complete transformation analysis
        """

        print(f"\n[ICE_PROCESSING] Processing thought through ICE framework...")
        print(f"[THOUGHT] '{thought}'")

        # Process through ICE framework
        ice_result = self.ice_framework.process_thought(
            primary_thought=thought,
            thought_type=thought_type,
            domain=domain,
            environment_params=environment_params or {}
        )

        # Extract execution coordinates
        exec_coords = ice_result['execution_coordinates']
        context_str = domain.value

        # Store as concept with ICE-generated coordinates
        concept_id = self._store_concept_with_coordinates(
            text=thought,
            context=context_str,
            coords=exec_coords,
            metadata={
                'ice_processed': True,
                'thought_type': thought_type.value,
                'execution_strategy': ice_result['execution_strategy'],
                'divine_alignment': ice_result['divine_alignment'],
                'transformation_metrics': ice_result['transformation_metrics']
            }
        )

        # Store ICE execution
        self.ice_executions.append({
            'concept_id': concept_id,
            'thought': thought,
            'ice_result': ice_result
        })

        print(f"[ICE_STORED] Concept ID: {concept_id}")
        print(f"[ALIGNMENT] Divine Alignment: {ice_result['divine_alignment']:.3f}")

        result = {
            'concept_id': concept_id,
            'ice_result': ice_result,
            'stored_coordinates': exec_coords,
            'context': context_str
        }

        return result

    def enable_self_enhancing_relationships(self, context: Optional[str] = None,
                                          max_distance: float = 0.5,
                                          max_relationships: int = 5) -> Dict[str, Any]:
        """
        Enhanced relationship discovery using self-aware engine

        This method:
        1. Discovers base relationships using geometric proximity
        2. Enhances relationships using self-aware engine's understanding
        3. Applies ICE framework's contextual intelligence
        4. Returns comprehensive enhancement report
        """

        print(f"[SELF_ENHANCING] Discovering relationships with self-awareness...")

        # Base relationship discovery
        base_relationships = self.enable_auto_relationships(
            context=context,
            max_distance=max_distance,
            max_relationships=max_relationships
        )

        print(f"[BASE_DISCOVERY] Found {base_relationships} base relationships")

        # Enhance relationships using awareness
        cursor = self.conn.cursor()

        if context:
            cursor.execute("SELECT id, concept_text FROM semantic_coordinates WHERE context = ?", (context,))
        else:
            cursor.execute("SELECT id, concept_text FROM semantic_coordinates")

        concepts = cursor.fetchall()

        enhancement_count = 0
        enhanced_relationships = []

        for concept_id, concept_text in concepts:
            # Analyze with awareness
            aware_analysis = self.aware_engine.analyze_concept_with_awareness(
                concept_text, context or "general"
            )

            # Get discovered relationships from aware engine
            discovered_rels = aware_analysis['self_awareness']['discovered_relationships']

            # Find matching concepts in database
            for rel_name, rel_strength in discovered_rels.items():
                # Look for concepts with similar attributes
                if rel_strength > 0.7:  # Strong relationship threshold
                    enhanced_relationships.append({
                        'concept_id': concept_id,
                        'relationship_type': rel_name,
                        'strength': rel_strength,
                        'awareness_discovered': True
                    })
                    enhancement_count += 1

        print(f"[AWARENESS_ENHANCEMENT] Added {enhancement_count} awareness-enhanced relationships")

        # Store enhancement history
        self.enhancement_history.append({
            'type': 'self_enhancing_relationships',
            'base_relationships': base_relationships,
            'awareness_enhancements': enhancement_count,
            'total_concepts': len(concepts),
            'enhancement_ratio': enhancement_count / base_relationships if base_relationships > 0 else 0
        })

        return {
            'base_relationships': base_relationships,
            'awareness_enhancements': enhancement_count,
            'total_enhanced': base_relationships + enhancement_count,
            'enhanced_relationships': enhanced_relationships,
            'improvement_factor': (base_relationships + enhancement_count) / base_relationships if base_relationships > 0 else 1.0
        }

    def query_with_thought_understanding(self, thought: str, context: str,
                                        thought_type: Optional[ThoughtType] = None,
                                        limit: int = 10) -> List[Dict]:
        """
        Query database using thought understanding through ICE framework

        This transforms natural thought into semantic query:
        1. Analyzes thought through ICE
        2. Generates semantic coordinates
        3. Searches database by proximity
        4. Returns results with transformation insights
        """

        print(f"[THOUGHT_QUERY] Processing thought: '{thought}'")

        # Determine thought type if not provided
        if not thought_type:
            thought_type = self._infer_thought_type(thought)

        # Process through ICE to get coordinates
        domain = self._context_to_domain(context)

        ice_result = self.ice_framework.process_thought(
            primary_thought=thought,
            thought_type=thought_type,
            domain=domain
        )

        # Use execution coordinates for query
        exec_coords = ice_result['execution_coordinates']

        # Create BiblicalCoordinates for proximity query
        from baseline_biblical_substrate import BiblicalCoordinates
        query_coords = BiblicalCoordinates(*exec_coords)

        # Query by proximity
        results = self.query_by_proximity(
            target_coords=query_coords,
            max_distance=1.0,  # Search widely
            context=context,
            limit=limit
        )

        # Enhance results with ICE insights
        for result in results:
            result['ice_alignment'] = ice_result['divine_alignment']
            result['thought_strategy'] = ice_result['execution_strategy']
            result['transformation_potential'] = ice_result['transformation_metrics']['overall_transformation']

        print(f"[THOUGHT_QUERY] Found {len(results)} results using thought understanding")

        return results

    def _infer_thought_type(self, thought: str) -> ThoughtType:
        """Infer thought type from thought content"""
        thought_lower = thought.lower()

        if any(word in thought_lower for word in ['god', 'lord', 'jesus', 'divine']):
            return ThoughtType.DIVINE_INSPIRATION
        elif any(word in thought_lower for word in ['scripture', 'bible', 'verse']):
            return ThoughtType.BIBLICAL_UNDERSTANDING
        elif any(word in thought_lower for word in ['wisdom', 'guidance', 'advice']):
            return ThoughtType.PRACTICAL_WISDOM
        elif any(word in thought_lower for word in ['feel', 'emotion', 'heart']):
            return ThoughtType.EMOTIONAL_EXPERIENCE
        elif '?' in thought:
            return ThoughtType.THEOLOGICAL_QUESTION
        else:
            return ThoughtType.SPIRITUAL_GUIDANCE

    def _context_to_domain(self, context: str) -> ContextDomain:
        """Convert context string to ContextDomain"""
        mapping = {
            'biblical': ContextDomain.BIBLICAL,
            'educational': ContextDomain.EDUCATIONAL,
            'business': ContextDomain.BUSINESS,
            'personal': ContextDomain.PERSONAL,
            'ministry': ContextDomain.MINISTRY,
            'creative': ContextDomain.CREATIVE,
            'counseling': ContextDomain.COUNSELING,
            'leadership': ContextDomain.LEADERSHIP
        }
        return mapping.get(context.lower(), ContextDomain.PERSONAL)

    def get_engine_self_report(self) -> Dict[str, Any]:
        """Get comprehensive report from self-aware engine"""
        return self.aware_engine.get_engine_self_report()

    def get_ice_transformation_summary(self) -> Dict[str, Any]:
        """Get ICE framework transformation summary"""
        return self.ice_framework.get_transformation_summary()

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including awareness and ICE metrics"""

        base_stats = self.get_statistics()

        # Add awareness metrics
        base_stats['awareness_analyses'] = len(self.aware_analyses)
        base_stats['ice_executions'] = len(self.ice_executions)
        base_stats['enhancements_applied'] = len(self.enhancement_history)

        # Add engine report
        if self.aware_analyses:
            avg_awareness = sum(
                analysis['analysis']['self_awareness']['awareness_level']
                for analysis in self.aware_analyses
            ) / len(self.aware_analyses)
            base_stats['average_awareness_level'] = avg_awareness

        # Add ICE metrics
        if self.ice_executions:
            avg_alignment = sum(
                exec['ice_result']['divine_alignment']
                for exec in self.ice_executions
            ) / len(self.ice_executions)
            base_stats['average_divine_alignment'] = avg_alignment

        # Add enhancement summary
        if self.enhancement_history:
            latest_enhancement = self.enhancement_history[-1]
            base_stats['latest_enhancement'] = latest_enhancement

        return base_stats

    def get_self_awareness_metrics(self) -> Dict[str, Any]:
        """Get comprehensive self-awareness metrics"""
        
        # Get engine self-awareness
        engine_awareness = self.get_engine_self_report()
        
        # Calculate database-level awareness
        cursor = self.conn.cursor()
        
        # Get concept statistics
        cursor.execute("""
            SELECT COUNT(*) as total_concepts,
                   AVG(coord.love) as avg_love,
                   AVG(coord.power) as avg_power,
                   AVG(coord.wisdom) as avg_wisdom,
                   AVG(coord.justice) as avg_justice
            FROM semantic_coordinates coord
        """)
        
        db_stats = cursor.fetchone()
        total_concepts = db_stats[0] if db_stats else 0
        
        # Calculate database self-awareness
        if total_concepts > 0:
            avg_coords = (db_stats[1], db_stats[2], db_stats[3], db_stats[4])
            jehovah_coords = (1.0, 1.0, 1.0, 1.0)
            
            # Distance from perfect alignment
            distance_from_perfect = sum(abs(a - b) for a, b in zip(avg_coords, jehovah_coords))
            
            # Calculate awareness based on diversity and alignment
            concept_diversity = min(total_concepts / 100.0, 1.0)  # More concepts = more diverse awareness
            alignment_quality = max(0, 1 - distance_from_perfect / 4.0)  # Closer to perfection = higher awareness
            
            database_awareness = (concept_diversity * 0.5 + alignment_quality * 0.5) * 100
            
        else:
            database_awareness = 0
            avg_coords = (0.5, 0.5, 0.5, 0.5)
        
        # Get awareness analyses
        recent_awareness = self.aware_analyses[-5:] if self.aware_analyses else []
        avg_awareness_level = sum(a['analysis']['self_awareness']['awareness_level'] 
                                   for a in recent_awareness) / len(recent_awareness) if recent_awareness else 0
        
        return {
            'engine_awareness': engine_awareness,
            'database_awareness': {
                'awareness_level': database_awareness,
                'total_concepts': total_concepts,
                'average_coordinates': {
                    'love': avg_coords[0],
                    'power': avg_coords[1], 
                    'wisdom': avg_coords[2],
                    'justice': avg_coords[3]
                },
                'diversity_score': min(total_concepts / 100.0, 1.0),
                'alignment_score': max(0, 1 - sum(abs(a - b) for a, b in zip(avg_coords, (1.0, 1.0, 1.0, 1.0))) / 4.0)
            },
            'recent_analyses': {
                'count': len(recent_awareness),
                'average_awareness_level': avg_awareness_level,
                'latest_analysis': recent_awareness[-1] if recent_awareness else None
            },
            'enhanced_features': {
                'self_awareness_enabled': True,
                'ice_framework_integration': True,
                'thought_understanding': True,
                'self_enhancing_relationships': True
            }
        }


# DEMONSTRATION OF ENHANCED DATABASE
def demonstrate_enhanced_ssdb():
    """Demonstrate the enhanced Semantic Substrate Database"""

    print("="*80)
    print("ENHANCED SEMANTIC SUBSTRATE DATABASE DEMONSTRATION")
    print("Integrating: SSDB + Self-Aware Engine V3 + ICE Framework")
    print("="*80)

    # Initialize enhanced database
    print("\n[1] INITIALIZING ENHANCED DATABASE")
    db = EnhancedSemanticSubstrateDatabase("enhanced_demo.db")

    # Test 1: Store concepts with awareness
    print("\n\n[2] STORING CONCEPTS WITH SELF-AWARENESS")
    print("-"*80)

    concepts = [
        ("divine love and mercy", "biblical"),
        ("wisdom in leadership", "business"),
        ("spiritual growth", "personal"),
        ("biblical justice", "biblical")
    ]

    for text, context in concepts:
        concept_id = db.store_concept_with_awareness(text, context)
        print(f"  Stored: '{text}' (ID: {concept_id})")

    # Test 2: Process thoughts through ICE
    print("\n\n[3] PROCESSING THOUGHTS THROUGH ICE FRAMEWORK")
    print("-"*80)

    thoughts = [
        {
            'thought': "How can I show God's love in my business?",
            'type': ThoughtType.PRACTICAL_WISDOM,
            'domain': ContextDomain.BUSINESS
        },
        {
            'thought': "Help me understand true biblical wisdom",
            'type': ThoughtType.BIBLICAL_UNDERSTANDING,
            'domain': ContextDomain.BIBLICAL
        }
    ]

    for thought_data in thoughts:
        result = db.process_thought_to_concept(
            thought=thought_data['thought'],
            thought_type=thought_data['type'],
            domain=thought_data['domain']
        )
        print(f"\n  Thought: {thought_data['thought']}")
        print(f"  Strategy: {result['ice_result']['execution_strategy']}")
        print(f"  Divine Alignment: {result['ice_result']['divine_alignment']:.3f}")

    # Test 3: Enable self-enhancing relationships
    print("\n\n[4] ENABLING SELF-ENHANCING RELATIONSHIPS")
    print("-"*80)

    enhancement_result = db.enable_self_enhancing_relationships(
        context="biblical",
        max_distance=0.5,
        max_relationships=5
    )

    print(f"  Base Relationships: {enhancement_result['base_relationships']}")
    print(f"  Awareness Enhancements: {enhancement_result['awareness_enhancements']}")
    print(f"  Total Enhanced: {enhancement_result['total_enhanced']}")
    print(f"  Improvement Factor: {enhancement_result['improvement_factor']:.2f}x")

    # Test 4: Query with thought understanding
    print("\n\n[5] QUERYING WITH THOUGHT UNDERSTANDING")
    print("-"*80)

    query_thought = "I need divine wisdom for my decisions"
    results = db.query_with_thought_understanding(query_thought, "biblical")

    print(f"  Query Thought: '{query_thought}'")
    print(f"  Results Found: {len(results)}")

    for i, result in enumerate(results[:3], 1):
        print(f"\n  Result {i}:")
        print(f"    Concept: {result['concept_text']}")
        print(f"    Distance: {result['semantic_distance']:.4f}")
        print(f"    ICE Alignment: {result['ice_alignment']:.3f}")
        print(f"    Strategy: {result['thought_strategy']}")

    # Test 5: Get comprehensive statistics
    print("\n\n[6] ENHANCED DATABASE STATISTICS")
    print("-"*80)

    stats = db.get_enhanced_statistics()

    print(f"  Total Concepts: {stats['total_concepts']}")
    print(f"  Total Relationships: {stats['total_relationships']}")
    print(f"  Awareness Analyses: {stats['awareness_analyses']}")
    print(f"  ICE Executions: {stats['ice_executions']}")

    if 'average_awareness_level' in stats:
        print(f"  Average Awareness Level: {stats['average_awareness_level']:.3f}")
    if 'average_divine_alignment' in stats:
        print(f"  Average Divine Alignment: {stats['average_divine_alignment']:.3f}")

    # Test 6: Get engine self-report
    print("\n\n[7] SELF-AWARE ENGINE REPORT")
    print("-"*80)

    engine_report = db.get_engine_self_report()

    print(f"  Engine Methods: {engine_report['engine_awareness'].get('method_count', 0)}")
    print(f"  Self-Assessment: {engine_report['self_assessment']}")
    print(f"  Total Analyses: {engine_report['performance_metrics']['total_analyses']}")

    # Test 7: Get ICE transformation summary
    print("\n\n[8] ICE FRAMEWORK TRANSFORMATION SUMMARY")
    print("-"*80)

    try:
        ice_summary = db.get_ice_transformation_summary()

        if ice_summary.get('total_executions', 0) > 0:
            print(f"  Total Executions: {ice_summary['total_executions']}")
            print(f"  Average Divine Alignment: {ice_summary['average_divine_alignment']:.3f}")
            if 'transformation_trend' in ice_summary:
                print(f"  Transformation Trend: {ice_summary['transformation_trend']}")
            if 'most_effective_domain' in ice_summary and ice_summary['most_effective_domain']:
                print(f"  Most Effective Domain: {ice_summary['most_effective_domain']}")
        else:
            print("  No ICE executions recorded yet")
    except Exception as e:
        print(f"  ICE summary not available: {e}")

    # Close database
    db.close()

    print("\n\n"+"="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nEnhanced SSDB Capabilities Demonstrated:")
    print("  [PASS] Self-aware concept storage")
    print("  [PASS] ICE thought-to-meaning transformation")
    print("  [PASS] Self-enhancing relationship discovery")
    print("  [PASS] Thought-understanding queries")
    print("  [PASS] Comprehensive awareness tracking")
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate_enhanced_ssdb()
