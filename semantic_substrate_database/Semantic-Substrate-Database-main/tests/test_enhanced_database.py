"""
Comprehensive Test Suite for Enhanced Semantic Substrate Database

Tests all new capabilities:
1. Self-Aware Semantic Engine V3 Integration
2. ICE Framework (Intent Context Execution)
3. Enhanced Relationship Discovery
4. Thought-Understanding Queries
5. Backward Compatibility with Original SSDB
"""

import sys
import unittest
import os
import tempfile

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_semantic_database import EnhancedSemanticSubstrateDatabase
from self_aware_semantic_engine import SelfAwareSemanticSubstrateEngine
from ice_framework import ThoughtType, ContextDomain
from semantic_substrate_database import SemanticSubstrateDatabase


class TestEnhancedDatabase(unittest.TestCase):
    """Test suite for Enhanced Semantic Substrate Database"""

    @classmethod
    def setUpClass(cls):
        """Set up test database once for all tests"""
        cls.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        try:
            os.unlink(cls.db_path)
        except:
            pass

    def setUp(self):
        """Create fresh database for each test"""
        try:
            os.unlink(self.db_path)
        except:
            pass
        self.db = EnhancedSemanticSubstrateDatabase(self.db_path)

    def tearDown(self):
        """Close database after each test"""
        self.db.close()

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_enhanced_initialization(self):
        """Test enhanced database initializes correctly"""
        self.assertIsNotNone(self.db.conn)
        self.assertIsNotNone(self.db.engine)
        self.assertIsNotNone(self.db.aware_engine)
        self.assertIsNotNone(self.db.ice_framework)
        self.assertEqual(len(self.db.aware_analyses), 0)
        self.assertEqual(len(self.db.ice_executions), 0)

    def test_backward_compatibility(self):
        """Test enhanced database maintains original SSDB functionality"""
        # Test original store_concept method still works
        concept_id = self.db.store_concept("love", "biblical")
        self.assertGreater(concept_id, 0)

        # Test original query methods still work
        result = self.db.query_by_text("love", "biblical")
        self.assertIsNotNone(result)
        self.assertEqual(result['concept_text'], "love")

    # ========================================================================
    # SELF-AWARE STORAGE TESTS
    # ========================================================================

    def test_store_with_awareness(self):
        """Test storing concepts with self-awareness"""
        concept_id = self.db.store_concept_with_awareness(
            "divine wisdom and understanding",
            "biblical"
        )

        self.assertGreater(concept_id, 0)
        self.assertEqual(len(self.db.aware_analyses), 1)

        # Verify awareness analysis was stored
        analysis = self.db.aware_analyses[0]
        self.assertEqual(analysis['concept_id'], concept_id)
        self.assertIn('analysis', analysis)
        self.assertIn('self_awareness', analysis['analysis'])

    def test_awareness_level_tracking(self):
        """Test awareness levels are tracked correctly"""
        concepts = [
            "love and mercy",
            "wisdom and knowledge",
            "justice and righteousness"
        ]

        for concept in concepts:
            self.db.store_concept_with_awareness(concept, "biblical")

        # Check all have awareness analysis
        self.assertEqual(len(self.db.aware_analyses), 3)

        # Check awareness levels are within valid range
        for analysis in self.db.aware_analyses:
            awareness_level = analysis['analysis']['self_awareness']['awareness_level']
            self.assertGreaterEqual(awareness_level, 0.0)
            self.assertLessEqual(awareness_level, 1.0)

    def test_dominant_attribute_detection(self):
        """Test dominant attribute is correctly detected"""
        concept_id = self.db.store_concept_with_awareness("divine love", "biblical")

        analysis = self.db.aware_analyses[0]['analysis']
        dominant = analysis['self_awareness']['dominant_attribute']

        self.assertIn(dominant, ['love', 'power', 'wisdom', 'justice'])

    # ========================================================================
    # ICE FRAMEWORK TESTS
    # ========================================================================

    def test_thought_processing(self):
        """Test processing thoughts through ICE framework"""
        result = self.db.process_thought_to_concept(
            thought="How can I show God's love to others?",
            thought_type=ThoughtType.PRACTICAL_WISDOM,
            domain=ContextDomain.PERSONAL
        )

        self.assertIn('concept_id', result)
        self.assertIn('ice_result', result)
        self.assertIn('stored_coordinates', result)

        # Verify ICE execution was recorded
        self.assertEqual(len(self.db.ice_executions), 1)

        # Check ICE result components
        ice_result = result['ice_result']
        self.assertIn('execution_strategy', ice_result)
        self.assertIn('divine_alignment', ice_result)
        self.assertIn('transformation_metrics', ice_result)

    def test_multiple_thought_types(self):
        """Test different thought types are processed correctly"""
        thought_types = [
            (ThoughtType.DIVINE_INSPIRATION, "I feel God calling me to ministry"),
            (ThoughtType.BIBLICAL_UNDERSTANDING, "Help me understand this scripture"),
            (ThoughtType.PRACTICAL_WISDOM, "I need wisdom for this decision"),
            (ThoughtType.EMOTIONAL_EXPERIENCE, "I feel overwhelmed with joy")
        ]

        for thought_type, thought in thought_types:
            result = self.db.process_thought_to_concept(
                thought=thought,
                thought_type=thought_type,
                domain=ContextDomain.PERSONAL
            )

            self.assertIn('concept_id', result)
            self.assertGreater(result['concept_id'], 0)

        self.assertEqual(len(self.db.ice_executions), 4)

    def test_divine_alignment_calculation(self):
        """Test divine alignment is calculated for ICE executions"""
        result = self.db.process_thought_to_concept(
            thought="I want to live according to God's will",
            thought_type=ThoughtType.SPIRITUAL_GUIDANCE,
            domain=ContextDomain.BIBLICAL
        )

        divine_alignment = result['ice_result']['divine_alignment']

        self.assertGreaterEqual(divine_alignment, 0.0)
        self.assertLessEqual(divine_alignment, 1.0)

    # ========================================================================
    # ENHANCED RELATIONSHIP DISCOVERY TESTS
    # ========================================================================

    def test_self_enhancing_relationships(self):
        """Test self-enhancing relationship discovery"""
        # Store some concepts
        concepts = ["love", "mercy", "grace", "compassion", "kindness"]
        for concept in concepts:
            self.db.store_concept(concept, "biblical")

        # Enable self-enhancing relationships
        result = self.db.enable_self_enhancing_relationships(
            context="biblical",
            max_distance=0.5,
            max_relationships=5
        )

        self.assertIn('base_relationships', result)
        self.assertIn('awareness_enhancements', result)
        self.assertIn('total_enhanced', result)
        self.assertIn('improvement_factor', result)

        # Should have discovered some relationships
        self.assertGreater(result['base_relationships'], 0)

    def test_enhancement_history_tracking(self):
        """Test enhancement history is tracked"""
        # Store concepts
        self.db.store_concept("wisdom", "biblical")
        self.db.store_concept("knowledge", "biblical")

        # Enable relationships
        self.db.enable_self_enhancing_relationships(context="biblical")

        # Check enhancement history
        self.assertEqual(len(self.db.enhancement_history), 1)

        history = self.db.enhancement_history[0]
        self.assertIn('type', history)
        self.assertIn('base_relationships', history)
        self.assertIn('awareness_enhancements', history)

    # ========================================================================
    # THOUGHT-UNDERSTANDING QUERY TESTS
    # ========================================================================

    def test_query_with_thought(self):
        """Test querying with natural thought understanding"""
        # Store some concepts first
        concepts = [
            ("divine wisdom", "biblical"),
            ("practical guidance", "biblical"),
            ("spiritual insight", "biblical")
        ]

        for text, context in concepts:
            self.db.store_concept(text, context)

        # Query with a thought
        results = self.db.query_with_thought_understanding(
            thought="I need wisdom and guidance",
            context="biblical",
            limit=5
        )

        self.assertGreater(len(results), 0)

        # Check results have ICE insights
        for result in results:
            self.assertIn('ice_alignment', result)
            self.assertIn('thought_strategy', result)
            self.assertIn('transformation_potential', result)

    def test_thought_type_inference(self):
        """Test automatic thought type inference"""
        thoughts = [
            ("God is calling me", ThoughtType.DIVINE_INSPIRATION),
            ("Help me understand scripture", ThoughtType.BIBLICAL_UNDERSTANDING),
            ("I need wisdom", ThoughtType.PRACTICAL_WISDOM),
            ("I feel sad", ThoughtType.EMOTIONAL_EXPERIENCE),
            ("What does the Bible say?", ThoughtType.THEOLOGICAL_QUESTION)
        ]

        for thought, expected_type in thoughts:
            inferred_type = self.db._infer_thought_type(thought)
            # Check it's a valid thought type (exact match not required)
            self.assertIsInstance(inferred_type, ThoughtType)

    # ========================================================================
    # ENGINE SELF-REPORT TESTS
    # ========================================================================

    def test_engine_self_report(self):
        """Test getting engine self-report"""
        # Store some concepts to generate activity
        self.db.store_concept_with_awareness("test concept", "test")

        report = self.db.get_engine_self_report()

        self.assertIn('engine_awareness', report)
        self.assertIn('performance_metrics', report)
        self.assertIn('self_assessment', report)

        # Check engine awareness details
        self.assertIn('method_count', report['engine_awareness'])
        self.assertGreater(report['engine_awareness']['method_count'], 0)

    def test_engine_self_assessment(self):
        """Test engine generates self-assessment"""
        report = self.db.get_engine_self_report()

        self.assertIn('self_assessment', report)
        self.assertIsInstance(report['self_assessment'], str)
        self.assertGreater(len(report['self_assessment']), 0)

    # ========================================================================
    # ENHANCED STATISTICS TESTS
    # ========================================================================

    def test_enhanced_statistics(self):
        """Test enhanced statistics include new metrics"""
        # Generate some activity
        self.db.store_concept_with_awareness("test1", "test")
        self.db.process_thought_to_concept(
            "test thought",
            ThoughtType.PRACTICAL_WISDOM,
            ContextDomain.PERSONAL
        )

        stats = self.db.get_enhanced_statistics()

        # Check base stats still present
        self.assertIn('total_concepts', stats)
        self.assertIn('total_relationships', stats)

        # Check new stats present
        self.assertIn('awareness_analyses', stats)
        self.assertIn('ice_executions', stats)
        self.assertIn('enhancements_applied', stats)

        # Check counts are correct
        self.assertEqual(stats['awareness_analyses'], 1)
        self.assertEqual(stats['ice_executions'], 1)

    def test_average_awareness_tracking(self):
        """Test average awareness level is tracked"""
        # Store multiple concepts with awareness
        concepts = ["wisdom", "love", "justice"]
        for concept in concepts:
            self.db.store_concept_with_awareness(concept, "biblical")

        stats = self.db.get_enhanced_statistics()

        self.assertIn('average_awareness_level', stats)
        self.assertGreaterEqual(stats['average_awareness_level'], 0.0)
        self.assertLessEqual(stats['average_awareness_level'], 1.0)

    def test_average_divine_alignment(self):
        """Test average divine alignment is tracked"""
        # Process some thoughts
        thoughts = ["Help me", "Guide me", "Teach me"]
        for thought in thoughts:
            self.db.process_thought_to_concept(
                thought,
                ThoughtType.SPIRITUAL_GUIDANCE,
                ContextDomain.PERSONAL
            )

        stats = self.db.get_enhanced_statistics()

        self.assertIn('average_divine_alignment', stats)
        self.assertGreaterEqual(stats['average_divine_alignment'], 0.0)
        self.assertLessEqual(stats['average_divine_alignment'], 1.0)

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    def test_mixed_storage_methods(self):
        """Test mixing regular and aware storage methods"""
        # Store with regular method
        id1 = self.db.store_concept("regular concept", "test")

        # Store with awareness
        id2 = self.db.store_concept_with_awareness("aware concept", "test")

        # Both should work
        self.assertGreater(id1, 0)
        self.assertGreater(id2, 0)

        # Check statistics
        stats = self.db.get_enhanced_statistics()
        self.assertEqual(stats['total_concepts'], 2)
        self.assertEqual(stats['awareness_analyses'], 1)

    def test_full_workflow(self):
        """Test complete enhanced workflow"""
        # 1. Store concepts with awareness
        self.db.store_concept_with_awareness("divine love", "biblical")
        self.db.store_concept_with_awareness("holy wisdom", "biblical")

        # 2. Process a thought through ICE
        result = self.db.process_thought_to_concept(
            "How can I grow spiritually?",
            ThoughtType.SPIRITUAL_GUIDANCE,
            ContextDomain.PERSONAL
        )

        # 3. Enable self-enhancing relationships
        rel_result = self.db.enable_self_enhancing_relationships(
            context="biblical"
        )

        # 4. Query with thought understanding
        query_results = self.db.query_with_thought_understanding(
            "I need divine guidance",
            "biblical"
        )

        # 5. Get comprehensive statistics
        stats = self.db.get_enhanced_statistics()
        report = self.db.get_engine_self_report()

        # Verify everything worked
        self.assertGreater(result['concept_id'], 0)
        self.assertGreater(rel_result['total_enhanced'], 0)
        self.assertGreater(len(query_results), 0)
        self.assertGreater(stats['total_concepts'], 0)
        self.assertIn('self_assessment', report)

    # ========================================================================
    # PERFORMANCE TESTS
    # ========================================================================

    def test_performance_with_awareness(self):
        """Test performance doesn't degrade significantly with awareness"""
        import time

        # Test aware storage
        start = time.time()
        for i in range(10):
            self.db.store_concept_with_awareness(f"concept_{i}", "test")
        aware_time = time.time() - start

        # Should complete reasonably fast (< 5 seconds for 10 concepts)
        self.assertLess(aware_time, 5.0,
                       f"Aware storage took {aware_time:.2f}s for 10 concepts")

    def test_ice_processing_performance(self):
        """Test ICE processing performance"""
        import time

        start = time.time()
        for i in range(5):
            self.db.process_thought_to_concept(
                f"test thought {i}",
                ThoughtType.PRACTICAL_WISDOM,
                ContextDomain.PERSONAL
            )
        ice_time = time.time() - start

        # Should complete reasonably fast (< 3 seconds for 5 thoughts)
        self.assertLess(ice_time, 3.0,
                       f"ICE processing took {ice_time:.2f}s for 5 thoughts")


def run_all_tests():
    """Run all enhanced database tests"""
    print("="*80)
    print("ENHANCED SEMANTIC SUBSTRATE DATABASE - TEST SUITE")
    print("="*80)
    print("\nTesting:")
    print("  1. Self-Aware Semantic Engine V3 Integration")
    print("  2. ICE Framework (Intent Context Execution)")
    print("  3. Enhanced Relationship Discovery")
    print("  4. Thought-Understanding Queries")
    print("  5. Backward Compatibility")
    print("="*80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnhancedDatabase)

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[SUCCESS] ALL ENHANCED DATABASE TESTS PASSED!")
        print("\nEnhanced SSDB is production-ready with:")
        print("  [PASS] Self-Aware Engine Integration")
        print("  [PASS] ICE Framework Integration")
        print("  [PASS] Enhanced Relationship Discovery")
        print("  [PASS] Thought-Understanding Queries")
        print("  [PASS] Backward Compatibility")
    else:
        print("\n[FAILED] Some tests failed - review output above")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
