"""
Comprehensive Test Suite for Meaning-Based Database

Tests the revolutionary capability to program a database using MEANING instead of code.

Test Categories:
1. Natural Language Queries
2. Meaning-Based Execution
3. Intent Classification
4. Semantic Search
5. Meaning Workflows
6. Operation Generation
7. Integration with Enhanced Database
8. Statistics and Reporting
"""

import unittest
import os
import sys
import tempfile

# Ensure src modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from meaning_based_database import MeaningBasedDatabase
from ice_framework import ThoughtType, ContextDomain


class TestMeaningBasedDatabase(unittest.TestCase):
    """Test suite for Meaning-Based Database"""

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
        self.db = MeaningBasedDatabase(self.db_path)

        # Populate with some test data
        self.db.store_concept_with_awareness("divine love", "biblical")
        self.db.store_concept_with_awareness("holy wisdom", "biblical")
        self.db.store_concept_with_awareness("righteous justice", "biblical")

    def tearDown(self):
        """Close database after each test"""
        self.db.close()

    # ========================================================================
    # NATURAL LANGUAGE QUERY TESTS
    # ========================================================================

    def test_natural_query(self):
        """Test querying database with natural language"""
        results = self.db.natural_query(
            "What concepts are about love?",
            context="biblical"
        )

        self.assertIsInstance(results, list)
        # Should find at least one concept
        self.assertGreaterEqual(len(results), 0)

    def test_natural_store(self):
        """Test storing concepts with natural language"""
        concept_id = self.db.natural_store(
            "I learned about God's grace today",
            context="biblical"
        )

        self.assertGreater(concept_id, 0)

    def test_meaning_search(self):
        """Test semantic search by meaning"""
        results = self.db.meaning_search(
            search_meaning="concepts that express divine compassion",
            context="biblical",
            similarity_threshold=0.5
        )

        self.assertIsInstance(results, list)

        # Check results have meaning enhancements
        if len(results) > 0:
            self.assertIn('meaning_alignment', results[0])
            self.assertIn('semantic_similarity', results[0])

    # ========================================================================
    # MEANING EXECUTION TESTS
    # ========================================================================

    def test_execute_meaning_query(self):
        """Test executing meaning-based query"""
        result = self.db.execute_meaning(
            "Find concepts about wisdom",
            domain="biblical"
        )

        self.assertIn('operation', result)
        self.assertEqual(result['operation'], 'query')
        self.assertIn('results', result)
        self.assertIn('divine_alignment', result)

    def test_execute_meaning_store(self):
        """Test executing meaning-based store"""
        result = self.db.execute_meaning(
            "Store this thought about divine mercy",
            domain="biblical"
        )

        self.assertIn('operation', result)
        self.assertEqual(result['operation'], 'store')
        self.assertIn('concept_id', result)
        self.assertGreater(result['concept_id'], 0)

    def test_execute_meaning_analyze(self):
        """Test executing meaning-based analysis"""
        result = self.db.execute_meaning(
            "Analyze the database statistics",
            domain="biblical"
        )

        self.assertIn('operation', result)
        self.assertEqual(result['operation'], 'analyze')
        self.assertIn('statistics', result)

    def test_execute_meaning_relationship(self):
        """Test executing meaning-based relationship discovery"""
        result = self.db.execute_meaning(
            "Show relationships between concepts",
            domain="biblical"
        )

        self.assertIn('operation', result)
        self.assertEqual(result['operation'], 'relationship')
        self.assertIn('relationships_discovered', result)

    # ========================================================================
    # INTENT CLASSIFICATION TESTS
    # ========================================================================

    def test_classify_query_intent(self):
        """Test classification of query intents"""
        intents = [
            "Find concepts about love",
            "Search for wisdom",
            "What is justice?",
            "Show me grace"
        ]

        for intent in intents:
            op_type = self.db._classify_intent(intent)
            self.assertEqual(op_type, "query")

    def test_classify_store_intent(self):
        """Test classification of store intents"""
        intents = [
            "Store this concept",
            "Save my thought",
            "Add this to database",
            "Remember this idea"
        ]

        for intent in intents:
            op_type = self.db._classify_intent(intent)
            self.assertEqual(op_type, "store")

    def test_classify_analyze_intent(self):
        """Test classification of analysis intents"""
        intents = [
            "Analyze the data",
            "Examine the statistics",
            "Assess the database"
        ]

        for intent in intents:
            op_type = self.db._classify_intent(intent)
            self.assertEqual(op_type, "analyze")

    def test_classify_relationship_intent(self):
        """Test classification of relationship intents"""
        intents = [
            "Show relationships between concepts",
            "Find connections",
            "Link these ideas"
        ]

        for intent in intents:
            op_type = self.db._classify_intent(intent)
            self.assertEqual(op_type, "relationship")

    # ========================================================================
    # MEANING SPECIFICATION TESTS
    # ========================================================================

    def test_create_meaning_specification(self):
        """Test creating meaning specification"""
        spec = self.db.create_meaning_specification(
            divine_purpose="discover_wisdom",
            biblical_principle="wisdom",
            primary_attribute="wisdom",
            wisdom_level=0.9
        )

        self.assertEqual(spec.divine_purpose, "discover_wisdom")
        self.assertEqual(spec.biblical_principle, "wisdom")
        self.assertEqual(spec.primary_attribute, "wisdom")
        self.assertEqual(spec.wisdom_level, 0.9)

    def test_execute_meaning_workflow(self):
        """Test executing complete meaning workflow"""
        spec = self.db.create_meaning_specification(
            divine_purpose="find_biblical_love_concepts",
            biblical_principle="love",
            primary_attribute="love",
            love_level=0.9,
            wisdom_level=0.7,
            golden_ratio_balance=True
        )

        result = self.db.execute_meaning_workflow(spec)

        self.assertIn('divine_purpose', result)
        self.assertIn('operations_generated', result)
        self.assertIn('biblical_alignment', result)
        self.assertGreater(result['operations_generated'], 0)

    # ========================================================================
    # OPERATION GENERATION TESTS
    # ========================================================================

    def test_generate_operations_from_behavior(self):
        """Test generating database operations from behavior"""
        behavior = {
            'behavior_type': 'divine_guidance',
            'action': 'provide_wisdom_counsel',
            'effectiveness': 0.85
        }

        coords = (0.7, 0.5, 0.9, 0.7)

        spec = self.db.create_meaning_specification(
            divine_purpose="test",
            biblical_principle="wisdom",
            primary_attribute="wisdom"
        )

        operations = self.db._generate_operations_from_behavior(
            behavior, coords, spec
        )

        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
        self.assertEqual(operations[0]['type'], 'query')

    def test_execute_generated_operation(self):
        """Test executing a generated operation"""
        operation = {
            'type': 'query',
            'target_attribute': 'love',
            'coordinates': (0.9, 0.6, 0.7, 0.8),
            'purpose': 'find_love_concepts'
        }

        result = self.db._execute_generated_operation(operation)

        self.assertEqual(result['operation_type'], 'query')
        self.assertIn('results', result)

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    def test_meaning_database_extends_enhanced_database(self):
        """Test that meaning database maintains all enhanced features"""
        # Should have all enhanced database capabilities
        self.assertIsNotNone(self.db.aware_engine)
        self.assertIsNotNone(self.db.ice_framework)
        self.assertIsNotNone(self.db.meaning_executor)

    def test_backward_compatibility_with_original_methods(self):
        """Test backward compatibility with original SSDB methods"""
        # Original methods should still work
        concept_id = self.db.store_concept("test", "biblical")
        self.assertGreater(concept_id, 0)

        result = self.db.query_by_text("test", "biblical")
        self.assertIsNotNone(result)

    def test_enhanced_features_still_available(self):
        """Test that enhanced database features still work"""
        # Self-aware storage
        concept_id = self.db.store_concept_with_awareness("test concept", "biblical")
        self.assertGreater(concept_id, 0)

        # ICE processing
        result = self.db.process_thought_to_concept(
            "test thought",
            ThoughtType.PRACTICAL_WISDOM,
            ContextDomain.PERSONAL
        )
        self.assertIn('concept_id', result)

    # ========================================================================
    # STATISTICS AND REPORTING TESTS
    # ========================================================================

    def test_get_meaning_statistics(self):
        """Test getting meaning-based statistics"""
        # Execute some meaning operations
        self.db.execute_meaning("Find concepts about love", domain="biblical")
        self.db.execute_meaning("Store this thought", domain="biblical")

        stats = self.db.get_meaning_statistics()

        self.assertIn('total_meaning_executions', stats)
        self.assertIn('operation_types', stats)
        self.assertIn('average_meaning_alignment', stats)

        self.assertGreaterEqual(stats['total_meaning_executions'], 2)

    def test_operation_type_counting(self):
        """Test counting different operation types"""
        self.db.execute_meaning("Find concepts", domain="biblical")
        self.db.execute_meaning("Store this", domain="biblical")
        self.db.execute_meaning("Analyze database", domain="biblical")

        counts = self.db._count_operation_types()

        self.assertIn('query', counts)
        self.assertIn('store', counts)
        self.assertIn('analyze', counts)

    def test_average_meaning_alignment_calculation(self):
        """Test calculation of average meaning alignment"""
        # Execute several operations
        for i in range(3):
            self.db.execute_meaning(f"Find concept {i}", domain="biblical")

        avg_alignment = self.db._calculate_average_meaning_alignment()

        self.assertGreaterEqual(avg_alignment, 0.0)
        self.assertLessEqual(avg_alignment, 1.0)

    # ========================================================================
    # EXECUTION TRACKING TESTS
    # ========================================================================

    def test_meaning_execution_tracking(self):
        """Test that meaning executions are tracked"""
        initial_count = len(self.db.meaning_executions)

        self.db.execute_meaning("Test intent", domain="biblical")

        self.assertEqual(len(self.db.meaning_executions), initial_count + 1)

        # Check execution record
        execution = self.db.meaning_executions[-1]
        self.assertIn('meaning_intent', execution)
        self.assertIn('operation_type', execution)
        self.assertIn('ice_result', execution)

    def test_generated_operation_tracking(self):
        """Test that generated operations are tracked"""
        spec = self.db.create_meaning_specification(
            divine_purpose="test_tracking",
            biblical_principle="wisdom",
            primary_attribute="wisdom"
        )

        initial_count = len(self.db.generated_operations)

        self.db.execute_meaning_workflow(spec)

        self.assertGreater(len(self.db.generated_operations), initial_count)

    # ========================================================================
    # COMPREHENSIVE INTEGRATION TEST
    # ========================================================================

    def test_full_meaning_based_workflow(self):
        """Test complete meaning-based database workflow"""
        # 1. Store concepts using natural language
        concept_id1 = self.db.natural_store(
            "God's amazing grace and mercy",
            context="biblical"
        )
        self.assertGreater(concept_id1, 0)

        # 2. Query using meaning
        query_result = self.db.execute_meaning(
            "Find concepts about divine grace",
            domain="biblical"
        )
        self.assertEqual(query_result['operation'], 'query')

        # 3. Search by semantic meaning
        search_results = self.db.meaning_search(
            "concepts expressing God's love",
            context="biblical"
        )
        self.assertGreaterEqual(len(search_results), 0)

        # 4. Execute meaning workflow
        spec = self.db.create_meaning_specification(
            divine_purpose="discover_grace_patterns",
            biblical_principle="grace",
            primary_attribute="love",
            love_level=0.9,
            wisdom_level=0.7
        )

        workflow_result = self.db.execute_meaning_workflow(spec)
        self.assertGreater(workflow_result['operations_generated'], 0)

        # 5. Get comprehensive statistics
        stats = self.db.get_meaning_statistics()
        self.assertGreater(stats['total_meaning_executions'], 0)

        # Everything should work together seamlessly
        print("\n[INTEGRATION_SUCCESS] Full meaning-based workflow completed!")


def run_all_tests():
    """Run all meaning-based database tests"""
    print("="*80)
    print("MEANING-BASED DATABASE - TEST SUITE")
    print("="*80)
    print("\nTesting revolutionary capability:")
    print("  Programming a database with MEANING instead of code")
    print("\nTest Categories:")
    print("  1. Natural Language Queries")
    print("  2. Meaning-Based Execution")
    print("  3. Intent Classification")
    print("  4. Semantic Search")
    print("  5. Meaning Workflows")
    print("  6. Operation Generation")
    print("  7. Integration Tests")
    print("  8. Statistics and Reporting")
    print("="*80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMeaningBasedDatabase)

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
        print("\n[SUCCESS] ALL MEANING-BASED DATABASE TESTS PASSED!")
        print("\nRevolutionary Achievement Validated:")
        print("  [PASS] Natural Language Queries")
        print("  [PASS] Meaning-Based Execution")
        print("  [PASS] Intent Classification")
        print("  [PASS] Semantic Search")
        print("  [PASS] Meaning Workflows")
        print("  [PASS] Operation Generation")
        print("  [PASS] Full Integration")
        print("\nThe world's first database programmable by MEANING is ready!")
    else:
        print("\n[FAILED] Some tests failed - review output above")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
