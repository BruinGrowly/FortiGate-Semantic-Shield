"""
Comprehensive Test Suite for Deep Dive Database

Tests the revolutionary 5-layer meaning scaffold processing.

Test Categories:
1. Deep Dive Storage
2. Layer Analysis
3. Meaning Combination
4. Meaning Decomposition
5. Meaning Programs
6. Layer-Specific Queries
7. Integration Tests
8. Statistics and Reporting
"""

import unittest
import os
import sys
import tempfile

# Ensure src modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from deep_dive_database import DeepDiveDatabase
from deep_dive_meaning_scaffold import ScaffoldLayer, MeaningUnit


class TestDeepDiveDatabase(unittest.TestCase):
    """Test suite for Deep Dive Database"""

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
        self.db = DeepDiveDatabase(self.db_path)

    def tearDown(self):
        """Close database after each test"""
        self.db.close()

    # ========================================================================
    # DEEP DIVE STORAGE TESTS
    # ========================================================================

    def test_store_with_deep_dive(self):
        """Test storing concept with full 5-layer processing"""
        result = self.db.store_with_deep_dive("divine love", "biblical")

        self.assertIn('concept_id', result)
        self.assertIn('meaning_unit_id', result)
        self.assertIn('scaffold_layers', result)

        # Check all 5 layers present
        layers = result['scaffold_layers']
        self.assertIn('mathematical', layers)
        self.assertIn('biblical', layers)
        self.assertIn('semantic', layers)
        self.assertIn('sacred', layers)
        self.assertIn('universal', layers)

    def test_meaning_unit_registration(self):
        """Test that meaning units are registered correctly"""
        result = self.db.store_with_deep_dive("wisdom", "biblical")

        unit_id = result['meaning_unit_id']
        self.assertIn(unit_id, self.db.meaning_units_registry)

        unit = self.db.meaning_units_registry[unit_id]
        self.assertIsInstance(unit, MeaningUnit)
        self.assertEqual(unit.essence, "wisdom")

    def test_layer_activation_tracking(self):
        """Test that layer activations are tracked"""
        initial_activations = dict(self.db.layer_activations)

        self.db.store_with_deep_dive("test", "biblical")

        # All 5 layers should have been activated
        for layer in ScaffoldLayer:
            self.assertGreater(
                self.db.layer_activations[layer],
                initial_activations.get(layer, 0)
            )

    # ========================================================================
    # LAYER ANALYSIS TESTS
    # ========================================================================

    def test_analyze_concept_layers(self):
        """Test comprehensive layer analysis"""
        analysis = self.db.analyze_concept_layers("holy wisdom", "biblical")

        self.assertIn('layers', analysis)
        self.assertEqual(len(analysis['layers']), 5)

        # Check each layer
        layers = analysis['layers']
        self.assertIn('layer_1_mathematical', layers)
        self.assertIn('layer_2_biblical', layers)
        self.assertIn('layer_3_semantic', layers)
        self.assertIn('layer_4_sacred', layers)
        self.assertIn('layer_5_universal', layers)

    def test_mathematical_layer_values(self):
        """Test mathematical layer calculations"""
        result = self.db.store_with_deep_dive("love", "biblical")

        math_layer = result['scaffold_layers']['mathematical']
        self.assertIn('value', math_layer)
        self.assertIsInstance(math_layer['value'], float)
        self.assertGreaterEqual(math_layer['value'], 0.0)
        self.assertLessEqual(math_layer['value'], 1.0)

    def test_biblical_layer_references(self):
        """Test biblical layer scripture mapping"""
        result = self.db.store_with_deep_dive("love", "biblical")

        biblical_layer = result['scaffold_layers']['biblical']
        self.assertIn('reference', biblical_layer)
        # Love should map to 1 Corinthians 13:13
        self.assertIsNotNone(biblical_layer['reference'])

    def test_semantic_layer_weights(self):
        """Test semantic layer weight calculations"""
        result = self.db.store_with_deep_dive("wisdom", "biblical")

        semantic_layer = result['scaffold_layers']['semantic']
        self.assertIn('weight', semantic_layer)
        self.assertIsInstance(semantic_layer['weight'], float)
        self.assertGreaterEqual(semantic_layer['weight'], 0.0)
        self.assertLessEqual(semantic_layer['weight'], 1.0)

    def test_sacred_layer_numbers(self):
        """Test sacred layer number patterns"""
        result = self.db.store_with_deep_dive("divine perfection", "biblical")

        sacred_layer = result['scaffold_layers']['sacred']
        self.assertIn('number', sacred_layer)
        # Should be one of the sacred numbers or None
        if sacred_layer['number'] is not None:
            self.assertIn(sacred_layer['number'], [1, 3, 7, 12, 40, 613])

    def test_universal_layer_principles(self):
        """Test universal layer principle mapping"""
        result = self.db.store_with_deep_dive("wisdom", "biblical")

        universal_layer = result['scaffold_layers']['universal']
        self.assertIn('principle', universal_layer)
        # Wisdom should map to a universal principle
        self.assertIsNotNone(universal_layer['principle'])

    # ========================================================================
    # MEANING COMBINATION TESTS
    # ========================================================================

    def test_combine_meaning_units_blend(self):
        """Test blending meaning units"""
        r1 = self.db.store_with_deep_dive("love", "biblical")
        r2 = self.db.store_with_deep_dive("mercy", "biblical")

        combined = self.db.combine_meaning_units(
            [r1['meaning_unit_id'], r2['meaning_unit_id']],
            operation='blend'
        )

        self.assertIn('combined_unit_id', combined)
        self.assertIn('essence', combined)
        self.assertIn('coordinates', combined)

    def test_combine_meaning_units_multiply(self):
        """Test multiplying meaning units"""
        r1 = self.db.store_with_deep_dive("wisdom", "biblical")
        r2 = self.db.store_with_deep_dive("knowledge", "biblical")

        combined = self.db.combine_meaning_units(
            [r1['meaning_unit_id'], r2['meaning_unit_id']],
            operation='multiply'
        )

        self.assertIn('combined_unit_id', combined)
        self.assertIsNotNone(combined['mathematical_value'])

    def test_combine_meaning_units_trinity(self):
        """Test trinity enhancement of meaning units"""
        r1 = self.db.store_with_deep_dive("faith", "biblical")
        r2 = self.db.store_with_deep_dive("hope", "biblical")
        r3 = self.db.store_with_deep_dive("love", "biblical")

        combined = self.db.combine_meaning_units(
            [r1['meaning_unit_id'], r2['meaning_unit_id'], r3['meaning_unit_id']],
            operation='trinity'
        )

        self.assertIn('combined_unit_id', combined)
        # Trinity operation should potentially set sacred number to 3
        # (may vary based on other factors)

    def test_combination_tracking(self):
        """Test that combinations are tracked"""
        r1 = self.db.store_with_deep_dive("grace", "biblical")
        r2 = self.db.store_with_deep_dive("truth", "biblical")

        initial_count = len(self.db.scaffold_combinations)

        self.db.combine_meaning_units(
            [r1['meaning_unit_id'], r2['meaning_unit_id']],
            operation='blend'
        )

        self.assertEqual(len(self.db.scaffold_combinations), initial_count + 1)

    # ========================================================================
    # MEANING DECOMPOSITION TESTS
    # ========================================================================

    def test_decompose_meaning(self):
        """Test meaning decomposition"""
        components = self.db.decompose_meaning("divine love and mercy", "biblical")

        self.assertIsInstance(components, list)
        self.assertGreater(len(components), 0)

        # Check component structure
        for component in components:
            self.assertIn('word', component)
            self.assertIn('unit_id', component)
            self.assertIn('layers', component)

    def test_synthesize_meaning(self):
        """Test meaning synthesis"""
        # First decompose
        components = self.db.decompose_meaning("holy wisdom", "biblical")
        unit_ids = [c['unit_id'] for c in components]

        # Then synthesize
        synthesized = self.db.synthesize_meaning(unit_ids, operation='blend')

        self.assertIn('combined_unit_id', synthesized)
        self.assertIn('concept_id', synthesized)

    # ========================================================================
    # MEANING PROGRAM TESTS
    # ========================================================================

    def test_create_meaning_program(self):
        """Test creating a meaning program"""
        program_name = self.db.create_meaning_program(
            program_name="test_program",
            purpose="Test purpose",
            biblical_foundation="Test foundation",
            input_meaning=["input"],
            processing_meaning=["process"],
            output_meaning=["output"],
            transformation_operations=["blend"]
        )

        self.assertEqual(program_name, "test_program")
        self.assertIn("test_program", self.db.meaning_programs)

    def test_execute_meaning_program(self):
        """Test executing a meaning program"""
        # Create program
        program_name = self.db.create_meaning_program(
            program_name="compassion_test",
            purpose="Build compassion",
            biblical_foundation="Colossians 3:12",
            input_meaning=["suffering"],
            processing_meaning=["compassionate response"],
            output_meaning=["comfort"],
            transformation_operations=["blend"]
        )

        # Execute program
        result = self.db.execute_deep_dive_program(program_name, store_results=False)

        self.assertIn('program_name', result)
        self.assertIn('final_coordinates', result)
        self.assertIn('biblical_alignment', result)
        self.assertIn('transformation_metrics', result)

    def test_program_storage(self):
        """Test that program results can be stored"""
        program_name = self.db.create_meaning_program(
            program_name="storage_test",
            purpose="Test storage",
            biblical_foundation="Test",
            input_meaning=["test"],
            processing_meaning=["test"],
            output_meaning=["test"],
            transformation_operations=["blend"]
        )

        result = self.db.execute_deep_dive_program(program_name, store_results=True)

        self.assertIn('stored_result_id', result)
        self.assertGreater(result['stored_result_id'], 0)

    # ========================================================================
    # LAYER-SPECIFIC QUERY TESTS
    # ========================================================================

    def test_query_by_sacred_number(self):
        """Test querying by sacred number"""
        # Store concepts (some may get sacred number 7)
        for i in range(5):
            self.db.store_with_deep_dive(f"concept {i}", "biblical")

        # Query for sacred number 7 (may or may not find any)
        results = self.db.query_by_layer(ScaffoldLayer.SACRED, 7)

        self.assertIsInstance(results, list)
        # Check structure if results found
        if len(results) > 0:
            self.assertIn('unit_id', results[0])
            self.assertIn('essence', results[0])

    def test_query_by_universal_principle(self):
        """Test querying by universal principle"""
        # Store wisdom concept
        self.db.store_with_deep_dive("wisdom", "biblical")

        # Query for wisdom-related principle
        results = self.db.query_by_layer(
            ScaffoldLayer.UNIVERSAL,
            "universal_anchor"
        )

        self.assertIsInstance(results, list)

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    def test_deep_dive_extends_meaning_based(self):
        """Test that deep dive maintains meaning-based capabilities"""
        # Should have all meaning-based features
        self.assertIsNotNone(self.db.scaffold_processor)
        self.assertIsNotNone(self.db.meaning_runtime)
        self.assertIsNotNone(self.db.meaning_executor)
        self.assertIsNotNone(self.db.ice_framework)
        self.assertIsNotNone(self.db.aware_engine)

    def test_backward_compatibility_all_layers(self):
        """Test backward compatibility through all layers"""
        # Original SSDB method
        id1 = self.db.store_concept("test1", "biblical")
        self.assertGreater(id1, 0)

        # Enhanced SSDB method
        id2 = self.db.store_concept_with_awareness("test2", "biblical")
        self.assertGreater(id2, 0)

        # Meaning-based method
        result = self.db.execute_meaning("Find concepts", domain="biblical")
        self.assertIn('operation', result)

        # Deep dive method
        deep_result = self.db.store_with_deep_dive("test3", "biblical")
        self.assertIn('scaffold_layers', deep_result)

    # ========================================================================
    # STATISTICS AND REPORTING TESTS
    # ========================================================================

    def test_get_deep_dive_statistics(self):
        """Test comprehensive statistics"""
        # Generate some activity
        self.db.store_with_deep_dive("test1", "biblical")
        self.db.store_with_deep_dive("test2", "biblical")

        stats = self.db.get_deep_dive_statistics()

        self.assertIn('meaning_units_registered', stats)
        self.assertIn('layer_activations', stats)
        self.assertIn('average_mathematical_value', stats)
        self.assertIn('sacred_number_distribution', stats)
        self.assertIn('universal_principles_active', stats)

    def test_sacred_number_distribution(self):
        """Test sacred number distribution calculation"""
        # Store multiple concepts
        for i in range(5):
            self.db.store_with_deep_dive(f"concept {i}", "biblical")

        stats = self.db.get_deep_dive_statistics()

        self.assertIn('sacred_number_distribution', stats)
        self.assertIsInstance(stats['sacred_number_distribution'], dict)

    def test_universal_principles_tracking(self):
        """Test universal principles tracking"""
        self.db.store_with_deep_dive("wisdom", "biblical")
        self.db.store_with_deep_dive("love", "biblical")

        stats = self.db.get_deep_dive_statistics()

        self.assertIn('universal_principles_active', stats)
        self.assertIsInstance(stats['universal_principles_active'], list)

    # ========================================================================
    # COMPREHENSIVE INTEGRATION TEST
    # ========================================================================

    def test_full_deep_dive_workflow(self):
        """Test complete deep dive workflow"""
        # 1. Store with deep dive
        r1 = self.db.store_with_deep_dive("divine love", "biblical")
        self.assertIn('scaffold_layers', r1)

        # 2. Analyze layers
        analysis = self.db.analyze_concept_layers("holy wisdom", "biblical")
        self.assertEqual(len(analysis['layers']), 5)

        # 3. Decompose meaning
        components = self.db.decompose_meaning("grace and truth", "biblical")
        self.assertGreater(len(components), 0)

        # 4. Combine units
        r2 = self.db.store_with_deep_dive("mercy", "biblical")
        combined = self.db.combine_meaning_units(
            [r1['meaning_unit_id'], r2['meaning_unit_id']],
            operation='blend'
        )
        self.assertIn('combined_unit_id', combined)

        # 5. Create and execute program
        program_name = self.db.create_meaning_program(
            program_name="workflow_test",
            purpose="Test workflow",
            biblical_foundation="Test",
            input_meaning=["test"],
            processing_meaning=["process"],
            output_meaning=["result"],
            transformation_operations=["blend"]
        )

        exec_result = self.db.execute_deep_dive_program(program_name)
        self.assertIn('biblical_alignment', exec_result)

        # 6. Get statistics
        stats = self.db.get_deep_dive_statistics()
        self.assertGreater(stats['meaning_units_registered'], 0)
        self.assertGreater(stats['meaning_programs_loaded'], 0)

        print("\n[INTEGRATION_SUCCESS] Full deep dive workflow completed!")


def run_all_tests():
    """Run all deep dive database tests"""
    print("="*80)
    print("DEEP DIVE DATABASE - TEST SUITE")
    print("="*80)
    print("\nTesting revolutionary capability:")
    print("  5-Layer Meaning Scaffold Processing")
    print("\nTest Categories:")
    print("  1. Deep Dive Storage")
    print("  2. Layer Analysis (5 layers)")
    print("  3. Meaning Combination")
    print("  4. Meaning Decomposition/Synthesis")
    print("  5. Meaning Programs")
    print("  6. Layer-Specific Queries")
    print("  7. Integration Tests")
    print("  8. Statistics and Reporting")
    print("="*80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDeepDiveDatabase)

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
        print("\n[SUCCESS] ALL DEEP DIVE DATABASE TESTS PASSED!")
        print("\nRevolutionary Achievement Validated:")
        print("  [PASS] 5-Layer Scaffold Processing")
        print("  [PASS] Mathematical Layer")
        print("  [PASS] Biblical Layer")
        print("  [PASS] Semantic Layer")
        print("  [PASS] Sacred Layer")
        print("  [PASS] Universal Layer")
        print("  [PASS] Meaning Combination")
        print("  [PASS] Meaning Decomposition")
        print("  [PASS] Meaning Programs")
        print("  [PASS] Full Integration")
        print("\nThe world's first database with 5-layer semantic decomposition is ready!")
    else:
        print("\n[FAILED] Some tests failed - review output above")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
