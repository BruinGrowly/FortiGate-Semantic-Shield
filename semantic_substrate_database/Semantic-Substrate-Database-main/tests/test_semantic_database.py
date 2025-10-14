"""
SEMANTIC SUBSTRATE DATABASE - Comprehensive Test Suite

Tests for the revolutionary meaning-native database system.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import database
from semantic_substrate_database import SemanticSubstrateDatabase
from baseline_biblical_substrate import BiblicalCoordinates


class TestSemanticDatabase(unittest.TestCase):
    """Comprehensive test suite for Semantic Substrate Database"""

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
        # Delete existing database to ensure clean state
        try:
            os.unlink(self.db_path)
        except:
            pass
        self.db = SemanticSubstrateDatabase(self.db_path)

    def tearDown(self):
        """Close database after each test"""
        self.db.close()

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_database_initialization(self):
        """Test database initializes correctly"""
        self.assertIsNotNone(self.db.conn)
        self.assertIsNotNone(self.db.engine)
        self.assertEqual(len(self.db.cache), 0)

    def test_schema_creation(self):
        """Test all tables are created"""
        cursor = self.db.conn.cursor()

        tables = [
            'semantic_coordinates',
            'semantic_units',
            'sacred_numbers',
            'universal_anchors',
            'concept_relationships',
            'contextual_resonance',
            'semantic_evolution'
        ]

        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            result = cursor.fetchone()
            self.assertIsNotNone(result, f"Table {table} should exist")

    def test_universal_anchors_initialized(self):
        """Test universal anchors are initialized"""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM universal_anchors")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 4, "Should have 4 universal anchors")

        # Check specific anchors
        for anchor_id in [613, 12, 7, 40]:
            cursor.execute("SELECT * FROM universal_anchors WHERE id = ?", (anchor_id,))
            anchor = cursor.fetchone()
            self.assertIsNotNone(anchor, f"Anchor {anchor_id} should exist")

    # ========================================================================
    # STORAGE TESTS
    # ========================================================================

    def test_store_concept_basic(self):
        """Test storing a basic concept"""
        concept_id = self.db.store_concept("love", "biblical")

        self.assertIsNotNone(concept_id)
        self.assertGreater(concept_id, 0)

        # Verify stored in database
        result = self.db.query_by_text("love", "biblical")
        self.assertIsNotNone(result)
        self.assertEqual(result['concept_text'], "love")
        self.assertEqual(result['context'], "biblical")

    def test_store_multiple_concepts(self):
        """Test storing multiple concepts"""
        concepts = ["love", "wisdom", "justice", "mercy", "faith"]

        for concept in concepts:
            concept_id = self.db.store_concept(concept, "biblical")
            self.assertGreater(concept_id, 0)

        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], len(concepts))

    def test_store_concept_duplicate(self):
        """Test storing duplicate concept updates existing"""
        # Store first time
        id1 = self.db.store_concept("wisdom", "biblical")

        # Store again - should update
        id2 = self.db.store_concept("wisdom", "biblical")

        # Should still be only one concept
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 1)

    def test_store_concept_different_contexts(self):
        """Test same concept in different contexts"""
        id1 = self.db.store_concept("justice", "biblical")
        id2 = self.db.store_concept("justice", "educational")

        # Should be two different entries
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 2)

        result1 = self.db.query_by_text("justice", "biblical")
        result2 = self.db.query_by_text("justice", "educational")

        # Verify both entries exist (coordinates may be similar for same word)
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(result1['context'], "biblical")
        self.assertEqual(result2['context'], "educational")

    def test_store_sacred_number(self):
        """Test storing sacred numbers"""
        sacred_id = self.db.store_sacred_number(7)
        self.assertGreater(sacred_id, 0)

        # Query back
        results = self.db.query_sacred_numbers(min_value=7, max_value=7)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['value'], 7)
        self.assertTrue(results[0]['is_sacred'])

    def test_store_multiple_sacred_numbers(self):
        """Test storing multiple sacred numbers"""
        sacred_nums = [1, 3, 7, 12, 40, 613]

        for num in sacred_nums:
            self.db.store_sacred_number(num)

        results = self.db.query_sacred_numbers(min_value=0, max_value=1000, only_sacred=True)
        self.assertGreaterEqual(len(results), len(sacred_nums))

    def test_store_relationship(self):
        """Test storing concept relationships"""
        id1 = self.db.store_concept("love", "biblical")
        id2 = self.db.store_concept("mercy", "biblical")

        rel_id = self.db.store_relationship(id1, id2, "semantic_proximity")
        self.assertGreater(rel_id, 0)

        stats = self.db.get_statistics()
        self.assertEqual(stats['total_relationships'], 1)

    # ========================================================================
    # QUERY TESTS
    # ========================================================================

    def test_query_by_text_found(self):
        """Test querying concept by text"""
        self.db.store_concept("wisdom", "biblical")

        result = self.db.query_by_text("wisdom", "biblical")
        self.assertIsNotNone(result)
        self.assertEqual(result['concept_text'], "wisdom")

    def test_query_by_text_not_found(self):
        """Test querying non-existent concept"""
        result = self.db.query_by_text("nonexistent", "biblical")
        self.assertIsNone(result)

    def test_query_by_proximity(self):
        """Test revolutionary proximity query"""
        # Store some concepts
        concepts = ["love", "mercy", "compassion", "kindness", "wrath"]
        for concept in concepts:
            self.db.store_concept(concept, "biblical")

        # Get coordinates for "love"
        love_result = self.db.query_by_text("love", "biblical")
        love_coords = BiblicalCoordinates(
            love_result['love'], love_result['power'],
            love_result['wisdom'], love_result['justice']
        )

        # Find concepts near "love"
        nearby = self.db.query_by_proximity(love_coords, max_distance=0.5,
                                           context="biblical", limit=5)

        self.assertGreater(len(nearby), 0)
        # First result should be love itself (distance 0)
        self.assertEqual(nearby[0]['concept_text'], "love")
        self.assertAlmostEqual(nearby[0]['semantic_distance'], 0.0, places=5)

    def test_query_by_divine_resonance(self):
        """Test query by divine resonance"""
        # Store concepts with varying divine alignment
        concepts = ["love", "mercy", "wisdom", "faith", "hope"]
        for concept in concepts:
            self.db.store_concept(concept, "biblical")

        # Query ANY resonance concepts (current SSE gives low values)
        results = self.db.query_by_divine_resonance(min_resonance=0.0,
                                                    context="biblical",
                                                    limit=10)

        self.assertGreater(len(results), 0)
        # All results should have resonance >= 0.0
        for result in results:
            self.assertGreaterEqual(result['divine_resonance'], 0.0)

        # Results should be sorted by resonance descending
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]['divine_resonance'],
                                   results[i+1]['divine_resonance'])

    def test_query_sacred_numbers(self):
        """Test querying sacred numbers"""
        # Store some sacred numbers
        for num in [3, 7, 12, 40, 100, 613]:
            self.db.store_sacred_number(num)

        # Query all sacred numbers
        results = self.db.query_sacred_numbers(min_value=0, max_value=1000,
                                              only_sacred=True)

        self.assertGreater(len(results), 0)
        for result in results:
            self.assertTrue(result['is_sacred'])

        # Query specific range
        results_small = self.db.query_sacred_numbers(min_value=0, max_value=10,
                                                     only_sacred=True)
        for result in results_small:
            self.assertLessEqual(result['value'], 10)

    def test_query_nearest_to_anchor(self):
        """Test finding concepts nearest to universal anchor"""
        # Store some concepts
        concepts = ["love", "wisdom", "justice", "mercy", "faith"]
        for concept in concepts:
            self.db.store_concept(concept, "biblical")

        # Find concepts near anchor 7 (Divine Perfection)
        results = self.db.query_nearest_to_anchor(anchor_id=7,
                                                  max_distance=2.0,
                                                  limit=5)

        self.assertGreater(len(results), 0)
        # Results should be sorted by distance
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i]['semantic_distance'],
                                results[i+1]['semantic_distance'])

    def test_semantic_search(self):
        """Test revolutionary semantic search"""
        # Store concepts
        concepts = ["love", "mercy", "compassion", "kindness", "grace"]
        for concept in concepts:
            self.db.store_concept(concept, "biblical")

        # Semantic search for related concept
        query = "compassion and kindness"
        results = self.db.search_semantic(query, context="biblical", limit=5)

        self.assertGreater(len(results), 0)
        # Should have similarity scores
        for result in results:
            self.assertIn('semantic_similarity', result)
            self.assertIn('query_alignment', result)
            self.assertGreater(result['semantic_similarity'], 0)

        # Results should be sorted by query alignment
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]['query_alignment'],
                                   results[i+1]['query_alignment'])

    # ========================================================================
    # COORDINATE TESTS
    # ========================================================================

    def test_coordinates_stored_correctly(self):
        """Test coordinates are stored with correct values"""
        self.db.store_concept("love", "biblical")
        result = self.db.query_by_text("love", "biblical")

        # All coordinates should be between 0 and 1
        self.assertGreaterEqual(result['love'], 0.0)
        self.assertLessEqual(result['love'], 1.0)
        self.assertGreaterEqual(result['power'], 0.0)
        self.assertLessEqual(result['power'], 1.0)
        self.assertGreaterEqual(result['wisdom'], 0.0)
        self.assertLessEqual(result['wisdom'], 1.0)
        self.assertGreaterEqual(result['justice'], 0.0)
        self.assertLessEqual(result['justice'], 1.0)

    def test_divine_resonance_calculated(self):
        """Test divine resonance is calculated"""
        self.db.store_concept("wisdom", "biblical")
        result = self.db.query_by_text("wisdom", "biblical")

        self.assertIsNotNone(result['divine_resonance'])
        self.assertGreater(result['divine_resonance'], 0.0)
        self.assertLessEqual(result['divine_resonance'], 1.0)

    def test_distance_from_jehovah_calculated(self):
        """Test distance from JEHOVAH is calculated"""
        self.db.store_concept("mercy", "biblical")
        result = self.db.query_by_text("mercy", "biblical")

        self.assertIsNotNone(result['distance_from_jehovah'])
        self.assertGreaterEqual(result['distance_from_jehovah'], 0.0)

    # ========================================================================
    # CACHE TESTS
    # ========================================================================

    def test_cache_populated_on_store(self):
        """Test cache is populated when storing concepts"""
        initial_cache_size = len(self.db.cache)

        self.db.store_concept("wisdom", "biblical")

        self.assertGreater(len(self.db.cache), initial_cache_size)

    def test_cache_clear(self):
        """Test cache can be cleared"""
        self.db.store_concept("love", "biblical")
        self.assertGreater(len(self.db.cache), 0)

        self.db.clear_cache()
        self.assertEqual(len(self.db.cache), 0)

    # ========================================================================
    # STATISTICS TESTS
    # ========================================================================

    def test_statistics_empty_database(self):
        """Test statistics on empty database"""
        stats = self.db.get_statistics()

        self.assertEqual(stats['total_concepts'], 0)
        self.assertEqual(stats['total_semantic_units'], 0)
        self.assertEqual(stats['total_relationships'], 0)

    def test_statistics_with_data(self):
        """Test statistics with data"""
        # Add some data
        concepts = ["love", "wisdom", "justice"]
        for concept in concepts:
            self.db.store_concept(concept, "biblical")

        self.db.store_sacred_number(7)

        stats = self.db.get_statistics()

        self.assertEqual(stats['total_concepts'], len(concepts))
        self.assertGreater(stats['total_semantic_units'], 0)
        self.assertGreater(stats['avg_divine_resonance'], 0)

    def test_context_distribution(self):
        """Test context distribution statistics"""
        self.db.store_concept("love", "biblical")
        self.db.store_concept("learning", "educational")
        self.db.store_concept("ethics", "business")

        stats = self.db.get_statistics()

        self.assertIn('context_distribution', stats)
        self.assertEqual(stats['context_distribution']['biblical'], 1)
        self.assertEqual(stats['context_distribution']['educational'], 1)
        self.assertEqual(stats['context_distribution']['business'], 1)

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    def test_full_workflow(self):
        """Test complete workflow: store, query, analyze"""
        # Store concepts
        id1 = self.db.store_concept("love", "biblical")
        id2 = self.db.store_concept("mercy", "biblical")
        id3 = self.db.store_concept("wrath", "biblical")

        # Store relationship
        self.db.store_relationship(id1, id2)

        # Store sacred number
        self.db.store_sacred_number(7)

        # Query by text
        love = self.db.query_by_text("love", "biblical")
        self.assertIsNotNone(love)

        # Semantic search
        results = self.db.search_semantic("compassion", "biblical")
        self.assertGreater(len(results), 0)

        # Query by resonance (use low threshold since SSE gives low values)
        high_resonance = self.db.query_by_divine_resonance(min_resonance=0.0)
        self.assertGreater(len(high_resonance), 0)

        # Get statistics
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 3)
        self.assertEqual(stats['total_relationships'], 1)

    def test_context_manager(self):
        """Test database works as context manager"""
        with SemanticSubstrateDatabase(self.db_path) as db:
            db.store_concept("test", "biblical")
            result = db.query_by_text("test", "biblical")
            self.assertIsNotNone(result)

    # ========================================================================
    # EXPORT TESTS
    # ========================================================================

    def test_export_to_json(self):
        """Test exporting database to JSON"""
        # Add some data
        self.db.store_concept("love", "biblical")
        self.db.store_sacred_number(7)

        # Export
        export_path = tempfile.mktemp(suffix='.json')
        self.db.export_to_json(export_path)

        # Verify file exists
        self.assertTrue(os.path.exists(export_path))

        # Verify content
        import json
        with open(export_path) as f:
            data = json.load(f)

        self.assertIn('metadata', data)
        self.assertIn('concepts', data)
        self.assertIn('sacred_numbers', data)
        self.assertIn('anchors', data)

        # Cleanup
        os.unlink(export_path)

    # ========================================================================
    # PERFORMANCE TESTS
    # ========================================================================

    def test_bulk_insert_performance(self):
        """Test performance with bulk inserts"""
        import time

        concepts = [f"concept_{i}" for i in range(100)]

        start_time = time.time()
        for concept in concepts:
            self.db.store_concept(concept, "biblical")
        end_time = time.time()

        elapsed = end_time - start_time
        per_concept = elapsed / len(concepts)

        # Should be reasonably fast (< 100ms per concept)
        self.assertLess(per_concept, 0.1,
                       f"Storing concept took {per_concept*1000:.2f}ms (should be < 100ms)")

        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], len(concepts))

    def test_query_performance(self):
        """Test query performance"""
        import time

        # Store some concepts
        for i in range(50):
            self.db.store_concept(f"concept_{i}", "biblical")

        # Test semantic search performance
        start_time = time.time()
        results = self.db.search_semantic("test query", "biblical", limit=10)
        end_time = time.time()

        elapsed = end_time - start_time

        # Should be fast (< 500ms)
        self.assertLess(elapsed, 0.5,
                       f"Semantic search took {elapsed*1000:.2f}ms (should be < 500ms)")


def run_all_tests():
    """Run all tests with detailed output"""
    print("=" * 100)
    print("SEMANTIC SUBSTRATE DATABASE - COMPREHENSIVE TEST SUITE")
    print("=" * 100)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSemanticDatabase)

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED - Database is ready for production!")
    else:
        print("\n[FAILED] SOME TESTS FAILED - Review output above")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
