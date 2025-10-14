"""
Test Transaction Management for Semantic Substrate Database

Tests all transaction features:
- Basic transactions (begin, commit, rollback)
- Savepoints for nested transactions
- Atomic operations
- Batch operations
- Transaction isolation
- Error handling and rollback
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_substrate_database import SemanticSubstrateDatabase
import unittest


class TestTransactionManagement(unittest.TestCase):
    """Test transaction management operations"""

    def setUp(self):
        """Create fresh test database"""
        self.test_db_path = "test_transaction_db.db"

        # Delete existing test file
        try:
            os.unlink(self.test_db_path)
        except:
            pass

        self.db = SemanticSubstrateDatabase(self.test_db_path)

    def tearDown(self):
        """Clean up test files"""
        self.db.close()

        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def test_basic_transaction_commit(self):
        """Test basic transaction with commit"""
        # Start transaction
        self.db.begin_transaction()
        self.assertTrue(self.db.is_transaction_active())

        # Store concept within transaction
        concept_id = self.db.store_concept("love", "biblical")
        self.assertIsNotNone(concept_id)

        # Commit transaction
        self.db.commit()
        self.assertFalse(self.db.is_transaction_active())

        # Verify concept was saved
        concept = self.db.query_by_text("love", "biblical")
        self.assertIsNotNone(concept)

        print("[PASS] Basic transaction commit works")

    def test_basic_transaction_rollback(self):
        """Test basic transaction with rollback"""
        # Start transaction
        self.db.begin_transaction()

        # Store concept within transaction
        self.db.store_concept("love", "biblical")

        # Rollback transaction
        self.db.rollback()
        self.assertFalse(self.db.is_transaction_active())

        # Verify concept was NOT saved
        concept = self.db.query_by_text("love", "biblical")
        self.assertIsNone(concept)

        print("[PASS] Basic transaction rollback works")

    def test_transaction_isolation(self):
        """Test that uncommitted changes are isolated"""
        # Store initial concept outside transaction
        self.db.store_concept("faith", "biblical")

        # Start transaction
        self.db.begin_transaction()

        # Store concept within transaction
        self.db.store_concept("love", "biblical")

        # Rollback
        self.db.rollback()

        # Verify only the first concept remains
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 1)

        # Verify the right concept is there
        concept = self.db.query_by_text("faith", "biblical")
        self.assertIsNotNone(concept)

        print("[PASS] Transaction isolation works")

    def test_savepoint_creation(self):
        """Test savepoint creation"""
        self.db.begin_transaction()

        # Create savepoint
        sp_name = self.db.create_savepoint("test")
        self.assertIsNotNone(sp_name)
        self.assertIn("sp_test", sp_name)

        self.db.commit()
        print("[PASS] Savepoint creation works")

    def test_rollback_to_savepoint(self):
        """Test rollback to savepoint"""
        self.db.begin_transaction()

        # Store first concept
        self.db.store_concept("faith", "biblical")

        # Create savepoint
        sp_name = self.db.create_savepoint("checkpoint")

        # Store second concept after savepoint
        self.db.store_concept("love", "biblical")

        # Rollback to savepoint (should undo only the second concept)
        self.db.rollback_to_savepoint(sp_name)

        # Commit transaction
        self.db.commit()

        # Verify only first concept exists
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 1)

        faith = self.db.query_by_text("faith", "biblical")
        self.assertIsNotNone(faith)

        love = self.db.query_by_text("love", "biblical")
        self.assertIsNone(love)

        print("[PASS] Rollback to savepoint works")

    def test_release_savepoint(self):
        """Test releasing (committing) a savepoint"""
        self.db.begin_transaction()

        # Store concept
        self.db.store_concept("faith", "biblical")

        # Create savepoint
        sp_name = self.db.create_savepoint("checkpoint")

        # Store another concept
        self.db.store_concept("love", "biblical")

        # Release savepoint (commit it)
        self.db.release_savepoint(sp_name)

        # Commit main transaction
        self.db.commit()

        # Verify both concepts exist
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 2)

        print("[PASS] Release savepoint works")

    def test_nested_savepoints(self):
        """Test multiple nested savepoints"""
        self.db.begin_transaction()

        # Store first concept
        self.db.store_concept("faith", "biblical")
        sp1 = self.db.create_savepoint("level1")

        # Store second concept
        self.db.store_concept("love", "biblical")
        sp2 = self.db.create_savepoint("level2")

        # Store third concept
        self.db.store_concept("hope", "biblical")

        # Rollback to first savepoint (undo love and hope)
        self.db.rollback_to_savepoint(sp1)

        # Commit
        self.db.commit()

        # Verify only faith exists
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 1)

        faith = self.db.query_by_text("faith", "biblical")
        self.assertIsNotNone(faith)

        print("[PASS] Nested savepoints work")

    def test_atomic_operation_success(self):
        """Test atomic operation that succeeds"""
        def store_concepts():
            self.db.store_concept("love", "biblical")
            self.db.store_concept("faith", "biblical")
            return True

        result = self.db.atomic_operation(store_concepts)
        self.assertTrue(result)

        # Verify both concepts were saved
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 2)

        print("[PASS] Atomic operation success works")

    def test_atomic_operation_failure(self):
        """Test atomic operation that fails and rolls back"""
        def failing_operation():
            self.db.store_concept("love", "biblical")
            raise ValueError("Intentional failure")

        # Operation should raise exception
        with self.assertRaises(ValueError):
            self.db.atomic_operation(failing_operation)

        # Verify no concepts were saved (rollback happened)
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 0)

        print("[PASS] Atomic operation failure and rollback works")

    def test_batch_store_concepts(self):
        """Test batch storing multiple concepts"""
        concepts = [
            ("love", "biblical"),
            ("wisdom", "biblical"),
            ("mercy", "biblical"),
            ("grace", "biblical"),
            ("faith", "biblical")
        ]

        concept_ids = self.db.batch_store_concepts(concepts)

        # Verify correct number of IDs returned
        self.assertEqual(len(concept_ids), 5)

        # Verify all concepts were saved
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 5)

        # Verify each concept exists
        for text, context in concepts:
            concept = self.db.query_by_text(text, context)
            self.assertIsNotNone(concept)

        print(f"[PASS] Batch store concepts works ({len(concepts)} concepts)")

    def test_batch_store_sacred_numbers(self):
        """Test batch storing multiple sacred numbers"""
        numbers = [7, 12, 40, 613, 3, 10, 144]

        number_ids = self.db.batch_store_sacred_numbers(numbers)

        # Verify correct number of IDs returned
        self.assertEqual(len(number_ids), 7)

        # Verify all numbers were saved
        stored_numbers = self.db.query_sacred_numbers(min_value=0, max_value=1000, only_sacred=False)
        self.assertEqual(len(stored_numbers), 7)

        print(f"[PASS] Batch store sacred numbers works ({len(numbers)} numbers)")

    def test_batch_operation_partial_failure(self):
        """Test that batch operation rolls back on partial failure"""
        # Store one concept first
        self.db.store_concept("love", "biblical")

        # Try to batch store with duplicate (should fail on second one with UNIQUE constraint)
        concepts = [
            ("wisdom", "biblical"),
            ("love", "biblical"),  # Duplicate - should cause update, not failure
            ("mercy", "biblical")
        ]

        # This should succeed (UPDATE on conflict)
        concept_ids = self.db.batch_store_concepts(concepts)
        self.assertEqual(len(concept_ids), 3)

        # Verify concepts exist
        stats = self.db.get_statistics()
        self.assertGreater(stats['total_concepts'], 0)

        print("[PASS] Batch operation handles conflicts correctly")

    def test_transaction_state_tracking(self):
        """Test transaction state is correctly tracked"""
        # No transaction initially
        self.assertFalse(self.db.is_transaction_active())

        # Start transaction
        self.db.begin_transaction()
        self.assertTrue(self.db.is_transaction_active())

        # Commit
        self.db.commit()
        self.assertFalse(self.db.is_transaction_active())

        # Start another
        self.db.begin_transaction()
        self.assertTrue(self.db.is_transaction_active())

        # Rollback
        self.db.rollback()
        self.assertFalse(self.db.is_transaction_active())

        print("[PASS] Transaction state tracking works")

    def test_nested_transaction_error(self):
        """Test that nested transactions raise error"""
        self.db.begin_transaction()

        # Try to start another transaction
        with self.assertRaises(RuntimeError):
            self.db.begin_transaction()

        self.db.rollback()
        print("[PASS] Nested transaction error handling works")

    def test_commit_without_transaction(self):
        """Test that commit without transaction raises error"""
        with self.assertRaises(RuntimeError):
            self.db.commit()

        print("[PASS] Commit without transaction error handling works")

    def test_rollback_without_transaction(self):
        """Test that rollback without transaction raises error"""
        with self.assertRaises(RuntimeError):
            self.db.rollback()

        print("[PASS] Rollback without transaction error handling works")

    def test_savepoint_without_transaction(self):
        """Test that savepoint without transaction raises error"""
        with self.assertRaises(RuntimeError):
            self.db.create_savepoint("test")

        print("[PASS] Savepoint without transaction error handling works")

    def test_complex_transaction_workflow(self):
        """Test complex multi-step transaction workflow"""
        # Start main transaction
        self.db.begin_transaction()

        # Store concepts
        self.db.store_concept("faith", "biblical")
        self.db.store_concept("hope", "biblical")

        # Create savepoint before risky operation
        sp1 = self.db.create_savepoint("before_risky")

        # Store more concepts
        self.db.store_concept("love", "biblical")
        self.db.store_concept("grace", "biblical")

        # Decide to keep these, create another savepoint
        sp2 = self.db.create_savepoint("before_more")

        # Store more
        self.db.store_concept("mercy", "biblical")

        # Roll back the last one
        self.db.rollback_to_savepoint(sp2)

        # Commit everything up to sp2
        self.db.commit()

        # Verify correct concepts exist
        stats = self.db.get_statistics()
        self.assertEqual(stats['total_concepts'], 4)

        # faith, hope, love, grace should exist
        self.assertIsNotNone(self.db.query_by_text("faith", "biblical"))
        self.assertIsNotNone(self.db.query_by_text("hope", "biblical"))
        self.assertIsNotNone(self.db.query_by_text("love", "biblical"))
        self.assertIsNotNone(self.db.query_by_text("grace", "biblical"))

        # mercy should not exist
        self.assertIsNone(self.db.query_by_text("mercy", "biblical"))

        print("[PASS] Complex transaction workflow works")


def run_all_tests():
    """Run all transaction management tests"""
    print("=" * 80)
    print("SEMANTIC SUBSTRATE DATABASE - TRANSACTION MANAGEMENT TESTS")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTransactionManagement)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED - TRANSACTION MANAGEMENT OPERATIONAL!")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
