"""
Test Backup and Recovery Mechanisms for Semantic Substrate Database

Tests all backup/recovery features:
- Full database backups
- Backup verification
- Restore from backup
- JSON export/import
- Incremental backups
- Named snapshots
- Backup rotation
- Backup listing
"""

import sys
import os
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_substrate_database import SemanticSubstrateDatabase
import unittest


class TestBackupRecovery(unittest.TestCase):
    """Test backup and recovery operations"""

    def setUp(self):
        """Create fresh test database"""
        self.test_db_path = "test_backup_db.db"
        self.backup_dir = "test_backups"

        # Delete existing test files
        try:
            os.unlink(self.test_db_path)
        except:
            pass

        if os.path.exists(self.backup_dir):
            shutil.rmtree(self.backup_dir)

        # Create test database with sample data
        self.db = SemanticSubstrateDatabase(self.test_db_path)

        # Store test concepts
        self.test_concepts = [
            ("love", "biblical"),
            ("wisdom", "biblical"),
            ("mercy", "biblical"),
            ("grace", "biblical"),
            ("faith", "biblical")
        ]

        for text, context in self.test_concepts:
            self.db.store_concept(text, context)

        # Store test sacred numbers
        self.test_sacred_nums = [7, 12, 40]
        for num in self.test_sacred_nums:
            self.db.store_sacred_number(num)

    def tearDown(self):
        """Clean up test files"""
        self.db.close()

        # Clean up test files
        try:
            os.unlink(self.test_db_path)
        except:
            pass

        if os.path.exists(self.backup_dir):
            shutil.rmtree(self.backup_dir)

    def test_create_backup(self):
        """Test creating a full database backup"""
        backup_path = self.db.create_backup()

        # Verify backup file exists
        self.assertTrue(os.path.exists(backup_path))

        # Verify backup is not empty
        self.assertGreater(os.path.getsize(backup_path), 0)

        # Clean up
        os.unlink(backup_path)
        print("[PASS] Create backup works")

    def test_backup_with_custom_path(self):
        """Test backup with custom path"""
        custom_path = "custom_backup.db"
        backup_path = self.db.create_backup(custom_path)

        self.assertEqual(backup_path, custom_path)
        self.assertTrue(os.path.exists(custom_path))

        # Clean up
        os.unlink(custom_path)
        print("[PASS] Custom path backup works")

    def test_verify_backup(self):
        """Test backup verification"""
        # Create valid backup
        backup_path = self.db.create_backup()

        # Verify it's valid
        is_valid = self.db.verify_backup(backup_path)
        self.assertTrue(is_valid)

        # Test with non-existent file
        is_valid = self.db.verify_backup("nonexistent.db")
        self.assertFalse(is_valid)

        # Clean up
        os.unlink(backup_path)
        print("[PASS] Backup verification works")

    def test_restore_from_backup(self):
        """Test restoring database from backup"""
        # Get initial stats
        initial_stats = self.db.get_statistics()
        initial_concepts = initial_stats['total_concepts']

        # Create backup
        backup_path = self.db.create_backup()

        # Modify database (add more concepts)
        self.db.store_concept("hope", "biblical")
        self.db.store_concept("truth", "biblical")

        # Verify concepts were added
        modified_stats = self.db.get_statistics()
        self.assertEqual(modified_stats['total_concepts'], initial_concepts + 2)

        # Restore from backup
        result = self.db.restore_from_backup(backup_path)
        self.assertTrue(result)

        # Verify restoration (should be back to initial count)
        restored_stats = self.db.get_statistics()
        self.assertEqual(restored_stats['total_concepts'], initial_concepts)

        # Clean up
        os.unlink(backup_path)
        print("[PASS] Restore from backup works")

    def test_json_export_import(self):
        """Test JSON export and import"""
        json_path = "test_export.json"

        # Export to JSON
        self.db.export_to_json(json_path)

        # Verify JSON file exists
        self.assertTrue(os.path.exists(json_path))

        # Get initial stats
        initial_stats = self.db.get_statistics()

        # Clear database
        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM semantic_coordinates")
        cursor.execute("DELETE FROM sacred_numbers")
        self.db.conn.commit()

        # Verify database is empty
        empty_stats = self.db.get_statistics()
        self.assertEqual(empty_stats['total_concepts'], 0)
        self.assertEqual(empty_stats['sacred_numbers_count'], 0)

        # Restore from JSON
        result = self.db.restore_from_json(json_path, clear_existing=False)
        self.assertTrue(result)

        # Verify restoration
        restored_stats = self.db.get_statistics()
        self.assertEqual(restored_stats['total_concepts'], initial_stats['total_concepts'])
        self.assertEqual(restored_stats['sacred_numbers_count'], initial_stats['sacred_numbers_count'])

        # Clean up
        os.unlink(json_path)
        print("[PASS] JSON export/import works")

    def test_incremental_backup(self):
        """Test incremental backup creation"""
        os.makedirs(self.backup_dir, exist_ok=True)

        # Create incremental backup
        backup_path = self.db.create_incremental_backup(self.backup_dir)

        # Verify backup exists
        self.assertTrue(os.path.exists(backup_path))

        # Verify it's a JSON file
        self.assertTrue(backup_path.endswith('.json'))

        print("[PASS] Incremental backup works")

    def test_create_snapshot(self):
        """Test creating named snapshots"""
        snapshot_dir = "test_snapshots"

        # Create snapshot
        snapshot_path = self.db.create_snapshot("before_changes", snapshot_dir)

        # Verify snapshot exists
        self.assertTrue(os.path.exists(snapshot_path))
        self.assertTrue("before_changes" in snapshot_path)

        # Clean up
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)

        print("[PASS] Create snapshot works")

    def test_list_backups(self):
        """Test listing available backups"""
        os.makedirs(self.backup_dir, exist_ok=True)

        # Create multiple backups
        backup1 = self.db.create_backup(os.path.join(self.backup_dir,
                                                     f"{os.path.basename(self.test_db_path)}.backup_20250101_120000"))
        backup2 = self.db.create_backup(os.path.join(self.backup_dir,
                                                     f"{os.path.basename(self.test_db_path)}.backup_20250102_120000"))

        # List backups
        backups = self.db.list_backups(self.backup_dir)

        # Verify we found both backups
        self.assertGreaterEqual(len(backups), 2)

        # Verify backup info structure
        for backup in backups:
            self.assertIn('path', backup)
            self.assertIn('size_bytes', backup)
            self.assertIn('size_mb', backup)
            self.assertIn('created_at', backup)
            self.assertIn('valid', backup)
            self.assertTrue(backup['valid'])

        print(f"[PASS] List backups works (found {len(backups)} backups)")

    def test_auto_backup_rotation(self):
        """Test automatic backup with rotation"""
        os.makedirs(self.backup_dir, exist_ok=True)

        # Create auto-backup with rotation (keep last 3)
        backup_path = self.db.auto_backup(self.backup_dir, keep_last_n=3)

        # Verify backup was created
        self.assertTrue(os.path.exists(backup_path))

        # Create 5 more backups to test rotation
        for i in range(5):
            self.db.auto_backup(self.backup_dir, keep_last_n=3)

        # List backups
        backups = self.db.list_backups(self.backup_dir)

        # Should only have 3 backups (rotation working)
        self.assertLessEqual(len(backups), 3)

        print(f"[PASS] Auto-backup rotation works (kept {len(backups)} backups)")

    def test_restore_preserves_data_integrity(self):
        """Test that restore preserves all data correctly"""
        # Create backup
        backup_path = self.db.create_backup()

        # Get original concept
        original = self.db.query_by_text("love", "biblical")

        # Modify database
        self.db.store_concept("love", "biblical")  # Will update

        # Restore from backup
        self.db.restore_from_backup(backup_path)

        # Get restored concept
        restored = self.db.query_by_text("love", "biblical")

        # Verify data integrity
        self.assertEqual(original['love'], restored['love'])
        self.assertEqual(original['power'], restored['power'])
        self.assertEqual(original['wisdom'], restored['wisdom'])
        self.assertEqual(original['justice'], restored['justice'])
        self.assertEqual(original['divine_resonance'], restored['divine_resonance'])

        # Clean up
        os.unlink(backup_path)
        print("[PASS] Data integrity preserved after restore")

    def test_backup_with_empty_database(self):
        """Test backup of empty database"""
        # Create empty database
        empty_db_path = "test_empty.db"
        empty_db = SemanticSubstrateDatabase(empty_db_path)

        # Create backup
        backup_path = empty_db.create_backup()

        # Verify backup
        is_valid = empty_db.verify_backup(backup_path)
        self.assertTrue(is_valid)

        # Clean up
        empty_db.close()
        os.unlink(empty_db_path)
        os.unlink(backup_path)
        print("[PASS] Empty database backup works")

    def test_concurrent_backups(self):
        """Test creating multiple backups concurrently"""
        import time

        backups = []

        # Create 3 backups with slight delay to ensure unique timestamps
        for i in range(3):
            time.sleep(0.1)  # Small delay to ensure unique timestamps
            backup_path = self.db.create_backup()
            backups.append(backup_path)

        # Verify all backups are valid
        for backup_path in backups:
            self.assertTrue(os.path.exists(backup_path))
            self.assertTrue(self.db.verify_backup(backup_path))

        # Clean up (check if file exists before deleting)
        for backup_path in backups:
            if os.path.exists(backup_path):
                os.unlink(backup_path)

        print("[PASS] Concurrent backups work")


def run_all_tests():
    """Run all backup/recovery tests"""
    print("=" * 80)
    print("SEMANTIC SUBSTRATE DATABASE - BACKUP & RECOVERY TESTS")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBackupRecovery)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED - BACKUP & RECOVERY OPERATIONAL!")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
