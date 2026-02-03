"""
Tests for NeuralBlitz Audit Logging System
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from neuralblitz.security.audit import AuditLogger, AuditEntry, create_audit_logger


class TestAuditLogger:
    """Test suite for audit logging functionality."""

    def setup_method(self):
        """Setup temporary audit log for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_audit.log")
        self.audit = AuditLogger(self.log_file)

    def teardown_method(self):
        """Cleanup temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audit_logger_initialization(self):
        """Test that audit logger initializes correctly."""
        assert self.audit.log_file == Path(self.log_file)
        assert self.audit.max_entries == 100000
        assert self.audit.previous_hash == "0" * 64
        assert os.path.exists(self.log_file)

    def test_log_operation_basic(self):
        """Test basic operation logging."""
        result = {"coherence": 0.85, "consciousness_level": "AWARE"}

        entry = self.audit.log_operation(
            operation="test_operation", intent_hash="abc123", result=result
        )

        assert entry.operation == "test_operation"
        assert entry.intent_hash == "abc123"
        assert entry.result_summary["coherence"] == 0.85
        assert entry.timestamp is not None
        assert entry.entry_hash is not None
        assert entry.previous_hash == "0" * 64

    def test_hash_chaining(self):
        """Test that hash chaining works correctly."""
        # Log first entry
        entry1 = self.audit.log_operation(
            operation="op1", intent_hash="hash1", result={}
        )

        # Log second entry
        entry2 = self.audit.log_operation(
            operation="op2", intent_hash="hash2", result={}
        )

        # Verify chaining
        assert entry1.previous_hash == "0" * 64  # Genesis hash
        assert entry2.previous_hash == entry1.entry_hash
        assert entry1.entry_hash != entry2.entry_hash

    def test_result_summarization(self):
        """Test result summarization for log size management."""
        detailed_result = {
            "coherence": 0.8574392106,
            "consciousness_level": "TRANSCENDENT",
            "confidence": 0.93451234,
            "processing_time_ms": 15.789234,
            "patterns_stored": 5,
            "large_data": "x" * 1000,  # Should be excluded
        }

        entry = self.audit.log_operation(
            operation="process_intent", intent_hash="test_hash", result=detailed_result
        )

        summary = entry.result_summary
        assert summary["coherence"] == 0.8574  # Rounded
        assert summary["consciousness_level"] == "TRANSCENDENT"
        assert summary["confidence"] == 0.9345  # Rounded
        assert summary["processing_time_ms"] == 15.7892  # Rounded
        assert summary["patterns_stored"] == 5
        assert "large_data" not in summary  # Large data excluded

    def test_log_persistence(self):
        """Test that logs persist across instances."""
        # Log entry in first instance
        self.audit.log_operation(
            operation="persistent_op",
            intent_hash="persist123",
            result={"coherence": 0.9},
        )

        # Create new instance
        audit2 = AuditLogger(self.log_file)

        # Verify entry was loaded
        entries = audit2.get_entries(limit=10)
        assert len(entries) == 1
        assert entries[0].operation == "persistent_op"
        assert entries[0].intent_hash == "persist123"

    def test_get_entries_with_filters(self):
        """Test filtering entries."""
        # Add test entries
        self.audit.log_operation("op1", "hash1", {"coherence": 0.8})
        self.audit.log_operation("op2", "hash2", {"coherence": 0.9})
        self.audit.log_operation("op1", "hash3", {"coherence": 0.85})

        # Test operation filter
        op1_entries = self.audit.get_entries(operation="op1")
        assert len(op1_entries) == 2
        assert all(e.operation == "op1" for e in op1_entries)

        # Test limit
        limited = self.audit.get_entries(limit=2)
        assert len(limited) == 2

    def test_integrity_verification(self):
        """Test log integrity verification."""
        # Add some entries
        for i in range(3):
            self.audit.log_operation(f"op{i}", f"hash{i}", {"coherence": 0.8 + i * 0.1})

        # Verify integrity
        assert self.audit.verify_integrity() == True

        # Tamper with log file
        with open(self.log_file, "w") as f:
            f.write('{"tampered": "data"}\n')

        # Verify integrity fails
        assert self.audit.verify_integrity() == False

    def test_get_statistics(self):
        """Test statistics generation."""
        # Add varied entries
        self.audit.log_operation("process_intent", "hash1", {"coherence": 0.8})
        self.audit.log_operation("process_intent", "hash2", {"coherence": 0.9})
        self.audit.log_operation("system_error", "hash3", {"coherence": 0.7})

        stats = self.audit.get_statistics()

        assert stats["total_entries"] == 3
        assert "process_intent" in stats["operations"]
        assert "system_error" in stats["operations"]
        assert stats["operations"]["process_intent"] == 2
        assert stats["operations"]["system_error"] == 1
        assert "coherence_stats" in stats
        assert stats["coherence_stats"]["mean"] == 0.8  # (0.8 + 0.9 + 0.7) / 3

    def test_export_to_json(self):
        """Test log export functionality."""
        # Add test entry
        self.audit.log_operation("export_test", "hash123", {"coherence": 0.85})

        # Export
        export_file = os.path.join(self.temp_dir, "export.json")
        self.audit.export_to_json(export_file)

        # Verify export
        assert os.path.exists(export_file)

        with open(export_file, "r") as f:
            exported_data = json.load(f)

        assert len(exported_data) == 1
        assert exported_data[0]["operation"] == "export_test"
        assert exported_data[0]["intent_hash"] == "hash123"


class TestAuditEntry:
    """Test suite for individual audit entries."""

    def test_entry_creation(self):
        """Test audit entry creation."""
        timestamp = datetime.utcnow().isoformat()
        entry = AuditEntry(
            operation="test_op",
            intent_hash="test_hash",
            result_summary={"coherence": 0.8},
            timestamp=timestamp,
            previous_hash="0" * 64,
        )

        assert entry.operation == "test_op"
        assert entry.intent_hash == "test_hash"
        assert entry.result_summary["coherence"] == 0.8
        assert entry.timestamp == timestamp
        assert entry.previous_hash == "0" * 64
        assert entry.entry_hash is not None

    def test_entry_hash_calculation(self):
        """Test hash calculation consistency."""
        entry_data = {
            "operation": "test_op",
            "intent_hash": "test_hash",
            "result_summary": {"coherence": 0.8},
            "timestamp": "2023-01-01T00:00:00",
            "previous_hash": "0" * 64,
        }

        entry1 = AuditEntry(**entry_data)
        entry2 = AuditEntry(**entry_data)

        # Same data should produce same hash
        assert entry1.entry_hash == entry2.entry_hash

    def test_entry_serialization(self):
        """Test entry to/from dict conversion."""
        original = AuditEntry(
            operation="serialize_test",
            intent_hash="hash123",
            result_summary={"coherence": 0.85},
        )

        # Convert to dict and back
        entry_dict = original.to_dict()
        restored = AuditEntry.from_dict(entry_dict)

        assert restored.operation == original.operation
        assert restored.intent_hash == original.intent_hash
        assert restored.result_summary == original.result_summary
        assert restored.timestamp == original.timestamp
        assert restored.previous_hash == original.previous_hash


class TestCreateAuditLogger:
    """Test audit logger factory function."""

    def test_create_audit_logger(self):
        """Test factory function creates logger with date-based filename."""
        temp_dir = tempfile.mkdtemp()

        try:
            logger = create_audit_logger(temp_dir)

            # Should create file with today's date
            today = datetime.utcnow().strftime("%Y%m%d")
            expected_filename = f"audit_{today}.log"
            expected_path = os.path.join(temp_dir, expected_filename)

            assert os.path.exists(expected_path)
            assert logger.log_file.name == expected_filename

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestAuditLoggerIntegration:
    """Integration tests with actual file system operations."""

    def setup_method(self):
        """Setup for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "integration_test.log")
        self.audit = AuditLogger(self.log_file)

    def teardown_method(self):
        """Cleanup."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_logging(self):
        """Test concurrent logging operations."""
        import threading
        import time

        results = []
        errors = []

        def log_worker(worker_id):
            try:
                for i in range(10):
                    entry = self.audit.log_operation(
                        operation=f"worker_{worker_id}_op_{i}",
                        intent_hash=f"hash_{worker_id}_{i}",
                        result={"coherence": 0.8 + i * 0.01},
                    )
                    results.append(entry)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for worker_id in range(3):
            t = threading.Thread(target=log_worker, args=(worker_id,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 30  # 3 workers * 10 operations

        # Verify all entries were logged
        entries = self.audit.get_entries(limit=100)
        assert len(entries) == 30

    def test_log_rotation(self):
        """Test log rotation when exceeding max entries."""
        # Create audit logger with small max for testing
        small_audit = AuditLogger(self.log_file, max_entries=5)

        # Add more entries than max
        for i in range(7):
            small_audit.log_operation(f"rotation_test_{i}", f"hash_{i}", {})

        # Should have rotated (file might be archived or truncated)
        assert os.path.exists(self.log_file)

        # New entries should still work
        entry = small_audit.log_operation("after_rotation", "new_hash", {})
        assert entry is not None

    def test_file_corruption_recovery(self):
        """Test behavior with corrupted log file."""
        # Write some valid entries
        for i in range(3):
            self.audit.log_operation(f"valid_{i}", f"hash_{i}", {})

        # Corrupt the file by appending invalid JSON
        with open(self.log_file, "a") as f:
            f.write('{"invalid": json}\n')

        # Create new instance - should handle corruption gracefully
        audit2 = AuditLogger(self.log_file)

        # Should still be able to log new entries
        entry = audit2.log_operation("after_corruption", "recovery_hash", {})
        assert entry is not None

        # Integrity check should fail due to corruption
        assert audit2.verify_integrity() == False
