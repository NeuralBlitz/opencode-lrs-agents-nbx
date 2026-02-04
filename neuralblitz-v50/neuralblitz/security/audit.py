"""
NeuralBlitz V50 - Audit Logging System
Tamper-evident logging with blockchain-style integrity.
"""

import hashlib
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import threading


class AuditEntry:
    """Single audit log entry with hash chaining."""

    def __init__(
        self,
        operation: str,
        intent_hash: str,
        result_summary: Dict[str, Any],
        timestamp: Optional[str] = None,
        previous_hash: str = "0" * 64,
        entry_hash: Optional[str] = None,
    ):
        self.operation = operation
        self.intent_hash = intent_hash
        self.result_summary = result_summary
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.previous_hash = previous_hash
        self.entry_hash = entry_hash or self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of entry data."""
        data = {
            "operation": self.operation,
            "intent_hash": self.intent_hash,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "intent_hash": self.intent_hash,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        return cls(
            operation=data["operation"],
            intent_hash=data["intent_hash"],
            result_summary=data["result_summary"],
            timestamp=data["timestamp"],
            previous_hash=data["previous_hash"],
            entry_hash=data["hash"],
        )


class AuditLogger:
    """
    Tamper-evident audit logger with blockchain-style integrity.

    Each log entry contains a hash of the previous entry, creating
    a chain that makes tampering detectable.
    """

    def __init__(self, log_file: str, max_entries: int = 100000):
        self.log_file = Path(log_file)
        self.max_entries = max_entries
        self.previous_hash = "0" * 64
        self._lock = threading.Lock()

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()

        # Load last hash if log has content
        if self.log_file.exists() and self.log_file.stat().st_size > 0:
            self._load_last_hash()

    def _load_last_hash(self):
        """Load the hash of the last entry."""
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    self.previous_hash = last_entry["hash"]
        except Exception as e:
            # If corrupted, start fresh chain
            self.previous_hash = "0" * 64

    def log_operation(
        self, operation: str, intent_hash: str, result: Dict[str, Any]
    ) -> AuditEntry:
        """
        Log an operation with tamper-evident chaining.

        Args:
            operation: Type of operation (e.g., 'process_intent')
            intent_hash: Hash of the intent
            result: Operation result summary

        Returns:
            AuditEntry that was logged
        """
        with self._lock:
            # Create entry
            entry = AuditEntry(
                operation=operation,
                intent_hash=intent_hash,
                result_summary=self._summarize_result(result),
                previous_hash=self.previous_hash,
            )

            # Append to log
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

            # Update previous hash for next entry
            self.previous_hash = entry.entry_hash

            # Rotate log if too large
            self._rotate_if_needed()

            return entry

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of result (to keep log size manageable)."""
        return {
            "consciousness_level": result.get("consciousness_level", "UNKNOWN"),
            "coherence": round(result.get("coherence", 0.5), 4),
            "confidence": round(result.get("confidence", 0.0), 4),
            "processing_time_ms": round(result.get("processing_time_ms", 0), 4),
            "patterns_stored": result.get("patterns_stored", 0),
        }

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max entries."""
        try:
            with open(self.log_file, "r") as f:
                line_count = sum(1 for _ in f)

            if line_count > self.max_entries:
                # Rotate: archive current log
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                archive_name = f"{self.log_file.stem}_{timestamp}{self.log_file.suffix}"
                archive_path = self.log_file.parent / archive_name

                # Rename current to archive
                self.log_file.rename(archive_path)

                # Reset hash chain for new log
                self.previous_hash = "0" * 64
        except Exception:
            pass  # Rotation is best-effort

    def verify_integrity(self) -> bool:
        """
        Verify the integrity of the entire audit log chain.

        Returns:
            True if chain is valid, False if tampering detected
        """
        with self._lock:
            try:
                with open(self.log_file, "r") as f:
                    lines = f.readlines()

                if not lines:
                    return True  # Empty log is valid

                expected_previous_hash = "0" * 64

                for i, line in enumerate(lines):
                    entry_data = json.loads(line)

                    # Verify previous hash matches
                    if entry_data["previous_hash"] != expected_previous_hash:
                        print(f"❌ Integrity check FAILED at entry {i}")
                        print(f"   Expected previous_hash: {expected_previous_hash}")
                        print(f"   Found: {entry_data['previous_hash']}")
                        return False

                    # Verify entry hash is correct
                    entry = AuditEntry.from_dict(entry_data)
                    if entry.entry_hash != entry_data["hash"]:
                        print(f"❌ Hash mismatch at entry {i}")
                        return False

                    expected_previous_hash = entry_data["hash"]

                print(f"✅ Integrity check PASSED for {len(lines)} entries")
                return True

            except Exception as e:
                print(f"❌ Error during integrity check: {e}")
                return False

    def get_entries(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Query audit log entries with filters.

        Args:
            start_time: ISO format timestamp
            end_time: ISO format timestamp
            operation: Filter by operation type
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        entries = []

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    if len(entries) >= limit:
                        break

                    entry_data = json.loads(line)

                    # Apply filters
                    if start_time and entry_data["timestamp"] < start_time:
                        continue
                    if end_time and entry_data["timestamp"] > end_time:
                        continue
                    if operation and entry_data["operation"] != operation:
                        continue

                    entries.append(AuditEntry.from_dict(entry_data))
        except Exception:
            pass

        return entries

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the audit log."""
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()

            if not lines:
                return {"total_entries": 0}

            operations = {}
            coherences = []

            for line in lines:
                entry_data = json.loads(line)
                op = entry_data["operation"]
                operations[op] = operations.get(op, 0) + 1

                coherence = entry_data["result_summary"].get("coherence")
                if coherence is not None:
                    coherences.append(coherence)

            import statistics

            return {
                "total_entries": len(lines),
                "operations": operations,
                "coherence_stats": {
                    "mean": statistics.mean(coherences) if coherences else 0,
                    "min": min(coherences) if coherences else 0,
                    "max": max(coherences) if coherences else 0,
                }
                if coherences
                else None,
                "log_file_size_bytes": self.log_file.stat().st_size,
            }
        except Exception as e:
            return {"error": str(e)}

    def export_to_json(self, output_file: str) -> None:
        """Export entire audit log to JSON file."""
        entries = self.get_entries(limit=1000000)

        with open(output_file, "w") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)


def create_audit_logger(log_dir: str = "./logs") -> AuditLogger:
    """
    Factory function to create an audit logger.

    Args:
        log_dir: Directory for log files

    Returns:
        Configured AuditLogger
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    log_file = Path(log_dir) / f"audit_{timestamp}.log"
    return AuditLogger(str(log_file))


# Export
__all__ = ["AuditLogger", "AuditEntry", "create_audit_logger"]
