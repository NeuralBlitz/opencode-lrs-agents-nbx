"""
NeuralBlitz v50.0 - SQL Database Integration
Production-ready database schema for Omega Singularity Architecture
"""

import sqlite3
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import hashlib
import os

logger = logging.getLogger(__name__)


@dataclass
class IntentRecord:
    """Intent record for database storage."""

    intent_id: str
    phi_1: float
    phi_22: float
    phi_omega: float
    phi_cognitive: float
    phi_emotional: float
    phi_creative: float
    phi_intuitive: float
    metadata: str  # JSON string
    timestamp: datetime
    processing_result: str  # JSON string
    coherence_score: float


@dataclass
class GoldenDAGRecord:
    """GoldenDAG record for database storage."""

    dag_id: str
    dag_hash: str
    dag_structure: str  # JSON string
    is_valid: bool
    timestamp: datetime
    coherence_level: float


@dataclass
class AttestationRecord:
    """Attestation record for database storage."""

    attestation_id: str
    intent_id: str
    dag_hash: str
    signature: str
    timestamp: datetime
    is_verified: bool
    verification_result: str  # JSON string


class NeuralBlitzDatabase:
    """Production-ready SQL database for NeuralBlitz v50.0."""

    def __init__(self, db_path: str = "neuralblitz_v50.db"):
        self.db_path = db_path
        self.connection = None
        self.initialize_database()
        logger.info(f"Initialized NeuralBlitz database: {db_path}")

    def initialize_database(self):
        """Initialize database with all required tables."""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dictionary-like access

            # Enable foreign key constraints
            self.connection.execute("PRAGMA foreign_keys = ON")

            # Create all tables
            self._create_intents_table()
            self._create_golden_dags_table()
            self._create_attestations_table()
            self._create_cognitive_states_table()
            self._create_ml_models_table()
            self._create_analytics_table()
            self._create_system_logs_table()

            # Create indexes for performance
            self._create_indexes()

            self.connection.commit()
            logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _create_intents_table(self):
        """Create intents table."""
        sql = """
        CREATE TABLE IF NOT EXISTS intents (
            intent_id TEXT PRIMARY KEY,
            phi_1 REAL NOT NULL,
            phi_22 REAL NOT NULL,
            phi_omega REAL NOT NULL,
            phi_cognitive REAL NOT NULL,
            phi_emotional REAL NOT NULL,
            phi_creative REAL NOT NULL,
            phi_intuitive REAL NOT NULL,
            metadata TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            processing_result TEXT NOT NULL,
            coherence_score REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.connection.execute(sql)

    def _create_golden_dags_table(self):
        """Create GoldenDAGs table."""
        sql = """
        CREATE TABLE IF NOT EXISTS golden_dags (
            dag_id TEXT PRIMARY KEY,
            dag_hash TEXT UNIQUE NOT NULL,
            dag_structure TEXT NOT NULL,
            is_valid BOOLEAN NOT NULL,
            timestamp DATETIME NOT NULL,
            coherence_level REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.connection.execute(sql)

    def _create_attestations_table(self):
        """Create attestations table."""
        sql = """
        CREATE TABLE IF NOT EXISTS attestations (
            attestation_id TEXT PRIMARY KEY,
            intent_id TEXT NOT NULL,
            dag_hash TEXT NOT NULL,
            signature TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            is_verified BOOLEAN NOT NULL,
            verification_result TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (intent_id) REFERENCES intents (intent_id),
            FOREIGN KEY (dag_hash) REFERENCES golden_dags (dag_hash)
        );
        """
        self.connection.execute(sql)

    def _create_cognitive_states_table(self):
        """Create cognitive states table for AI engine."""
        sql = """
        CREATE TABLE IF NOT EXISTS cognitive_states (
            state_id TEXT PRIMARY KEY,
            consciousness_level TEXT NOT NULL,
            global_coherence REAL NOT NULL,
            self_awareness REAL NOT NULL,
            creativity_index REAL NOT NULL,
            wisdom_factor REAL NOT NULL,
            emotional_state TEXT NOT NULL,
            processing_metrics TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.connection.execute(sql)

    def _create_ml_models_table(self):
        """Create ML models table."""
        sql = """
        CREATE TABLE IF NOT EXISTS ml_models (
            model_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            accuracy REAL NOT NULL,
            parameters TEXT NOT NULL,
            training_data_size INTEGER NOT NULL,
            last_trained DATETIME,
            is_active BOOLEAN NOT NULL,
            performance_metrics TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.connection.execute(sql)

    def _create_analytics_table(self):
        """Create analytics table for real-time metrics."""
        sql = """
        CREATE TABLE IF NOT EXISTS analytics (
            metric_id TEXT PRIMARY KEY,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            tags TEXT,
            timestamp DATETIME NOT NULL,
            anomaly_detected BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.connection.execute(sql)

    def _create_system_logs_table(self):
        """Create system logs table."""
        sql = """
        CREATE TABLE IF NOT EXISTS system_logs (
            log_id TEXT PRIMARY KEY,
            log_level TEXT NOT NULL,
            component TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            timestamp DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.connection.execute(sql)

    def _create_indexes(self):
        """Create performance indexes."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_intents_timestamp ON intents(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_intents_coherence ON intents(coherence_score);",
            "CREATE INDEX IF NOT EXISTS idx_golden_dags_hash ON golden_dags(dag_hash);",
            "CREATE INDEX IF NOT EXISTS idx_attestations_intent ON attestations(intent_id);",
            "CREATE INDEX IF NOT EXISTS idx_cognitive_states_timestamp ON cognitive_states(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_analytics_name ON analytics(metric_name);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);",
        ]

        for index_sql in indexes:
            self.connection.execute(index_sql)

        logger.info("Database indexes created successfully")

    # Intent Management Methods
    def insert_intent(self, intent: IntentRecord) -> bool:
        """Insert a new intent record."""
        try:
            sql = """
            INSERT INTO intents 
            (intent_id, phi_1, phi_22, phi_omega, phi_cognitive, phi_emotional, 
             phi_creative, phi_intuitive, metadata, timestamp, processing_result, coherence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """

            self.connection.execute(
                sql,
                (
                    intent.intent_id,
                    intent.phi_1,
                    intent.phi_22,
                    intent.phi_omega,
                    intent.phi_cognitive,
                    intent.phi_emotional,
                    intent.phi_creative,
                    intent.phi_intuitive,
                    intent.metadata,
                    intent.timestamp,
                    intent.processing_result,
                    intent.coherence_score,
                ),
            )

            self.connection.commit()
            logger.debug(f"Inserted intent: {intent.intent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert intent: {e}")
            return False

    def get_intent(self, intent_id: str) -> Optional[Dict[str, Any]]:
        """Get an intent by ID."""
        try:
            sql = "SELECT * FROM intents WHERE intent_id = ?;"
            cursor = self.connection.execute(sql, (intent_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

        except Exception as e:
            logger.error(f"Failed to get intent {intent_id}: {e}")
            return None

    def get_recent_intents(
        self, limit: int = 100, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent intents within specified time window."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            sql = """
            SELECT * FROM intents 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC 
            LIMIT ?;
            """

            cursor = self.connection.execute(sql, (cutoff_time, limit))
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get recent intents: {e}")
            return []

    # GoldenDAG Management Methods
    def insert_golden_dag(self, dag: GoldenDAGRecord) -> bool:
        """Insert a new GoldenDAG record."""
        try:
            sql = """
            INSERT OR REPLACE INTO golden_dags 
            (dag_id, dag_hash, dag_structure, is_valid, timestamp, coherence_level)
            VALUES (?, ?, ?, ?, ?, ?);
            """

            self.connection.execute(
                sql,
                (
                    dag.dag_id,
                    dag.dag_hash,
                    dag.dag_structure,
                    dag.is_valid,
                    dag.timestamp,
                    dag.coherence_level,
                ),
            )

            self.connection.commit()
            logger.debug(f"Inserted GoldenDAG: {dag.dag_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert GoldenDAG: {e}")
            return False

    def get_golden_dag_by_hash(self, dag_hash: str) -> Optional[Dict[str, Any]]:
        """Get a GoldenDAG by its hash."""
        try:
            sql = "SELECT * FROM golden_dags WHERE dag_hash = ?;"
            cursor = self.connection.execute(sql, (dag_hash,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

        except Exception as e:
            logger.error(f"Failed to get GoldenDAG {dag_hash}: {e}")
            return None

    def get_valid_dags(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get valid GoldenDAGs."""
        try:
            sql = """
            SELECT * FROM golden_dags 
            WHERE is_valid = TRUE 
            ORDER BY coherence_level DESC, timestamp DESC 
            LIMIT ?;
            """

            cursor = self.connection.execute(sql, (limit,))
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get valid DAGs: {e}")
            return []

    # Attestation Management Methods
    def insert_attestation(self, attestation: AttestationRecord) -> bool:
        """Insert a new attestation record."""
        try:
            sql = """
            INSERT INTO attestations 
            (attestation_id, intent_id, dag_hash, signature, timestamp, 
             is_verified, verification_result)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """

            self.connection.execute(
                sql,
                (
                    attestation.attestation_id,
                    attestation.intent_id,
                    attestation.dag_hash,
                    attestation.signature,
                    attestation.timestamp,
                    attestation.is_verified,
                    attestation.verification_result,
                ),
            )

            self.connection.commit()
            logger.debug(f"Inserted attestation: {attestation.attestation_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert attestation: {e}")
            return False

    def get_attestations_by_intent(self, intent_id: str) -> List[Dict[str, Any]]:
        """Get attestations for a specific intent."""
        try:
            sql = """
            SELECT * FROM attestations 
            WHERE intent_id = ? 
            ORDER BY timestamp DESC;
            """

            cursor = self.connection.execute(sql, (intent_id,))
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get attestations for intent {intent_id}: {e}")
            return []

    # Analytics Methods
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Record an analytics metric."""
        try:
            metric_id = str(uuid.uuid4())
            tags_json = json.dumps(tags) if tags else "{}"

            sql = """
            INSERT INTO analytics 
            (metric_id, metric_name, metric_value, tags, timestamp)
            VALUES (?, ?, ?, ?, ?);
            """

            self.connection.execute(
                sql,
                (metric_id, metric_name, metric_value, tags_json, datetime.utcnow()),
            )

            self.connection.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False

    def get_metrics_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            sql = """
            SELECT 
                COUNT(*) as count,
                AVG(metric_value) as mean,
                MIN(metric_value) as min,
                MAX(metric_value) as max,
                ROUND(AVG(metric_value) * 1.0, 3) as avg_value
            FROM analytics 
            WHERE metric_name = ? AND timestamp >= ?;
            """

            cursor = self.connection.execute(sql, (metric_name, cutoff_time))
            row = cursor.fetchone()

            if row and row["count"] > 0:
                return dict(row)
            return {}

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}

    # System Health Methods
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}

            # Count records in each table
            tables = [
                "intents",
                "golden_dags",
                "attestations",
                "cognitive_states",
                "ml_models",
                "analytics",
                "system_logs",
            ]

            for table in tables:
                sql = f"SELECT COUNT(*) as count FROM {table};"
                cursor = self.connection.execute(sql)
                row = cursor.fetchone()
                stats[f"{table}_count"] = row["count"] if row else 0

            # Get database size
            if os.path.exists(self.db_path):
                stats["database_size_bytes"] = os.path.getsize(self.db_path)

            # Get recent activity
            recent_cutoff = datetime.utcnow() - timedelta(hours=1)

            # Recent intents
            sql = "SELECT COUNT(*) as count FROM intents WHERE timestamp >= ?;"
            cursor = self.connection.execute(sql, (recent_cutoff,))
            row = cursor.fetchone()
            stats["recent_intents_hour"] = row["count"] if row else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def log_system_event(
        self, level: str, component: str, message: str, details: Optional[str] = None
    ) -> bool:
        """Log a system event."""
        try:
            log_id = str(uuid.uuid4())
            details_json = json.dumps(details) if details else "{}"

            sql = """
            INSERT INTO system_logs 
            (log_id, log_level, component, message, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?);
            """

            self.connection.execute(
                sql,
                (log_id, level, component, message, details_json, datetime.utcnow()),
            )

            self.connection.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return False

    def get_recent_logs(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get recent system logs."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            sql = "SELECT * FROM system_logs WHERE timestamp >= ?"
            params = [cutoff_time]

            if level:
                sql += " AND log_level = ?"
                params.append(level)

            if component:
                sql += " AND component = ?"
                params.append(component)

            sql += " ORDER BY timestamp DESC LIMIT ?;"
            params.append(limit)

            cursor = self.connection.execute(sql, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get recent logs: {e}")
            return []

    # Cleanup Methods
    def cleanup_old_data(self, days: int = 30) -> bool:
        """Clean up old data to maintain database size."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)

            # Clean up old analytics data (keep only recent metrics)
            sql = "DELETE FROM analytics WHERE timestamp < ?;"
            cursor = self.connection.execute(sql, (cutoff_time,))
            analytics_deleted = cursor.rowcount

            # Clean up old system logs
            sql = "DELETE FROM system_logs WHERE timestamp < ?;"
            cursor = self.connection.execute(sql, (cutoff_time,))
            logs_deleted = cursor.rowcount

            self.connection.commit()

            logger.info(
                f"Cleaned up {analytics_deleted} analytics records and {logs_deleted} log records"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Global database instance
_database = None


def get_database() -> NeuralBlitzDatabase:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = NeuralBlitzDatabase()
        logger.info("Initialized Global Database Instance")
    return _database


def initialize_database(db_path: Optional[str] = None) -> NeuralBlitzDatabase:
    """Initialize database with custom path."""
    global _database
    if db_path:
        _database = NeuralBlitzDatabase(db_path)
    else:
        _database = NeuralBlitzDatabase()
    logger.info("Database initialized with custom configuration")
    return _database
