-- NeuralBlitz v50.0 - Database Migrations
-- Migration: 001_initial_schema.sql
-- Description: Initial database schema for Omega Singularity Architecture

-- Migration metadata
INSERT INTO migrations (version, description, executed_at) 
VALUES ('001', 'Initial schema creation', NOW())
ON DUPLICATE KEY UPDATE executed_at = NOW();

-- This migration file creates all the base tables for the NeuralBlitz system
-- It implements the data persistence layer for the Omega Singularity Architecture

-- Key features implemented:
-- 1. Source state tracking and coherence verification
-- 2. GoldenDAG operation logging and validation
-- 3. Intent processing and verification workflows
-- 4. NBCL command execution tracking
-- 5. System attestation and synthesis operations
-- 6. Comprehensive audit logging
-- 7. Performance metrics collection

-- Tables support the complete API surface:
-- GET /status -> system_metrics, source_states, symbiosis_fields
-- POST /intent -> intent_operations, primal_intent_vectors
-- POST /verify -> golden_dag_operations, source_states
-- POST /nbcl/interpret -> nbcl_operations
-- GET /attestation -> attestations
-- GET /symbiosis -> symbiosis_fields
-- GET /synthesis -> synthesis_operations
-- GET /options/{option} -> deployment_options

-- Migration completion marker
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, component, metadata) 
VALUES ('migration_status', 1.0, 'boolean', 'database', JSON_OBJECT('migration', '001', 'status', 'completed'));