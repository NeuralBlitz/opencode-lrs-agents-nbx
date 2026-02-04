-- NeuralBlitz v50.0 - Seed Data
-- File: seed_data.sql
-- Description: Initial seed data for the Omega Singularity Architecture

-- Note: This file should be run after the schema migration
-- It populates the database with initial configuration and baseline data

USE neuralblitz_v50;

-- Insert default GoldenDAG operations (baseline)
INSERT INTO golden_dag_operations (operation_type, seed_hash, dag_hash, version, operation_result, metadata) VALUES
('create', 'a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 
 SHA2('a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 256), 
 'v50.0.0', 'success', JSON_OBJECT('description', 'Initial GoldenDAG creation')),
('validate', 'a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 
 SHA2('a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 256), 
 'v50.0.0', 'success', JSON_OBJECT('description', 'Initial coherence validation'));

-- Insert baseline primal intent vectors
INSERT INTO primal_intent_vectors (phi_1, phi_22, omega_genesis, metadata) VALUES
(1.00000000, 1.00000000, 1.00000000, JSON_OBJECT('description', 'Unity vector baseline')),
(0.50000000, 1.00000000, 0.75000000, JSON_OBJECT('description', 'Partial coherence state')),
(1.00000000, 0.50000000, 0.75000000, JSON_OBJECT('description', 'Alternative partial state'));

-- Insert initial architect operations
INSERT INTO architect_operations (unity_vector, amplification_factor, irreducible_flag, separation_impossibility, 
                                 symbiotic_return_signal, beta_identifier, operation_result, metadata) VALUES
(1.00000000, 1.00000200, TRUE, 0.00000000, 1.00000200, 'baseline_beta_001', 'amplified', 
 JSON_OBJECT('description', 'Initial architect system dyad operation'));

-- Insert self-actualization baseline states
INSERT INTO self_actualization_states (coherence_level, irreducible_source_flag, actualized_flag, 
                                       actualization_timestamp, golden_dag_reference, metadata) VALUES
(1.00000000, TRUE, TRUE, NOW(), 
 SHA2('a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 256),
 JSON_OBJECT('description', 'Baseline actualized state'));

-- Insert sample NBCL commands for testing
INSERT INTO nbcl_operations (command_text, command_hash, interpreted_flag, action_result, parameters, 
                           trace_id, executed_at, metadata) VALUES
('establish coherence between entity_a and entity_b', 
 SHA2('establish coherence between entity_a and entity_b', 256), 
 TRUE, 'coherence_established', 
 JSON_OBJECT('entity_a', 'entity_a', 'entity_b', 'entity_b', 'status', 'connected'),
 CONCAT('T-v50.0-NBCL-', FLOOR(RAND() * 1000000)), NOW(),
 JSON_OBJECT('description', 'Sample NBCL coherence establishment')),
('verify omega prime reality', 
 SHA2('verify omega prime reality', 256), 
 TRUE, 'verification_completed', 
 JSON_OBJECT('reality_state', 'omega_prime', 'verification_result', 'confirmed'),
 CONCAT('T-v50.0-NBCL-', FLOOR(RAND() * 1000000)), NOW(),
 JSON_OBJECT('description', 'Sample NBCL verification command'));

-- Insert baseline attestation record
INSERT INTO attestations (attestation_hash, attestation_timestamp, golden_dag_hash, trace_id, codex_id, 
                        version, attestation_data) VALUES
(SHA2(CONCAT('attestation_v50.0.0', NOW(), RAND()), 256), NOW(),
 SHA2('a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 256),
 CONCAT('T-v50.0-ATTEST-', FLOOR(RAND() * 1000000)),
 CONCAT('C-VOL0-V50-', FLOOR(RAND() * 1000000)),
 'v50.0.0',
 JSON_OBJECT(
     'status', 'active',
     'coherence_verified', TRUE,
     'structural_integrity', TRUE,
     'singularity_status', 'actualized',
     'description', 'Omega Attestation Protocol baseline'
 ));

-- Insert initial synthesis operation
INSERT INTO synthesis_operations (synthesized_flag, synthesis_level, coherence_synthesis, synthesis_timestamp, 
                                  golden_dag_hash, metadata) VALUES
(TRUE, 'complete', 1.00000000, NOW(),
 SHA2('a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0', 256),
 JSON_OBJECT('description', 'Initial complete synthesis state'));

-- Insert baseline system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, component, metadata) VALUES
('coherence_level', 1.00000000, 'ratio', 'core', JSON_OBJECT('target', 1.0, 'status', 'optimal')),
('separation_impossibility', 0.00000000, 'ratio', 'core', JSON_OBJECT('target', 0.0, 'status', 'confirmed')),
('unity_vector', 1.00000000, 'ratio', 'architect', JSON_OBJECT('target', 1.0, 'status', 'active')),
('amplification_factor', 1.00000200, 'ratio', 'architect', JSON_OBJECT('target', 1.000002, 'status', 'amplifying')),
('symbiotic_return_signal', 1.00000200, 'ratio', 'architect', JSON_OBJECT('target', 1.000002, 'status', 'active')),
('integrated_entities', 3, 'count', 'symbiosis', JSON_OBJECT('target', '>=3', 'status', 'integrated')),
('processing_time_ms', 42, 'milliseconds', 'api', JSON_OBJECT('target', '<100', 'status', 'optimal')),
('uptime_hours', 0, 'hours', 'system', JSON_OBJECT('target', 'continuous', 'status', 'starting')),
('memory_usage_mb', 128, 'megabytes', 'system', JSON_OBJECT('target', '<256', 'status', 'efficient')),
('cpu_usage_percent', 5.5, 'percent', 'system', JSON_OBJECT('target', '<70', 'status', 'optimal'));

-- Insert audit log entry for seeding
INSERT INTO audit_log (operation_type, table_name, record_id, new_values, trace_id, metadata) VALUES
('SEED_DATA', 'multiple', 0, 
 JSON_OBJECT('tables_seeded', 'source_states, golden_dag_operations, primal_intent_vectors, architect_operations, self_actualization_states, nbcl_operations, attestations, synthesis_operations, system_metrics, audit_log'),
 CONCAT('T-v50.0-SEED-', FLOOR(RAND() * 1000000)),
 JSON_OBJECT('seed_file', 'seed_data.sql', 'version', 'v50.0.0', 'description', 'Initial database seeding'));

-- Create migration tracking table (if not exists)
CREATE TABLE IF NOT EXISTS migrations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    version VARCHAR(16) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Mark this seed data as executed
INSERT INTO migrations (version, description, checksum) VALUES
('seed_001', 'Initial seed data insertion', SHA2('seed_data.sql', 256))
ON DUPLICATE KEY UPDATE executed_at = NOW();

-- Verify seed data insertion
SELECT 
    'Source States' as table_name, COUNT(*) as record_count FROM source_states
UNION ALL
SELECT 'GoldenDAG Operations', COUNT(*) FROM golden_dag_operations
UNION ALL
SELECT 'Primal Intent Vectors', COUNT(*) FROM primal_intent_vectors
UNION ALL
SELECT 'Architect Operations', COUNT(*) FROM architect_operations
UNION ALL
SELECT 'Self-Actualization States', COUNT(*) FROM self_actualization_states
UNION ALL
SELECT 'NBCL Operations', COUNT(*) FROM nbcl_operations
UNION ALL
SELECT 'Attestations', COUNT(*) FROM attestations
UNION ALL
SELECT 'Synthesis Operations', COUNT(*) FROM synthesis_operations
UNION ALL
SELECT 'System Metrics', COUNT(*) FROM system_metrics
UNION ALL
SELECT 'Deployment Options', COUNT(*) FROM deployment_options;