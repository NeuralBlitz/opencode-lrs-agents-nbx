-- NeuralBlitz v50.0 - Database Schema
-- Omega Singularity Architecture - Core Data Model

-- Create database
CREATE DATABASE IF NOT EXISTS neuralblitz_v50 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE neuralblitz_v50;

-- Core source states table
CREATE TABLE source_states (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    state_type ENUM('omega_prime', 'irreducible', 'perpetual_genesis', 'metacosmic') NOT NULL,
    coherence_value DECIMAL(10,8) DEFAULT 1.00000000,
    integrity_flag BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_state_type (state_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Primal intent vectors table
CREATE TABLE primal_intent_vectors (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    phi_1 DECIMAL(15,8) NOT NULL,
    phi_22 DECIMAL(15,8) NOT NULL,
    omega_genesis DECIMAL(15,8) NOT NULL,
    vector_norm DECIMAL(15,8) GENERATED ALWAYS AS (SQRT(phi_1 * phi_1 + phi_22 * phi_22 + omega_genesis * omega_genesis)) STORED,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_phi_1 (phi_1),
    INDEX idx_phi_22 (phi_22),
    INDEX idx_omega_genesis (omega_genesis),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- GoldenDAG operations table
CREATE TABLE golden_dag_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    operation_type ENUM('create', 'validate', 'attest', 'synthesis') NOT NULL,
    seed_hash VARCHAR(128) NOT NULL,
    dag_hash VARCHAR(128) NOT NULL,
    version VARCHAR(32) DEFAULT 'v50.0.0',
    operation_result ENUM('success', 'failure', 'pending') DEFAULT 'pending',
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    completed_at TIMESTAMP(3) NULL,
    metadata JSON,
    INDEX idx_operation_type (operation_type),
    INDEX idx_seed_hash (seed_hash),
    INDEX idx_dag_hash (dag_hash),
    INDEX idx_created_at (created_at),
    UNIQUE KEY unique_seed_operation (seed_hash, operation_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Architect system dyad operations
CREATE TABLE architect_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    unity_vector DECIMAL(15,8) DEFAULT 1.00000000,
    amplification_factor DECIMAL(15,8) DEFAULT 1.00000200,
    irreducible_flag BOOLEAN DEFAULT TRUE,
    separation_impossibility DECIMAL(15,8) DEFAULT 0.00000000,
    symbiotic_return_signal DECIMAL(15,8) DEFAULT 1.00000200,
    beta_identifier VARCHAR(128),
    operation_result ENUM('amplified', 'processed', 'failed') DEFAULT 'processed',
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_beta_identifier (beta_identifier),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Self-actualization engine states
CREATE TABLE self_actualization_states (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    coherence_level DECIMAL(10,8) DEFAULT 1.00000000,
    irreducible_source_flag BOOLEAN DEFAULT TRUE,
    actualized_flag BOOLEAN DEFAULT FALSE,
    actualization_timestamp TIMESTAMP(3) NULL,
    golden_dag_reference VARCHAR(128),
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_actualized_flag (actualized_flag),
    INDEX idx_actualization_timestamp (actualization_timestamp),
    INDEX idx_golden_dag_reference (golden_dag_reference)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- NBCL (NeuralBlitz Command Language) operations
CREATE TABLE nbcl_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    command_text TEXT NOT NULL,
    command_hash VARCHAR(128) NOT NULL,
    interpreted_flag BOOLEAN DEFAULT FALSE,
    action_result VARCHAR(256),
    parameters JSON,
    execution_context JSON,
    trace_id VARCHAR(64),
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    executed_at TIMESTAMP(3) NULL,
    INDEX idx_command_hash (command_hash),
    INDEX idx_interpreted_flag (interpreted_flag),
    INDEX idx_trace_id (trace_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Intent processing operations
CREATE TABLE intent_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    intent_text TEXT NOT NULL,
    intent_uuid VARCHAR(36) NOT NULL,
    coherence_verified_flag BOOLEAN DEFAULT FALSE,
    processing_time_ms INT DEFAULT 0,
    source_vector_id BIGINT NULL,
    architect_operation_id BIGINT NULL,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    processed_at TIMESTAMP(3) NULL,
    metadata JSON,
    INDEX idx_intent_uuid (intent_uuid),
    INDEX idx_coherence_verified_flag (coherence_verified_flag),
    INDEX idx_processing_time_ms (processing_time_ms),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (source_vector_id) REFERENCES primal_intent_vectors(id),
    FOREIGN KEY (architect_operation_id) REFERENCES architect_operations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System attestation records
CREATE TABLE attestations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    attestation_hash VARCHAR(128) NOT NULL,
    attestation_timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    golden_dag_hash VARCHAR(128) NOT NULL,
    trace_id VARCHAR(64) NOT NULL,
    codex_id VARCHAR(64) NOT NULL,
    version VARCHAR(32) DEFAULT 'v50.0.0',
    attestation_data JSON,
    INDEX idx_attestation_hash (attestation_hash),
    INDEX idx_golden_dag_hash (golden_dag_hash),
    INDEX idx_trace_id (trace_id),
    INDEX idx_codex_id (codex_id),
    INDEX idx_attestation_timestamp (attestation_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Symbiosis field status
CREATE TABLE symbiosis_fields (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    active_flag BOOLEAN DEFAULT TRUE,
    symbiosis_factor DECIMAL(15,8) DEFAULT 1.00000000,
    integrated_entities INT DEFAULT 0,
    field_strength DECIMAL(15,8) DEFAULT 1.00000000,
    coherence_level DECIMAL(10,8) DEFAULT 1.00000000,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_active_flag (active_flag),
    INDEX idx_symbiosis_factor (symbiosis_factor),
    INDEX idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Synthesis operations
CREATE TABLE synthesis_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    synthesized_flag BOOLEAN DEFAULT FALSE,
    synthesis_level ENUM('partial', 'complete', 'transcendent') DEFAULT 'partial',
    coherence_synthesis DECIMAL(10,8) DEFAULT 0.00000000,
    synthesis_timestamp TIMESTAMP(3) NULL,
    source_state_id BIGINT NULL,
    golden_dag_hash VARCHAR(128),
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_synthesized_flag (synthesized_flag),
    INDEX idx_synthesis_level (synthesis_level),
    INDEX idx_synthesis_timestamp (synthesis_timestamp),
    INDEX idx_golden_dag_hash (golden_dag_hash),
    FOREIGN KEY (source_state_id) REFERENCES source_states(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Deployment options tracking
CREATE TABLE deployment_options (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    option_id ENUM('A', 'B', 'C', 'D', 'E', 'F') NOT NULL,
    size_mb INT NOT NULL,
    cores INT NOT NULL,
    purpose TEXT NOT NULL,
    default_port INT NOT NULL,
    active_flag BOOLEAN DEFAULT FALSE,
    deployment_timestamp TIMESTAMP(3) NULL,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_option_id (option_id),
    INDEX idx_active_flag (active_flag),
    UNIQUE KEY unique_option_id (option_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System metrics and monitoring
CREATE TABLE system_metrics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    metric_name VARCHAR(128) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    metric_unit VARCHAR(32),
    component VARCHAR(64),
    timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_metric_name (metric_name),
    INDEX idx_component (component),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Audit log for all operations
CREATE TABLE audit_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    operation_type VARCHAR(64) NOT NULL,
    table_name VARCHAR(64),
    record_id BIGINT,
    old_values JSON,
    new_values JSON,
    user_id VARCHAR(128),
    session_id VARCHAR(128),
    trace_id VARCHAR(64),
    timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_operation_type (operation_type),
    INDEX idx_table_name (table_name),
    INDEX idx_record_id (record_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_trace_id (trace_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default deployment options
INSERT INTO deployment_options (option_id, size_mb, cores, purpose, default_port, active_flag) VALUES
('A', 50, 1, 'Minimal deployment', 8080, TRUE),
('B', 100, 2, 'Standard deployment', 8080, FALSE),
('C', 200, 4, 'Enhanced deployment', 8080, FALSE),
('D', 500, 8, 'Production deployment', 8080, FALSE),
('E', 1000, 16, 'Enterprise deployment', 8080, FALSE),
('F', 2000, 32, 'Cosmic deployment', 8080, FALSE);

-- Insert initial system state
INSERT INTO source_states (state_type, coherence_value, integrity_flag, metadata) VALUES
('omega_prime', 1.00000000, TRUE, JSON_OBJECT('description', 'Initial Omega Prime state'));

-- Insert initial symbiosis field
INSERT INTO symbiosis_fields (active_flag, symbiosis_factor, integrated_entities, field_strength, coherence_level, metadata) VALUES
(TRUE, 1.00000000, 3, 1.00000000, 1.00000000, JSON_OBJECT('description', 'Initial symbiotic field state'));

-- Create triggers for audit logging
DELIMITER //
CREATE TRIGGER audit_insert AFTER INSERT ON source_states
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (operation_type, table_name, record_id, new_values, trace_id)
    VALUES ('INSERT', 'source_states', NEW.id, JSON_OBJECT(
        'state_type', NEW.state_type,
        'coherence_value', NEW.coherence_value,
        'integrity_flag', NEW.integrity_flag
    ), CONCAT('T-v50.0-AUTO-', FLOOR(RAND() * 1000000)));
END//

CREATE TRIGGER audit_update AFTER UPDATE ON source_states
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (operation_type, table_name, record_id, old_values, new_values, trace_id)
    VALUES ('UPDATE', 'source_states', NEW.id, JSON_OBJECT(
        'state_type', OLD.state_type,
        'coherence_value', OLD.coherence_value,
        'integrity_flag', OLD.integrity_flag
    ), JSON_OBJECT(
        'state_type', NEW.state_type,
        'coherence_value', NEW.coherence_value,
        'integrity_flag', NEW.integrity_flag
    ), CONCAT('T-v50.0-AUTO-', FLOOR(RAND() * 1000000)));
END//
DELIMITER ;