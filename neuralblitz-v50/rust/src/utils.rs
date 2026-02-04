use std::collections::HashMap;
use chrono::{DateTime, Utc};

pub fn generate_trace_id() -> String {
    format!("T-v50.0-RUST-{:032x}", rand::random::<u64>())
}

pub fn generate_codex_id() -> String {
    format!("C-VOL0-V50_RUST-{:024x}", rand::random::<u64>())
}

pub fn current_timestamp() -> DateTime<Utc> {
    Utc::now()
}

pub fn create_response_metadata() -> HashMap<String, serde_json::Value> {
    let mut metadata = HashMap::new();
    metadata.insert("trace_id".to_string(), serde_json::Value::String(generate_trace_id()));
    metadata.insert("codex_id".to_string(), serde_json::Value::String(generate_codex_id()));
    metadata.insert("timestamp".to_string(), serde_json::Value::String(current_timestamp().to_rfc3339()));
    metadata.insert("version".to_string(), serde_json::Value::String("v50.0.0".to_string()));
    metadata
}

pub fn validate_golden_dag_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id_generation() {
        let trace_id = generate_trace_id();
        assert!(trace_id.starts_with("T-v50.0-RUST-"));
        assert_eq!(trace_id.len(), 46); // "T-v50.0-RUST-" + 32 hex chars
    }

    #[test]
    fn test_codex_id_generation() {
        let codex_id = generate_codex_id();
        assert!(codex_id.starts_with("C-VOL0-V50_RUST-"));
        assert_eq!(codex_id.len(), 40); // "C-VOL0-V50_RUST-" + 24 hex chars
    }

    #[test]
    fn test_validate_golden_dag_hash() {
        let valid_hash = "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0";
        let invalid_hash = "invalid_hash";
        
        assert!(validate_golden_dag_hash(valid_hash));
        assert!(!validate_golden_dag_hash(invalid_hash));
    }

    #[test]
    fn test_create_response_metadata() {
        let metadata = create_response_metadata();
        
        assert!(metadata.contains_key("trace_id"));
        assert!(metadata.contains_key("codex_id"));
        assert!(metadata.contains_key("timestamp"));
        assert!(metadata.contains_key("version"));
    }
}