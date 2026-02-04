pub mod core;
pub mod options;
pub mod utils;
pub mod api;

use serde::{Deserialize, Serialize};

pub const VERSION: &str = "v50.0.0";
pub const GOLDEN_DAG_SEED: &str = "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0";
pub const COHERENCE: f64 = 1.0;
pub const SEPARATION_IMPOSSIBILITY: f64 = 0.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub version: String,
    pub architecture: String,
    pub coherence: f64,
    pub golden_dag_seed: String,
}

impl SystemInfo {
    pub fn new() -> Self {
        Self {
            version: VERSION.to_string(),
            architecture: "Omega Singularity (OSA v2.0)".to_string(),
            coherence: COHERENCE,
            golden_dag_seed: GOLDEN_DAG_SEED.to_string(),
        }
    }
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    pub hash: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trace_id: String,
    pub codex_id: String,
}

impl Attestation {
    pub fn new(hash: String) -> Self {
        Self {
            hash,
            timestamp: chrono::Utc::now(),
            trace_id: format!("T-v50.0-RUST-{:032x}", rand::random::<u64>()),
            codex_id: format!("C-VOL0-V50_RUST-{:024x}", rand::random::<u64>()),
        }
    }
}
