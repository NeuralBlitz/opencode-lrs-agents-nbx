// Placeholder for utils module - Cryptographic utilities
pub mod crypto {
    use sha2::{Sha256, Digest};
    
    pub fn hash_data(data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}
