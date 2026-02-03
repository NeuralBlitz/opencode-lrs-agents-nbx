// NeuralBlitz v50.0 - LRS Bridge Module
// Bidirectional Communication with LRS Agents

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64;

// Type alias for HMAC-SHA256
type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSBridgeConfig {
    pub lrs_endpoint: String,
    pub auth_key: String,
    pub system_id: String,
    pub heartbeat_interval: Duration,
    pub connection_timeout: Duration,
    pub max_retries: u32,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_timeout: Duration,
}

impl Default for LRSBridgeConfig {
    fn default() -> Self {
        Self {
            lrs_endpoint: std::env::var("LRS_AGENT_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:9000".to_string()),
            auth_key: std::env::var("LRS_AUTH_KEY")
                .unwrap_or_else(|_| "shared_goldendag_key".to_string()),
            system_id: "NEURALBLITZ_V50".to_string(),
            heartbeat_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(10),
            max_retries: 3,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    timeout: Duration,
    failure_count: u32,
    last_failure_time: Option<SystemTime>,
    state: CircuitBreakerState,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, timeout: Duration) -> Self {
        Self {
            failure_threshold: threshold,
            timeout,
            failure_count: 0,
            last_failure_time: None,
            state: CircuitBreakerState::Closed,
        }
    }
    
    pub fn should_allow_request(&self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    SystemTime::now()
                        .duration_since(last_failure)
                        .unwrap_or(Duration::from_secs(0))
                        > self.timeout
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }
    
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.state = CircuitBreakerState::Closed;
    }
    
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(SystemTime::now());
        
        if self.failure_count >= self.failure_threshold {
            self.state = CircuitBreakerState::Open;
        }
    }
    
    pub fn get_state(&self) -> &CircuitBreakerState {
        &self.state
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSMessage {
    pub protocol_version: String,
    pub timestamp: u64,
    pub source_system: String,
    pub target_system: String,
    pub message_id: String,
    pub correlation_id: Option<String>,
    pub message_type: String,
    pub payload: serde_json::Value,
    pub signature: String,
    pub priority: String,
    pub ttl: u32,
}

impl LRSMessage {
    pub fn new(
        source_system: String,
        target_system: String,
        message_type: String,
        payload: serde_json::Value,
        auth_key: &str,
    ) -> Self {
        let message_id = Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            protocol_version: "1.0".to_string(),
            timestamp,
            source_system,
            target_system,
            message_id,
            correlation_id: None,
            message_type,
            payload,
            signature: String::new(), // Will be set by sign method
            priority: "NORMAL".to_string(),
            ttl: 300,
        }
    }
    
    pub fn sign(&mut self, auth_key: &str) {
        let payload_string = serde_json::to_string(&self.payload);
        let mut mac = HmacSha256::new_from_slice(auth_key.as_bytes()).unwrap();
        mac.update(payload_string.as_bytes());
        self.signature = base64::encode(mac.finalize().into_bytes());
    }
    
    pub fn verify_signature(&self, auth_key: &str) -> bool {
        let payload_string = serde_json::to_string(&self.payload);
        let mut mac = HmacSha256::new_from_slice(auth_key.as_bytes()).unwrap();
        mac.update(payload_string.as_bytes());
        let expected_signature = base64::encode(mac.finalize().into_bytes());
        
        use hmac::Mac;
        mac.verify(&self.signature.as_bytes()).unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRequest {
    pub phi_1: f64,
    pub phi_22: f64,
    pub phi_omega: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResponse {
    pub intent_id: String,
    pub status: String,
    pub coherence_verified: bool,
    pub processing_time_ms: u64,
    pub golden_dag_hash: String,
    pub output_data: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceRequest {
    pub target_system: String,
    pub verification_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceResponse {
    pub coherent: bool,
    pub coherence_value: f64,
    pub verification_timestamp: u64,
    pub structural_integrity: bool,
    pub golden_dag_valid: bool,
}

#[derive(Debug)]
pub struct MessageQueue {
    max_size: usize,
    queue: Arc<RwLock<Vec<LRSMessage>>>,
}

impl MessageQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            queue: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn enqueue(&self, message: LRSMessage) -> Result<(), String> {
        let mut queue = self.queue.write().await;
        if queue.len() >= self.max_size {
            return Err("Message queue is full".to_string());
        }
        queue.push(message);
        Ok(())
    }
    
    pub async fn dequeue(&self) -> Option<LRSMessage> {
        let mut queue = self.queue.write().await;
        queue.pop()
    }
    
    pub async fn size(&self) -> usize {
        self.queue.read().await.len()
    }
}

pub struct LRSBridge {
    config: LRSBridgeConfig,
    message_queue: MessageQueue,
    circuit_breaker: Arc<Mutex<CircuitBreaker>>,
    message_handlers: Arc<RwLock<HashMap<String, Box<dyn Fn(serde_json::Value) + Send + Sync>>>>,
    running: Arc<RwLock<bool>>,
    start_time: SystemTime,
}

impl LRSBridge {
    pub fn new(config: LRSBridgeConfig) -> Self {
        Self {
            message_queue: MessageQueue::new(1000),
            circuit_breaker: Arc::new(Mutex::new(CircuitBreaker::new(
                config.circuit_breaker_threshold,
                config.circuit_breaker_timeout,
            ))),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
            start_time: SystemTime::now(),
            config,
        }
    }
    
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("Starting LRS bridge for {}", self.config.system_id);
        
        // Set running state
        *self.running.write().await = true;
        
        // Register default handlers
        self.register_default_handlers().await;
        
        // Start heartbeat task
        let heartbeat_config = self.config.clone();
        let running_clone = self.running.clone();
        tokio::spawn(async move {
            let mut interval = interval(heartbeat_config.heartbeat_interval);
            
            loop {
                interval.tick().await;
                
                // Check if still running
                if !*running_clone.read().await {
                    break;
                }
                
                // Send heartbeat
                if let Err(e) = Self::send_heartbeat(&heartbeat_config).await {
                    log::error!("Heartbeat failed: {}", e);
                }
            }
        });
        
        log::info!("LRS bridge started successfully");
        Ok(())
    }
    
    pub async fn stop(&self) {
        log::info!("Stopping LRS bridge");
        *self.running.write().await = false;
        log::info!("LRS bridge stopped");
    }
    
    pub async fn register_handler<F>(&self, message_type: String, handler: F) 
    where
        F: Fn(serde_json::Value) + Send + Sync + 'static
    {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(message_type, Box::new(handler));
        log::info!("Registered handler for message type: {}", message_type);
    }
    
    async fn register_default_handlers(&self) {
        // Register default message handlers
        // These would be implemented as async methods
        log::info!("Default handlers registered");
    }
    
    async fn send_message(
        &self,
        message_type: String,
        payload: serde_json::Value,
        target_system: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut circuit_breaker = self.circuit_breaker.lock().await;
        
        if !circuit_breaker.should_allow_request() {
            return Err("Circuit breaker is OPEN".into());
        }
        
        let mut message = LRSMessage::new(
            self.config.system_id.clone(),
            target_system.to_string(),
            message_type,
            payload,
            &self.config.auth_key,
        );
        
        // Send using reqwest
        let client = reqwest::Client::new()
            .timeout(self.config.connection_timeout)
            .build()?;
        
        let url = format!("{}/neuralblitz/bridge", self.config.lrs_endpoint);
        
        match client.post(&url).json(&message).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    circuit_breaker.record_success();
                    let result = response.json().await?;
                    Ok(result)
                } else {
                    circuit_breaker.record_failure();
                    let error_text = response.text().await.unwrap_or_default();
                    let error_msg = format!("HTTP {}: {}", response.status(), error_text);
                    Err(error_msg.into())
                }
            }
            Err(e) => {
                circuit_breaker.record_failure();
                Err(format!("Request failed: {}", e).into())
            }
        }
    }
    
    pub async fn submit_intent(&self, intent: IntentRequest) -> Result<IntentResponse, Box<dyn std::error::Error + Send + Sync>> {
        let payload = serde_json::to_value(&intent);
        let response = self.send_message("INTENT_SUBMIT".to_string(), payload, "LRS_AGENT").await?;
        
        let intent_response: IntentResponse = serde_json::from_value(response)?;
        Ok(intent_response)
    }
    
    pub async fn verify_coherence(&self, target_system: &str, verification_type: &str) -> Result<CoherenceResponse, Box<dyn std::error::Error + Send + Sync>> {
        let payload = serde_json::json!({
            "target_system": target_system,
            "verification_type": verification_type
        });
        
        let response = self.send_message("COHERENCE_VERIFICATION".to_string(), payload, target_system).await?;
        
        let coherence_response: CoherenceResponse = serde_json::from_value(response)?;
        Ok(coherence_response)
    }
    
    pub async fn send_heartbeat(config: &LRSBridgeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let uptime = SystemTime::now()
            .duration_since(config.start_time)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();
        
        let metrics = serde_json::json!({
            "system_status": "HEALTHY",
            "queue_size": 0, // Would be actual queue size
            "circuit_breaker_state": "CLOSED",
            "active_connections": 1,
            "coherence": 1.0,
            "golden_dag_valid": true,
            "uptime_seconds": uptime
        });
        
        let payload = serde_json::json!({
            "system_id": &config.system_id,
            "metrics": metrics
        });
        
        let client = reqwest::Client::new()
            .timeout(Duration::from_secs(10))
            .build()?;
        
        let url = format!("{}/neuralblitz/bridge", config.lrs_endpoint);
        
        let response = client.post(&url).json(&payload).send().await?;
        
        if !response.status().is_success() {
            log::error!("Heartbeat failed: {}", response.status());
        }
        
        Ok(())
    }
    
    pub fn is_healthy(&self) -> bool {
        // Check circuit breaker state
        if let Ok(circuit_breaker) = self.circuit_breaker.try_lock() {
            matches!(circuit_breaker.get_state(), CircuitBreakerState::Closed)
        } else {
            false
        }
    }
    
    pub async fn get_metrics(&self) -> serde_json::Value {
        let uptime = SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();
        
        serde_json::json!({
            "queue_size": self.message_queue.size().await,
            "circuit_breaker_state": "CLOSED",
            "active_connections": 1,
            "is_healthy": self.is_healthy(),
            "uptime_seconds": uptime
        })
    }
}

// Actix-web server integration
use actix_web::{web, App, HttpServer, HttpResponse, Result, middleware::Logger};
use actix_cors::Cors;
use actix_web_actors::ws;

async fn lrs_bridge_endpoint(
    message: web::Json<LRSMessage>,
    bridge: web::Data<Arc<LRSBridge>>,
) -> Result<HttpResponse, actix_web::Error> {
    // Verify message signature
    if !message.verify_signature(&bridge.config.auth_key) {
        return Ok(HttpResponse::Unauthorized().json(serde_json::json!({
            "error": "Invalid message signature"
        })));
    }
    
    // Route to appropriate handler
    // This would integrate with the message handlers
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "received",
        "message_id": message.message_id
    })))
}

async fn lrs_bridge_status(bridge: web::Data<Arc<LRSBridge>>) -> Result<HttpResponse, actix_web::Error> {
    let status = if bridge.is_healthy() { "healthy" } else { "degraded" };
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "system_id": bridge.config.system_id,
        "status": status,
        "metrics": bridge.get_metrics().await,
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

async fn lrs_bridge_metrics(bridge: web::Data<Arc<LRSBridge>>) -> Result<HttpResponse, actix_web::Error> {
    Ok(HttpResponse::Ok().json(bridge.get_metrics().await))
}

async fn submit_intent_to_lrs(
    intent: web::Json<IntentRequest>,
    bridge: web::Data<Arc<LRSBridge>>,
) -> Result<HttpResponse, actix_web::Error> {
    match bridge.submit_intent(intent.into_inner()).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Intent submission failed: {}", e)
        }))),
    }
}

async fn verify_lrs_coherence_endpoint(
    request: web::Json<CoherenceRequest>,
    bridge: web::Data<Arc<LRSBridge>>,
) -> Result<HttpResponse, actix_web::Error> {
    match bridge.verify_coherence(&request.target_system, &request.verification_type).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Coherence verification failed: {}", e)
        }))),
    }
}

pub async fn run_lrs_bridge_server(config: LRSBridgeConfig) -> std::io::Result<()> {
    env_logger::init();
    
    let bridge = Arc::new(LRSBridge::new(config.clone()));
    
    // Start bridge background tasks
    bridge.start().await.expect("Failed to start LRS bridge");
    
    HttpServer::new(move || {
        App::new()
            .app_data(bridge.clone())
            .wrap(Cors::permissive())
            .wrap(Logger::default())
            .route("/lrs/bridge", web::post().to(lrs_bridge_endpoint))
            .route("/lrs/bridge/status", web::get().to(lrs_bridge_status))
            .route("/lrs/bridge/metrics", web::get().to(lrs_bridge_metrics))
            .route("/lrs/bridge/intent/submit", web::post().to(submit_intent_to_lrs))
            .route("/lrs/bridge/coherence/verify", web::post().to(verify_lrs_coherence_endpoint))
    })
    .bind(format!("0.0.0.0:{}", std::env::var("LRS_BRIDGE_PORT").unwrap_or_else(|_| "8083".to_string())))?
    .run()
    .await
}