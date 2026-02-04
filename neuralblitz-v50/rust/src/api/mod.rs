use actix_web::{web, App, HttpServer, HttpResponse, Result, middleware::Logger};
use actix_web::middleware::DefaultHeaders;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::core::{
    SourceState, SourceStateInfo, PrimalIntentVector, ArchitectSystemDyad,
    SelfActualizationEngine, IrreducibleSourceField, GoldenDAG, COHERENCE_VALUE
};

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: String,
    pub coherence: f64,
    pub separation: f64,
    pub golden_dag_seed: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntentRequest {
    pub intent: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntentResponse {
    pub intent_id: String,
    pub coherence_verified: bool,
    pub processing_time_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VerificationResponse {
    pub coherent: bool,
    pub coherence_value: f64,
    pub verification_timestamp: DateTime<Utc>,
    pub structural_integrity: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NBCLRequest {
    pub command: String,
    pub context: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NBCLResponse {
    pub interpreted: bool,
    pub action: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AttestationResponse {
    pub attested: bool,
    pub attestation_hash: String,
    pub attestation_timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SymbiosisResponse {
    pub active: bool,
    pub symbiosis_factor: f64,
    pub integrated_entities: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SynthesisResponse {
    pub synthesized: bool,
    pub synthesis_level: String,
    pub coherence_synthesis: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptionResponse {
    pub option: String,
    pub size_mb: i32,
    pub cores: i32,
    pub purpose: String,
    pub default_port: i32,
}

// Shared application state
pub struct AppState {
    pub dyad: ArchitectSystemDyad,
    pub engine: SelfActualizationEngine,
    pub field: IrreducibleSourceField,
    pub golden_dag: GoldenDAG,
}

impl AppState {
    pub fn new() -> Self {
        let dyad = ArchitectSystemDyad::new();
        let engine = SelfActualizationEngine::new(&dyad);
        let field = IrreducibleSourceField::new();
        let golden_dag = GoldenDAG::new("a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0");
        
        Self {
            dyad,
            engine,
            field,
            golden_dag,
        }
    }
}

// API Handlers
pub async fn get_status(data: web::Data<AppState>) -> Result<HttpResponse> {
    let response = StatusResponse {
        status: "operational".to_string(),
        coherence: COHERENCE_VALUE,
        separation: 0.0,
        golden_dag_seed: data.golden_dag.seed.clone(),
        timestamp: Utc::now(),
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn submit_intent(
    req: web::Json<IntentRequest>,
    data: web::Data<AppState>
) -> Result<HttpResponse> {
    let start_time = std::time::Instant::now();
    
    // Process intent through the dyad
    let intent_vector = PrimalIntentVector::new(1.0, 1.0, 1.0);
    let _result = data.dyad.architect_process(intent_vector);
    
    let processing_time = start_time.elapsed().as_millis() as u64;
    
    let response = IntentResponse {
        intent_id: Uuid::new_v4().to_string(),
        coherence_verified: true,
        processing_time_ms: processing_time,
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn verify_coherence(data: web::Data<AppState>) -> Result<HttpResponse> {
    let coherence = data.field.get_coherence();
    let integrity = data.golden_dag.validate();
    
    let response = VerificationResponse {
        coherent: coherence == COHERENCE_VALUE,
        coherence_value: coherence,
        verification_timestamp: Utc::now(),
        structural_integrity: integrity,
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn interpret_nbcl(req: web::Json<NBCLRequest>) -> Result<HttpResponse> {
    let mut parameters = HashMap::new();
    
    // Simple command parsing for demonstration
    if req.command.contains("coherence") {
        parameters.insert("status".to_string(), serde_json::Value::String("coherence_established".to_string()));
    }
    
    let response = NBCLResponse {
        interpreted: true,
        action: "command_processed".to_string(),
        parameters,
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn get_attestation(data: web::Data<AppState>) -> Result<HttpResponse> {
    let response = AttestationResponse {
        attested: true,
        attestation_hash: data.golden_dag.hash.clone(),
        attestation_timestamp: Utc::now(),
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn get_symbiosis(data: web::Data<AppState>) -> Result<HttpResponse> {
    let response = SymbiosisResponse {
        active: true,
        symbiosis_factor: data.dyad.get_symbiotic_return_signal(),
        integrated_entities: 3,
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn get_synthesis(data: web::Data<AppState>) -> Result<HttpResponse> {
    let response = SynthesisResponse {
        synthesized: data.engine.actualized,
        synthesis_level: "complete".to_string(),
        coherence_synthesis: data.engine.get_coherence(),
    };
    
    Ok(HttpResponse::Ok().json(response))
}

pub async fn get_option(path: web::Path<String>) -> Result<HttpResponse> {
    let option = path.into_inner();
    
    let option_data = match option.as_str() {
        "A" => Some(("A", 50, 1, "Minimal deployment", 8080)),
        "B" => Some(("B", 100, 2, "Standard deployment", 8080)),
        "C" => Some(("C", 200, 4, "Enhanced deployment", 8080)),
        "D" => Some(("D", 500, 8, "Production deployment", 8080)),
        "E" => Some(("E", 1000, 16, "Enterprise deployment", 8080)),
        "F" => Some(("F", 2000, 32, "Cosmic deployment", 8080)),
        _ => None,
    };
    
    if let Some((opt, size, cores, purpose, port)) = option_data {
        let response = OptionResponse {
            option: opt.to_string(),
            size_mb: size,
            cores,
            purpose: purpose.to_string(),
            default_port: port,
        };
        Ok(HttpResponse::Ok().json(response))
    } else {
        Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Option not found"
        })))
    }
}

pub async fn run_server() -> std::io::Result<()> {
    run_server_on_port(8081).await
}

pub async fn run_server_on_port(port: u16) -> std::io::Result<()> {
    env_logger::init();
    
    let app_state = web::Data::new(AppState::new());
    
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(
                DefaultHeaders::new()
                    .add(("Server", "NeuralBlitz/v50.0"))
                    .add(("X-GoldenDAG-Seed", "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"))
            )
            .wrap(Logger::default())
            .route("/status", web::get().to(get_status))
            .route("/intent", web::post().to(submit_intent))
            .route("/verify", web::post().to(verify_coherence))
            .route("/nbcl/interpret", web::post().to(interpret_nbcl))
            .route("/attestation", web::get().to(get_attestation))
            .route("/symbiosis", web::get().to(get_symbiosis))
            .route("/synthesis", web::get().to(get_synthesis))
            .route("/options/{option}", web::get().to(get_option))
    })
    .bind(format!("0.0.0.0:{}", port))?
    .run()
    .await
}

pub mod server {
    pub use super::run_server;
}
