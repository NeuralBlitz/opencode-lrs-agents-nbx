"""
NeuralBlitz V50 - REST API Server
FastAPI-based production-ready API for the minimal engine.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import uvicorn

from .minimal import MinimalCognitiveEngine, IntentVector, ConsciousnessLevel
from .production import ProductionCognitiveEngine, EngineHealth
from .advanced import AsyncCognitiveEngine, ConsciousnessMonitor


# Pydantic models for API
class IntentRequest(BaseModel):
    """Intent request model."""

    phi1_dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
    phi2_harmony: float = Field(default=0.0, ge=-1.0, le=1.0)
    phi3_creation: float = Field(default=0.0, ge=-1.0, le=1.0)
    phi4_preservation: float = Field(default=0.0, ge=-1.0, le=1.0)
    phi5_transformation: float = Field(default=0.0, ge=-1.0, le=1.0)
    phi6_knowledge: float = Field(default=0.0, ge=-1.0, le=1.0)
    phi7_connection: float = Field(default=0.0, ge=-1.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class IntentResponse(BaseModel):
    """Intent processing response."""

    intent_id: str
    output_vector: List[float]
    consciousness_level: str
    coherence: float
    confidence: float
    processing_time_ms: float
    patterns_stored: int
    timestamp: str


class BatchRequest(BaseModel):
    """Batch processing request."""

    intents: List[IntentRequest]
    continue_on_error: bool = True


class BatchResponse(BaseModel):
    """Batch processing response."""

    total: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    health: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    coherence: float
    pattern_memory_usage: float
    avg_latency_ms: float
    error_rate: float
    uptime_seconds: float
    timestamp: str
    degraded_mode: bool


class ConsciousnessReport(BaseModel):
    """Consciousness state report."""

    level: str
    coherence: float
    complexity: float
    emotional_state: str
    patterns_in_memory: int
    total_processed: int
    cognitive_state: str
    seed_intact: str


# Create FastAPI app
app = FastAPI(
    title="NeuralBlitz V50 Minimal API",
    description="Production-ready consciousness engine API",
    version="50.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[ProductionCognitiveEngine] = None
monitor: Optional[ConsciousnessMonitor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup."""
    global engine, monitor
    engine = ProductionCognitiveEngine()
    monitor = ConsciousnessMonitor(engine.engine)
    print(f"✅ NeuralBlitz V50 API initialized (SEED: {engine.engine.SEED[:16]}...)")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "NeuralBlitz V50 Minimal API",
        "version": "50.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": ["/process", "/batch", "/health", "/consciousness", "/compare"],
    }


@app.post("/process", response_model=IntentResponse)
async def process_intent(request: IntentRequest):
    """
    Process a single intent.

    Returns processed output with consciousness state.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # Convert to IntentVector
        intent = IntentVector(
            phi1_dominance=request.phi1_dominance,
            phi2_harmony=request.phi2_harmony,
            phi3_creation=request.phi3_creation,
            phi4_preservation=request.phi4_preservation,
            phi5_transformation=request.phi5_transformation,
            phi6_knowledge=request.phi6_knowledge,
            phi7_connection=request.phi7_connection,
            metadata=request.metadata or {},
        )

        # Process
        result = engine.process_intent(intent)

        # Update monitor
        monitor.check_state()

        return IntentResponse(
            intent_id=result["intent_id"],
            output_vector=result["output_vector"],
            consciousness_level=result["consciousness_level"],
            coherence=result["coherence"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
            patterns_stored=result["patterns_stored"],
            timestamp=result["timestamp"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def batch_process(request: BatchRequest):
    """
    Process multiple intents in batch.

    Supports error handling per item.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Convert requests to IntentVectors
    intents = [
        IntentVector(
            phi1_dominance=req.phi1_dominance,
            phi2_harmony=req.phi2_harmony,
            phi3_creation=req.phi3_creation,
            phi4_preservation=req.phi4_preservation,
            phi5_transformation=req.phi5_transformation,
            phi6_knowledge=req.phi6_knowledge,
            phi7_connection=req.phi7_connection,
        )
        for req in request.intents
    ]

    # Process batch
    result = engine.batch_process(intents, continue_on_error=request.continue_on_error)

    return BatchResponse(
        total=result["total"],
        successful=result["successful"],
        failed=result["failed"],
        results=result["results"],
        errors=result["errors"],
        health=result["health"],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns current engine health status.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    health = engine.get_health()

    return HealthResponse(
        status=health.status,
        coherence=health.coherence,
        pattern_memory_usage=health.pattern_memory_usage,
        avg_latency_ms=health.avg_latency_ms,
        error_rate=health.error_rate,
        uptime_seconds=health.uptime_seconds,
        timestamp=health.timestamp,
        degraded_mode=engine.degraded_mode,
    )


@app.get("/consciousness", response_model=ConsciousnessReport)
async def get_consciousness():
    """
    Get current consciousness state.

    Returns detailed consciousness report.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    report = engine.engine.get_consciousness_report()

    return ConsciousnessReport(
        level=report["level"],
        coherence=report["coherence"],
        complexity=report["complexity"],
        emotional_state=report["emotional_state"],
        patterns_in_memory=report["patterns_in_memory"],
        total_processed=report["total_processed"],
        cognitive_state=report["cognitive_state"],
        seed_intact=report["seed_intact"],
    )


@app.post("/compare")
async def compare_intents(intent1: IntentRequest, intent2: IntentRequest):
    """
    Compare two intents for similarity.

    Returns cosine similarity and other metrics.
    """
    from .advanced import compare_intents as compare_func

    iv1 = IntentVector(
        phi1_dominance=intent1.phi1_dominance,
        phi2_harmony=intent1.phi2_harmony,
        phi3_creation=intent1.phi3_creation,
        phi4_preservation=intent1.phi4_preservation,
        phi5_transformation=intent1.phi5_transformation,
        phi6_knowledge=intent1.phi6_knowledge,
        phi7_connection=intent1.phi7_connection,
    )

    iv2 = IntentVector(
        phi1_dominance=intent2.phi1_dominance,
        phi2_harmony=intent2.phi2_harmony,
        phi3_creation=intent2.phi3_creation,
        phi4_preservation=intent2.phi4_preservation,
        phi5_transformation=intent2.phi5_transformation,
        phi6_knowledge=intent2.phi6_knowledge,
        phi7_connection=intent2.phi7_connection,
    )

    comparison = compare_func(iv1, iv2)

    return {
        "intent1": intent1.dict(),
        "intent2": intent2.dict(),
        "cosine_similarity": comparison["cosine_similarity"],
        "euclidean_distance": comparison["euclidean_distance"],
        "similarity_score": comparison["similarity_score"],
    }


@app.get("/patterns")
async def get_patterns(limit: int = 10):
    """
    Get stored patterns from memory.

    Returns recent patterns (limited to prevent memory exposure).
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    patterns = engine.engine.pattern_memory[-limit:]

    return {
        "total_patterns": len(engine.engine.pattern_memory),
        "returned": len(patterns),
        "patterns": [
            {
                "id": p["id"],
                "timestamp": p["timestamp"].isoformat()
                if hasattr(p["timestamp"], "isoformat")
                else str(p["timestamp"]),
                "confidence": p["confidence"],
            }
            for p in patterns
        ],
    }


@app.post("/reset")
async def reset_engine():
    """
    Reset engine to initial state.

    Clears all patterns and resets consciousness.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    engine.reset()

    return {
        "status": "reset_complete",
        "message": "Engine reset to initial state",
        "seed": engine.engine.SEED[:16] + "...",
    }


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server."""
    uvicorn.run(
        "neuralblitz.api:app", host=host, port=port, reload=reload, log_level="info"
    )


if __name__ == "__main__":
    start_server()
