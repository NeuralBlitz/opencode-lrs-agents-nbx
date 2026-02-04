"""
Example 5: Production Integration
Demonstrates enterprise patterns for production deployment.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from neuralblitz import MinimalCognitiveEngine, IntentVector
from neuralblitz.advanced import AsyncCognitiveEngine, ConsciousnessMonitor


class ProductionService:
    """Example production service wrapper."""

    def __init__(self):
        self.engine = AsyncCognitiveEngine()
        self.monitor = ConsciousnessMonitor(self.engine.engine, alert_threshold=0.35)
        self.health_checks = []

    async def process_request(self, request_data: dict) -> dict:
        """Process a single request with full monitoring."""
        # Convert request to intent
        intent = IntentVector(
            phi1_dominance=request_data.get("dominance", 0.5),
            phi2_harmony=request_data.get("harmony", 0.5),
            phi3_creation=request_data.get("creativity", 0.5),
            phi4_preservation=request_data.get("stability", 0.5),
            phi5_transformation=request_data.get("adaptability", 0.5),
            phi6_knowledge=request_data.get("analytical", 0.5),
            phi7_connection=request_data.get("social", 0.5),
        )

        # Process
        result = await self.engine.process_async(intent)

        # Check consciousness state
        alert = self.monitor.check_state()

        return {
            "output_vector": result["output_vector"],
            "confidence": result["confidence"],
            "processing_time_ms": result["processing_time_ms"],
            "consciousness_level": result["consciousness_level"].name,
            "alert": alert,
            "timestamp": time.time(),
        }

    def get_health(self) -> dict:
        """Get service health status."""
        metrics = self.engine.get_metrics()
        trends = self.monitor.get_trends(window=100)

        # Determine health status
        status = "healthy"
        if metrics["avg_latency_ms"] > 5:
            status = "degraded"
        if trends["avg_coherence"] < 0.35:
            status = "critical"

        return {
            "status": status,
            "metrics": metrics,
            "trends": trends,
            "uptime": len(self.monitor.history),
        }


def simulate_production_load():
    """Simulate production workload."""
    print("\n3. Simulating production workload...")
    service = ProductionService()

    # Simulate 500 requests with varying patterns
    request_patterns = [
        {"name": "API Request", "dominance": 0.4, "analytical": 0.8},
        {"name": "UI Action", "harmony": 0.7, "social": 0.6},
        {"name": "Data Process", "creativity": 0.3, "analytical": 0.9},
        {"name": "User Auth", "stability": 0.9, "dominance": 0.3},
        {"name": "Background Job", "adaptability": 0.8, "creativity": 0.7},
    ]

    import asyncio

    async def run_simulation():
        for i in range(500):
            pattern = request_patterns[i % len(request_patterns)]
            result = await service.process_request(pattern)

            if i % 100 == 0:
                print(
                    f"   Request {i}: {result['consciousness_level']} "
                    f"({result['processing_time_ms']:.2f}ms)"
                )

        # Get final health
        health = service.get_health()
        return health

    return asyncio.run(run_simulation())


def main():
    print("=" * 60)
    print("NeuralBlitz V50 Minimal - Example 5: Production Integration")
    print("=" * 60)

    # Service initialization
    print("\n1. Initializing production service...")
    service = ProductionService()
    print("   ✓ Async engine initialized")
    print("   ✓ Consciousness monitor active (threshold: 0.35)")
    print("   ✓ Health check system ready")

    # Health check
    print("\n2. Initial health check:")
    health = service.get_health()
    print(f"   ✓ Status: {health['status']}")
    print(f"   ✓ Avg latency: {health['metrics']['avg_latency_ms']:.3f}ms")
    print(f"   ✓ Current level: {health['metrics']['current_consciousness']['level']}")

    # Simulate load
    final_health = simulate_production_load()

    # Final report
    print("\n4. Final production report:")
    print(f"   ✓ Status: {final_health['status']}")
    print(f"   ✓ Total processed: {final_health['metrics']['total_processed']}")
    print(f"   ✓ Avg coherence: {final_health['trends']['avg_coherence']:.3f}")
    print(f"   ✓ Coherence trend: {final_health['trends']['trend']}")

    # Performance analysis
    print("\n5. Performance characteristics:")
    if final_health["metrics"]["avg_latency_ms"] < 1:
        print("   ✓ EXCELLENT: Sub-millisecond latency")
    elif final_health["metrics"]["avg_latency_ms"] < 2:
        print("   ✓ GOOD: Under 2ms target")
    else:
        print("   ⚠ WARNING: Above 2ms target")

    if final_health["trends"]["avg_coherence"] > 0.4:
        print("   ✓ STABLE: Healthy coherence levels")
    else:
        print("   ⚠ WARNING: Low coherence detected")

    # Best practices
    print("\n6. Production best practices:")
    print("   ✓ Use AsyncCognitiveEngine for concurrent workloads")
    print("   ✓ Monitor consciousness trends for system health")
    print("   ✓ Set alert threshold based on your SLA (0.35-0.45)")
    print("   ✓ Pattern memory auto-limits to 100 (no memory leaks)")
    print("   ✓ SEED ensures deterministic behavior across restarts")
    print("   ✓ NumPy-only = no GPU/CUDA dependencies to manage")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Production service ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
