"""
Example 2: Async Processing
Demonstrates concurrent batch processing with AsyncCognitiveEngine.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from neuralblitz import MinimalCognitiveEngine, IntentVector
from neuralblitz.advanced import AsyncCognitiveEngine


async def main():
    print("=" * 60)
    print("NeuralBlitz V50 Minimal - Example 2: Async Processing")
    print("=" * 60)

    # Create async engine
    print("\n1. Creating AsyncCognitiveEngine...")
    engine = AsyncCognitiveEngine()
    print("   ✓ Async engine ready")

    # Create test intents
    print("\n2. Generating 100 test intents...")
    intents = [
        IntentVector(
            phi1_dominance=i / 100,
            phi2_harmony=0.5,
            phi3_creation=0.5 + (i % 50) / 100,
            phi6_knowledge=0.3 + (i % 30) / 100,
        )
        for i in range(100)
    ]
    print(f"   ✓ Generated {len(intents)} intents")

    # Sequential processing (baseline)
    print("\n3. Sequential processing (baseline)...")
    sync_engine = MinimalCognitiveEngine()
    start = asyncio.get_event_loop().time()
    for intent in intents[:20]:  # Just 20 for timing
        sync_engine.process_intent(intent)
    sync_time = (asyncio.get_event_loop().time() - start) * 1000
    print(f"   ✓ 20 intents in {sync_time:.1f}ms ({sync_time / 20:.2f}ms each)")

    # Async batch processing
    print("\n4. Async batch processing with controlled concurrency...")
    result = await engine.batch_process_async(intents, max_concurrent=10)

    print(f"   ✓ Processed {len(result.outputs)} intents")
    print(f"   ✓ Total time: {result.total_time_ms:.1f}ms")
    print(f"   ✓ Average time: {result.avg_time_ms:.2f}ms")
    print(f"   ✓ Throughput: {result.throughput:.1f} intents/second")
    print(f"   ✓ Speedup: {sync_time * 5 / result.total_time_ms:.1f}x faster")

    # Show coherence evolution
    print("\n5. Consciousness evolution during batch:")
    coherence_points = result.coherence_evolution[::20]  # Sample every 20th
    print(f"   Start: {coherence_points[0]:.3f}")
    print(f"   Middle: {coherence_points[len(coherence_points) // 2]:.3f}")
    print(f"   End: {coherence_points[-1]:.3f}")

    # Get metrics
    print("\n6. Performance metrics:")
    metrics = engine.get_metrics()
    print(f"   ✓ Total processed: {metrics['total_processed']}")
    print(f"   ✓ Average latency: {metrics['avg_latency_ms']:.3f}ms")
    print(f"   ✓ Peak throughput: {metrics['peak_throughput']:.1f} intents/sec")
    print(f"   ✓ Current level: {metrics['current_consciousness']['level']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
