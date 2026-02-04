"""
NeuralBlitz V50 Minimal - Advanced Features & New Advancements
Extension module adding async support, streaming, and optimizations to the minimal engine.
"""

from typing import Optional, Callable, Iterator, AsyncIterator
import asyncio
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from .minimal import (
    MinimalCognitiveEngine,
    IntentVector,
    ConsciousnessLevel,
    ConsciousnessModel,
)


@dataclass
class StreamConfig:
    """Configuration for streaming consciousness processing."""

    chunk_size: int = 10
    interval_ms: float = 0.1
    buffer_size: int = 100


@dataclass
class BatchResult:
    """Result from batch processing with metrics."""

    outputs: list
    total_time_ms: float
    avg_time_ms: float
    throughput: float  # intents per second
    coherence_evolution: list


class AsyncCognitiveEngine:
    """Async wrapper for MinimalCognitiveEngine with streaming and batch optimizations."""

    def __init__(
        self,
        engine: Optional[MinimalCognitiveEngine] = None,
        stream_config: Optional[StreamConfig] = None,
    ):
        self.engine = engine or MinimalCognitiveEngine()
        self.stream_config = stream_config or StreamConfig()
        self._lock = asyncio.Lock()
        self._metrics = {
            "total_processed": 0,
            "avg_latency_ms": 0.0,
            "peak_throughput": 0.0,
        }

    async def process_async(self, intent: IntentVector) -> dict:
        """Async wrapper for process_intent."""
        async with self._lock:
            # Run CPU-intensive work in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.engine.process_intent, intent
            )

            # Update metrics
            self._metrics["total_processed"] += 1
            self._update_latency(result["processing_time_ms"])

            return result

    async def batch_process_async(
        self, intents: list[IntentVector], max_concurrent: int = 10
    ) -> BatchResult:
        """Process batch of intents with controlled concurrency."""
        import time

        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(intent: IntentVector) -> dict:
            async with semaphore:
                return await self.process_async(intent)

        # Process all intents concurrently with limit
        tasks = [process_with_limit(intent) for intent in intents]
        results = await asyncio.gather(*tasks)

        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(intents) if intents else 0
        throughput = len(intents) / (total_time / 1000) if total_time > 0 else 0

        # Track coherence evolution
        coherence_evolution = [
            r["consciousness_snapshot"]["coherence"] for r in results
        ]

        # Update peak throughput
        if throughput > self._metrics["peak_throughput"]:
            self._metrics["peak_throughput"] = throughput

        return BatchResult(
            outputs=results,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            throughput=throughput,
            coherence_evolution=coherence_evolution,
        )

    async def stream_process(
        self,
        intents: list[IntentVector],
        callback: Optional[Callable[[dict, int], None]] = None,
    ) -> AsyncIterator[dict]:
        """Stream processing results with optional callbacks."""
        for i, intent in enumerate(intents):
            result = await self.process_async(intent)

            if callback:
                callback(result, i)

            yield result

            # Throttle to configured interval
            if self.stream_config.interval_ms > 0:
                await asyncio.sleep(self.stream_config.interval_ms / 1000)

    def _update_latency(self, latency_ms: float):
        """Update running average latency."""
        n = self._metrics["total_processed"]
        current_avg = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = (current_avg * (n - 1) + latency_ms) / n

    def get_metrics(self) -> dict:
        """Get engine performance metrics."""
        return {
            **self._metrics,
            "current_consciousness": {
                "level": self.engine.consciousness.consciousness_level.name,
                "coherence": self.engine.consciousness.coherence,
                "complexity": self.engine.consciousness.complexity,
            },
            "pattern_memory_size": len(self.engine.pattern_memory),
        }


class ConsciousnessMonitor:
    """Real-time consciousness state monitoring and alerting."""

    def __init__(self, engine: MinimalCognitiveEngine, alert_threshold: float = 0.3):
        self.engine = engine
        self.alert_threshold = alert_threshold
        self.history: list[dict] = []
        self._observers: list[Callable] = []

    def add_observer(self, callback: Callable[[str, dict], None]):
        """Add observer for consciousness state changes."""
        self._observers.append(callback)

    def check_state(self) -> Optional[str]:
        """Check consciousness state and return alert if threshold crossed."""
        current = self.engine.consciousness

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": current.consciousness_level.name,
            "coherence": current.coherence,
            "complexity": current.complexity,
        }
        self.history.append(snapshot)

        # Keep only last 1000 snapshots
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        # Check for degradation
        if current.coherence < self.alert_threshold:
            alert = f"LOW_COHERENCE: {current.coherence:.3f}"
            for observer in self._observers:
                observer(alert, snapshot)
            return alert

        return None

    def get_trends(self, window: int = 100) -> dict:
        """Analyze consciousness trends over recent window."""
        if len(self.history) < window:
            window = len(self.history)

        recent = self.history[-window:]
        coherences = [h["coherence"] for h in recent]

        return {
            "window_size": window,
            "avg_coherence": np.mean(coherences),
            "coherence_std": np.std(coherences),
            "trend": "increasing" if coherences[-1] > coherences[0] else "decreasing",
            "min_coherence": min(coherences),
            "max_coherence": max(coherences),
        }


# Convenience functions for common use cases
def quick_process(intent_dict: dict) -> dict:
    """One-liner to process an intent from a dictionary."""
    engine = MinimalCognitiveEngine()
    intent = IntentVector(**intent_dict)
    return engine.process_intent(intent)


def compare_intents(intent1: IntentVector, intent2: IntentVector) -> dict:
    """Compare two intents and return similarity metrics."""
    vec1 = intent1.to_vector()
    vec2 = intent2.to_vector()

    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    euclidean_dist = np.linalg.norm(vec1 - vec2)

    return {
        "cosine_similarity": float(cosine_sim),
        "euclidean_distance": float(euclidean_dist),
        "similarity_score": float((cosine_sim + 1) / 2),  # Normalize to 0-1
    }


# Export new classes
__all__ = [
    "AsyncCognitiveEngine",
    "StreamConfig",
    "BatchResult",
    "ConsciousnessMonitor",
    "quick_process",
    "compare_intents",
]
