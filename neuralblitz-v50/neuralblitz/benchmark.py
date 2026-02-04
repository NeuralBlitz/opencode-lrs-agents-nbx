"""
NeuralBlitz V50 - Comprehensive Benchmark Suite
Statistical performance analysis with visualization.
"""

import time
import statistics
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json

from .minimal import MinimalCognitiveEngine, IntentVector


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark results."""

    name: str
    sample_size: int
    total_time_ms: float
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sample_size": self.sample_size,
            "total_time_ms": round(self.total_time_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "throughput": round(self.throughput, 1),
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return f"""
Benchmark: {self.name}
{"=" * 50}
Sample Size:     {self.sample_size:,} intents
Total Time:      {self.total_time_ms:.2f} ms
Mean:            {self.mean_ms:.3f} ms ± {self.std_ms:.3f} ms
Median:          {self.median_ms:.3f} ms
Range:           [{self.min_ms:.3f}, {self.max_ms:.3f}] ms
P95 / P99:       {self.p95_ms:.3f} / {self.p99_ms:.3f} ms
Throughput:      {self.throughput:,.0f} intents/sec
{"=" * 50}
        """


class BenchmarkSuite:
    """Comprehensive benchmarking for NeuralBlitz."""

    def __init__(self, engine: MinimalCognitiveEngine = None):
        self.engine = engine or MinimalCognitiveEngine()
        self.results: List[BenchmarkResult] = []

    def benchmark_single_intent(
        self, sample_size: int = 1000, warmup: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark single intent processing.

        Args:
            sample_size: Number of intents to process
            warmup: Number of warmup iterations

        Returns:
            BenchmarkResult with statistics
        """
        # Warmup
        for i in range(warmup):
            intent = IntentVector(phi1_dominance=i / warmup)
            self.engine.process_intent(intent)

        # Benchmark
        latencies = []
        start = time.perf_counter()

        for i in range(sample_size):
            intent = IntentVector(phi1_dominance=i / sample_size)

            t0 = time.perf_counter()
            self.engine.process_intent(intent)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)

        total_time = (time.perf_counter() - start) * 1000

        return self._compute_stats(
            "Single Intent Processing", sample_size, latencies, total_time
        )

    def benchmark_batch_processing(
        self, batch_sizes: List[int] = [10, 100, 1000], trials: int = 5
    ) -> List[BenchmarkResult]:
        """
        Benchmark batch processing at different scales.

        Args:
            batch_sizes: List of batch sizes to test
            trials: Number of trials per batch size

        Returns:
            List of BenchmarkResults
        """
        results = []

        for batch_size in batch_sizes:
            latencies = []

            for trial in range(trials):
                intents = [
                    IntentVector(phi3_creation=i / batch_size)
                    for i in range(batch_size)
                ]

                t0 = time.perf_counter()
                for intent in intents:
                    self.engine.process_intent(intent)
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)

            # Average across trials
            mean_latency = statistics.mean(latencies)

            result = BenchmarkResult(
                name=f"Batch Processing (n={batch_size})",
                sample_size=batch_size * trials,
                total_time_ms=mean_latency,
                mean_ms=mean_latency / batch_size,
                median_ms=mean_latency / batch_size,
                std_ms=statistics.stdev(latencies) / batch_size
                if len(latencies) > 1
                else 0,
                min_ms=min(latencies) / batch_size,
                max_ms=max(latencies) / batch_size,
                p95_ms=mean_latency / batch_size,
                p99_ms=mean_latency / batch_size,
                throughput=(batch_size / mean_latency) * 1000,
                timestamp=datetime.utcnow().isoformat(),
            )

            results.append(result)

        return results

    def benchmark_consciousness_evolution(
        self, stages: int = 10, intents_per_stage: int = 50
    ) -> BenchmarkResult:
        """
        Benchmark with consciousness evolution over time.

        Args:
            stages: Number of processing stages
            intents_per_stage: Intents per stage

        Returns:
            BenchmarkResult with coherence tracking
        """
        latencies = []
        coherence_readings = []

        for stage in range(stages):
            # Vary intent profile by stage
            dominance = 0.3 + (0.6 * stage / stages)
            harmony = 0.7 - (0.5 * stage / stages)

            for i in range(intents_per_stage):
                intent = IntentVector(
                    phi1_dominance=dominance, phi2_harmony=harmony, phi3_creation=0.5
                )

                t0 = time.perf_counter()
                result = self.engine.process_intent(intent)
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)
                coherence_readings.append(result["coherence"])

        total_time = sum(latencies)

        return self._compute_stats(
            f"Consciousness Evolution ({stages} stages)",
            stages * intents_per_stage,
            latencies,
            total_time,
        )

    def benchmark_memory_pressure(self, max_patterns: int = 200) -> Dict[str, Any]:
        """
        Benchmark memory pressure with FIFO eviction.

        Args:
            max_patterns: Number of patterns to store (exceeds 100 limit)

        Returns:
            Dict with memory pressure metrics
        """
        latencies_before = []
        latencies_after = []

        # First 50 - no pressure
        for i in range(50):
            intent = IntentVector(phi6_knowledge=i / 50)
            t0 = time.perf_counter()
            self.engine.process_intent(intent)
            latencies_before.append((time.perf_counter() - t0) * 1000)

        # Next 150 - with pressure (FIFO eviction active)
        for i in range(150):
            intent = IntentVector(phi6_knowledge=0.5 + i / 150)
            t0 = time.perf_counter()
            self.engine.process_intent(intent)
            latencies_after.append((time.perf_counter() - t0) * 1000)

        return {
            "memory_pressure_test": {
                "patterns_stored": len(self.engine.pattern_memory),
                "patterns_processed": max_patterns,
                "latencies_before_eviction": {
                    "mean": round(statistics.mean(latencies_before), 3),
                    "std": round(
                        statistics.stdev(latencies_before)
                        if len(latencies_before) > 1
                        else 0,
                        3,
                    ),
                },
                "latencies_after_eviction": {
                    "mean": round(statistics.mean(latencies_after[:50]), 3),
                    "std": round(
                        statistics.stdev(latencies_after[:50])
                        if len(latencies_after[:50]) > 1
                        else 0,
                        3,
                    ),
                },
                "memory_overhead_percent": round(
                    (
                        (
                            statistics.mean(latencies_after)
                            - statistics.mean(latencies_before)
                        )
                        / statistics.mean(latencies_before)
                    )
                    * 100,
                    1,
                ),
            }
        }

    def run_full_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.

        Returns:
            Complete benchmark report
        """
        print("Running NeuralBlitz V50 Benchmark Suite...")
        print("=" * 60)

        # Reset engine for clean state
        self.engine = MinimalCognitiveEngine()

        # Run benchmarks
        results = {
            "single_intent": self.benchmark_single_intent(sample_size=1000),
            "batch_processing": self.benchmark_batch_processing(),
            "consciousness_evolution": self.benchmark_consciousness_evolution(),
            "memory_pressure": self.benchmark_memory_pressure(),
        }

        # Print results
        print("\n" + results["single_intent"])
        for batch_result in results["batch_processing"]:
            print(batch_result)
        print(results["consciousness_evolution"])

        # Memory pressure summary
        mp = results["memory_pressure"]["memory_pressure_test"]
        print(f"\nMemory Pressure Test:")
        print(
            f"  Patterns stored: {mp['patterns_stored']} / {mp['patterns_processed']}"
        )
        print(f"  Latency impact: +{mp['memory_overhead_percent']}%")

        # Generate report
        report = {
            "engine_info": {
                "seed": self.engine.SEED[:16] + "...",
                "version": "50.0.0-minimal",
                "implementation": "NumPy-only",
            },
            "benchmarks": {
                "single_intent": results["single_intent"].to_dict(),
                "batch_processing": [r.to_dict() for r in results["batch_processing"]],
                "consciousness_evolution": results["consciousness_evolution"].to_dict(),
                "memory_pressure": results["memory_pressure"],
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return report

    def _compute_stats(
        self, name: str, sample_size: int, latencies: List[float], total_time: float
    ) -> BenchmarkResult:
        """Compute statistical metrics from latency data."""
        sorted_latencies = sorted(latencies)

        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        return BenchmarkResult(
            name=name,
            sample_size=sample_size,
            total_time_ms=total_time,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p95_ms=sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)],
            p99_ms=sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)],
            throughput=(sample_size / total_time) * 1000,
            timestamp=datetime.utcnow().isoformat(),
        )

    def save_report(
        self, report: Dict[str, Any], filename: str = "benchmark_report.json"
    ) -> None:
        """Save benchmark report to file."""
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {filename}")


# Convenience function
def quick_benchmark() -> Dict[str, Any]:
    """Run quick benchmark and return results."""
    suite = BenchmarkSuite()
    return suite.run_full_suite()


__all__ = ["BenchmarkSuite", "BenchmarkResult", "quick_benchmark"]
