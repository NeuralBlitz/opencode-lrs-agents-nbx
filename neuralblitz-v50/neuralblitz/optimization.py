"""
NeuralBlitz V50 - Optimization Utilities
JIT compilation, GPU acceleration, and performance optimizations.
"""

import numpy as np
from typing import Optional, Callable, Any
import time
import warnings

# Optional imports for optimizations
try:
    import numba
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. JIT optimizations disabled.")

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class OptimizationConfig:
    """Configuration for optimization features."""

    def __init__(
        self,
        use_jit: bool = True,
        use_gpu: bool = False,
        parallel: bool = True,
        cache_results: bool = True,
    ):
        self.use_jit = use_jit and NUMBA_AVAILABLE
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.parallel = parallel
        self.cache_results = cache_results


def optimized_neural_forward(config: OptimizationConfig = None):
    """
    Create optimized neural forward pass.

    Returns optimized forward function based on config.
    """
    config = config or OptimizationConfig()

    if config.use_jit and NUMBA_AVAILABLE:
        # JIT compiled version
        @jit(nopython=True, cache=config.cache_results, parallel=config.parallel)
        def forward_jit(input_vec, w1, b1, w2, b2, w3, b3):
            # Layer 1
            h1 = np.maximum(0, np.dot(input_vec, w1) + b1)
            # Layer 2
            h2 = np.maximum(0, np.dot(h1, w2) + b2)
            # Output
            output = np.tanh(np.dot(h2, w3) + b3)
            return output

        return forward_jit

    elif config.use_gpu and CUDA_AVAILABLE:
        # GPU version using CuPy
        def forward_gpu(input_vec, w1, b1, w2, b2, w3, b3):
            # Transfer to GPU
            x = cp.asarray(input_vec)
            w1_g = cp.asarray(w1)
            b1_g = cp.asarray(b1)
            w2_g = cp.asarray(w2)
            b2_g = cp.asarray(b2)
            w3_g = cp.asarray(w3)
            b3_g = cp.asarray(b3)

            # Compute on GPU
            h1 = cp.maximum(0, cp.dot(x, w1_g) + b1_g)
            h2 = cp.maximum(0, cp.dot(h1, w2_g) + b2_g)
            output = cp.tanh(cp.dot(h2, w3_g) + b3_g)

            # Transfer back
            return cp.asnumpy(output)

        return forward_gpu

    else:
        # Standard NumPy version (already efficient)
        def forward_standard(input_vec, w1, b1, w2, b2, w3, b3):
            h1 = np.maximum(0, np.dot(input_vec, w1) + b1)
            h2 = np.maximum(0, np.dot(h1, w2) + b2)
            output = np.tanh(np.dot(h2, w3) + b3)
            return output

        return forward_standard


class OptimizedEngine:
    """Engine with optimization features."""

    def __init__(self, config: OptimizationConfig = None):
        from .minimal import MinimalCognitiveEngine

        self.engine = MinimalCognitiveEngine()
        self.config = config or OptimizationConfig()
        self.forward_func = optimized_neural_forward(self.config)

        # Cache for repeated similar intents
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def process_intent_optimized(
        self, intent_vector: np.ndarray, use_cache: bool = True
    ) -> np.ndarray:
        """
        Process intent with optimizations.

        Args:
            intent_vector: 7-dimensional intent vector
            use_cache: Whether to use result caching

        Returns:
            Output vector
        """
        # Check cache
        if use_cache and self.config.cache_results:
            cache_key = hash(intent_vector.tobytes())
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1

        # Get weights
        w1 = self.engine.weights["layer1"]
        b1 = self.engine.biases["layer1"]
        w2 = self.engine.weights["layer2"]
        b2 = self.engine.biases["layer2"]
        w3 = self.engine.weights["layer3"]
        b3 = self.engine.biases["layer3"]

        # Process with optimized function
        output = self.forward_func(intent_vector, w1, b1, w2, b2, w3, b3)

        # Cache result
        if use_cache and self.config.cache_results:
            if len(self._cache) < 1000:  # Limit cache size
                self._cache[cache_key] = output

        return output

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear result cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class PerformanceProfiler:
    """Profile and identify hot paths."""

    def __init__(self):
        self.timings = {}
        self.call_counts = {}

    def profile(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function call.

        Args:
            func: Function to profile
            *args, **kwargs: Function arguments

        Returns:
            Function result
        """
        func_name = func.__name__

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000

        # Update stats
        if func_name not in self.timings:
            self.timings[func_name] = []
            self.call_counts[func_name] = 0

        self.timings[func_name].append(elapsed)
        self.call_counts[func_name] += 1

        # Keep only last 1000 timings
        if len(self.timings[func_name]) > 1000:
            self.timings[func_name] = self.timings[func_name][-1000:]

        return result

    def get_report(self) -> dict:
        """Get profiling report."""
        import statistics

        report = {}
        for func_name, timings in self.timings.items():
            if timings:
                report[func_name] = {
                    "calls": self.call_counts[func_name],
                    "mean_ms": statistics.mean(timings),
                    "median_ms": statistics.median(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
                }

        return report

    def print_report(self):
        """Print profiling report to console."""
        report = self.get_report()

        print("\n" + "=" * 70)
        print("PERFORMANCE PROFILE REPORT")
        print("=" * 70)
        print(f"{'Function':<30} {'Calls':<10} {'Mean (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)

        # Sort by mean time
        sorted_funcs = sorted(
            report.items(), key=lambda x: x[1]["mean_ms"], reverse=True
        )

        for func_name, stats in sorted_funcs:
            print(
                f"{func_name:<30} {stats['calls']:<10} "
                f"{stats['mean_ms']:<12.3f} {stats['max_ms']:<12.3f}"
            )

        print("=" * 70)


def benchmark_optimizations():
    """
    Benchmark different optimization configurations.

    Returns comparison of standard vs optimized performance.
    """
    from .minimal import MinimalCognitiveEngine, IntentVector

    print("Benchmarking optimization configurations...")
    print("=" * 60)

    results = {}

    # Standard
    print("\n1. Standard NumPy...")
    engine = MinimalCognitiveEngine()

    latencies = []
    for i in range(1000):
        intent = IntentVector(phi1_dominance=i / 1000)
        t0 = time.perf_counter()
        engine.process_intent(intent)
        latencies.append((time.perf_counter() - t0) * 1000)

    results["standard"] = {"mean": np.mean(latencies), "std": np.std(latencies)}
    print(
        f"   Mean: {results['standard']['mean']:.3f}ms ± {results['standard']['std']:.3f}ms"
    )

    # JIT (if available)
    if NUMBA_AVAILABLE:
        print("\n2. JIT Compiled (Numba)...")
        opt_engine = OptimizedEngine(OptimizationConfig(use_jit=True))

        latencies = []
        for i in range(1000):
            intent = IntentVector(phi1_dominance=i / 1000)
            t0 = time.perf_counter()
            opt_engine.process_intent_optimized(intent.to_vector())
            latencies.append((time.perf_counter() - t0) * 1000)

        results["jit"] = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "speedup": results["standard"]["mean"] / np.mean(latencies),
        }
        print(
            f"   Mean: {results['jit']['mean']:.3f}ms ± {results['jit']['std']:.3f}ms"
        )
        print(f"   Speedup: {results['jit']['speedup']:.1f}x")

    # Cached
    print("\n3. With Caching...")
    cached_engine = OptimizedEngine(OptimizationConfig(cache_results=True))

    # First pass (populate cache)
    for i in range(100):
        intent = IntentVector(phi1_dominance=i / 100)
        cached_engine.process_intent_optimized(intent.to_vector())

    # Second pass (cache hits)
    latencies = []
    for i in range(100):
        intent = IntentVector(phi1_dominance=i / 100)
        t0 = time.perf_counter()
        cached_engine.process_intent_optimized(intent.to_vector())
        latencies.append((time.perf_counter() - t0) * 1000)

    results["cached"] = {
        "mean": np.mean(latencies),
        "cache_stats": cached_engine.get_cache_stats(),
    }
    print(f"   Mean: {results['cached']['mean']:.3f}ms")
    print(
        f"   Cache hit rate: {results['cached']['cache_stats']['hit_rate_percent']:.1f}%"
    )

    print("\n" + "=" * 60)

    return results


__all__ = [
    "OptimizationConfig",
    "OptimizedEngine",
    "PerformanceProfiler",
    "benchmark_optimizations",
    "NUMBA_AVAILABLE",
    "CUDA_AVAILABLE",
]
