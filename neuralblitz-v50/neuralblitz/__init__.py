"""
NeuralBlitz V50 Minimal - Consciousness Engine
Package initialization with minimal exports.
"""

__version__ = "50.0.0-minimal"

# Minimal implementation exports (V50 Minimal)
from .minimal import (
    MinimalCognitiveEngine,
    IntentVector,
    ConsciousnessModel,
    ConsciousnessLevel,
    CognitiveState,
)

# Advanced features (optional)
try:
    from .advanced import (
        AsyncCognitiveEngine,
        StreamConfig,
        BatchResult,
        ConsciousnessMonitor,
        quick_process,
        compare_intents,
    )

    _advanced_available = True
except ImportError:
    _advanced_available = False

# Version info
__all__ = [
    "MinimalCognitiveEngine",
    "IntentVector",
    "ConsciousnessModel",
    "ConsciousnessLevel",
    "CognitiveState",
    "__version__",
]

# Add advanced exports if available
if _advanced_available:
    __all__.extend(
        [
            "AsyncCognitiveEngine",
            "StreamConfig",
            "BatchResult",
            "ConsciousnessMonitor",
            "quick_process",
            "compare_intents",
        ]
    )

# Production features (optional)
try:
    from .production import (
        ProductionCognitiveEngine,
        EngineHealth,
        NeuralBlitzError,
        InvalidIntentError,
        CoherenceDegradationError,
        EngineDegradationError,
    )

    _production_available = True
except ImportError:
    _production_available = False

if _production_available:
    __all__.extend(
        [
            "ProductionCognitiveEngine",
            "EngineHealth",
            "NeuralBlitzError",
            "InvalidIntentError",
            "CoherenceDegradationError",
            "EngineDegradationError",
        ]
    )

# Benchmark features (optional)
try:
    from .benchmark import (
        BenchmarkSuite,
        BenchmarkResult,
        quick_benchmark,
    )

    _benchmark_available = True
except ImportError:
    _benchmark_available = False

if _benchmark_available:
    __all__.extend(
        [
            "BenchmarkSuite",
            "BenchmarkResult",
            "quick_benchmark",
        ]
    )

# Optimization features (optional)
try:
    from .optimization import (
        OptimizationConfig,
        OptimizedEngine,
        PerformanceProfiler,
        benchmark_optimizations,
    )

    _optimization_available = True
except ImportError:
    _optimization_available = False

if _optimization_available:
    __all__.extend(
        [
            "OptimizationConfig",
            "OptimizedEngine",
            "PerformanceProfiler",
            "benchmark_optimizations",
        ]
    )

# Metrics & Monitoring (optional)
try:
    from .metrics import (
        NeuralBlitzMetrics,
        enable_metrics,
        PROMETHEUS_AVAILABLE,
    )

    _metrics_available = True
except ImportError:
    _metrics_available = False

if _metrics_available:
    __all__.extend(
        [
            "NeuralBlitzMetrics",
            "enable_metrics",
            "PROMETHEUS_AVAILABLE",
        ]
    )

# Persistence (optional)
try:
    from .persistence import (
        EngineSerializer,
        MemoryMappedStorage,
        PersistentEngine,
    )

    _persistence_available = True
except ImportError:
    _persistence_available = False

if _persistence_available:
    __all__.extend(
        [
            "EngineSerializer",
            "MemoryMappedStorage",
            "PersistentEngine",
        ]
    )
