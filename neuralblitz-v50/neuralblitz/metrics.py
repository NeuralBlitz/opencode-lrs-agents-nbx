"""
NeuralBlitz V50 - Prometheus Metrics Exporter
Production observability for monitoring and alerting.
"""

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

import time
import threading
from typing import Optional

from .minimal import MinimalCognitiveEngine
from .production import ProductionCognitiveEngine


class NeuralBlitzMetrics:
    """Prometheus metrics exporter for NeuralBlitz."""

    def __init__(self, engine: MinimalCognitiveEngine, port: int = 9090):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client not installed. Run: pip install prometheus-client"
            )

        self.engine = engine
        self.port = port

        # Counters
        self.intents_total = Counter(
            "neuralblitz_intents_total",
            "Total number of intents processed",
            ["consciousness_level"],
        )

        self.errors_total = Counter(
            "neuralblitz_errors_total",
            "Total number of processing errors",
            ["error_type"],
        )

        self.patterns_stored_total = Counter(
            "neuralblitz_patterns_stored_total",
            "Total number of patterns stored in memory",
        )

        # Histograms
        self.latency_seconds = Histogram(
            "neuralblitz_latency_seconds",
            "Intent processing latency in seconds",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        self.coherence_distribution = Histogram(
            "neuralblitz_coherence_distribution",
            "Distribution of coherence values",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Gauges (current state)
        self.coherence = Gauge(
            "neuralblitz_coherence", "Current consciousness coherence level (0-1)"
        )

        self.complexity = Gauge(
            "neuralblitz_complexity", "Current consciousness complexity (0-1)"
        )

        self.pattern_memory_usage = Gauge(
            "neuralblitz_pattern_memory_usage",
            "Percentage of pattern memory used (0-1)",
        )

        self.consciousness_level = Gauge(
            "neuralblitz_consciousness_level_value",
            "Current consciousness level as numeric value (0-1)",
        )

        self.uptime_seconds = Gauge(
            "neuralblitz_uptime_seconds", "Engine uptime in seconds"
        )

        # Info
        self.build_info = Info("neuralblitz_build", "Build information")
        self.build_info.info(
            {
                "version": "50.0.0-minimal",
                "seed_prefix": engine.SEED[:16],
                "implementation": "NumPy-only",
            }
        )

        self.server = None
        self._running = False
        self._update_thread = None

    def record_intent(self, result: dict, latency_ms: float):
        """Record metrics for a processed intent."""
        level = result.get("consciousness_level", "UNKNOWN")
        self.intents_total.labels(consciousness_level=level).inc()

        # Convert ms to seconds for Prometheus
        self.latency_seconds.observe(latency_ms / 1000)

        # Record coherence
        coherence = result.get("coherence", 0.5)
        self.coherence_distribution.observe(coherence)
        self.coherence.set(coherence)

        # Update pattern count
        patterns = result.get("patterns_stored", 0)
        self.pattern_memory_usage.set(patterns / 100.0)  # 100 is max

    def record_error(self, error_type: str):
        """Record an error."""
        self.errors_total.labels(error_type=error_type).inc()

    def update_gauges(self):
        """Update all gauge metrics."""
        self.coherence.set(self.engine.consciousness.coherence)
        self.complexity.set(self.engine.consciousness.complexity)
        self.pattern_memory_usage.set(len(self.engine.pattern_memory) / 100.0)
        self.consciousness_level.set(
            self.engine.consciousness.consciousness_level.value
        )

    def _update_loop(self, interval: float = 5.0):
        """Background thread to update gauges periodically."""
        start_time = time.time()

        while self._running:
            try:
                self.update_gauges()
                self.uptime_seconds.set(time.time() - start_time)
                time.sleep(interval)
            except Exception as e:
                print(f"Metrics update error: {e}")
                time.sleep(interval)

    def start(self, interval: float = 5.0):
        """Start the Prometheus metrics HTTP server."""
        if self._running:
            return

        self._running = True

        # Start HTTP server
        self.server = start_http_server(self.port)
        print(f"✅ Prometheus metrics server started on port {self.port}")
        print(f"   Metrics available at: http://localhost:{self.port}/metrics")

        # Start background update thread
        self._update_thread = threading.Thread(
            target=self._update_loop, args=(interval,), daemon=True
        )
        self._update_thread.start()

    def stop(self):
        """Stop the metrics server."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        if self.server:
            self.server.shutdown()
        print("Prometheus metrics server stopped")


class MetricsMiddleware:
    """Middleware for FastAPI/Flask to auto-record metrics."""

    def __init__(self, metrics: NeuralBlitzMetrics):
        self.metrics = metrics

    async def __call__(self, request, call_next):
        """ASGI middleware for automatic metrics."""
        import time

        start = time.time()

        try:
            response = await call_next(request)

            # Record metrics if this was an intent processing request
            if "/process" in request.url.path:
                latency_ms = (time.time() - start) * 1000
                # Note: In real implementation, you'd extract result from response

            return response

        except Exception as e:
            self.metrics.record_error(type(e).__name__)
            raise


def enable_metrics(
    engine: MinimalCognitiveEngine, port: int = 9090
) -> NeuralBlitzMetrics:
    """
    Convenience function to enable Prometheus metrics.

    Args:
        engine: NeuralBlitz engine to monitor
        port: Prometheus metrics port

    Returns:
        NeuralBlitzMetrics instance

    Example:
        >>> from neuralblitz import MinimalCognitiveEngine, enable_metrics
        >>> engine = MinimalCognitiveEngine()
        >>> metrics = enable_metrics(engine, port=9090)
        >>> metrics.start()
    """
    metrics = NeuralBlitzMetrics(engine, port)
    return metrics


# Grafana dashboard JSON template
GRAFANA_DASHBOARD_JSON = """
{
  "dashboard": {
    "title": "NeuralBlitz V50 Consciousness Monitor",
    "panels": [
      {
        "title": "Intents Processed (rate)",
        "targets": [
          {
            "expr": "rate(neuralblitz_intents_total[5m])",
            "legendFormat": "{{consciousness_level}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Processing Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(neuralblitz_latency_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(neuralblitz_latency_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Consciousness Coherence",
        "targets": [
          {
            "expr": "neuralblitz_coherence",
            "legendFormat": "Current Coherence"
          }
        ],
        "type": "gauge",
        "fieldConfig": {
          "max": 1,
          "min": 0,
          "thresholds": {
            "steps": [
              {"color": "red", "value": 0},
              {"color": "yellow", "value": 0.3},
              {"color": "green", "value": 0.7}
            ]
          }
        }
      },
      {
        "title": "Pattern Memory Usage",
        "targets": [
          {
            "expr": "neuralblitz_pattern_memory_usage * 100",
            "legendFormat": "Memory %"
          }
        ],
        "type": "gauge",
        "fieldConfig": {
          "unit": "percent",
          "max": 100,
          "min": 0
        }
      },
      {
        "title": "Errors (rate)",
        "targets": [
          {
            "expr": "rate(neuralblitz_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
"""


def export_grafana_dashboard(filename: str = "neuralblitz_dashboard.json"):
    """Export Grafana dashboard JSON."""
    with open(filename, "w") as f:
        f.write(GRAFANA_DASHBOARD_JSON)
    print(f"Grafana dashboard exported to: {filename}")


__all__ = [
    "NeuralBlitzMetrics",
    "MetricsMiddleware",
    "enable_metrics",
    "export_grafana_dashboard",
    "PROMETHEUS_AVAILABLE",
]
