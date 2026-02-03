"""
Simplified Real-Time Dashboard for NeuralBlitz
Core metrics without Dash dependency.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SimpleMetrics:
    """Simplified metrics storage."""

    # System metrics
    coherence_history: deque = field(default_factory=lambda: deque(maxlen=100))
    processing_time_history: deque = field(default_factory=lambda: deque(maxlen=100))
    consciousness_level_counts: Dict[str, int] = field(default_factory=dict)

    # Performance metrics
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0

    # Alert metrics
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update_coherence(self, coherence: float):
        """Update coherence metric."""
        self.coherence_history.append(coherence)
        self.last_updated = datetime.utcnow()

    def update_processing_time(self, processing_time_ms: float):
        """Update processing time metric."""
        self.processing_time_history.append(processing_time_ms)
        self.last_updated = datetime.utcnow()

    def update_consciousness_level(self, level: str):
        """Update consciousness level distribution."""
        self.consciousness_level_counts[level] = (
            self.consciousness_level_counts.get(level, 0) + 1
        )
        self.last_updated = datetime.utcnow()

    def add_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Add alert to dashboard."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity,
        }
        self.active_alerts.append(alert)
        self.last_updated = datetime.utcnow()

        # Keep only recent alerts (last 50)
        if len(self.active_alerts) > 50:
            self.active_alerts = self.active_alerts[-50:]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_coherence = 0.0
        if len(self.coherence_history) > 0:
            avg_coherence = sum(self.coherence_history) / len(self.coherence_history)

        avg_processing_time = 0.0
        if len(self.processing_time_history) > 0:
            avg_processing_time = sum(self.processing_time_history) / len(
                self.processing_time_history
            )

        return {
            "avg_coherence": avg_coherence,
            "avg_processing_time_ms": avg_processing_time,
            "total_requests": len(self.processing_time_history),
            "consciousness_distribution": dict(self.consciousness_level_counts),
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate,
            "active_alerts_count": len(self.active_alerts),
            "last_updated": self.last_updated.isoformat(),
            "coherence_trend": "stable"
            if len(self.coherence_history) < 2
            else "improving"
            if self.coherence_history[-1] > self.coherence_history[-2]
            else "declining",
        }


class SimpleDashboard:
    """Simple dashboard without Dash dependency."""

    def __init__(self):
        self.metrics = SimpleMetrics()
        self.running = False

    def update_from_engine(self, engine_result: Dict[str, Any]):
        """Update metrics from engine processing result."""
        coherence = engine_result.get("coherence", 0.5)
        processing_time = engine_result.get("processing_time_ms", 0)
        consciousness_level = engine_result.get("consciousness_level", "UNKNOWN")

        self.metrics.update_coherence(coherence)
        self.metrics.update_processing_time(processing_time)
        self.metrics.update_consciousness_level(consciousness_level)

        # Check for alerts
        if coherence < 0.3:
            self.metrics.add_alert(
                "LOW_COHERENCE", f"Coherence dropped to {coherence:.3f}", "error"
            )
        elif processing_time > 100:
            self.metrics.add_alert(
                "SLOW_PROCESSING", f"Processing time {processing_time:.1f}ms", "warning"
            )

        # Update simulated metrics
        self.requests_per_second = 25.0 + (time.time() % 10)
        self.error_rate = 0.01 + (time.time() % 100) / 10000
        self.memory_usage_mb = 45.0 + (time.time() % 20)

    def print_status(self):
        """Print current dashboard status."""
        summary = self.metrics.get_summary()

        print("\n" + "=" * 60)
        print("🧠 NEURALBLITZ V50 - DASHBOARD STATUS")
        print("=" * 60)

        print(f"📊 Avg Coherence: {summary['avg_coherence']:.3f}")
        print(f"⚡ Avg Processing Time: {summary['avg_processing_time_ms']:.2f}ms")
        print(f"📈 Total Requests: {summary['total_requests']}")
        print(f"🔥 Rate: {summary['requests_per_second']:.1f} req/s")
        print(f"🚨 Error Rate: {summary['error_rate']:.2%}")
        print(f"💾 Memory Usage: {summary['memory_usage_mb']:.1f} MB")
        print(f"🧠 Coherence Trend: {summary['coherence_trend'].upper()}")

        print(f"\n🧠 Consciousness Levels:")
        for level, count in summary["consciousness_distribution"].items():
            print(f"   {level}: {count}")

        if summary["active_alerts_count"] > 0:
            print(f"\n🚨 Recent Alerts ({summary['active_alerts_count']}):")
            for alert in self.metrics.active_alerts[-3:]:  # Show last 3
                print(
                    f"   [{alert['severity'].upper()}] {alert['type']}: {alert['message']}"
                )

        print(f"🕐 Last Updated: {summary['last_updated']}")
        print("=" * 60)

    def start_monitoring(self):
        """Start background monitoring."""
        self.running = True

        def monitor_loop():
            while self.running:
                self.print_status()
                time.sleep(5)  # Update every 5 seconds

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("🚀 Dashboard monitoring started (updates every 5 seconds)")
        print("Press Ctrl+C to stop")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        print("\n👋 Dashboard monitoring stopped")


def demo_simple_dashboard():
    """Demo simple dashboard functionality."""
    dashboard = SimpleDashboard()

    # Simulate some engine processing
    print("🚀 Starting Simple Dashboard Demo...")
    print("Simulating NeuralBlitz engine processing...\n")

    # Simulate multiple intent processing results
    results = [
        {"coherence": 0.7, "processing_time_ms": 20, "consciousness_level": "AWARE"},
        {"coherence": 0.8, "processing_time_ms": 18, "consciousness_level": "FOCUSED"},
        {"coherence": 0.6, "processing_time_ms": 25, "consciousness_level": "AWARE"},
        {
            "coherence": 0.9,
            "processing_time_ms": 15,
            "consciousness_level": "TRANSCENDENT",
        },
        {
            "coherence": 0.4,
            "processing_time_ms": 120,
            "consciousness_level": "DORMANT",
        },  # Alert condition
        {"coherence": 0.85, "processing_time_ms": 16, "consciousness_level": "FOCUSED"},
    ]

    for i, result in enumerate(results):
        print(f"Processing intent {i + 1}...")
        dashboard.update_from_engine(result)
        time.sleep(1)

    # Show final status
    print("\n📊 Final Dashboard Status:")
    dashboard.print_status()


# Export
__all__ = ["SimpleMetrics", "SimpleDashboard", "demo_simple_dashboard"]
