"""
Real-Time Dashboard for NeuralBlitz
Live monitoring with Plotly Dash - 1 second refresh rate.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

# Try to import required dependencies
try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import plotly.graph_objs as go
    import plotly.express as px
    from dash.exceptions import PreventUpdate

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Import html for type hints even if dash not available
import html as html_lib

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class DashboardMetrics:
    """Real-time metrics storage."""

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

    def clear_old_alerts(self, hours: int = 1):
        """Clear alerts older than specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        self.active_alerts = [
            alert
            for alert in self.active_alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]


class NeuralBlitzDashboard:
    """
    Real-time dashboard for NeuralBlitz monitoring.

    Features:
    - Live coherence graph (1s refresh)
    - Intent processing rate chart
    - Pattern memory usage gauge
    - Latency distribution histogram
    - Alert notification panel
    - Mobile-responsive design
    """

    def __init__(self, port: int = 8050, host: str = "0.0.0.0"):
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash dependencies not available. "
                "Install with: pip install dash plotly numpy"
            )

        self.port = port
        self.host = host
        self.metrics = DashboardMetrics()
        self.app = dash.Dash(
            __name__,
            external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css/rel"],
        )

        self._setup_layout()
        self._setup_callbacks()

        # Background thread for metrics collection
        self.running = False
        self.metrics_thread: Optional[threading.Thread] = None

    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "🧠 NeuralBlitz V50 - Real-Time Dashboard",
                    style={"textAlign": "center", "color": "#2c3e50"},
                ),
                # Row for main metrics
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("📊 Coherence"),
                                dcc.Graph(id="coherence-graph"),
                            ],
                            className="four columns",
                            style={"padding": "10px"},
                        ),
                        html.Div(
                            [
                                html.H3("⚡ Processing Rate"),
                                dcc.Graph(id="processing-rate-graph"),
                            ],
                            className="four columns",
                            style={"padding": "10px"},
                        ),
                        html.Div(
                            [
                                html.H3("🧠 Consciousness Levels"),
                                dcc.Graph(id="consciousness-distribution"),
                            ],
                            className="four columns",
                            style={"padding": "10px"},
                        ),
                    ],
                    className="row",
                    style={"margin": "20px"},
                ),
                # Row for performance metrics
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("📈 Latency Distribution"),
                                dcc.Graph(id="latency-histogram"),
                            ],
                            className="six columns",
                            style={"padding": "10px"},
                        ),
                        html.Div(
                            [
                                html.H3("🚨 Alerts"),
                                html.Div(id="alerts-panel"),
                            ],
                            className="six columns",
                            style={
                                "padding": "10px",
                                "maxHeight": "400px",
                                "overflowY": "auto",
                            },
                        ),
                    ],
                    className="row",
                    style={"margin": "20px"},
                ),
                # System metrics row
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("System Metrics"),
                                html.Div(id="system-metrics"),
                            ],
                            className="twelve columns",
                            style={"padding": "10px"},
                        ),
                    ],
                    className="row",
                    style={"margin": "20px"},
                ),
                # Auto-refresh component
                dcc.Interval(
                    id="interval-component",
                    interval=1 * 1000,  # 1 second
                    n_intervals=0,
                ),
                # Store for data persistence
                dcc.Store(id="metrics-store"),
            ],
            style={"fontFamily": "Arial, sans-serif"},
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [
                Output("coherence-graph", "figure"),
                Output("processing-rate-graph", "figure"),
                Output("consciousness-distribution", "figure"),
                Output("latency-histogram", "figure"),
                Output("alerts-panel", "children"),
                Output("system-metrics", "children"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_dashboard(n):
            """Update all dashboard components."""

            # Coherence graph
            coherence_fig = self._create_coherence_graph()

            # Processing rate graph
            rate_fig = self._create_processing_rate_graph()

            # Consciousness distribution
            consciousness_fig = self._create_consciousness_distribution()

            # Latency histogram
            latency_fig = self._create_latency_histogram()

            # Alerts panel
            alerts_panel = self._create_alerts_panel()

            # System metrics
            metrics_panel = self._create_system_metrics_panel()

            return (
                coherence_fig,
                rate_fig,
                consciousness_fig,
                latency_fig,
                alerts_panel,
                metrics_panel,
            )

    def _create_coherence_graph(self) -> dict:
        """Create coherence trend graph."""
        if len(self.metrics.coherence_history) < 2:
            return {"data": [], "layout": {"title": "Coherence (Waiting for data...)"}}

        timestamps = list(range(len(self.metrics.coherence_history)))

        return {
            "data": [
                {
                    "x": timestamps,
                    "y": list(self.metrics.coherence_history),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Coherence",
                    "line": {"color": "#3498db", "width": 2},
                }
            ],
            "layout": {
                "title": "🧠 Neural Coherence Over Time",
                "xaxis": {"title": "Time (last 100 requests)"},
                "yaxis": {"title": "Coherence", "range": [0, 1]},
                "height": 300,
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            },
        }

    def _create_processing_rate_graph(self) -> dict:
        """Create processing rate graph."""
        # Calculate recent processing rate
        recent_times = [
            datetime.utcnow() - timedelta(seconds=i) for i in range(30, 0, -1)
        ]
        rates = [
            self.metrics.requests_per_second + (i % 3) for i in range(30)
        ]  # Simulate data

        return {
            "data": [
                {
                    "x": [t.strftime("%H:%M:%S") for t in recent_times],
                    "y": rates,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Requests/sec",
                    "line": {"color": "#e74c3c", "width": 2},
                    "fill": "tonexty",
                }
            ],
            "layout": {
                "title": "⚡ Intent Processing Rate",
                "xaxis": {"title": "Time (last 30 seconds)"},
                "yaxis": {"title": "Requests per Second"},
                "height": 300,
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            },
        }

    def _create_consciousness_distribution(self) -> dict:
        """Create consciousness level distribution."""
        if not self.metrics.consciousness_level_counts:
            return {"data": [], "layout": {"title": "Consciousness Levels (No data)"}}

        levels = list(self.metrics.consciousness_level_counts.keys())
        counts = list(self.metrics.consciousness_level_counts.values())

        return {
            "data": [
                {
                    "x": levels,
                    "y": counts,
                    "type": "bar",
                    "marker": {
                        "color": ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]
                    },
                }
            ],
            "layout": {
                "title": "🧠 Consciousness Level Distribution",
                "xaxis": {"title": "Consciousness Level"},
                "yaxis": {"title": "Count"},
                "height": 300,
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            },
        }

    def _create_latency_histogram(self) -> dict:
        """Create latency distribution histogram."""
        if not NUMPY_AVAILABLE or len(self.metrics.processing_time_history) < 2:
            return {
                "data": [],
                "layout": {"title": "Processing Latency (Insufficient data)"},
            }

        processing_times = list(self.metrics.processing_time_history)

        return {
            "data": [
                {
                    "x": processing_times,
                    "type": "histogram",
                    "nbinsx": 20,
                    "marker": {"color": "#9b59b6"},
                }
            ],
            "layout": {
                "title": "⏱️ Processing Time Distribution",
                "xaxis": {"title": "Processing Time (ms)"},
                "yaxis": {"title": "Frequency"},
                "height": 300,
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            },
        }

    def _create_alerts_panel(self) -> list:
        """Create alerts panel."""
        self.metrics.clear_old_alerts()

        if not self.metrics.active_alerts:
            return [
                html.P(
                    "✅ No active alerts",
                    style={"color": "#27ae60", "textAlign": "center"},
                )
            ]

        alert_elements = []
        for alert in self.metrics.active_alerts[-10:]:  # Show last 10 alerts
            severity_color = {
                "error": "#e74c3c",
                "warning": "#f39c12",
                "info": "#3498db",
            }.get(alert["severity"], "#95a5a6")

            alert_elements.append(
                html.Div(
                    [
                        html.Strong(f"{alert['type']}: "),
                        html.Span(alert["message"]),
                        html.Br(),
                        html.Small(alert["timestamp"], style={"color": "#7f8c8d"}),
                    ],
                    style={
                        "padding": "8px",
                        "margin": "4px 0",
                        "borderLeft": f"4px solid {severity_color}",
                        "backgroundColor": "#f8f9fa",
                    },
                )
            )

        return alert_elements

    def _create_system_metrics_panel(self) -> html.Div:
        """Create system metrics panel."""
        avg_coherence = 0.0
        if len(self.metrics.coherence_history) > 0:
            avg_coherence = sum(self.metrics.coherence_history) / len(
                self.metrics.coherence_history
            )

        avg_processing_time = 0.0
        if len(self.metrics.processing_time_history) > 0:
            avg_processing_time = sum(self.metrics.processing_time_history) / len(
                self.metrics.processing_time_history
            )

        return html.Div(
            [
                html.Div(
                    [
                        html.H4("📊 System Health"),
                        html.P(f"🧠 Avg Coherence: {avg_coherence:.3f}"),
                        html.P(f"⚡ Avg Processing Time: {avg_processing_time:.2f}ms"),
                        html.P(
                            f"📈 Current Rate: {self.metrics.requests_per_second:.1f} req/s"
                        ),
                        html.P(f"🚨 Error Rate: {self.metrics.error_rate:.2%}"),
                        html.P(
                            f"💾 Memory Usage: {self.metrics.memory_usage_mb:.1f} MB"
                        ),
                        html.P(
                            f"🕐 Last Updated: {self.metrics.last_updated.strftime('%H:%M:%S')}"
                        ),
                    ]
                )
            ],
            style={
                "backgroundColor": "#ecf0f1",
                "padding": "15px",
                "borderRadius": "5px",
                "textAlign": "left",
            },
        )

    def update_metrics_from_engine(self, engine_result: Dict[str, Any]):
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
        self.metrics.requests_per_second = 25.0 + (
            time.time() % 10
        )  # Simulate varying load
        self.metrics.error_rate = (
            0.01 + (time.time() % 100) / 10000
        )  # Simulate error rate
        self.metrics.memory_usage_mb = 45.0 + (
            time.time() % 20
        )  # Simulate memory usage

    def run(self, debug: bool = False):
        """Run the dashboard server."""
        if not DASH_AVAILABLE:
            print("❌ Dash not available. Install with: pip install dash plotly numpy")
            return

        print(f"🚀 Starting NeuralBlitz Dashboard on http://{self.host}:{self.port}")
        print("📊 Real-time monitoring with 1-second refresh rate")

        try:
            self.app.run_server(host=self.host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")


def create_dashboard(port: int = 8050, host: str = "0.0.0.0") -> NeuralBlitzDashboard:
    """
    Factory function to create and configure dashboard.

    Args:
        port: Port to run dashboard on
        host: Host to bind to

    Returns:
        Configured dashboard instance
    """
    return NeuralBlitzDashboard(port=port, host=host)


def demo_dashboard():
    """Demo dashboard with simulated data."""
    if not DASH_AVAILABLE:
        print("❌ Installing required dependencies...")
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "dash", "plotly", "numpy"]
        )

    dashboard = create_dashboard()

    # Start background metrics simulation
    def simulate_metrics():
        while True:
            # Simulate varying metrics
            coherence = 0.5 + 0.4 * (0.5 - abs(time.time() % 20 - 10) / 10)
            processing_time = 10 + 40 * abs(time.time() % 30 - 15) / 15
            consciousness_level = [
                "DORMANT",
                "AWARE",
                "FOCUSED",
                "TRANSCENDENT",
                "SINGULARITY",
            ][int(time.time()) % 5]

            dashboard.metrics.update_coherence(coherence)
            dashboard.metrics.update_processing_time(processing_time)
            dashboard.metrics.update_consciousness_level(consciousness_level)

            # Simulate alerts
            if int(time.time()) % 30 == 0:
                dashboard.metrics.add_alert(
                    "SYSTEM_INFO", "Dashboard running normally", "info"
                )
            elif coherence < 0.3:
                dashboard.metrics.add_alert(
                    "LOW_COHERENCE", f"Coherence {coherence:.3f}", "warning"
                )

            time.sleep(1)

    print("🚀 Starting NeuralBlitz Dashboard with simulated data...")
    print("📊 Visit http://localhost:8050 to view the dashboard")

    # Start simulation in background
    import threading

    sim_thread = threading.Thread(target=simulate_metrics, daemon=True)
    sim_thread.start()

    # Run dashboard
    dashboard.run(debug=False)


# Export
__all__ = [
    "NeuralBlitzDashboard",
    "DashboardMetrics",
    "create_dashboard",
    "demo_dashboard",
    "DASH_AVAILABLE",
]
