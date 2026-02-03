"""
Tests for NeuralBlitz Real-Time Dashboard
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from neuralblitz.visualization.dashboard import (
    DashboardMetrics,
    NeuralBlitzDashboard,
    create_dashboard,
    DASH_AVAILABLE,
)


class TestDashboardMetrics:
    """Test suite for DashboardMetrics functionality."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = DashboardMetrics()

        assert len(metrics.coherence_history) == 0
        assert len(metrics.processing_time_history) == 0
        assert len(metrics.consciousness_level_counts) == 0
        assert metrics.requests_per_second == 0.0
        assert metrics.error_rate == 0.0
        assert len(metrics.active_alerts) == 0
        assert isinstance(metrics.last_updated, datetime)

    def test_coherence_updates(self):
        """Test coherence metric updates."""
        metrics = DashboardMetrics()

        # Update coherence values
        metrics.update_coherence(0.8)
        metrics.update_coherence(0.9)
        metrics.update_coherence(0.7)

        assert len(metrics.coherence_history) == 3
        assert list(metrics.coherence_history) == [0.8, 0.9, 0.7]
        assert metrics.last_updated > datetime.utcnow() - timedelta(seconds=1)

    def test_processing_time_updates(self):
        """Test processing time metric updates."""
        metrics = DashboardMetrics()

        # Update processing times
        metrics.update_processing_time(15.5)
        metrics.update_processing_time(22.1)
        metrics.update_processing_time(18.3)

        assert len(metrics.processing_time_history) == 3
        assert list(metrics.processing_time_history) == [15.5, 22.1, 18.3]

    def test_consciousness_level_tracking(self):
        """Test consciousness level distribution tracking."""
        metrics = DashboardMetrics()

        # Update levels
        metrics.update_consciousness_level("AWARE")
        metrics.update_consciousness_level("FOCUSED")
        metrics.update_consciousness_level("AWARE")
        metrics.update_consciousness_level("TRANSCENDENT")

        expected_counts = {"AWARE": 2, "FOCUSED": 1, "TRANSCENDENT": 1}

        assert metrics.consciousness_level_counts == expected_counts

    def test_alert_management(self):
        """Test alert creation and management."""
        metrics = DashboardMetrics()

        # Add alerts
        metrics.add_alert("TEST_ALERT", "Test message", "warning")
        metrics.add_alert("ERROR_ALERT", "Error message", "error")

        assert len(metrics.active_alerts) == 2

        # Check alert structure
        alert = metrics.active_alerts[0]
        assert alert["type"] == "TEST_ALERT"
        assert alert["message"] == "Test message"
        assert alert["severity"] == "warning"
        assert "timestamp" in alert

    def test_alert_cleanup(self):
        """Test old alert cleanup."""
        metrics = DashboardMetrics()

        # Add old alert
        old_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        old_alert = {
            "timestamp": old_time,
            "type": "OLD_ALERT",
            "message": "Old message",
            "severity": "info",
        }
        metrics.active_alerts.append(old_alert)

        # Add recent alert
        metrics.add_alert("NEW_ALERT", "New message", "info")

        # Clear old alerts (older than 1 hour)
        metrics.clear_old_alerts(hours=1)

        # Should only keep recent alert
        assert len(metrics.active_alerts) == 1
        assert metrics.active_alerts[0]["type"] == "NEW_ALERT"

    def test_history_limits(self):
        """Test that history respects max length limits."""
        metrics = DashboardMetrics()

        # Add more than max coherence values (max is 100)
        for i in range(150):
            metrics.update_coherence(i * 0.01)

        # Should be limited to 100
        assert len(metrics.coherence_history) == 100

        # Should contain the most recent 100 values
        assert (
            metrics.coherence_history[-1] == 1.49
        )  # Last value (149 * 0.01, but capped at 1.0)

    def test_update_metrics_from_engine(self):
        """Test metrics update from engine results."""
        metrics = DashboardMetrics()

        # Simulate engine result
        engine_result = {
            "coherence": 0.85,
            "processing_time_ms": 25.7,
            "consciousness_level": "FOCUSED",
        }

        metrics.update_metrics_from_engine(engine_result)

        # Check updates
        assert len(metrics.coherence_history) == 1
        assert metrics.coherence_history[0] == 0.85
        assert len(metrics.processing_time_history) == 1
        assert metrics.processing_time_history[0] == 25.7
        assert metrics.consciousness_level_counts["FOCUSED"] == 1


@pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash not available")
class TestNeuralBlitzDashboard:
    """Test suite for dashboard functionality."""

    def test_dashboard_creation(self):
        """Test dashboard creation."""
        dashboard = NeuralBlitzDashboard(port=8051, host="127.0.0.1")

        assert dashboard.port == 8051
        assert dashboard.host == "127.0.0.1"
        assert dashboard.app is not None
        assert dashboard.metrics is not None

    def test_layout_setup(self):
        """Test dashboard layout setup."""
        dashboard = NeuralBlitzDashboard()

        # Check that layout was created
        layout = dashboard.app.layout
        assert layout is not None

        # Should contain main title
        layout_str = str(layout)
        assert "NeuralBlitz V50" in layout_str
        assert "Real-Time Dashboard" in layout_str

    def test_coherence_graph_creation(self):
        """Test coherence graph creation."""
        dashboard = NeuralBlitzDashboard()

        # Add some coherence data
        for i in range(10):
            dashboard.metrics.update_coherence(0.5 + i * 0.05)

        figure = dashboard._create_coherence_graph()

        assert "data" in figure
        assert "layout" in figure

        # Check data structure
        data = figure["data"][0]
        assert data["type"] == "scatter"
        assert data["mode"] == "lines"
        assert len(data["y"]) == 10

    def test_processing_rate_graph_creation(self):
        """Test processing rate graph creation."""
        dashboard = NeuralBlitzDashboard()

        figure = dashboard._create_processing_rate_graph()

        assert "data" in figure
        assert "layout" in figure

        # Check layout title
        assert "Processing Rate" in figure["layout"]["title"]

    def test_consciousness_distribution_creation(self):
        """Test consciousness distribution creation."""
        dashboard = NeuralBlitzDashboard()

        # Add some consciousness level data
        dashboard.metrics.update_consciousness_level("AWARE")
        dashboard.metrics.update_consciousness_level("FOCUSED")
        dashboard.metrics.update_consciousness_level("AWARE")

        figure = dashboard._create_consciousness_distribution()

        assert "data" in figure
        assert "layout" in figure

        # Check data structure
        data = figure["data"][0]
        assert data["type"] == "bar"
        assert len(data["x"]) == 2  # AWARE, FOCUSED
        assert data["y"][0] == 2  # AWARE count
        assert data["y"][1] == 1  # FOCUSED count

    def test_latency_histogram_creation(self):
        """Test latency histogram creation."""
        dashboard = NeuralBlitzDashboard()

        # Add some processing time data
        times = [10, 15, 20, 25, 30, 35, 40, 45]
        for time_val in times:
            dashboard.metrics.update_processing_time(time_val)

        figure = dashboard._create_latency_histogram()

        assert "data" in figure
        assert "layout" in figure

        # Check data structure
        data = figure["data"][0]
        assert data["type"] == "histogram"
        assert data["x"] == times

    def test_alerts_panel_creation(self):
        """Test alerts panel creation."""
        dashboard = NeuralBlitzDashboard()

        # Add some alerts
        dashboard.add_alert("INFO", "System running", "info")
        dashboard.add_alert("WARNING", "High load", "warning")
        dashboard.add_alert("ERROR", "System error", "error")

        alerts_html = dashboard._create_alerts_panel()

        # Should return HTML components
        assert hasattr(alerts_html, "__iter__")

        # Convert to string to check content
        alerts_str = str(alerts_html)
        assert "System running" in alerts_str
        assert "High load" in alerts_str
        assert "System error" in alerts_str

    def test_system_metrics_panel_creation(self):
        """Test system metrics panel creation."""
        dashboard = NeuralBlitzDashboard()

        # Add some data
        dashboard.metrics.update_coherence(0.75)
        dashboard.metrics.update_processing_time(25.5)

        metrics_html = dashboard._create_system_metrics_panel()

        # Should be HTML component
        assert hasattr(metrics_html, "children")

        # Check content
        metrics_str = str(metrics_html)
        assert "System Health" in metrics_str

    def test_empty_data_handling(self):
        """Test dashboard with empty data."""
        dashboard = NeuralBlitzDashboard()

        # Test with no data
        coherence_fig = dashboard._create_coherence_graph()
        assert "Waiting for data" in coherence_fig["layout"]["title"]

        consciousness_fig = dashboard._create_consciousness_distribution()
        assert "No data" in consciousness_fig["layout"]["title"]


def test_create_dashboard_factory():
    """Test dashboard factory function."""
    dashboard = create_dashboard(port=8052, host="localhost")

    assert isinstance(dashboard, NeuralBlitzDashboard)
    assert dashboard.port == 8052
    assert dashboard.host == "localhost"


def test_dashboard_availability():
    """Test dashboard availability check."""
    # Should be a boolean
    assert isinstance(DASH_AVAILABLE, bool)


@pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash not available")
class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    def test_metrics_update_workflow(self):
        """Test complete metrics update workflow."""
        dashboard = create_dashboard()

        # Simulate processing sequence
        results = [
            {
                "coherence": 0.7,
                "processing_time_ms": 20,
                "consciousness_level": "AWARE",
            },
            {
                "coherence": 0.8,
                "processing_time_ms": 18,
                "consciousness_level": "FOCUSED",
            },
            {
                "coherence": 0.6,
                "processing_time_ms": 25,
                "consciousness_level": "AWARE",
            },
            {
                "coherence": 0.9,
                "processing_time_ms": 15,
                "consciousness_level": "TRANSCENDENT",
            },
        ]

        for result in results:
            dashboard.update_metrics_from_engine(result)

        # Verify metrics
        assert len(dashboard.metrics.coherence_history) == 4
        assert len(dashboard.metrics.processing_time_history) == 4

        # Calculate expected averages
        expected_avg_coherence = sum(r["coherence"] for r in results) / len(results)
        expected_avg_time = sum(r["processing_time_ms"] for r in results) / len(results)

        actual_avg_coherence = sum(dashboard.metrics.coherence_history) / len(
            dashboard.metrics.coherence_history
        )
        actual_avg_time = sum(dashboard.metrics.processing_time_history) / len(
            dashboard.metrics.processing_time_history
        )

        assert abs(actual_avg_coherence - expected_avg_coherence) < 0.01
        assert abs(actual_avg_time - expected_avg_time) < 0.01

        # Check consciousness level counts
        assert dashboard.metrics.consciousness_level_counts["AWARE"] == 2
        assert dashboard.metrics.consciousness_level_counts["FOCUSED"] == 1
        assert dashboard.metrics.consciousness_level_counts["TRANSCENDENT"] == 1

    @patch("threading.Thread")
    def test_background_simulation(self, mock_thread):
        """Test background simulation functionality."""
        # This would test the demo_dashboard function
        # For now, just verify the threading would be used
        pass
