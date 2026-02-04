"""
Tests for NeuralBlitz Distributed Tracing System
"""

import pytest
import time
import json
import tempfile
import os
from datetime import datetime
from neuralblitz.tracing import (
    Tracer,
    TraceSpan,
    SpanKind,
    SpanStatus,
    SpanEvent,
    SpanContextManager,
    ConsoleTraceExporter,
    JSONFileTraceExporter,
    initialize_tracing,
    get_tracer,
    trace_operation,
    trace_intent_processing,
    trace_neural_forward_pass,
    trace_consciousness_update,
)


class TestTraceSpan:
    """Test suite for TraceSpan functionality."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = TraceSpan(
            trace_id="test_trace",
            span_id="test_span",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow(),
            end_time=None,
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        assert span.trace_id == "test_trace"
        assert span.span_id == "test_span"
        assert span.parent_span_id is None
        assert span.operation_name == "test_operation"
        assert span.status == SpanStatus.OK
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes == {}
        assert span.events == []

    def test_span_attributes(self):
        """Test span attribute management."""
        span = TraceSpan(
            trace_id="test",
            span_id="span",
            parent_span_id=None,
            operation_name="test",
            start_time=datetime.utcnow(),
            end_time=None,
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        # Test setting attributes
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 123)
        span.set_attribute("key3", True)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123
        assert span.attributes["key3"] == True

    def test_span_events(self):
        """Test span event management."""
        span = TraceSpan(
            trace_id="test",
            span_id="span",
            parent_span_id=None,
            operation_name="test",
            start_time=datetime.utcnow(),
            end_time=None,
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        # Add events
        span.add_event("event1", {"attr1": "value1"})
        span.add_event("event2", {"attr2": "value2"})

        assert len(span.events) == 2
        assert span.events[0].name == "event1"
        assert span.events[0].attributes["attr1"] == "value1"
        assert span.events[1].name == "event2"
        assert span.events[1].attributes["attr2"] == "value2"

    def test_span_duration(self):
        """Test span duration calculation."""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()

        span = TraceSpan(
            trace_id="test",
            span_id="span",
            parent_span_id=None,
            operation_name="test",
            start_time=start_time,
            end_time=end_time,
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        duration = span.duration_ms()
        assert duration is not None
        assert duration >= 0

        # Test without end time
        span_no_end = TraceSpan(
            trace_id="test",
            span_id="span2",
            parent_span_id=None,
            operation_name="test",
            start_time=start_time,
            end_time=None,
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        assert span_no_end.duration_ms() is None

    def test_span_to_dict(self):
        """Test span serialization to dict."""
        span = TraceSpan(
            trace_id="test",
            span_id="span",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("test_attr", "test_value")
        span.add_event("test_event")

        span_dict = span.to_dict()

        assert span_dict["trace_id"] == "test"
        assert span_dict["span_id"] == "span"
        assert span_dict["operation_name"] == "test_operation"
        assert span_dict["status"] == "OK"
        assert span_dict["kind"] == "INTERNAL"
        assert span_dict["attributes"]["test_attr"] == "test_value"
        assert len(span_dict["events"]) == 1
        assert span_dict["events"][0]["name"] == "test_event"

    def test_span_finish(self):
        """Test span finishing."""
        span = TraceSpan(
            trace_id="test",
            span_id="span",
            parent_span_id=None,
            operation_name="test",
            start_time=datetime.utcnow(),
            end_time=None,
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        assert span.end_time is None

        # Finish span
        time.sleep(0.01)  # Small delay
        span.finish(SpanStatus.ERROR)

        assert span.end_time is not None
        assert span.status == SpanStatus.ERROR
        assert span.duration_ms() is not None
        assert span.duration_ms() > 0


class TestTracer:
    """Test suite for Tracer functionality."""

    def setup_method(self):
        """Setup tracer for testing."""
        self.tracer = Tracer("test_service")

    def test_tracer_initialization(self):
        """Test tracer initialization."""
        assert self.tracer.service_name == "test_service"
        assert len(self.tracer.spans) == 0
        assert self.tracer.context.current_span() is None

    def test_start_span(self):
        """Test starting a span."""
        span = self.tracer.start_span("test_operation")

        assert span.operation_name == "test_operation"
        assert span.kind == SpanKind.INTERNAL
        assert span.status == SpanStatus.OK
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.parent_span_id is None

        # Span should be added to tracer
        assert len(self.tracer.spans) == 1
        assert self.tracer.spans[0] == span

        # Span should be current
        assert self.tracer.context.current_span() == span

    def test_start_span_with_parent(self):
        """Test starting a span with parent."""
        parent = self.tracer.start_span("parent_operation")
        child = self.tracer.start_span("child_operation", parent_span=parent)

        assert child.parent_span_id == parent.span_id
        assert child.trace_id == parent.trace_id

        # Child should be current
        assert self.tracer.context.current_span() == child

    def test_finish_span(self):
        """Test finishing a span."""
        span = self.tracer.start_span("test_operation")

        # Finish span
        self.tracer.finish_span(span)

        assert span.end_time is not None
        assert span.status == SpanStatus.OK

        # Span should no longer be current
        assert self.tracer.context.current_span() is None

    def test_span_context_manager(self):
        """Test span context manager."""
        with self.tracer.create_span("context_operation") as span:
            assert span.operation_name == "context_operation"
            assert self.tracer.context.current_span() == span

            # Span should be active within context
            self.tracer.set_span_attribute("test_attr", "test_value")
            self.tracer.add_span_event("test_event")

        # Span should be finished after context
        assert span.end_time is not None
        assert self.tracer.context.current_span() is None
        assert span.attributes["test_attr"] == "test_value"
        assert len(span.events) == 1
        assert span.events[0].name == "test_event"

    def test_nested_spans(self):
        """Test nested span creation."""
        with self.tracer.create_span("parent") as parent:
            parent.set_attribute("level", "parent")

            with self.tracer.create_span("child") as child:
                child.set_attribute("level", "child")

                assert self.tracer.context.current_span() == child
                assert child.parent_span_id == parent.span_id

            # After child context, parent should be current again
            assert self.tracer.context.current_span() == parent

        # After all contexts, no span should be current
        assert self.tracer.context.current_span() is None

        # Both spans should be finished
        assert parent.end_time is not None
        assert child.end_time is not None

    def test_current_span_operations(self):
        """Test operations on current span."""
        with self.tracer.create_span("test_current") as span:
            # Test attribute setting
            self.tracer.set_span_attribute("key1", "value1")
            assert span.attributes["key1"] == "value1"

            # Test event adding
            self.tracer.add_span_event("current_event", {"attr": "value"})
            assert len(span.events) == 1
            assert span.events[0].name == "current_event"

    def test_get_spans_by_filters(self):
        """Test filtering spans."""
        # Create spans with different operations
        self.tracer.start_span("op1")
        self.tracer.start_span("op2")
        self.tracer.start_span("op1")

        # Filter by operation name
        op1_spans = self.tracer.get_spans_by_operation("op1")
        assert len(op1_spans) == 2
        assert all(s.operation_name == "op1" for s in op1_spans)

        op2_spans = self.tracer.get_spans_by_operation("op2")
        assert len(op2_spans) == 1
        assert op2_spans[0].operation_name == "op2"

    def test_get_statistics(self):
        """Test tracing statistics."""
        # Create and finish some spans
        with self.tracer.create_span("operation1"):
            time.sleep(0.001)

        with self.tracer.create_span("operation2"):
            time.sleep(0.002)

        # Finish all spans
        for span in self.tracer.spans:
            if not span.end_time:
                span.finish()

        stats = self.tracer.get_statistics()

        assert stats["total_spans"] == 2
        assert stats["finished_spans"] == 2
        assert stats["active_spans"] == 0
        assert stats["avg_duration_ms"] > 0
        assert stats["operations"]["operation1"] == 1
        assert stats["operations"]["operation2"] == 1
        assert stats["traces"] == 2  # Different trace IDs


class TestTraceExporters:
    """Test suite for trace exporters."""

    def test_console_exporter(self):
        """Test console exporter."""
        exporter = ConsoleTraceExporter()

        span = TraceSpan(
            trace_id="test",
            span_id="span",
            parent_span_id=None,
            operation_name="test_operation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            status=SpanStatus.OK,
            kind=SpanKind.INTERNAL,
        )

        # Should not raise exception
        exporter.export(span)

    def test_json_file_exporter(self):
        """Test JSON file exporter."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            exporter = JSONFileTraceExporter(temp_file)

            span = TraceSpan(
                trace_id="test",
                span_id="span",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                status=SpanStatus.OK,
                kind=SpanKind.INTERNAL,
            )
            span.set_attribute("test_attr", "test_value")

            # Export span
            exporter.export(span)

            # Verify file content
            with open(temp_file, "r") as f:
                content = f.read().strip()
                exported_data = json.loads(content)

                assert exported_data["operation_name"] == "test_operation"
                assert exported_data["trace_id"] == "test"
                assert exported_data["attributes"]["test_attr"] == "test_value"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestGlobalTracing:
    """Test global tracing functions."""

    def teardown_method(self):
        """Cleanup global tracer."""
        import neuralblitz.tracing as tracing

        tracing._global_tracer = None

    def test_initialize_tracing(self):
        """Test global tracer initialization."""
        tracer = initialize_tracing("test_global")

        assert tracer is not None
        assert tracer.service_name == "test_global"
        assert get_tracer() == tracer

    def test_trace_operation_decorator(self):
        """Test operation tracing decorator."""
        initialize_tracing("decorator_test")

        @trace_operation("decorated_function")
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)

        assert result == 5

        # Check that span was created
        tracer = get_tracer()
        assert len(tracer.spans) == 1

        span = tracer.spans[0]
        assert span.operation_name == "decorated_function"
        assert span.attributes["function.name"] == "test_function"
        assert span.attributes["function.args_count"] == 2

    def test_trace_operation_decorator_with_exception(self):
        """Test decorator with exception."""
        initialize_tracing("exception_test")

        @trace_operation("exception_function")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Check that error was recorded
        tracer = get_tracer()
        assert len(tracer.spans) == 1

        span = tracer.spans[0]
        assert span.attributes["error.type"] == "ValueError"
        assert span.attributes["error.message"] == "Test error"

    def test_neuralblitz_specific_tracing(self):
        """Test NeuralBlitz specific tracing functions."""
        initialize_tracing("neuralblitz_test")

        # Test intent processing trace
        trace_intent_processing("intent_hash_123")

        tracer = get_tracer()
        current_span = tracer.context.current_span()
        if current_span:
            assert (
                current_span.attributes["neuralblitz.intent_hash"] == "intent_hash_123"
            )

        # Test neural forward pass trace
        trace_neural_forward_pass(layer_count=5, input_size=100)

        current_span = tracer.context.current_span()
        if current_span:
            assert current_span.attributes["neuralblitz.layer_count"] == 5
            assert current_span.attributes["neuralblitz.input_size"] == 100

        # Test consciousness update trace
        trace_consciousness_update("TRANSCENDENT", 0.95)

        current_span = tracer.context.current_span()
        if current_span:
            assert (
                current_span.attributes["neuralblitz.consciousness_level"]
                == "TRANSCENDENT"
            )
            assert current_span.attributes["neuralblitz.coherence"] == 0.95


class TestTracingIntegration:
    """Integration tests for complete tracing workflow."""

    def test_complete_tracing_workflow(self):
        """Test complete tracing workflow."""
        # Initialize tracing with file exporter
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            exporter = JSONFileTraceExporter(temp_file)
            tracer = Tracer("integration_test", exporter)

            # Simulate intent processing workflow
            with tracer.create_span(
                "process_intent", kind=SpanKind.SERVER
            ) as main_span:
                main_span.set_attribute("intent.type", "creative")
                main_span.set_attribute("intent.confidence", 0.85)

                # Neural forward pass
                with tracer.create_span("neural_forward_pass") as neural_span:
                    neural_span.set_attribute("layer_count", 3)
                    neural_span.add_event("layer_1_completed")
                    neural_span.add_event("layer_2_completed")
                    neural_span.add_event("layer_3_completed")
                    time.sleep(0.001)  # Simulate processing

                # Consciousness update
                with tracer.create_span("consciousness_update") as consciousness_span:
                    consciousness_span.set_attribute("previous_level", "AWARE")
                    consciousness_span.set_attribute("new_level", "FOCUSED")
                    consciousness_span.set_attribute("coherence", 0.92)
                    time.sleep(0.001)

            # Verify spans were created
            assert len(tracer.spans) == 3

            # Verify hierarchy
            main_span = tracer.spans[0]
            neural_span = tracer.spans[1]
            consciousness_span = tracer.spans[2]

            assert main_span.parent_span_id is None
            assert neural_span.parent_span_id == main_span.span_id
            assert consciousness_span.parent_span_id == main_span.span_id

            assert (
                main_span.trace_id
                == neural_span.trace_id
                == consciousness_span.trace_id
            )

            # Verify all spans finished
            assert all(span.end_time is not None for span in tracer.spans)

            # Verify export
            with open(temp_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 3  # 3 spans exported

                # Verify content
                for line in lines:
                    span_data = json.loads(line.strip())
                    assert "operation_name" in span_data
                    assert "duration_ms" in span_data
                    assert span_data["duration_ms"] >= 0

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
