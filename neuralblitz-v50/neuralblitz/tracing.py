"""
Distributed Tracing System for NeuralBlitz
OpenTelemetry integration with span creation for each processing phase.
"""

import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger("NeuralBlitz.Tracing")


class SpanKind(Enum):
    """Types of spans."""

    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class SpanStatus(Enum):
    """Span status codes."""

    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass
class SpanEvent:
    """Event that occurred during a span."""

    timestamp: datetime
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "attributes": self.attributes,
        }


@dataclass
class SpanLink:
    """Link to another span."""

    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Individual span in a trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: SpanStatus
    kind: SpanKind
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)

    def __post_init__(self):
        """Generate span and trace IDs if not provided."""
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        if not self.span_id:
            self.span_id = str(uuid.uuid4())[:16]

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        event = SpanEvent(
            timestamp=datetime.utcnow(), name=name, attributes=attributes or {}
        )
        self.events.append(event)

    def add_link(
        self, trace_id: str, span_id: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """Add link to another span."""
        link = SpanLink(trace_id=trace_id, span_id=span_id, attributes=attributes or {})
        self.links.append(link)

    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span."""
        self.end_time = datetime.utcnow()
        self.status = status

    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        duration = self.duration_ms()
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": duration,
            "status": self.status.value,
            "kind": self.kind.value,
            "attributes": self.attributes,
            "events": [event.to_dict() for event in self.events],
            "links": [
                {
                    "trace_id": link.trace_id,
                    "span_id": link.span_id,
                    "attributes": link.attributes,
                }
                for link in self.links
            ],
        }


class TraceContext:
    """Context for managing active spans."""

    def __init__(self):
        self._current_span: Optional[TraceSpan] = None
        self._lock = threading.Lock()

    def current_span(self) -> Optional[TraceSpan]:
        """Get current active span."""
        with self._lock:
            return self._current_span

    def set_current_span(self, span: TraceSpan):
        """Set current active span."""
        with self._lock:
            self._current_span = span

    def clear(self):
        """Clear current span."""
        with self._lock:
            self._current_span = None


class Tracer:
    """OpenTelemetry-compatible tracer."""

    def __init__(self, service_name: str, trace_exporter=None):
        """
        Initialize tracer.

        Args:
            service_name: Name of the service
            trace_exporter: Optional trace exporter
        """
        self.service_name = service_name
        self.trace_exporter = trace_exporter
        self.context = TraceContext()
        self.spans: List[TraceSpan] = []
        self.lock = threading.Lock()

        # Default service attributes
        self.service_attributes = {
            "service.name": service_name,
            "service.version": "50.0.0",
            "service.instance.id": str(uuid.uuid4()),
        }

    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[TraceSpan] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """Start a new span."""
        # Determine parent
        current_parent = parent_span or self.context.current_span()
        parent_span_id = current_parent.span_id if current_parent else None
        trace_id = current_parent.trace_id if current_parent else None

        # Create span
        span = TraceSpan(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            end_time=None,
            status=SpanStatus.OK,
            kind=kind,
            attributes=attributes or {},
        )

        # Add service attributes
        span.attributes.update(self.service_attributes)

        # Set as current span
        self.context.set_current_span(span)

        with self.lock:
            self.spans.append(span)

        logger.debug(f"Started span: {operation_name} ({span.span_id})")
        return span

    def finish_span(self, span: TraceSpan, status: SpanStatus = SpanStatus.OK):
        """Finish a span."""
        span.finish(status)

        # Clear from context if it's the current span
        if self.context.current_span() == span:
            self.context.clear()

        # Export if exporter available
        if self.trace_exporter:
            self.trace_exporter.export(span)

        logger.debug(
            f"Finished span: {span.operation_name} ({span.span_id}) in {span.duration_ms():.2f}ms"
        )

    def create_span(
        self,
        operation_name: str,
        parent_span: Optional[TraceSpan] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> "SpanContextManager":
        """Create a span context manager."""
        return SpanContextManager(self, operation_name, parent_span, kind)

    def get_current_span(self) -> Optional[TraceSpan]:
        """Get current active span."""
        return self.context.current_span()

    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        current = self.context.current_span()
        if current:
            current.add_event(name, attributes)

    def set_span_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        current = self.context.current_span()
        if current:
            current.set_attribute(key, value)

    def get_trace_by_id(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace ID."""
        with self.lock:
            return [span for span in self.spans if span.trace_id == trace_id]

    def get_spans_by_operation(self, operation_name: str) -> List[TraceSpan]:
        """Get spans by operation name."""
        with self.lock:
            return [
                span for span in self.spans if span.operation_name == operation_name
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self.lock:
            if not self.spans:
                return {"total_spans": 0}

            # Calculate statistics
            total_spans = len(self.spans)
            finished_spans = [s for s in self.spans if s.end_time]

            durations = [
                s.duration_ms() for s in finished_spans if s.duration_ms() is not None
            ]

            operation_counts = {}
            status_counts = {}

            for span in self.spans:
                operation_counts[span.operation_name] = (
                    operation_counts.get(span.operation_name, 0) + 1
                )
                status_counts[span.status.value] = (
                    status_counts.get(span.status.value, 0) + 1
                )

            return {
                "total_spans": total_spans,
                "finished_spans": len(finished_spans),
                "active_spans": total_spans - len(finished_spans),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "operations": operation_counts,
                "status_distribution": status_counts,
                "traces": len(set(s.trace_id for s in self.spans)),
            }


class SpanContextManager:
    """Context manager for spans."""

    def __init__(
        self,
        tracer: Tracer,
        operation_name: str,
        parent_span: Optional[TraceSpan] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ):
        self.tracer = tracer
        self.operation_name = operation_name
        self.parent_span = parent_span
        self.kind = kind
        self.span: Optional[TraceSpan] = None

    def __enter__(self) -> TraceSpan:
        """Enter context and start span."""
        self.span = self.tracer.start_span(
            operation_name=self.operation_name,
            parent_span=self.parent_span,
            kind=self.kind,
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and finish span."""
        if self.span:
            if exc_type:
                # Exception occurred, mark span as error
                self.span.set_attribute("error", True)
                self.span.set_attribute("error.type", exc_type.__name__)
                self.span.set_attribute("error.message", str(exc_val))
                self.tracer.finish_span(self.span, SpanStatus.ERROR)
            else:
                self.tracer.finish_span(self.span, SpanStatus.OK)


class TraceExporter:
    """Base class for trace exporters."""

    def export(self, span: TraceSpan):
        """Export a span."""
        raise NotImplementedError


class ConsoleTraceExporter(TraceExporter):
    """Export spans to console."""

    def export(self, span: TraceSpan):
        """Print span to console."""
        print(
            f"[TRACE] {span.operation_name}: {span.duration_ms():.2f}ms ({span.status.value})"
        )


class JSONFileTraceExporter(TraceExporter):
    """Export spans to JSON file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock = threading.Lock()

    def export(self, span: TraceSpan):
        """Write span to JSON file."""
        with self.lock:
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    json.dump(span.to_dict(), f)
                    f.write("\n")
            except Exception as e:
                logger.error(f"Failed to export span to {self.file_path}: {e}")


class JaegerTraceExporter(TraceExporter):
    """Export spans to Jaeger."""

    def __init__(self, endpoint: str = "http://localhost:14268/api/traces"):
        self.endpoint = endpoint
        self.session = None

        # Try to import requests
        try:
            import requests

            self.session = requests.Session()
        except ImportError:
            logger.warning("requests not available, Jaeger export disabled")

    def export(self, span: TraceSpan):
        """Send span to Jaeger."""
        if not self.session:
            return

        try:
            # Convert to Jaeger format
            jaeger_span = self._convert_to_jaeger_format(span)

            response = self.session.post(
                f"{self.endpoint}", json=jaeger_span, timeout=5
            )

            if response.status_code not in [200, 202]:
                logger.warning(f"Jaeger export failed: {response.status_code}")

        except Exception as e:
            logger.warning(f"Failed to export to Jaeger: {e}")

    def _convert_to_jaeger_format(self, span: TraceSpan) -> Dict[str, Any]:
        """Convert span to Jaeger format."""
        return {
            "traceID": span.trace_id.replace("-", ""),
            "spanID": span.span_id.replace("-", ""),
            "parentSpanID": span.parent_span_id.replace("-", "")
            if span.parent_span_id
            else None,
            "operationName": span.operation_name,
            "startTime": int(span.start_time.timestamp() * 1000000),  # microseconds
            "duration": int(span.duration_ms() * 1000)
            if span.duration_ms()
            else 0,  # microseconds
            "tags": [{"key": k, "value": str(v)} for k, v in span.attributes.items()],
            "logs": [
                {
                    "timestamp": int(event.timestamp.timestamp() * 1000000),
                    "fields": [{"key": "event", "value": event.name}]
                    + [
                        {"key": k, "value": str(v)} for k, v in event.attributes.items()
                    ],
                }
                for event in span.events
            ],
            "status": span.status.value,
        }


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def initialize_tracing(
    service_name: str = "neuralblitz", exporter: Optional[TraceExporter] = None
) -> Tracer:
    """Initialize global tracing."""
    global _global_tracer

    if exporter is None:
        exporter = ConsoleTraceExporter()

    _global_tracer = Tracer(service_name, exporter)

    logger.info(f"Initialized tracing for service: {service_name}")
    return _global_tracer


def get_tracer() -> Optional[Tracer]:
    """Get global tracer."""
    return _global_tracer


def trace_operation(operation_name: str):
    """Decorator for tracing function calls."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer:
                return func(*args, **kwargs)

            with tracer.create_span(operation_name) as span:
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Add input parameters info (sanitized)
                if args:
                    span.set_attribute("function.args_count", len(args))
                if kwargs:
                    span.set_attribute("function.kwargs_count", len(kwargs))

                try:
                    result = func(*args, **kwargs)

                    # Add result info
                    if result is not None:
                        span.set_attribute(
                            "function.result_type", type(result).__name__
                        )

                    return result

                except Exception as e:
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    raise

        return wrapper

    return decorator


# Convenience functions for NeuralBlitz specific tracing
def trace_intent_processing(intent_hash: str):
    """Trace intent processing."""
    tracer = get_tracer()
    if tracer:
        tracer.set_span_attribute("neuralblitz.intent_hash", intent_hash)
        tracer.add_span_event("intent_processing_started")


def trace_neural_forward_pass(layer_count: int, input_size: int):
    """Trace neural network forward pass."""
    tracer = get_tracer()
    if tracer:
        tracer.set_span_attribute("neuralblitz.layer_count", layer_count)
        tracer.set_span_attribute("neuralblitz.input_size", input_size)
        tracer.add_span_event("neural_forward_pass_started")


def trace_consciousness_update(level: str, coherence: float):
    """Trace consciousness state update."""
    tracer = get_tracer()
    if tracer:
        tracer.set_span_attribute("neuralblitz.consciousness_level", level)
        tracer.set_span_attribute("neuralblitz.coherence", coherence)
        tracer.add_span_event("consciousness_updated")


# Export
__all__ = [
    "Tracer",
    "TraceSpan",
    "TraceContext",
    "SpanContextManager",
    "TraceExporter",
    "ConsoleTraceExporter",
    "JSONFileTraceExporter",
    "JaegerTraceExporter",
    "SpanKind",
    "SpanStatus",
    "SpanEvent",
    "SpanLink",
    "initialize_tracing",
    "get_tracer",
    "trace_operation",
    "trace_intent_processing",
    "trace_neural_forward_pass",
    "trace_consciousness_update",
]
