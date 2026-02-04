"""
NeuralBlitz V50 Minimal - Production Hardened Version
Adds logging, error handling, health checks, and graceful degradation.
"""

import logging
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import numpy as np
import uuid
import time
import pickle
from pathlib import Path

from .minimal import (
    MinimalCognitiveEngine,
    IntentVector,
    ConsciousnessModel,
    ConsciousnessLevel,
    CognitiveState,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NeuralBlitz")


class NeuralBlitzError(Exception):
    """Base exception for NeuralBlitz errors."""

    pass


class InvalidIntentError(NeuralBlitzError):
    """Raised when intent validation fails."""

    pass


class CoherenceDegradationError(NeuralBlitzError):
    """Raised when consciousness coherence falls below critical threshold."""

    pass


class EngineDegradationError(NeuralBlitzError):
    """Raised when engine performance degrades significantly."""

    pass


@dataclass
class EngineHealth:
    """Health status snapshot."""

    status: str  # 'healthy', 'degraded', 'critical'
    coherence: float
    pattern_memory_usage: float  # percentage
    avg_latency_ms: float
    error_rate: float
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "coherence": self.coherence,
            "pattern_memory_usage": self.pattern_memory_usage,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate": self.error_rate,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


class ProductionCognitiveEngine:
    """Production-hardened wrapper around MinimalCognitiveEngine."""

    def __init__(
        self,
        coherence_threshold: float = 0.3,
        max_latency_ms: float = 10.0,
        persistence_path: Optional[str] = None,
        enable_logging: bool = True,
    ):
        self.engine = MinimalCognitiveEngine()
        self.coherence_threshold = coherence_threshold
        self.max_latency_ms = max_latency_ms
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.enable_logging = enable_logging

        # Health tracking
        self.start_time = time.time()
        self.error_count = 0
        self.total_requests = 0
        self.latencies: List[float] = []
        self.max_latency_history = 100

        # Graceful degradation state
        self.degraded_mode = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 30  # seconds
        self.last_failure_time = 0

        # Load persisted state if available
        if self.persistence_path and self.persistence_path.exists():
            self._load_state()

        if self.enable_logging:
            logger.info(
                f"ProductionCognitiveEngine initialized (SEED: {self.engine.SEED[:16]}...)"
            )

    def process_intent(
        self, intent: IntentVector, validate: bool = True, allow_degraded: bool = True
    ) -> Dict[str, Any]:
        """
        Process intent with full error handling and health monitoring.

        Args:
            intent: IntentVector to process
            validate: Whether to validate intent before processing
            allow_degraded: Whether to allow processing in degraded mode

        Returns:
            Processing result with health metadata

        Raises:
            InvalidIntentError: If intent validation fails
            CoherenceDegradationError: If coherence is critically low
            EngineDegradationError: If circuit breaker is open
        """
        self.total_requests += 1

        try:
            # Check circuit breaker
            if self._circuit_breaker_open():
                if not allow_degraded:
                    raise EngineDegradationError(
                        "Circuit breaker is open - too many failures"
                    )
                self.degraded_mode = True
                logger.warning("Operating in degraded mode due to circuit breaker")

            # Validate intent
            if validate:
                self._validate_intent(intent)

            # Process with timing
            start = time.perf_counter()
            result = self.engine.process_intent(intent)
            latency = (time.perf_counter() - start) * 1000

            # Track latency
            self.latencies.append(latency)
            if len(self.latencies) > self.max_latency_history:
                self.latencies.pop(0)

            # Check performance degradation
            if latency > self.max_latency_ms:
                logger.warning(f"High latency detected: {latency:.2f}ms")
                if latency > self.max_latency_ms * 2:
                    self.circuit_breaker_failures += 1
                    self.last_failure_time = time.time()

            # Check coherence
            coherence = result.get("coherence", 0.5)
            if coherence < self.coherence_threshold:
                error_msg = f"Critical coherence degradation: {coherence:.3f}"
                logger.error(error_msg)
                self.circuit_breaker_failures += 1
                self.last_failure_time = time.time()
                raise CoherenceDegradationError(error_msg)

            # Reset circuit breaker on success
            if self.circuit_breaker_failures > 0:
                self.circuit_breaker_failures -= 1

            # Add health metadata
            result["health"] = self.get_health().to_dict()
            result["degraded_mode"] = self.degraded_mode

            if self.enable_logging:
                logger.debug(
                    f"Processed intent in {latency:.3f}ms (coherence: {coherence:.3f})"
                )

            return result

        except (InvalidIntentError, CoherenceDegradationError) as e:
            self.error_count += 1
            self.circuit_breaker_failures += 1
            self.last_failure_time = time.time()
            logger.error(f"Processing error: {e}")
            raise
        except Exception as e:
            self.error_count += 1
            self.circuit_breaker_failures += 1
            self.last_failure_time = time.time()
            logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
            raise NeuralBlitzError(f"Processing failed: {e}") from e

    def batch_process(
        self, intents: List[IntentVector], continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Process batch with comprehensive error handling.

        Args:
            intents: List of IntentVectors
            continue_on_error: Whether to continue processing on individual errors

        Returns:
            Dict with results, errors, and health summary
        """
        results = []
        errors = []

        for i, intent in enumerate(intents):
            try:
                result = self.process_intent(intent)
                results.append({"index": i, "success": True, "result": result})
            except Exception as e:
                error_info = {"index": i, "success": False, "error": str(e)}
                errors.append(error_info)

                if not continue_on_error:
                    break

                logger.warning(f"Batch item {i} failed: {e}")

        return {
            "total": len(intents),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "health": self.get_health().to_dict(),
        }

    def get_health(self) -> EngineHealth:
        """Get current engine health status."""
        coherence = self.engine.consciousness.coherence
        pattern_usage = len(self.engine.pattern_memory) / 100.0

        # Calculate average latency
        avg_latency = np.mean(self.latencies) if self.latencies else 0.0

        # Calculate error rate
        error_rate = (
            self.error_count / self.total_requests if self.total_requests > 0 else 0.0
        )

        # Determine status
        status = "healthy"
        if coherence < self.coherence_threshold * 1.5 or error_rate > 0.1:
            status = "degraded"
        if coherence < self.coherence_threshold or error_rate > 0.5:
            status = "critical"
        if self._circuit_breaker_open():
            status = "critical"

        uptime = time.time() - self.start_time

        return EngineHealth(
            status=status,
            coherence=coherence,
            pattern_memory_usage=pattern_usage,
            avg_latency_ms=avg_latency,
            error_rate=error_rate,
            uptime_seconds=uptime,
        )

    def health_check(self) -> bool:
        """
        Quick health check endpoint.

        Returns:
            True if healthy, False otherwise
        """
        health = self.get_health()
        return health.status == "healthy"

    def _validate_intent(self, intent: IntentVector) -> None:
        """Validate intent vector."""
        vector = intent.to_vector()

        # Check for NaN or Inf
        if not np.all(np.isfinite(vector)):
            raise InvalidIntentError("Intent vector contains NaN or Inf values")

        # Check bounds
        if not np.all((vector >= -1) & (vector <= 1)):
            logger.warning("Intent values outside [-1, 1] range, clipping applied")
            # Clip in place
            np.clip(vector, -1, 1, out=vector)

    def _circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False

        # Check if enough time has passed to try again
        if time.time() - self.last_failure_time > self.circuit_breaker_timeout:
            self.circuit_breaker_failures = 0
            self.degraded_mode = False
            return False

        return True

    def save_state(self, path: Optional[str] = None) -> None:
        """
        Persist engine state to disk.

        Args:
            path: Path to save to (uses persistence_path if None)
        """
        save_path = Path(path) if path else self.persistence_path
        if not save_path:
            raise ValueError("No persistence path specified")

        state = {
            "consciousness": asdict(self.engine.consciousness),
            "pattern_memory": self.engine.pattern_memory,
            "processing_count": self.engine.processing_count,
            "seed": self.engine.SEED,
            "metadata": {
                "saved_at": datetime.utcnow().isoformat(),
                "version": "50.0.0-minimal-prod",
            },
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Engine state saved to {save_path}")

    def _load_state(self) -> None:
        """Load engine state from disk."""
        try:
            with open(self.persistence_path, "rb") as f:
                state = pickle.load(f)

            # Restore consciousness
            self.engine.consciousness = ConsciousnessModel(**state["consciousness"])
            self.engine.pattern_memory = state["pattern_memory"]
            self.engine.processing_count = state["processing_count"]

            logger.info(f"Engine state loaded from {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            raise NeuralBlitzError(f"State loading failed: {e}") from e

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.engine = MinimalCognitiveEngine()
        self.error_count = 0
        self.total_requests = 0
        self.latencies = []
        self.circuit_breaker_failures = 0
        self.degraded_mode = False
        self.start_time = time.time()

        logger.info("Engine reset to initial state")


# Export production classes
__all__ = [
    "ProductionCognitiveEngine",
    "EngineHealth",
    "NeuralBlitzError",
    "InvalidIntentError",
    "CoherenceDegradationError",
    "EngineDegradationError",
]
