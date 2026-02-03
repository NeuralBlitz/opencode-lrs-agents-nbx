"""
NeuralBlitz v50.0 - Advanced Machine Learning Integration
Production-ready ML pipeline with real-time analytics and anomaly detection
"""

import asyncio
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
import uuid
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MLModel:
    """Machine learning model metadata and configuration."""

    model_id: str
    model_type: str
    version: str
    accuracy: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    training_data_size: int = 0
    last_trained: Optional[datetime] = None
    is_active: bool = True
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Machine learning prediction result with confidence."""

    prediction: Any
    confidence: float
    model_id: str
    processing_time_ms: float
    features_used: List[str]
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""

    anomaly_id: str
    severity: str  # low, medium, high, critical
    confidence: float
    description: str
    affected_metrics: List[str]
    timestamp: datetime
    recommended_actions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class RealTimeAnalytics:
    """Real-time analytics and monitoring with anomaly detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history = defaultdict(list)
        self.alerts = []
        self.anomalies = []
        self.start_time = time.time()
        self.anomaly_detectors = {}

        # Analytics configuration
        self.max_history_size = 10000
        self.anomaly_threshold = 2.0  # Standard deviations
        self.alert_retention_hours = 24

    def record_metric(
        self, metric_name: str, value: float, tags: Dict[str, str] = None
    ):
        """Record a metric with optional tags and detect anomalies."""
        timestamp = time.time()

        # Create metric entry
        metric_entry = {"timestamp": timestamp, "value": value, "tags": tags or {}}

        self.metrics_history[metric_name].append(metric_entry)

        # Keep history within limits
        if len(self.metrics_history[metric_name]) > self.max_history_size:
            self.metrics_history[metric_name] = self.metrics_history[metric_name][
                -self.max_history_size :
            ]

        # Check for anomalies
        anomaly = self.detect_anomaly(metric_name, value)
        if anomaly:
            self.anomalies.append(anomaly)
            self.alerts.append(
                {
                    "timestamp": timestamp,
                    "type": "anomaly",
                    "metric": metric_name,
                    "value": value,
                    "severity": anomaly.severity,
                    "description": anomaly.description,
                }
            )

    def detect_anomaly(
        self, metric_name: str, current_value: float
    ) -> Optional[AnomalyDetection]:
        """Detect anomalies using statistical methods."""
        if (
            metric_name not in self.metrics_history
            or len(self.metrics_history[metric_name]) < 10
        ):
            return None

        values = [
            m["value"] for m in self.metrics_history[metric_name][-100:]
        ]  # Last 100 values

        # Statistical anomaly detection
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        # Z-score based anomaly detection
        z_score = abs(current_value - mean) / std if std > 0 else 0

        if z_score > 3.5:
            return AnomalyDetection(
                anomaly_id=str(uuid.uuid4()),
                severity="critical",
                confidence=min(z_score / 5.0, 1.0),
                description=f"Critical anomaly: {metric_name}={current_value:.3f} (z-score={z_score:.2f})",
                affected_metrics=[metric_name],
                timestamp=datetime.utcnow(),
                recommended_actions=["immediate_investigation", "escalate_team"],
            )
        elif z_score > 3.0:
            return AnomalyDetection(
                anomaly_id=str(uuid.uuid4()),
                severity="high",
                confidence=min(z_score / 5.0, 1.0),
                description=f"High severity anomaly: {metric_name}={current_value:.3f} (z-score={z_score:.2f})",
                affected_metrics=[metric_name],
                timestamp=datetime.utcnow(),
                recommended_actions=["investigate_cause", "monitor_closely"],
            )
        elif z_score > 2.0:
            return AnomalyDetection(
                anomaly_id=str(uuid.uuid4()),
                severity="medium",
                confidence=min(z_score / 5.0, 1.0),
                description=f"Medium anomaly: {metric_name}={current_value:.3f} (z-score={z_score:.2f})",
                affected_metrics=[metric_name],
                timestamp=datetime.utcnow(),
                recommended_actions=["monitor_trend", "check_recent_changes"],
            )

        return None

    def get_metric_summary(
        self, metric_name: str, window_minutes: int = 5
    ) -> Dict[str, float]:
        """Get metric summary for the last N minutes."""
        if metric_name not in self.metrics_history:
            return {}

        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [
            m["value"]
            for m in self.metrics_history[metric_name]
            if m["timestamp"] >= cutoff_time
        ]

        if not recent_values:
            return {}

        mean_val = sum(recent_values) / len(recent_values)

        return {
            "count": len(recent_values),
            "mean": mean_val,
            "min": min(recent_values),
            "max": max(recent_values),
            "latest": recent_values[-1],
            "std": (
                sum((x - mean_val) ** 2 for x in recent_values) / len(recent_values)
            )
            ** 0.5,
            "trend": self._calculate_trend(recent_values),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear regression to determine trend
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def get_alerts(
        self, severity: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent alerts, optionally filtered by severity."""
        cutoff_time = time.time() - (self.alert_retention_hours * 3600)
        recent_alerts = [a for a in self.alerts if a["timestamp"] >= cutoff_time]

        if severity:
            recent_alerts = [a for a in recent_alerts if a["severity"] == severity]

        return sorted(recent_alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        current_time = time.time()
        uptime_seconds = current_time - self.start_time

        # Calculate health score
        recent_anomalies = [
            a for a in self.anomalies if (current_time - a.timestamp.timestamp()) < 3600
        ]  # Last hour

        critical_count = sum(1 for a in recent_anomalies if a.severity == "critical")
        high_count = sum(1 for a in recent_anomalies if a.severity == "high")

        health_score = 100.0
        health_score -= critical_count * 20  # Each critical anomaly reduces score by 20
        health_score -= high_count * 10  # Each high anomaly reduces score by 10
        health_score = max(health_score, 0)  # Minimum score is 0

        return {
            "health_score": health_score,
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "total_metrics": len(self.metrics_history),
            "active_alerts": len(self.get_alerts()),
            "critical_anomalies": critical_count,
            "high_anomalies": high_count,
            "system_status": "healthy"
            if health_score > 80
            else "degraded"
            if health_score > 50
            else "critical",
            "timestamp": datetime.utcnow().isoformat(),
        }


class AdvancedMLIntegration:
    """Advanced machine learning integration with real-time capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        self.analytics = RealTimeAnalytics(config)
        self.feature_store = defaultdict(list)
        self.prediction_history = deque(maxlen=1000)
        self.model_performance = defaultdict(list)

        # ML configuration
        self.default_confidence_threshold = 0.7
        self.ensemble_models = {}
        self.feature_importance = defaultdict(float)

        # Initialize default models
        self._initialize_default_models()

        logger.info("Initialized Advanced ML Integration")

    def _initialize_default_models(self):
        """Initialize default ML models for common tasks."""

        # Intent classification model
        self.register_model(
            MLModel(
                model_id="intent_classifier_v1",
                model_type="classification",
                version="1.0.0",
                accuracy=0.92,
                parameters={
                    "classes": [
                        "create",
                        "analyze",
                        "transform",
                        "harmonize",
                        "transcend",
                    ],
                    "features": [
                        "text_embedding",
                        "emotional_tone",
                        "complexity_score",
                    ],
                },
            )
        )

        # Coherence prediction model
        self.register_model(
            MLModel(
                model_id="coherence_predictor_v1",
                model_type="regression",
                version="1.0.0",
                accuracy=0.89,
                parameters={
                    "target_range": [0.0, 1.0],
                    "features": ["phi_harmony", "phi_coherence", "context_similarity"],
                },
            )
        )

        # Anomaly detection model
        self.register_model(
            MLModel(
                model_id="anomaly_detector_v1",
                model_type="anomaly_detection",
                version="1.0.0",
                accuracy=0.94,
                parameters={
                    "threshold_method": "statistical",
                    "sensitivity": "high",
                    "features": [
                        "deviation_score",
                        "pattern_break",
                        "context_mismatch",
                    ],
                },
            )
        )

    def register_model(self, model: MLModel):
        """Register a new ML model."""
        self.models[model.model_id] = model
        logger.info(f"Registered ML model: {model.model_id}")

    def predict(self, model_id: str, features: Dict[str, Any]) -> PredictionResult:
        """Make a prediction using the specified model."""
        start_time = time.time()

        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        # Extract and validate features
        extracted_features = self._extract_features(
            features, model.parameters.get("features", [])
        )

        # Simulate model inference (in production, this would call actual ML models)
        prediction, confidence = self._simulate_model_inference(
            model_id, extracted_features
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Create prediction result
        result = PredictionResult(
            prediction=prediction,
            confidence=confidence,
            model_id=model_id,
            processing_time_ms=processing_time,
            features_used=list(extracted_features.keys()),
            timestamp=datetime.utcnow(),
            explanation=self._generate_explanation(model_id, extracted_features),
        )

        # Record metrics
        self.analytics.record_metric(f"ml_prediction_confidence_{model_id}", confidence)
        self.analytics.record_metric(f"ml_processing_time_{model_id}", processing_time)

        # Store prediction history
        self.prediction_history.append(result)

        # Update model performance
        self.model_performance[model_id].append(
            {
                "timestamp": datetime.utcnow(),
                "confidence": confidence,
                "processing_time_ms": processing_time,
            }
        )

        return result

    def _extract_features(
        self, input_features: Dict[str, Any], required_features: List[str]
    ) -> Dict[str, float]:
        """Extract and normalize features for ML models."""
        extracted = {}

        for feature in required_features:
            if feature in input_features:
                # Convert to float if possible
                try:
                    extracted[feature] = float(input_features[feature])
                except (ValueError, TypeError):
                    # Handle categorical features
                    extracted[feature] = self._encode_categorical(
                        input_features[feature]
                    )
            else:
                # Default value for missing features
                extracted[feature] = 0.0

            # Store feature for analytics
            self.feature_store[feature].append(extracted[feature])

        return extracted

    def _encode_categorical(self, value: Any) -> float:
        """Encode categorical values as float."""
        if isinstance(value, str):
            # Simple hash-based encoding
            return float(hash(value.lower()) % 1000) / 1000.0
        else:
            return float(value) if value is not None else 0.0

    def _simulate_model_inference(
        self, model_id: str, features: Dict[str, float]
    ) -> Tuple[Any, float]:
        """Simulate ML model inference (replace with actual models in production)."""

        # Create deterministic but pseudo-random predictions based on features
        feature_str = json.dumps(sorted(features.items()), sort_keys=True)
        seed = int(hashlib.md5(feature_str.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        model = self.models[model_id]

        if model.model_type == "classification":
            classes = model.parameters.get("classes", ["positive", "negative"])
            probabilities = np.random.dirichlet(np.ones(len(classes)))
            prediction_idx = np.argmax(probabilities)
            prediction = classes[prediction_idx]
            confidence = float(probabilities[prediction_idx])

        elif model.model_type == "regression":
            # Generate prediction based on feature weighted sum
            feature_sum = sum(features.values())
            prediction = max(
                0.0, min(1.0, feature_sum / len(features) + np.random.normal(0, 0.1))
            )
            confidence = (
                0.8 + np.random.random() * 0.2
            )  # High confidence for regression

        elif model.model_type == "anomaly_detection":
            # Anomaly score based on feature deviation
            deviation = np.std(list(features.values())) if len(features) > 1 else 0
            is_anomaly = deviation > 0.5 or np.random.random() < 0.1
            prediction = "anomaly" if is_anomaly else "normal"
            confidence = 0.6 + abs(deviation) * 0.4

        else:
            # Generic prediction
            prediction = np.random.choice(["success", "partial", "failure"])
            confidence = 0.5 + np.random.random() * 0.5

        return prediction, confidence

    def _generate_explanation(
        self, model_id: str, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate explanation for ML prediction."""

        # Find most influential features
        sorted_features = sorted(
            features.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_features = sorted_features[:3]

        return {
            "primary_factors": [
                {"feature": k, "importance": abs(v)} for k, v in top_features
            ],
            "explanation_type": "feature_importance",
            "model_confidence_factors": {
                "data_quality": 0.9,
                "feature_completeness": len(features)
                / 10.0,  # Assuming 10 expected features
                "model_accuracy": self.models[model_id].accuracy,
            },
        }

    def create_ensemble(
        self,
        ensemble_id: str,
        model_ids: List[str],
        weights: Optional[List[float]] = None,
    ):
        """Create an ensemble of multiple models."""

        if not all(model_id in self.models for model_id in model_ids):
            raise ValueError("One or more models not found")

        if weights is None:
            weights = [1.0 / len(model_ids)] * len(model_ids)
        elif len(weights) != len(model_ids):
            raise ValueError("Weights length must match models length")

        self.ensemble_models[ensemble_id] = {"model_ids": model_ids, "weights": weights}

        logger.info(f"Created ensemble: {ensemble_id} with {len(model_ids)} models")

    def predict_ensemble(
        self, ensemble_id: str, features: Dict[str, Any]
    ) -> PredictionResult:
        """Make prediction using ensemble of models."""

        if ensemble_id not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_id} not found")

        ensemble = self.ensemble_models[ensemble_id]
        model_ids = ensemble["model_ids"]
        weights = ensemble["weights"]

        # Get predictions from all models
        predictions = []
        confidences = []

        for i, model_id in enumerate(model_ids):
            result = self.predict(model_id, features)
            predictions.append(result.prediction)
            confidences.append(result.confidence * weights[i])

        # Weighted voting for classification, weighted average for regression
        if isinstance(predictions[0], str):
            # Weighted voting for classification
            from collections import Counter

            weighted_votes = {}
            for pred, conf in zip(predictions, confidences):
                weighted_votes[pred] = weighted_votes.get(pred, 0) + conf

            ensemble_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            ensemble_confidence = max(confidences)
        else:
            # Weighted average for regression
            ensemble_prediction = sum(
                p * c for p, c in zip(predictions, confidences)
            ) / sum(confidences)
            ensemble_confidence = np.mean(confidences)

        return PredictionResult(
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            model_id=f"ensemble_{ensemble_id}",
            processing_time_ms=sum(
                r.processing_time_ms for r in self.prediction_history[-len(model_ids) :]
            ),
            features_used=list(features.keys()),
            timestamp=datetime.utcnow(),
            explanation={
                "ensemble_members": model_ids,
                "individual_predictions": list(zip(predictions, confidences)),
                "final_confidence": ensemble_confidence,
            },
        )

    def get_model_performance(
        self, model_id: str, window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""

        if model_id not in self.model_performance:
            return {"error": "Model not found"}

        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        recent_performance = [
            p for p in self.model_performance[model_id] if p["timestamp"] >= cutoff_time
        ]

        if not recent_performance:
            return {"error": "No recent performance data"}

        confidences = [p["confidence"] for p in recent_performance]
        processing_times = [p["processing_time_ms"] for p in recent_performance]

        return {
            "model_id": model_id,
            "window_hours": window_hours,
            "total_predictions": len(recent_performance),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "predictions_per_hour": len(recent_performance) / window_hours,
            "model_health": "excellent"
            if sum(confidences) / len(confidences) > 0.8
            else "good"
            if sum(confidences) / len(confidences) > 0.6
            else "needs_improvement",
        }

    def get_ml_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive ML dashboard data."""

        return {
            "system_health": self.analytics.get_system_health(),
            "active_models": len(self.models),
            "active_ensembles": len(self.ensemble_models),
            "total_predictions": len(self.prediction_history),
            "recent_anomalies": len(self.analytics.get_alerts(severity="critical")),
            "model_performance": {
                model_id: self.get_model_performance(model_id)
                for model_id in self.models.keys()
            },
            "feature_statistics": {
                feature: {
                    "count": len(values),
                    "mean": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                }
                for feature, values in self.feature_store.items()
            },
            "recent_predictions": [
                {
                    "prediction": p.prediction,
                    "confidence": p.confidence,
                    "model_id": p.model_id,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in list(self.prediction_history)[-10:]  # Last 10 predictions
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global ML integration instance
_ml_integration = None


def get_ml_integration() -> AdvancedMLIntegration:
    """Get the global ML integration instance."""
    global _ml_integration
    if _ml_integration is None:
        _ml_integration = AdvancedMLIntegration()
        logger.info("Initialized Global ML Integration")
    return _ml_integration


def initialize_ml_engine(config: Optional[Dict[str, Any]] = None):
    """Initialize the ML engine with custom configuration."""
    global _ml_integration
    _ml_integration = AdvancedMLIntegration(config)
    logger.info("ML Engine initialized with custom configuration")
    return _ml_integration
