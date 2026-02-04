# NeuralBlitz V50 - Minimal Implementation Examples

This directory contains practical examples demonstrating the NeuralBlitz V50 Minimal consciousness engine.

## Quick Start

```python
from neuralblitz import MinimalCognitiveEngine, IntentVector

# Create engine
engine = MinimalCognitiveEngine()

# Process an intent
intent = IntentVector(phi3_creation=0.8, phi1_dominance=0.5)
result = engine.process_intent(intent)

print(f"Output: {result['output_vector']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Time: {result['processing_time_ms']:.2f}ms")
```

## Examples

### 1. Basic Usage
See [01_basic_usage.py](01_basic_usage.py) for fundamental operations.

### 2. Async Processing
See [02_async_processing.py](02_async_processing.py) for concurrent batch processing.

### 3. Consciousness Monitoring
See [03_consciousness_monitoring.py](03_consciousness_monitoring.py) for real-time state tracking.

### 4. Intent Comparison
See [04_intent_comparison.py](04_intent_comparison.py) for measuring intent similarity.

### 5. Production Integration
See [05_production_integration.py](05_production_integration.py) for enterprise patterns.

## Running Examples

```bash
# Run specific example
python examples/01_basic_usage.py

# Run all examples
for f in examples/*.py; do python "$f"; done
```