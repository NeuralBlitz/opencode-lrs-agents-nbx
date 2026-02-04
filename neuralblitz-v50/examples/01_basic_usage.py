"""
Example 1: Basic Usage
Demonstrates fundamental operations with the MinimalCognitiveEngine.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuralblitz import MinimalCognitiveEngine, IntentVector


def main():
    print("=" * 60)
    print("NeuralBlitz V50 Minimal - Example 1: Basic Usage")
    print("=" * 60)

    # Create engine
    print("\n1. Creating MinimalCognitiveEngine...")
    engine = MinimalCognitiveEngine()
    print(f"   ✓ Engine initialized with SEED: {engine.SEED[:16]}...{engine.SEED[-8:]}")
    print(f"   ✓ Initial coherence: {engine.consciousness.coherence}")
    print(f"   ✓ Initial level: {engine.consciousness.consciousness_level.name}")

    # Process a single intent
    print("\n2. Processing single intent (creative focus)...")
    intent = IntentVector(
        phi1_dominance=0.3,
        phi2_harmony=0.4,
        phi3_creation=0.9,  # High creativity
        phi4_preservation=0.2,
        phi5_transformation=0.7,
        phi6_knowledge=0.5,
        phi7_connection=0.6,
    )

    result = engine.process_intent(intent)

    print(f"   ✓ Processing time: {result['processing_time_ms']:.3f}ms")
    print(f"   ✓ Confidence: {result['confidence']:.2%}")
    print(
        f"   ✓ Output vector: [{', '.join(f'{x:.3f}' for x in result['output_vector'])}]"
    )
    print(f"   ✓ New coherence: {result['coherence']:.3f}")
    print(f"   ✓ Consciousness level: {result['consciousness_level']}")

    # Process multiple intents
    print("\n3. Processing 5 intents sequentially...")
    intents = [
        IntentVector(phi1_dominance=0.9, phi6_knowledge=0.8),  # Analytical
        IntentVector(phi7_connection=0.9, phi2_harmony=0.8),  # Social
        IntentVector(phi3_creation=0.9, phi5_transformation=0.9),  # Creative
        IntentVector(phi4_preservation=0.9, phi1_dominance=0.3),  # Conservative
        IntentVector(
            phi2_harmony=0.5, phi6_knowledge=0.5, phi7_connection=0.5
        ),  # Balanced
    ]

    for i, intent in enumerate(intents, 1):
        result = engine.process_intent(intent)
        print(
            f"   Intent {i}: {result['consciousness_level']} "
            f"(coherence: {result['coherence']:.3f}) "
            f"in {result['processing_time_ms']:.3f}ms"
        )

    # Check pattern memory
    print(f"\n4. Pattern memory status:")
    print(f"   ✓ Stored patterns: {len(engine.pattern_memory)} / 100")

    # Get consciousness report
    print("\n5. Consciousness Report:")
    report = engine.get_consciousness_report()
    for key, value in report.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
