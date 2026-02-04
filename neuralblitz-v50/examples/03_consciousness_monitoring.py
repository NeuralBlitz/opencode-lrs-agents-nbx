"""
Example 3: Consciousness Monitoring
Demonstrates real-time consciousness state tracking and alerting.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuralblitz import MinimalCognitiveEngine, IntentVector
from neuralblitz.advanced import ConsciousnessMonitor


def alert_handler(alert_type: str, snapshot: dict):
    """Callback for consciousness alerts."""
    print(f"   🚨 ALERT: {alert_type}")
    print(f"      State: {snapshot['level']} | Coherence: {snapshot['coherence']:.3f}")


def main():
    print("=" * 60)
    print("NeuralBlitz V50 Minimal - Example 3: Consciousness Monitoring")
    print("=" * 60)

    # Create engine and monitor
    print("\n1. Setting up consciousness monitoring...")
    engine = MinimalCognitiveEngine()
    monitor = ConsciousnessMonitor(engine, alert_threshold=0.4)
    monitor.add_observer(alert_handler)
    print("   ✓ Monitor initialized with threshold 0.4")
    print("   ✓ Alert observer registered")

    # Generate normal activity
    print("\n2. Simulating normal cognitive activity (intents 1-30)...")
    for i in range(30):
        intent = IntentVector(phi1_dominance=0.5, phi2_harmony=0.5, phi3_creation=0.5)
        engine.process_intent(intent)
        alert = monitor.check_state()
        if alert:
            print(f"   Intent {i + 1}: {alert}")

    print("   ✓ No alerts during normal activity")

    # Generate low coherence (disruptive) activity
    print("\n3. Simulating disruptive activity (intents 31-60)...")
    print("   (High dominance, low harmony should degrade coherence)")
    for i in range(30):
        intent = IntentVector(
            phi1_dominance=0.95,  # Very high dominance
            phi2_harmony=0.1,  # Very low harmony
            phi3_creation=0.1,
        )
        engine.process_intent(intent)
        alert = monitor.check_state()
        if alert and i > 20:  # Only show alerts after some degradation
            print(f"   Intent {i + 31}: {alert}")

    # Show history
    print(f"\n4. Monitor history:")
    print(f"   ✓ Total snapshots: {len(monitor.history)}")

    # Analyze trends
    print("\n5. Trend analysis (last 50 snapshots):")
    trends = monitor.get_trends(window=50)
    print(f"   ✓ Window size: {trends['window_size']}")
    print(f"   ✓ Average coherence: {trends['avg_coherence']:.3f}")
    print(f"   ✓ Coherence std: {trends['coherence_std']:.3f}")
    print(f"   ✓ Trend: {trends['trend']}")
    print(f"   ✓ Range: [{trends['min_coherence']:.3f}, {trends['max_coherence']:.3f}]")

    # Recovery
    print("\n6. Simulating recovery (intents 61-90)...")
    for i in range(30):
        intent = IntentVector(
            phi1_dominance=0.3,
            phi2_harmony=0.8,  # High harmony for recovery
            phi3_creation=0.6,
        )
        engine.process_intent(intent)
        monitor.check_state()

    final_trends = monitor.get_trends(window=30)
    print(f"   ✓ Post-recovery coherence: {final_trends['avg_coherence']:.3f}")
    print(f"   ✓ Trend: {final_trends['trend']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
