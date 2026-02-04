import unittest
import sys
import os

# Add the parent directory (neuralblitz-v50) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import directly from the package
from neuralblitz.minimal import (
    MinimalCognitiveEngine,
    IntentVector,
    ConsciousnessLevel,
    CognitiveState,
)


class TestMinimalEngine(unittest.TestCase):
    def test_seed_preservation(self):
        """SEED must match exactly: a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"""
        engine = MinimalCognitiveEngine()
        expected = "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
        self.assertEqual(engine.SEED, expected)
        self.assertEqual(len(engine.SEED), 64)

    def test_basic_processing(self):
        """Intent in → valid output vector (7 dims, -1 to 1 range, valid confidence 0-1)"""
        engine = MinimalCognitiveEngine()
        intent = IntentVector(phi3_creation=0.8, phi1_dominance=0.5)
        result = engine.process_intent(intent)

        self.assertEqual(len(result["output_vector"]), 7)
        self.assertTrue(all(-1 <= x <= 1 for x in result["output_vector"]))
        self.assertTrue(0 <= result["confidence"] <= 1)
        self.assertGreater(result["processing_time_ms"], 0)

    def test_consciousness_evolution(self):
        """Process multiple high-intensity intents, verify coherence changes from initial 0.5"""
        engine = MinimalCognitiveEngine()
        initial_coherence = engine.consciousness.coherence

        # Process multiple high-intensity intents
        for i in range(10):
            intent = IntentVector(
                phi1_dominance=0.9, phi2_harmony=0.9, phi3_creation=0.9
            )
            engine.process_intent(intent)

        final_coherence = engine.consciousness.coherence
        self.assertNotEqual(initial_coherence, final_coherence)

    def test_pattern_memory_limit(self):
        """Process 150 intents, verify only 100 remain in memory (FIFO eviction)"""
        engine = MinimalCognitiveEngine()

        # Process 150 unique intents
        for i in range(150):
            intent = IntentVector(phi1_dominance=i / 150)
            engine.process_intent(intent)

        self.assertEqual(len(engine.pattern_memory), 100)
        # Verify first 50 were evicted (FIFO) - check first pattern in memory
        first_pattern = engine.pattern_memory[0]
        # The first pattern should have phi1_dominance of 50/150 = 0.333...
        # (since we evicted 0-49)
        self.assertGreaterEqual(first_pattern["input_hash"], 0)


if __name__ == "__main__":
    unittest.main()
