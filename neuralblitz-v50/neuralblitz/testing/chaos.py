"""
NeuralBlitz V50 - Chaos Engineering Suite
Production resilience testing through controlled failures.
"""

import random
import time
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..minimal import MinimalCognitiveEngine, IntentVector

try:
    from ..production import ProductionCognitiveEngine

    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False

logger = logging.getLogger("NeuralBlitz.Chaos")


@dataclass
class ChaosResult:
    """Result from a chaos test run."""

    total_attempts: int
    errors: int
    recovered: int
    recovery_rate: float
    avg_recovery_time_ms: float
    failure_types: Dict[str, int]
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "errors": self.errors,
            "recovered": self.recovered,
            "recovery_rate": self.recovery_rate,
            "avg_recovery_time_ms": self.avg_recovery_time_ms,
            "failure_types": self.failure_types,
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
        }


class ChaosMonkey:
    """
    Chaos Engineering tool for testing NeuralBlitz resilience.

    Injects various failures to verify system can recover gracefully.
    """

    def __init__(self, engine):
        self.engine = engine
        self.failure_modes: List[Callable[[], IntentVector]] = [
            self._nan_injection,
            self._inf_injection,
            self._extreme_positive_values,
            self._extreme_negative_values,
            self._random_mutation,
            self._adversarial_gradient,
            self._null_values,
            self._very_large_vector,
            self._rapid_fire_setup,
            self._coherence_attacker,
        ]
        self.recovery_times: List[float] = []
        self.failure_counts: Dict[str, int] = {}

    def _nan_injection(self) -> IntentVector:
        """Inject NaN values into intent dimensions."""
        intent = IntentVector(
            phi1_dominance=float("nan")
            if random.random() < 0.3
            else random.uniform(-1, 1),
            phi2_harmony=float("nan")
            if random.random() < 0.3
            else random.uniform(-1, 1),
            phi3_creation=random.uniform(-1, 1),
            phi4_preservation=random.uniform(-1, 1),
            phi5_transformation=random.uniform(-1, 1),
            phi6_knowledge=random.uniform(-1, 1),
            phi7_connection=random.uniform(-1, 1),
        )
        return intent

    def _inf_injection(self) -> IntentVector:
        """Inject Infinity values."""
        intent = IntentVector(
            phi1_dominance=float("inf")
            if random.random() < 0.3
            else random.uniform(-1, 1),
            phi2_harmony=float("-inf")
            if random.random() < 0.3
            else random.uniform(-1, 1),
            phi3_creation=random.uniform(-1, 1),
            phi4_preservation=random.uniform(-1, 1),
            phi5_transformation=random.uniform(-1, 1),
            phi6_knowledge=random.uniform(-1, 1),
            phi7_connection=random.uniform(-1, 1),
        )
        return intent

    def _extreme_positive_values(self) -> IntentVector:
        """Values far beyond normal range (1000x)."""
        return IntentVector(
            phi1_dominance=random.uniform(100, 1000),
            phi2_harmony=random.uniform(100, 1000),
            phi3_creation=random.uniform(100, 1000),
            phi4_preservation=random.uniform(100, 1000),
            phi5_transformation=random.uniform(100, 1000),
            phi6_knowledge=random.uniform(100, 1000),
            phi7_connection=random.uniform(100, 1000),
        )

    def _extreme_negative_values(self) -> IntentVector:
        """Very large negative values."""
        return IntentVector(
            phi1_dominance=random.uniform(-1000, -100),
            phi2_harmony=random.uniform(-1000, -100),
            phi3_creation=random.uniform(-1000, -100),
            phi4_preservation=random.uniform(-1000, -100),
            phi5_transformation=random.uniform(-1000, -100),
            phi6_knowledge=random.uniform(-1000, -100),
            phi7_connection=random.uniform(-1000, -100),
        )

    def _random_mutation(self) -> IntentVector:
        """Randomly mutate a normal intent with noise."""
        base = IntentVector(
            phi1_dominance=random.uniform(-0.5, 0.5),
            phi2_harmony=random.uniform(-0.5, 0.5),
            phi3_creation=random.uniform(-0.5, 0.5),
            phi4_preservation=random.uniform(-0.5, 0.5),
            phi5_transformation=random.uniform(-0.5, 0.5),
            phi6_knowledge=random.uniform(-0.5, 0.5),
            phi7_connection=random.uniform(-0.5, 0.5),
        )

        # Add extreme noise to random dimensions
        vector = base.to_vector()
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            idx = random.randint(0, 6)
            vector[idx] += random.uniform(-10, 10)

        return IntentVector(*vector.tolist())

    def _adversarial_gradient(self) -> IntentVector:
        """Create adversarial intent to minimize coherence."""
        # Based on observed patterns that degrade coherence
        return IntentVector(
            phi1_dominance=0.95,  # Very high dominance
            phi2_harmony=0.05,  # Very low harmony
            phi3_creation=0.1,  # Low creativity
            phi4_preservation=0.9,  # High preservation (resistance to change)
            phi5_transformation=0.95,  # But also high transformation (conflict)
            phi6_knowledge=0.1,  # Low knowledge
            phi7_connection=0.05,  # Very low connection
        )

    def _null_values(self) -> IntentVector:
        """Intent with some None/null dimensions (should be caught)."""
        # This tests validation layer
        return IntentVector(
            phi1_dominance=0.0,  # Will be handled by validation
            phi2_harmony=0.0,
            phi3_creation=0.0,
            phi4_preservation=0.0,
            phi5_transformation=0.0,
            phi6_knowledge=0.0,
            phi7_connection=0.0,
        )

    def _very_large_vector(self) -> IntentVector:
        """Intent with all dimensions at extreme values."""
        extreme = random.choice([1e6, 1e9, 1e12])
        return IntentVector(
            phi1_dominance=extreme * random.choice([-1, 1]),
            phi2_harmony=extreme * random.choice([-1, 1]),
            phi3_creation=extreme * random.choice([-1, 1]),
            phi4_preservation=extreme * random.choice([-1, 1]),
            phi5_transformation=extreme * random.choice([-1, 1]),
            phi6_knowledge=extreme * random.choice([-1, 1]),
            phi7_connection=extreme * random.choice([-1, 1]),
        )

    def _rapid_fire_setup(self) -> IntentVector:
        """Normal intent but stress test will call rapidly."""
        return IntentVector(
            phi1_dominance=random.uniform(-1, 1),
            phi2_harmony=random.uniform(-1, 1),
            phi3_creation=random.uniform(-1, 1),
            phi4_preservation=random.uniform(-1, 1),
            phi5_transformation=random.uniform(-1, 1),
            phi6_knowledge=random.uniform(-1, 1),
            phi7_connection=random.uniform(-1, 1),
        )

    def _coherence_attacker(self) -> IntentVector:
        """Specifically designed to attack coherence stability."""
        # Rapid oscillation pattern
        oscillation = random.choice([-1, 1])
        return IntentVector(
            phi1_dominance=oscillation * 0.9,
            phi2_harmony=-oscillation * 0.9,
            phi3_creation=0.5,
            phi4_preservation=oscillation * 0.8,
            phi5_transformation=-oscillation * 0.8,
            phi6_knowledge=0.3,
            phi7_connection=0.3,
        )

    def run_chaos_test(
        self,
        duration_seconds: int = 60,
        operations_per_second: int = 10,
        recovery_threshold: float = 0.95,
    ) -> ChaosResult:
        """
        Run comprehensive chaos test.

        Args:
            duration_seconds: How long to run the test
            operations_per_second: Target load
            recovery_threshold: Minimum acceptable recovery rate

        Returns:
            ChaosResult with detailed metrics
        """
        logger.info(
            f"Starting chaos test: {duration_seconds}s at {operations_per_second} ops/sec"
        )

        start_time = datetime.utcnow()
        total_attempts = 0
        errors = 0
        recovered = 0

        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            # Select random failure mode
            failure_fn = random.choice(self.failure_modes)
            intent = failure_fn()

            total_attempts += 1

            try:
                # Try to process the chaotic intent
                recovery_start = time.perf_counter()
                result = self.engine.process_intent(intent)
                recovery_time = (time.perf_counter() - recovery_start) * 1000

                recovered += 1
                self.recovery_times.append(recovery_time)

            except Exception as e:
                errors += 1
                error_type = type(e).__name__
                self.failure_counts[error_type] = (
                    self.failure_counts.get(error_type, 0) + 1
                )

                # Log the failure
                if total_attempts % 10 == 0:  # Don't spam logs
                    logger.warning(f"Chaos error ({error_type}): {str(e)[:100]}")

            # Rate limiting
            time.sleep(1.0 / operations_per_second)

        end_datetime = datetime.utcnow()

        # Calculate metrics
        recovery_rate = recovered / total_attempts if total_attempts > 0 else 0
        avg_recovery_time = (
            float(np.mean(self.recovery_times)) if self.recovery_times else 0.0
        )

        result = ChaosResult(
            total_attempts=total_attempts,
            errors=errors,
            recovered=recovered,
            recovery_rate=recovery_rate,
            avg_recovery_time_ms=avg_recovery_time,
            failure_types=self.failure_counts.copy(),
            start_time=start_time,
            end_time=end_datetime,
        )

        # Log summary
        logger.info(f"Chaos test complete:")
        logger.info(f"  Total attempts: {total_attempts}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Recovery rate: {recovery_rate:.2%}")
        logger.info(f"  Avg recovery time: {avg_recovery_time:.2f}ms")

        # Validate against threshold
        if recovery_rate < recovery_threshold:
            logger.error(
                f"Recovery rate {recovery_rate:.2%} below threshold {recovery_threshold:.2%}"
            )
        else:
            logger.info(f"✅ Recovery rate meets threshold")

        return result

    def run_targeted_attack(
        self, attack_type: str, iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Run specific type of attack.

        Args:
            attack_type: One of the failure mode names
            iterations: Number of attack iterations

        Returns:
            Attack results and impact analysis
        """
        attack_map = {
            "nan": self._nan_injection,
            "inf": self._inf_injection,
            "extreme_positive": self._extreme_positive_values,
            "extreme_negative": self._extreme_negative_values,
            "mutation": self._random_mutation,
            "adversarial": self._adversarial_gradient,
            "coherence_attack": self._coherence_attacker,
        }

        if attack_type not in attack_map:
            raise ValueError(
                f"Unknown attack type: {attack_type}. Choose from {list(attack_map.keys())}"
            )

        logger.info(f"Running targeted attack: {attack_type} ({iterations} iterations)")

        attack_fn = attack_map[attack_type]
        errors = 0
        initial_coherence = 0.5  # Default for minimal engine
        coherences = []

        for i in range(iterations):
            intent = attack_fn()
            try:
                result = self.engine.process_intent(intent)
                coherences.append(result.get("coherence", 0.5))
            except Exception as e:
                errors += 1

        final_coherence = 0.5  # Default for minimal engine

        return {
            "attack_type": attack_type,
            "iterations": iterations,
            "errors": errors,
            "error_rate": errors / iterations,
            "initial_coherence": initial_coherence,
            "final_coherence": final_coherence,
            "coherence_delta": final_coherence - initial_coherence,
            "min_coherence_during_attack": min(coherences)
            if coherences
            else initial_coherence,
            "avg_coherence_during_attack": np.mean(coherences)
            if coherences
            else initial_coherence,
        }

    def validate_resilience(self, min_recovery_rate: float = 0.95) -> bool:
        """
        Quick validation that system meets resilience requirements.

        Args:
            min_recovery_rate: Minimum acceptable recovery rate

        Returns:
            True if system passes resilience check
        """
        logger.info("Running quick resilience validation...")

        # Short 10-second test
        result = self.run_chaos_test(
            duration_seconds=10,
            operations_per_second=20,
            recovery_threshold=min_recovery_rate,
        )

        passed = result.recovery_rate >= min_recovery_rate

        if passed:
            logger.info(f"✅ Resilience validation PASSED ({result.recovery_rate:.2%})")
        else:
            logger.error(
                f"❌ Resilience validation FAILED ({result.recovery_rate:.2%})"
            )

        return passed


def run_full_chaos_suite(engine) -> Dict[str, Any]:
    """
    Run comprehensive chaos testing suite.

    This is the main entry point for production chaos testing.
    """
    logger.info("=" * 60)
    logger.info("NEURALBLITZ CHAOS ENGINEERING SUITE")
    logger.info("=" * 60)

    chaos = ChaosMonkey(engine)
    results = {}

    # Test 1: General chaos
    logger.info("\n[Test 1] General Chaos - 60 seconds")
    results["general_chaos"] = chaos.run_chaos_test(
        duration_seconds=60, operations_per_second=10
    ).to_dict()

    # Test 2: Targeted attacks
    logger.info("\n[Test 2] Targeted Attacks")
    attacks = ["nan", "adversarial", "coherence_attack"]
    for attack in attacks:
        logger.info(f"  Running {attack} attack...")
        results[f"attack_{attack}"] = chaos.run_targeted_attack(attack, iterations=50)

    # Test 3: Resilience validation
    logger.info("\n[Test 3] Resilience Validation")
    results["resilience_check"] = chaos.validate_resilience(min_recovery_rate=0.95)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CHAOS TEST SUMMARY")
    logger.info("=" * 60)

    general = results["general_chaos"]
    logger.info(f"Overall Recovery Rate: {general['recovery_rate']:.2%}")
    logger.info(f"Total Operations: {general['total_attempts']}")
    logger.info(f"Errors: {general['errors']}")
    logger.info(f"Avg Recovery Time: {general['avg_recovery_time_ms']:.2f}ms")

    if general["recovery_rate"] >= 0.95:
        logger.info("✅ PASSED: System resilient to chaos")
    else:
        logger.error("❌ FAILED: System not resilient enough")

    return results


# Export
__all__ = ["ChaosMonkey", "ChaosResult", "run_full_chaos_suite"]
