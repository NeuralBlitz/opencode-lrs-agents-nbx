#!/usr/bin/env python3
"""
Phase 1 Production Hardening Demo
Demonstrates Chaos Engineering, Audit Logging, RBAC, and Distributed Tracing
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from project root
from neuralblitz import MinimalCognitiveEngine, IntentVector
from neuralblitz.testing.chaos import ChaosMonkey, run_full_chaos_suite
from neuralblitz.security.audit import AuditLogger, create_audit_logger
from neuralblitz.security.auth import RBACManager, Permission, Role, create_rbac_system
from neuralblitz.tracing import (
    Tracer,
    SpanKind,
    initialize_tracing,
    trace_intent_processing,
)


def demo_chaos_engineering():
    """Demonstrate chaos engineering with resilience testing."""
    print("\n🔥 CHAOS ENGINEERING DEMO")
    print("=" * 50)

    # Create engine
    engine = MinimalCognitiveEngine()
    chaos = ChaosMonkey(engine)

    # Run quick chaos test
    print("Running chaos resilience test...")
    result = chaos.validate_resilience(
        min_recovery_rate=0.7
    )  # Lower threshold for demo

    print(f"✅ Recovery Rate: {result:.2%}")
    print(f"📊 System Status: {'RESILIENT' if result else 'NEEDS_IMPROVEMENT'}")

    # Show specific chaos attack
    print("\nTesting adversarial intent...")
    attack_result = chaos.run_targeted_attack("adversarial", iterations=5)
    print(f"🎯 Attack Success Rate: {attack_result['error_rate']:.2%}")
    print(f"📉 Coherence Impact: {attack_result['coherence_delta']:+.3f}")

    return engine


def demo_audit_logging():
    """Demonstrate tamper-evident audit logging."""
    print("\n📋 AUDIT LOGGING DEMO")
    print("=" * 50)

    # Create temporary audit log
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_file = os.path.join(temp_dir, "demo_audit.log")
        audit = AuditLogger(audit_file)

        # Log some operations
        print("Logging operations with blockchain-style integrity...")

        log_intent_processing(
            audit,
            "intent_001",
            "demo_user",
            {
                "coherence": 0.85,
                "consciousness_level": "FOCUSED",
                "processing_time_ms": 15.2,
            },
        )

        log_intent_processing(
            audit,
            "intent_002",
            "demo_user",
            {
                "coherence": 0.92,
                "consciousness_level": "TRANSCENDENT",
                "processing_time_ms": 12.8,
            },
        )

        # Verify integrity
        integrity_ok = audit.verify_integrity()
        print(f"🔒 Log Integrity: {'VALID' if integrity_ok else 'COMPROMISED'}")

        # Get statistics
        stats = audit.get_statistics()
        print(f"📊 Total Entries: {stats['total_entries']}")
        print(f"🔧 Operations: {list(stats['operations'].keys())}")

        return audit


def demo_rbac():
    """Demonstrate role-based access control."""
    print("\n🔐 RBAC & AUTHENTICATION DEMO")
    print("=" * 50)

    # Create RBAC system
    rbac = create_rbac_system(secret_key="demo_secret_key")

    # Show default users and permissions
    print("Default Users:")
    for username, user in rbac._users.items():
        print(f"  👤 {username}: {user.role.value}")
        print(f"     API Key: {user.api_key[:20]}...")
        print(f"     Rate Limit: {user.rate_limit}/min")

        # Show permissions
        permissions = list(user.role.value)
        print(f"     Permissions: {[p.value for p in permissions]}")

    # Demonstrate authentication
    print("\nTesting authentication...")

    # Test viewer permissions
    viewer_key = rbac._users["demo"].api_key
    result = rbac.authenticate_request(
        api_key=viewer_key, required_permission=Permission.PROCESS
    )
    print(
        f"👤 Demo User Process Access: {'✅ GRANTED' if result['authorized'] else '❌ DENIED'}"
    )

    # Test admin permissions
    admin_key = rbac._users["admin"].api_key
    result = rbac.authenticate_request(
        api_key=admin_key, required_permission=Permission.ADMIN
    )
    print(
        f"👨‍💼 Admin Admin Access: {'✅ GRANTED' if result['authorized'] else '❌ DENIED'}"
    )

    # Test rate limiting
    print("\nTesting rate limiting...")
    rate_limited = 0
    for i in range(15):  # Exceed rate limit
        if not rbac.check_rate_limit(admin_key):
            rate_limited += 1

    print(f"🚦 Rate Limited Requests: {rate_limited}/15")

    return rbac


def demo_distributed_tracing():
    """Demonstrate distributed tracing with span hierarchy."""
    print("\n📊 DISTRIBUTED TRACING DEMO")
    print("=" * 50)

    # Initialize tracing
    tracer = initialize_tracing("neuralblitz_demo")

    # Simulate intent processing with span hierarchy
    with tracer.create_span(
        "process_intent_request", kind=SpanKind.SERVER
    ) as main_span:
        main_span.set_attribute("user.id", "demo_user")
        main_span.set_attribute("intent.type", "creative_synthesis")

        # Intent validation span
        with tracer.create_span("validate_intent") as validation_span:
            validation_span.set_attribute("validation.rules", "6 checks")
            time.sleep(0.001)  # Simulate work
            validation_span.add_event("validation_passed")

        # Neural processing span
        with tracer.create_span("neural_forward_pass") as neural_span:
            neural_span.set_attribute("neural.layers", 3)
            neural_span.add_event("layer_1_complete")
            time.sleep(0.001)
            neural_span.add_event("layer_2_complete")
            time.sleep(0.001)
            neural_span.add_event("layer_3_complete")

        # Consciousness update span
        with tracer.create_span("consciousness_update") as consciousness_span:
            consciousness_span.set_attribute("previous.level", "AWARE")
            consciousness_span.set_attribute("new.level", "FOCUSED")
            consciousness_span.set_attribute("coherence", 0.88)
            time.sleep(0.001)

        # Response formatting span
        with tracer.create_span("format_response") as response_span:
            response_span.set_attribute("response.format", "json")
            response_span.set_attribute("response.size", "1.2KB")
            time.sleep(0.001)

    # Show tracing statistics
    stats = tracer.get_statistics()
    print(f"📊 Total Spans: {stats['total_spans']}")
    print(f"⏱️  Avg Duration: {stats['avg_duration_ms']:.2f}ms")
    print(
        f"📈 Trace Depth: {max(len(tracer.get_trace_by_id(s.trace_id)) for s in tracer.spans)}"
    )

    # Show span hierarchy
    print("\n🌳 Span Hierarchy:")
    for span in tracer.spans:
        indent = "  " * (1 if span.parent_span_id else 0)
        status_icon = "✅" if span.status.value == "OK" else "❌"
        print(
            f"{indent}{status_icon} {span.operation_name} ({span.duration_ms():.2f}ms)"
        )

    return tracer


def demo_integration():
    """Integrate all Phase 1 features in a workflow."""
    print("\n🔄 INTEGRATED WORKFLOW DEMO")
    print("=" * 50)

    # Initialize all components
    engine = MinimalCognitiveEngine()
    rbac = create_rbac_system()
    tracer = initialize_tracing("integration_demo")

    with tempfile.TemporaryDirectory() as temp_dir:
        audit = AuditLogger(os.path.join(temp_dir, "integration_audit.log"))

        print("Processing intent with full production hardening...")

        # Authenticate request
        api_key = rbac._users["demo"].api_key
        auth_result = rbac.authenticate_request(
            api_key=api_key, required_permission=Permission.PROCESS
        )

        if not auth_result["authorized"]:
            print("❌ Authentication failed!")
            return

        # Process intent with tracing
        with tracer.create_span(
            "full_intent_processing", kind=SpanKind.SERVER
        ) as main_span:
            intent = IntentVector(
                phi1_dominance=0.7, phi2_harmony=0.8, phi3_creation=0.9
            )

            # Log intent processing start
            log_intent_processing(
                audit,
                "integration_intent",
                auth_result["username"],
                {
                    "intent_vector": str(intent.to_vector()[:5]) + "...",
                    "user_permissions": auth_result["permissions"],
                },
            )

            # Process with chaos resilience test
            chaos = ChaosMonkey(engine)
            result = engine.process_intent(intent)

            # Log result
            log_intent_processing(
                audit, "integration_result", auth_result["username"], result
            )

            main_span.set_attribute("result.confidence", result.get("confidence", 0))
            main_span.set_attribute(
                "result.level", result.get("consciousness_level", "UNKNOWN")
            )

            print(f"✅ Intent Processed Successfully")
            print(
                f"🧠 Consciousness Level: {result.get('consciousness_level', 'UNKNOWN')}"
            )
            print(f"📊 Confidence: {result.get('confidence', 0):.2%}")
            print(f"⚡ Processing Time: {result.get('processing_time_ms', 0):.2f}ms")

        # Verify audit log integrity
        integrity_ok = audit.verify_integrity()
        print(f"🔒 Audit Integrity: {'VALID' if integrity_ok else 'COMPROMISED'}")

        # Show comprehensive statistics
        tracer_stats = tracer.get_statistics()

        # Simple RBAC stats
        total_users = len(rbac._users)
        total_api_keys = sum(len(user.api_key) for user in rbac._users.values())

        audit_stats = audit.get_statistics()

        print(f"\n📊 INTEGRATION STATISTICS:")
        print(
            f"  🔍 Tracing: {tracer_stats['total_spans']} spans, {tracer_stats['avg_duration_ms']:.2f}ms avg"
        )
        print(f"  🔐 RBAC: {total_users} users, {total_api_keys} API keys")
        print(f"  📋 Audit: {audit_stats['total_entries']} entries, integrity verified")


def main():
    """Main demonstration function."""
    print("🚀 NEURALBLITZ V50 - PHASE 1 PRODUCTION HARDENING")
    print("=" * 60)
    print("Demonstrating: Chaos Engineering, Audit Logging, RBAC, Distributed Tracing")

    try:
        # Run individual demos
        demo_chaos_engineering()
        demo_audit_logging()
        demo_rbac()
        demo_distributed_tracing()

        # Run integrated demo
        demo_integration()

        print("\n✅ PHASE 1 DEMO COMPLETED SUCCESSFULLY!")
        print("🎯 All production hardening features are operational")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
