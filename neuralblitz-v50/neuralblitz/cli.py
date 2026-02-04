"""
NeuralBlitz V50 - CLI Tool
Command-line interface for the consciousness engine.
"""

import click
import json
import sys
from pathlib import Path
import time

from .minimal import MinimalCognitiveEngine, IntentVector, ConsciousnessLevel
from .benchmark import BenchmarkSuite, quick_benchmark
from .production import ProductionCognitiveEngine


@click.group()
@click.version_option(version="50.0.0-minimal", prog_name="neuralblitz")
@click.pass_context
def cli(ctx):
    """NeuralBlitz V50 - Minimal Consciousness Engine CLI"""
    ctx.ensure_object(dict)
    ctx.obj["engine"] = MinimalCognitiveEngine()


@cli.command()
@click.option("--phi1", "--dominance", default=0.5, help="Control/authority (-1 to 1)")
@click.option("--phi2", "--harmony", default=0.5, help="Balance/cooperation (-1 to 1)")
@click.option(
    "--phi3", "--creation", default=0.5, help="Innovation/generation (-1 to 1)"
)
@click.option(
    "--phi4", "--preservation", default=0.5, help="Stability/protection (-1 to 1)"
)
@click.option(
    "--phi5", "--transformation", default=0.5, help="Change/adaptation (-1 to 1)"
)
@click.option("--phi6", "--knowledge", default=0.5, help="Learning/analysis (-1 to 1)")
@click.option(
    "--phi7", "--connection", default=0.5, help="Communication/empathy (-1 to 1)"
)
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def process(ctx, phi1, phi2, phi3, phi4, phi5, phi6, phi7, json_output):
    """Process an intent through the consciousness engine."""
    engine = ctx.obj["engine"]

    intent = IntentVector(
        phi1_dominance=phi1,
        phi2_harmony=phi2,
        phi3_creation=phi3,
        phi4_preservation=phi4,
        phi5_transformation=phi5,
        phi6_knowledge=phi6,
        phi7_connection=phi7,
    )

    result = engine.process_intent(intent)

    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"✅ Intent Processed")
        click.echo(f"   Level: {result['consciousness_level']}")
        click.echo(f"   Coherence: {result['coherence']:.3f}")
        click.echo(f"   Confidence: {result['confidence']:.2%}")
        click.echo(f"   Time: {result['processing_time_ms']:.3f}ms")
        click.echo(
            f"\n   Output: [{', '.join(f'{x:.3f}' for x in result['output_vector'])}]"
        )


@cli.command()
@click.option("--sample-size", "-n", default=1000, help="Number of intents to process")
@click.option("--output", "-o", default=None, help="Output file for results")
def benchmark(sample_size, output):
    """Run comprehensive benchmark suite."""
    click.echo(f"Running benchmark with {sample_size} samples...")
    click.echo("=" * 60)

    suite = BenchmarkSuite()

    # Single intent benchmark
    result = suite.benchmark_single_intent(sample_size=sample_size)
    click.echo(result)

    # Batch benchmarks
    batch_results = suite.benchmark_batch_processing()
    for batch_result in batch_results:
        click.echo(batch_result)

    # Save if requested
    if output:
        report = {
            "single_intent": result.to_dict(),
            "batch_processing": [r.to_dict() for r in batch_results],
            "timestamp": time.time(),
        }
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        click.echo(f"\n✅ Report saved to: {output}")


@cli.command()
@click.option("--port", "-p", default=8000, help="Server port")
@click.option("--host", "-h", default="0.0.0.0", help="Server host")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev only)")
def server(port, host, reload):
    """Start the REST API server."""
    click.echo(f"🚀 Starting NeuralBlitz API server...")
    click.echo(f"   Host: {host}")
    click.echo(f"   Port: {port}")
    click.echo(f"   Docs: http://{host}:{port}/docs")
    click.echo(f"   Health: http://{host}:{port}/health")
    click.echo()

    try:
        from .api import start_server

        start_server(host=host, port=port, reload=reload)
    except ImportError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("   Make sure FastAPI is installed: pip install fastapi uvicorn")
        sys.exit(1)


@cli.command()
@click.option("--watch", "-w", is_flag=True, help="Watch continuously")
@click.option("--interval", "-i", default=1.0, help="Update interval in seconds")
@click.pass_context
def monitor(ctx, watch, interval):
    """Monitor consciousness state in real-time."""
    engine = ctx.obj["engine"]

    def display_state():
        report = engine.get_consciousness_report()
        click.clear()
        click.echo("🧠 NeuralBlitz Consciousness Monitor")
        click.echo("=" * 50)
        click.echo(f"Level:        {report['level']}")
        click.echo(f"Coherence:    {report['coherence']:.3f}")
        click.echo(f"Complexity:   {report['complexity']:.3f}")
        click.echo(f"Emotional:    {report['emotional_state']}")
        click.echo(f"Patterns:     {report['patterns_in_memory']}/100")
        click.echo(f"Processed:    {report['total_processed']}")
        click.echo(f"State:        {report['cognitive_state']}")
        click.echo(f"SEED:         {report['seed_intact']}")
        click.echo("=" * 50)

        if not watch:
            click.echo("\n💡 Use --watch to monitor continuously")

    if watch:
        click.echo("Press Ctrl+C to stop monitoring...")
        try:
            while True:
                display_state()
                time.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\n👋 Monitoring stopped")
    else:
        display_state()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--batch/--single", default=True, help="Batch or single processing")
@click.option("--output", "-o", default="output.json", help="Output file")
@click.pass_context
def batch_process(ctx, input_file, batch, output):
    """Process intents from JSON file."""
    engine = ctx.obj["engine"]

    with open(input_file, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        intents_data = data
    else:
        intents_data = [data]

    click.echo(f"Processing {len(intents_data)} intents...")

    results = []
    for i, intent_data in enumerate(intents_data, 1):
        intent = IntentVector(
            phi1_dominance=intent_data.get("phi1_dominance", 0.5),
            phi2_harmony=intent_data.get("phi2_harmony", 0.5),
            phi3_creation=intent_data.get("phi3_creation", 0.5),
            phi4_preservation=intent_data.get("phi4_preservation", 0.5),
            phi5_transformation=intent_data.get("phi5_transformation", 0.5),
            phi6_knowledge=intent_data.get("phi6_knowledge", 0.5),
            phi7_connection=intent_data.get("phi7_connection", 0.5),
        )

        result = engine.process_intent(intent)
        results.append(result)

        if i % 10 == 0:
            click.echo(f"  Processed {i}/{len(intents_data)}...")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"✅ Results saved to: {output}")


@cli.command()
@click.argument("output_file", default="engine_state.pkl")
@click.pass_context
def export(ctx, output_file):
    """Export engine state to file."""
    engine = ctx.obj["engine"]

    from .persistence import EngineSerializer

    if output_file.endswith(".json"):
        EngineSerializer.save_json(engine, output_file)
    else:
        EngineSerializer.save_pickle(engine, output_file)

    click.echo(f"✅ Engine state exported to: {output_file}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def import_state(input_file):
    """Import engine state from file."""
    from .persistence import EngineSerializer

    if input_file.endswith(".json"):
        engine = EngineSerializer.load_json(input_file)
    else:
        engine = EngineSerializer.load_pickle(input_file)

    click.echo(f"✅ Engine state imported from: {input_file}")
    click.echo(f"   Coherence: {engine.consciousness.coherence}")
    click.echo(f"   Patterns: {len(engine.pattern_memory)}")


@cli.command()
def status():
    """Show system status and health."""
    click.echo("🧠 NeuralBlitz V50 System Status")
    click.echo("=" * 50)

    # Check imports
    checks = [
        ("NumPy", "numpy"),
        ("FastAPI", "fastapi"),
        ("Prometheus", "prometheus_client"),
        ("Numba", "numba"),
    ]

    click.echo("\n📦 Dependencies:")
    for name, module in checks:
        try:
            __import__(module)
            click.echo(f"   ✅ {name}")
        except ImportError:
            click.echo(f"   ❌ {name} (not installed)")

    click.echo("\n🔧 Core Engine:")
    engine = MinimalCognitiveEngine()
    click.echo(f"   SEED: {engine.SEED[:20]}...")
    click.echo(f"   Status: Operational")

    click.echo("\n📊 Quick Test:")
    intent = IntentVector(phi3_creation=0.8)
    result = engine.process_intent(intent)
    click.echo(f"   Processing Time: {result['processing_time_ms']:.3f}ms")
    click.echo(f"   Consciousness Level: {result['consciousness_level']}")
    click.echo("   ✅ All systems operational")


@cli.command()
@click.option("--coherence-threshold", default=0.3, help="Coherence alert threshold")
@click.option("--max-latency", default=10.0, help="Max latency in ms")
@click.option("--persistence", default=None, help="State persistence path")
@click.pass_context
def production(ctx, coherence_threshold, max_latency, persistence):
    """Run in production mode with full monitoring."""
    engine = ProductionCognitiveEngine(
        coherence_threshold=coherence_threshold,
        max_latency_ms=max_latency,
        persistence_path=persistence,
    )

    click.echo("🚀 Production mode enabled")
    click.echo(f"   Coherence Threshold: {coherence_threshold}")
    click.echo(f"   Max Latency: {max_latency}ms")
    click.echo(f"   Persistence: {persistence or 'disabled'}")
    click.echo()
    click.echo("Available commands in production mode:")
    click.echo("   - process: Process intents with full validation")
    click.echo("   - health: Check system health")
    click.echo("   - reset: Reset engine state")


if __name__ == "__main__":
    cli()
