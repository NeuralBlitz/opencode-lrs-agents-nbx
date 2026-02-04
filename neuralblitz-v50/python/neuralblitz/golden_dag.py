"""Backward-compatibility shim for golden_dag.py
Redirects to ComputationalAxioms.python.goldendag_core
Mathematical Coherence must remain 1.0

This file maintains the original import interface while delegating
to the new Three-Pillar architecture location.
"""

import sys
import os

# Add ComputationalAxioms to path if needed
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import from new Three-Pillar location
from ComputationalAxioms.python.goldendag_core import (
    GoldenDAG,
    NBHSCryptographicHash,
    TraceID,
    CodexID,
    SEED,
    generate_goldendag,
    generate_trace_id,
    generate_codex_id,
)

__version__ = "50.0.0"
__all__ = [
    "GoldenDAG",
    "NBHSCryptographicHash",
    "TraceID",
    "CodexID",
    "SEED",
    "generate_goldendag",
    "generate_trace_id",
    "generate_codex_id",
]

# Preserve original module name for backwards compatibility
__name__ = "neuralblitz.golden_dag"
