"""Computational Axioms - Mathematical Foundation Layer
NeuralBlitz v50.0 Three-Pillar Architecture - Pillar 1

This package provides the cryptographic and mathematical foundation for
the NeuralBlitz v50.0 ontological framework, including NBHS-1024 hashing,
GoldenDAG signatures, and identifier generation protocols.
"""

from .goldendag_core import (
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
__author__ = "NeuralBlitz Project"
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
