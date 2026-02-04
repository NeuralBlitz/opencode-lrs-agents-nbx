"""
NeuralBlitz V50 - Serialization & Persistence
State saving/loading and memory-mapped storage.
"""

import pickle
import json
import mmap
import os
from typing import Dict, Any, Optional, BinaryIO
from pathlib import Path
from dataclasses import asdict
import numpy as np
import struct

from .minimal import MinimalCognitiveEngine, IntentVector, ConsciousnessModel
from .production import ProductionCognitiveEngine


class EngineSerializer:
    """Serialize and deserialize engine state."""

    @staticmethod
    def to_dict(engine: MinimalCognitiveEngine) -> Dict[str, Any]:
        """Convert engine state to dictionary."""
        return {
            "version": "50.0.0-minimal",
            "seed": engine.SEED,
            "consciousness": asdict(engine.consciousness),
            "pattern_memory": [
                {
                    "id": p["id"],
                    "timestamp": p["timestamp"].isoformat()
                    if hasattr(p["timestamp"], "isoformat")
                    else str(p["timestamp"]),
                    "input_hash": p["input_hash"],
                    "output_vector": p["output_vector"].tolist(),
                    "confidence": p["confidence"],
                }
                for p in engine.pattern_memory
            ],
            "processing_count": engine.processing_count,
            "weights": {k: v.tolist() for k, v in engine.weights.items()},
            "biases": {k: v.tolist() for k, v in engine.biases.items()},
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> MinimalCognitiveEngine:
        """Create engine from dictionary."""
        engine = MinimalCognitiveEngine()

        # Restore consciousness
        engine.consciousness = ConsciousnessModel(**data["consciousness"])

        # Restore patterns (simplified)
        engine.pattern_memory = []
        for p in data.get("pattern_memory", []):
            engine.pattern_memory.append(
                {
                    "id": p["id"],
                    "timestamp": p["timestamp"],
                    "input_hash": p["input_hash"],
                    "output_vector": np.array(p["output_vector"]),
                    "confidence": p["confidence"],
                }
            )

        # Restore counts
        engine.processing_count = data.get("processing_count", 0)

        # Restore weights and biases
        for key, value in data.get("weights", {}).items():
            engine.weights[key] = np.array(value)
        for key, value in data.get("biases", {}).items():
            engine.biases[key] = np.array(value)

        return engine

    @staticmethod
    def save_json(engine: MinimalCognitiveEngine, filepath: str) -> None:
        """Save engine state to JSON."""
        data = EngineSerializer.to_dict(engine)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(filepath: str) -> MinimalCognitiveEngine:
        """Load engine state from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return EngineSerializer.from_dict(data)

    @staticmethod
    def save_pickle(engine: MinimalCognitiveEngine, filepath: str) -> None:
        """Save engine state using pickle (faster, smaller)."""
        data = {
            "consciousness": engine.consciousness,
            "pattern_memory": engine.pattern_memory,
            "processing_count": engine.processing_count,
            "weights": engine.weights,
            "biases": engine.biases,
            "seed": engine.SEED,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(filepath: str) -> MinimalCognitiveEngine:
        """Load engine state from pickle."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        engine = MinimalCognitiveEngine()
        engine.consciousness = data["consciousness"]
        engine.pattern_memory = data["pattern_memory"]
        engine.processing_count = data["processing_count"]
        engine.weights = data["weights"]
        engine.biases = data["biases"]

        return engine


class MemoryMappedStorage:
    """Memory-mapped pattern storage for large-scale applications."""

    def __init__(self, filepath: str, max_patterns: int = 10000):
        self.filepath = Path(filepath)
        self.max_patterns = max_patterns
        self.pattern_size = 7 * 8 + 8 + 4  # 7 doubles + 1 long + 1 float
        self.header_size = 16  # version + count

        self._ensure_file()

    def _ensure_file(self):
        """Create file if it doesn't exist."""
        if not self.filepath.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

            # Create file with header
            with open(self.filepath, "wb") as f:
                # Version (4 bytes)
                f.write(struct.pack("I", 1))
                # Pattern count (4 bytes)
                f.write(struct.pack("I", 0))
                # Reserved (8 bytes)
                f.write(b"\x00" * 8)

                # Pre-allocate space for patterns
                for _ in range(self.max_patterns):
                    f.write(b"\x00" * self.pattern_size)

    def append_pattern(
        self, output_vector: np.ndarray, confidence: float, pattern_id: int
    ) -> bool:
        """
        Append pattern to memory-mapped storage.

        Returns True if successful, False if storage is full.
        """
        with open(self.filepath, "r+b") as f:
            # Read current count
            f.seek(4)
            count = struct.unpack("I", f.read(4))[0]

            if count >= self.max_patterns:
                return False

            # Write pattern at position
            offset = self.header_size + (count * self.pattern_size)
            f.seek(offset)

            # Write output vector (7 doubles)
            for val in output_vector[:7]:
                f.write(struct.pack("d", float(val)))

            # Write pattern ID (uint64)
            f.write(struct.pack("Q", pattern_id))

            # Write confidence (float32)
            f.write(struct.pack("f", confidence))

            # Update count
            f.seek(4)
            f.write(struct.pack("I", count + 1))

        return True

    def get_pattern(self, index: int) -> Optional[Dict[str, Any]]:
        """Retrieve pattern by index."""
        with open(self.filepath, "rb") as f:
            # Read count
            f.seek(4)
            count = struct.unpack("I", f.read(4))[0]

            if index >= count:
                return None

            # Read pattern
            offset = self.header_size + (index * self.pattern_size)
            f.seek(offset)

            # Read vector
            vector = np.array([struct.unpack("d", f.read(8))[0] for _ in range(7)])

            # Read ID
            pattern_id = struct.unpack("Q", f.read(8))[0]

            # Read confidence
            confidence = struct.unpack("f", f.read(4))[0]

            return {
                "index": index,
                "output_vector": vector,
                "pattern_id": pattern_id,
                "confidence": confidence,
            }

    def get_count(self) -> int:
        """Get number of stored patterns."""
        with open(self.filepath, "rb") as f:
            f.seek(4)
            return struct.unpack("I", f.read(4))[0]

    def clear(self):
        """Clear all patterns."""
        with open(self.filepath, "r+b") as f:
            f.seek(4)
            f.write(struct.pack("I", 0))


class PersistentEngine:
    """Engine with automatic persistence."""

    def __init__(
        self,
        persistence_path: str = "./neuralblitz_state.pkl",
        auto_save_interval: int = 100,
    ):
        self.persistence_path = Path(persistence_path)
        self.auto_save_interval = auto_save_interval
        self.operations_since_save = 0

        # Load or create
        if self.persistence_path.exists():
            self.engine = EngineSerializer.load_pickle(str(self.persistence_path))
            print(f"Loaded engine state from {persistence_path}")
        else:
            self.engine = MinimalCognitiveEngine()
            print("Created new engine")

    def process_intent(self, intent: IntentVector) -> Dict[str, Any]:
        """Process intent with automatic persistence."""
        result = self.engine.process_intent(intent)

        self.operations_since_save += 1
        if self.operations_since_save >= self.auto_save_interval:
            self.save()
            self.operations_since_save = 0

        return result

    def save(self):
        """Manually trigger save."""
        EngineSerializer.save_pickle(self.engine, str(self.persistence_path))
        print(f"Engine state saved to {self.persistence_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure save."""
        self.save()


__all__ = ["EngineSerializer", "MemoryMappedStorage", "PersistentEngine"]
