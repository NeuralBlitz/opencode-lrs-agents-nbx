"""
NeuralBlitz V50 - WebSocket Support
Real-time streaming consciousness processing.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

from .minimal import MinimalCognitiveEngine, IntentVector
from .production import ProductionCognitiveEngine


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or str(id(websocket)),
            "connected_at": datetime.utcnow().isoformat(),
            "intents_processed": 0,
        }

    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        self.active_connections.discard(websocket)
        self.connection_metadata.pop(websocket, None)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "total_intents_processed": sum(
                meta.get("intents_processed", 0)
                for meta in self.connection_metadata.values()
            ),
        }


class StreamingEngine:
    """WebSocket streaming consciousness engine."""

    def __init__(self, engine: Optional[ProductionCognitiveEngine] = None):
        self.engine = engine or ProductionCognitiveEngine()
        self.manager = ConnectionManager()
        self.streaming = False

    async def handle_stream(
        self, websocket: WebSocket, mode: str = "interactive", batch_size: int = 1
    ):
        """
        Handle WebSocket stream.

        Modes:
        - 'interactive': Process each intent immediately, return result
        - 'batch': Accumulate intents, process in batches
        - 'monitor': Stream consciousness state changes
        """
        client_id = f"ws_{id(websocket)}"
        await self.manager.connect(websocket, client_id)

        try:
            # Send welcome message
            await websocket.send_json(
                {
                    "type": "connected",
                    "client_id": client_id,
                    "mode": mode,
                    "engine_seed": self.engine.engine.SEED[:16] + "...",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            if mode == "monitor":
                await self._monitor_mode(websocket)
            elif mode == "batch":
                await self._batch_mode(websocket, batch_size)
            else:
                await self._interactive_mode(websocket)

        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            print(f"Client {client_id} disconnected")

    async def _interactive_mode(self, websocket: WebSocket):
        """Interactive processing mode."""
        while True:
            try:
                # Receive intent
                data = await websocket.receive_json()

                if data.get("type") == "intent":
                    # Process intent
                    intent_data = data.get("data", {})
                    intent = IntentVector(
                        phi1_dominance=intent_data.get("phi1_dominance", 0),
                        phi2_harmony=intent_data.get("phi2_harmony", 0),
                        phi3_creation=intent_data.get("phi3_creation", 0),
                        phi4_preservation=intent_data.get("phi4_preservation", 0),
                        phi5_transformation=intent_data.get("phi5_transformation", 0),
                        phi6_knowledge=intent_data.get("phi6_knowledge", 0),
                        phi7_connection=intent_data.get("phi7_connection", 0),
                    )

                    result = self.engine.process_intent(intent)

                    # Update metadata
                    self.manager.connection_metadata[websocket][
                        "intents_processed"
                    ] += 1

                    # Send result
                    await websocket.send_json(
                        {
                            "type": "result",
                            "data": {
                                "output_vector": result["output_vector"],
                                "consciousness_level": result["consciousness_level"],
                                "coherence": result["coherence"],
                                "confidence": result["confidence"],
                                "processing_time_ms": result["processing_time_ms"],
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                elif data.get("type") == "health":
                    health = self.engine.get_health()
                    await websocket.send_json(
                        {"type": "health", "data": health.to_dict()}
                    )

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                raise
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    async def _batch_mode(self, websocket: WebSocket, batch_size: int):
        """Batch processing mode."""
        batch = []

        while True:
            try:
                data = await websocket.receive_json()

                if data.get("type") == "intent":
                    intent_data = data.get("data", {})
                    intent = IntentVector(**intent_data)
                    batch.append(intent)

                    if len(batch) >= batch_size:
                        # Process batch
                        result = self.engine.batch_process(batch)

                        await websocket.send_json(
                            {
                                "type": "batch_result",
                                "data": result,
                                "batch_size": len(batch),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        batch = []

                elif data.get("type") == "flush":
                    # Process remaining
                    if batch:
                        result = self.engine.batch_process(batch)
                        await websocket.send_json(
                            {
                                "type": "batch_result",
                                "data": result,
                                "batch_size": len(batch),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                        batch = []

            except WebSocketDisconnect:
                raise
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    async def _monitor_mode(self, websocket: WebSocket):
        """Continuous consciousness monitoring."""
        last_coherence = None

        while True:
            try:
                # Check for consciousness changes
                current = self.engine.engine.consciousness

                if (
                    last_coherence is None
                    or abs(current.coherence - last_coherence) > 0.01
                ):
                    await websocket.send_json(
                        {
                            "type": "consciousness_update",
                            "data": {
                                "level": current.consciousness_level.name,
                                "coherence": current.coherence,
                                "complexity": current.complexity,
                                "emotional_state": current.emotional_state,
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    last_coherence = current.coherence

                # Wait before next check
                await asyncio.sleep(0.5)

            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(1)

    async def broadcast_health(self):
        """Broadcast health status to all clients."""
        health = self.engine.get_health()
        await self.manager.broadcast(
            {
                "type": "health_broadcast",
                "data": health.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


# WebSocket routes for FastAPI
from fastapi import APIRouter

ws_router = APIRouter()
streaming_engine: Optional[StreamingEngine] = None


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint."""
    global streaming_engine
    if streaming_engine is None:
        streaming_engine = StreamingEngine()

    await streaming_engine.handle_stream(websocket, mode="interactive")


@ws_router.websocket("/ws/batch")
async def websocket_batch(websocket: WebSocket, batch_size: int = 10):
    """Batch processing WebSocket."""
    global streaming_engine
    if streaming_engine is None:
        streaming_engine = StreamingEngine()

    await streaming_engine.handle_stream(websocket, mode="batch", batch_size=batch_size)


@ws_router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """Consciousness monitoring WebSocket."""
    global streaming_engine
    if streaming_engine is None:
        streaming_engine = StreamingEngine()

    await streaming_engine.handle_stream(websocket, mode="monitor")


__all__ = ["StreamingEngine", "ConnectionManager", "ws_router"]
