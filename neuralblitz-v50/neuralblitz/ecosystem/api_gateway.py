"""
NeuralBlitz V50 - Ecosystem API Gateway
REST and WebSocket endpoints for ecosystem access.
"""

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import asyncio
import json
from typing import Dict, Any, Optional, List
import logging

from .protocol import ComponentType, MessageType, Message, Protocol
from .orchestrator import EcosystemOrchestrator, WorkflowTemplates
from ..security.auth import RBACManager, Permission

logger = logging.getLogger("NeuralBlitz.EcosystemAPI")

if FASTAPI_AVAILABLE:
    security = HTTPBearer()


class EcosystemAPI:
    """
    FastAPI-based API gateway for the ecosystem.

    Provides:
    - REST endpoints for component interaction
    - WebSocket for real-time bidirectional streaming
    - Authentication and authorization
    - Workflow management
    - Health monitoring
    """

    def __init__(
        self,
        orchestrator: EcosystemOrchestrator,
        rbac: Optional[RBACManager] = None,
        enable_cors: bool = True,
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        self.orchestrator = orchestrator
        self.rbac = rbac
        self.app = FastAPI(
            title="NeuralBlitz Ecosystem API",
            description="Unified API for LRS Agents, OpenCode, NeuralBlitz, and ecosystem components",
            version="50.0.0",
        )

        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        self._setup_routes()
        self._active_websockets: Dict[str, WebSocket] = {}

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.get("/")
        async def root():
            return {
                "name": "NeuralBlitz Ecosystem API",
                "version": "50.0.0",
                "components": [
                    "lrs_agent",
                    "opencode",
                    "neuralblitz",
                    "advanced_research",
                    "computational_axioms",
                    "emergent_prompt",
                ],
                "endpoints": {
                    "rest": "/api/v1",
                    "websocket": "/ws",
                    "health": "/health",
                },
            }

        @self.app.get("/health")
        async def health():
            return self.orchestrator.get_health()

        @self.app.post("/api/v1/{component}/process")
        async def process_component(
            component: str,
            data: Dict[str, Any],
            credentials: HTTPAuthorizationCredentials = Depends(security)
            if self.rbac
            else None,
        ):
            """Process data through a specific component."""
            if self.rbac and credentials:
                result = self.rbac.authenticate_request(
                    jwt_token=credentials.credentials,
                    required_permission=Permission.PROCESS,
                )
                if not result["authenticated"] or not result["authorized"]:
                    raise HTTPException(
                        status_code=403, detail=result.get("error", "Unauthorized")
                    )

            try:
                component_type = ComponentType(component)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Unknown component: {component}"
                )

            # Create and send message
            message = Protocol.create_process_message(
                source=ComponentType.GATEWAY, target=component_type, data=data
            )

            response = await self.orchestrator.send_to_component(
                component_type, message
            )

            if response:
                return {
                    "status": "success",
                    "component": component,
                    "result": response.payload.get("result"),
                    "processing_time_ms": response.payload.get("processing_time_ms", 0),
                }
            else:
                raise HTTPException(
                    status_code=503, detail=f"Component {component} not available"
                )

        @self.app.post("/api/v1/{component}/query")
        async def query_component(
            component: str, query: str, parameters: Optional[Dict[str, Any]] = None
        ):
            """Query a component."""
            try:
                component_type = ComponentType(component)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Unknown component: {component}"
                )

            message = Protocol.create_query_message(
                source=ComponentType.GATEWAY,
                target=component_type,
                query=query,
                parameters=parameters or {},
            )

            response = await self.orchestrator.send_to_component(
                component_type, message
            )

            if response:
                return {
                    "status": "success",
                    "component": component,
                    "result": response.payload.get("result"),
                }
            else:
                raise HTTPException(
                    status_code=503, detail="No response from component"
                )

        @self.app.get("/api/v1/components")
        async def list_components():
            """List all registered components."""
            health = self.orchestrator.get_health()
            return {
                "components": health["registered_components"],
                "total_services": sum(health["registered_components"].values()),
            }

        @self.app.post("/api/v1/workflows/research-and-code")
        async def workflow_research_code(research_topic: str, coding_task: str):
            """Execute research then code workflow."""
            workflow = WorkflowTemplates.research_and_code(research_topic, coding_task)
            result = await self.orchestrator.execute_workflow(workflow)

            return {
                "workflow_id": workflow.id,
                "status": workflow.status,
                "steps_completed": workflow.current_step + 1,
                "total_steps": len(workflow.steps),
                "results": workflow.results,
            }

        @self.app.post("/api/v1/workflows/conscious-coding")
        async def workflow_conscious_coding(requirements: str):
            """Execute consciousness-guided coding workflow."""
            workflow = WorkflowTemplates.consciousness_guided_coding(requirements)
            result = await self.orchestrator.execute_workflow(workflow)

            return {
                "workflow_id": workflow.id,
                "status": workflow.status,
                "steps": [
                    {"step": i + 1, "component": step["component"], "result": res}
                    for i, (step, res) in enumerate(
                        zip(workflow.steps, workflow.results)
                    )
                ],
            }

        @self.app.post("/api/v1/broadcast")
        async def broadcast_message(
            payload: Dict[str, Any], exclude: Optional[List[str]] = None
        ):
            """Broadcast message to all components."""
            exclude_types = [ComponentType(e) for e in (exclude or [])]
            await self.orchestrator.broadcast(payload, exclude=exclude_types)
            return {"status": "broadcast_sent", "exclude": exclude or []}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for bidirectional streaming."""
            await websocket.accept()
            client_id = f"ws_{id(websocket)}"
            self._active_websockets[client_id] = websocket

            try:
                # Send welcome
                await websocket.send_json(
                    {
                        "type": "connected",
                        "client_id": client_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                # Handle messages
                while True:
                    try:
                        data = await websocket.receive_json()

                        # Parse message
                        if "component" in data and "payload" in data:
                            component_type = ComponentType(data["component"])
                            message = Message(
                                source=ComponentType.GATEWAY,
                                target=component_type,
                                msg_type=MessageType(data.get("type", "process")),
                                payload=data["payload"],
                            )

                            # Send to component
                            response = await self.orchestrator.send_to_component(
                                component_type, message
                            )

                            # Send response back
                            if response:
                                await websocket.send_json(
                                    {
                                        "type": "response",
                                        "component": data["component"],
                                        "result": response.payload,
                                        "timestamp": datetime.utcnow().isoformat(),
                                    }
                                )
                            else:
                                await websocket.send_json(
                                    {
                                        "type": "error",
                                        "error": "No response from component",
                                        "component": data["component"],
                                    }
                                )
                        else:
                            await websocket.send_json(
                                {"type": "error", "error": "Invalid message format"}
                            )

                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        await websocket.send_json({"type": "error", "error": str(e)})

            finally:
                del self._active_websockets[client_id]

        @self.app.websocket("/ws/stream/{component}")
        async def websocket_stream(websocket: WebSocket, component: str):
            """WebSocket for streaming to specific component."""
            await websocket.accept()

            try:
                component_type = ComponentType(component)
                stream_id = await self.orchestrator.create_stream(
                    ComponentType.GATEWAY, component_type
                )

                if not stream_id:
                    await websocket.send_json({"error": "Failed to create stream"})
                    await websocket.close()
                    return

                await websocket.send_json(
                    {
                        "type": "stream_created",
                        "stream_id": stream_id,
                        "component": component,
                    }
                )

                # Keep connection open for streaming
                while True:
                    data = await websocket.receive_json()
                    # Forward to stream
                    await websocket.send_json({"type": "stream_data", "received": data})

            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")

    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


# Export
__all__ = ["EcosystemAPI"]
