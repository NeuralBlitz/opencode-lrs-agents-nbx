"""
NeuralBlitz V50 - Ecosystem Orchestrator
Central coordinator for all ecosystem components.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

from .protocol import (
    ComponentType,
    MessageType,
    Message,
    Priority,
    Protocol,
    ComponentAdapter,
    NeuralBlitzAdapter,
    LRSAgentAdapter,
    OpenCodeAdapter,
    AdvancedResearchAdapter,
    ComputationalAxiomsAdapter,
    EmergentPromptAdapter,
)
from .service_bus import ServiceBus, BidirectionalStream

logger = logging.getLogger("NeuralBlitz.Orchestrator")


@dataclass
class Workflow:
    """Multi-step workflow across components."""

    id: str
    name: str
    steps: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "pending"
    results: List[Any] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []


class EcosystemOrchestrator:
    """
    Central orchestrator for the NeuralBlitz ecosystem.

    Coordinates:
    - LRS Agents (Learning/Reasoning/Skill)
    - OpenCode (AI coding)
    - NeuralBlitz (Consciousness)
    - Advanced Research
    - Computational Axioms
    - Emergent Prompt Architecture

    Provides:
    - Workflow management
    - Cross-component communication
    - Load balancing
    - Health monitoring
    - Event coordination
    """

    def __init__(self):
        self.bus = ServiceBus()
        self._adapters: Dict[ComponentType, List[ComponentAdapter]] = {}
        self._workflows: Dict[str, Workflow] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._streams: Dict[str, BidirectionalStream] = {}
        self._running = False

    async def start(self):
        """Start the orchestrator."""
        await self.bus.start()
        self._running = True
        logger.info("Ecosystem orchestrator started")

    async def stop(self):
        """Stop the orchestrator."""
        self._running = False

        # Stop all streams
        for stream in self._streams.values():
            await stream.stop()

        await self.bus.stop()
        logger.info("Ecosystem orchestrator stopped")

    def register_component(self, adapter: ComponentAdapter):
        """Register a component adapter."""
        component_type = adapter.component_type

        if component_type not in self._adapters:
            self._adapters[component_type] = []

        self._adapters[component_type].append(adapter)
        self.bus.register_adapter(adapter)

        # Set up default handlers
        self._setup_default_handlers(adapter)

        logger.info(f"Registered {component_type.value}: {adapter.instance_id}")

    def _setup_default_handlers(self, adapter: ComponentAdapter):
        """Set up default message handlers."""

        # Handler for process messages
        async def handle_process(msg: Message) -> Message:
            if hasattr(adapter, "process"):
                result = await adapter.process(msg.payload.get("data"))
                return msg.create_reply({"result": result})
            return msg.create_reply({"error": "No process method"})

        # Handler for query messages
        async def handle_query(msg: Message) -> Message:
            if hasattr(adapter, "query"):
                result = await adapter.query(
                    msg.payload.get("query"), msg.payload.get("parameters", {})
                )
                return msg.create_reply({"result": result})
            return msg.create_reply({"error": "No query method"})

        adapter.register_handler(MessageType.PROCESS, handle_process)
        adapter.register_handler(MessageType.QUERY, handle_query)

    async def send_to_component(
        self, target_type: ComponentType, message: Message
    ) -> Optional[Message]:
        """Send message to a specific component type."""
        adapters = self._adapters.get(target_type, [])

        if not adapters:
            logger.warning(f"No adapters for {target_type.value}")
            return None

        # Round-robin selection
        adapter = adapters[0]  # Simplified; could implement actual round-robin
        message.target = target_type

        return await self.bus.send(message)

    async def broadcast(
        self,
        payload: Dict[str, Any],
        msg_type: MessageType = MessageType.UPDATE,
        exclude: Optional[List[ComponentType]] = None,
    ):
        """Broadcast to all components."""
        message = Message(
            source=ComponentType.GATEWAY,
            target=None,
            msg_type=msg_type,
            payload=payload,
        )

        await self.bus.send(message)

    async def create_stream(
        self, source_type: ComponentType, target_type: ComponentType
    ) -> Optional[str]:
        """Create bidirectional stream between components."""
        source_adapters = self._adapters.get(source_type, [])
        target_adapters = self._adapters.get(target_type, [])

        if not source_adapters or not target_adapters:
            return None

        stream_id = f"stream_{source_type.value}_{target_type.value}_{datetime.utcnow().timestamp()}"

        stream = BidirectionalStream(
            stream_id=stream_id,
            source=source_adapters[0],
            target=target_adapters[0],
            bus=self.bus,
        )

        await stream.start()
        self._streams[stream_id] = stream

        return stream_id

    async def execute_workflow(
        self, workflow: Workflow, context: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """Execute a multi-step workflow."""
        workflow.status = "running"
        self._workflows[workflow.id] = workflow

        try:
            for i, step in enumerate(workflow.steps):
                workflow.current_step = i

                # Execute step
                result = await self._execute_step(step, context)
                workflow.results.append(result)

                logger.info(
                    f"Workflow {workflow.id}: Step {i + 1}/{len(workflow.steps)} complete"
                )

            workflow.status = "completed"

        except Exception as e:
            logger.error(f"Workflow {workflow.id} failed: {e}")
            workflow.status = "failed"

        return workflow

    async def _execute_step(
        self, step: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a single workflow step."""
        component_type = ComponentType(step["component"])
        operation = step["operation"]
        data = step.get("data", {})

        # Merge with context
        if context:
            data = {**context, **data}

        # Create message
        message = Message(
            source=ComponentType.GATEWAY,
            target=component_type,
            msg_type=MessageType(operation),
            payload=data,
        )

        # Send and wait for response
        response = await self.bus.send_and_wait(
            message, timeout=step.get("timeout", 30)
        )

        if response:
            return response.payload.get("result")
        return None

    def on_event(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all handlers."""
        await self.bus.publish_event(event_type, data)

        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def get_health(self) -> Dict[str, Any]:
        """Get ecosystem health status."""
        return {
            "orchestrator_status": "running" if self._running else "stopped",
            "registered_components": {
                ct.value: len(adapters) for ct, adapters in self._adapters.items()
            },
            "active_streams": len(self._streams),
            "active_workflows": len(
                [w for w in self._workflows.values() if w.status == "running"]
            ),
            "bus_stats": self.bus.get_stats(),
        }


# Pre-built workflow templates


class WorkflowTemplates:
    """Common workflow templates for the ecosystem."""

    @staticmethod
    def research_and_code(research_topic: str, coding_task: str) -> Workflow:
        """Research then code workflow."""
        return Workflow(
            id=f"research_code_{datetime.utcnow().timestamp()}",
            name="Research and Code",
            steps=[
                {
                    "component": "advanced_research",
                    "operation": "process",
                    "data": {"topic": research_topic, "depth": "comprehensive"},
                    "timeout": 60,
                },
                {
                    "component": "emergent_prompt",
                    "operation": "process",
                    "data": {
                        "task": "generate_prompts",
                        "context": "{{step_0.result}}",
                    },
                    "timeout": 30,
                },
                {
                    "component": "opencode",
                    "operation": "process",
                    "data": {
                        "task": coding_task,
                        "research_context": "{{step_0.result}}",
                    },
                    "timeout": 120,
                },
                {
                    "component": "computational_axioms",
                    "operation": "process",
                    "data": {"task": "verify_code", "code": "{{step_2.result}}"},
                    "timeout": 30,
                },
            ],
        )

    @staticmethod
    def consciousness_guided_coding(requirements: str) -> Workflow:
        """Use consciousness engine to guide coding."""
        return Workflow(
            id=f"conscious_coding_{datetime.utcnow().timestamp()}",
            name="Consciousness-Guided Coding",
            steps=[
                {
                    "component": "neuralblitz",
                    "operation": "process",
                    "data": {"intent": {"phi3_creation": 0.8, "phi6_knowledge": 0.7}},
                    "timeout": 10,
                },
                {
                    "component": "opencode",
                    "operation": "process",
                    "data": {
                        "task": requirements,
                        "consciousness_state": "{{step_0.result}}",
                    },
                    "timeout": 120,
                },
                {
                    "component": "neuralblitz",
                    "operation": "process",
                    "data": {
                        "intent": {"phi1_dominance": 0.6, "phi4_preservation": 0.7}
                    },
                    "timeout": 10,
                },
            ],
        )

    @staticmethod
    def multi_agent_problem_solving(problem: str) -> Workflow:
        """Multiple agents collaborate to solve a problem."""
        return Workflow(
            id=f"multi_agent_{datetime.utcnow().timestamp()}",
            name="Multi-Agent Problem Solving",
            steps=[
                {
                    "component": "computational_axioms",
                    "operation": "process",
                    "data": {"task": "formalize_problem", "problem": problem},
                    "timeout": 30,
                },
                {
                    "component": "lrs_agent",
                    "operation": "process",
                    "data": {"task": "reasoning", "formalized": "{{step_0.result}}"},
                    "timeout": 60,
                },
                {
                    "component": "advanced_research",
                    "operation": "process",
                    "data": {"task": "find_similar_solutions", "problem": problem},
                    "timeout": 60,
                },
                {
                    "component": "neuralblitz",
                    "operation": "process",
                    "data": {
                        "intent": {"phi2_harmony": 0.8, "phi5_transformation": 0.6}
                    },
                    "timeout": 10,
                },
            ],
        )


__all__ = ["EcosystemOrchestrator", "Workflow", "WorkflowTemplates"]
