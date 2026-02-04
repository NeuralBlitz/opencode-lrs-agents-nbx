"""
NeuralBlitz V50 - Unified Communication Protocol
Bidirectional messaging system for distributed ecosystem.

Components:
- LRS Agents (Learning/Reasoning/Skill)
- OpenCode (AI coding assistant)
- NeuralBlitz-V50 (Consciousness engine)
- Advanced-Research (Research automation)
- ComputationalAxioms (Math/logic engine)
- Emergent-Prompt-Architecture (Prompt engineering)
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service_bus import ServiceBus


class ComponentType(Enum):
    """Registered components in the ecosystem."""

    LRS_AGENT = "lrs_agent"
    OPENCODE = "opencode"
    NEURALBLITZ = "neuralblitz"
    ADVANCED_RESEARCH = "advanced_research"
    COMPUTATIONAL_AXIOMS = "computational_axioms"
    EMERGENT_PROMPT = "emergent_prompt"
    GATEWAY = "gateway"


class MessageType(Enum):
    """Types of messages in the system."""

    # Command messages
    PROCESS = "process"
    QUERY = "query"
    STREAM = "stream"
    EXECUTE = "execute"

    # Event messages
    UPDATE = "update"
    STATUS = "status"
    ALERT = "alert"
    COMPLETE = "complete"
    ERROR = "error"

    # Meta messages
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    DISCOVER = "discover"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    AUTH = "auth"


class Priority(Enum):
    """Message priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Message:
    """
    Universal message format for all components.

    All communication happens through this structured format.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Routing
    source: ComponentType = ComponentType.GATEWAY
    target: Optional[ComponentType] = None  # None = broadcast
    reply_to: Optional[str] = None  # Message ID to reply to

    # Content
    msg_type: MessageType = MessageType.PROCESS
    priority: Priority = Priority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    correlation_id: Optional[str] = None
    ttl: int = 30  # Time-to-live in seconds
    trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "target": self.target.value if self.target else None,
            "reply_to": self.reply_to,
            "msg_type": self.msg_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
            "trace": self.trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=ComponentType(data["source"]),
            target=ComponentType(data["target"]) if data["target"] else None,
            reply_to=data.get("reply_to"),
            msg_type=MessageType(data["msg_type"]),
            priority=Priority(data.get("priority", 2)),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl", 30),
            trace=data.get("trace", []),
        )

    def create_reply(
        self, payload: Dict[str, Any], msg_type: MessageType = MessageType.COMPLETE
    ) -> "Message":
        """Create a reply message."""
        return Message(
            source=self.target if self.target else ComponentType.GATEWAY,
            target=self.source,
            reply_to=self.id,
            msg_type=msg_type,
            payload=payload,
            correlation_id=self.correlation_id or self.id,
        )


@dataclass
class ComponentInfo:
    """Information about a registered component."""

    component_type: ComponentType
    instance_id: str
    endpoint: str  # URL or connection string
    capabilities: List[str]
    status: str = "online"
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Protocol:
    """
    Communication protocol for the NeuralBlitz ecosystem.

    Defines standard message patterns for each component interaction.
    """

    @staticmethod
    def create_process_message(
        source: ComponentType,
        target: ComponentType,
        data: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
    ) -> Message:
        """Create a processing request message."""
        return Message(
            source=source,
            target=target,
            msg_type=MessageType.PROCESS,
            priority=priority,
            payload={
                "data": data,
                "sync": False,  # Async by default
                "timeout": 30,
            },
        )

    @staticmethod
    def create_query_message(
        source: ComponentType,
        target: ComponentType,
        query: str,
        parameters: Dict[str, Any],
    ) -> Message:
        """Create a query message."""
        return Message(
            source=source,
            target=target,
            msg_type=MessageType.QUERY,
            payload={"query": query, "parameters": parameters},
        )

    @staticmethod
    def create_stream_message(
        source: ComponentType,
        target: ComponentType,
        stream_type: str,
        config: Dict[str, Any],
    ) -> Message:
        """Create a streaming request."""
        return Message(
            source=source,
            target=target,
            msg_type=MessageType.STREAM,
            payload={
                "stream_type": stream_type,
                "config": config,
                "direction": "bidirectional",
            },
        )

    @staticmethod
    def create_handshake_message(
        component_type: ComponentType,
        instance_id: str,
        capabilities: List[str],
        endpoint: str,
    ) -> Message:
        """Create a handshake message for service registration."""
        return Message(
            source=component_type,
            target=ComponentType.GATEWAY,
            msg_type=MessageType.HANDSHAKE,
            priority=Priority.HIGH,
            payload={
                "instance_id": instance_id,
                "capabilities": capabilities,
                "endpoint": endpoint,
                "version": "50.0.0",
                "supported_protocols": ["websocket", "grpc", "rest"],
            },
        )

    @staticmethod
    def create_heartbeat_message(component_type: ComponentType, status: str) -> Message:
        """Create a heartbeat message."""
        return Message(
            source=component_type,
            target=ComponentType.GATEWAY,
            msg_type=MessageType.HEARTBEAT,
            priority=Priority.HIGH,
            payload={"status": status, "metrics": {}},
        )

    @staticmethod
    def create_subscribe_message(
        subscriber: ComponentType,
        event_types: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Create a subscription message."""
        return Message(
            source=subscriber,
            target=ComponentType.GATEWAY,
            msg_type=MessageType.SUBSCRIBE,
            payload={"event_types": event_types, "filters": filters or {}},
        )


class ComponentAdapter:
    """
    Base adapter for all components.

    Each component (LRS, OpenCode, etc.) extends this class.
    """

    def __init__(
        self,
        component_type: ComponentType,
        instance_id: str,
        capabilities: List[str],
        bus: Optional["ServiceBus"] = None,
    ):
        self.component_type = component_type
        self.instance_id = instance_id
        self.capabilities = capabilities
        self.bus = bus
        self.handlers: Dict[MessageType, Callable[[Message], Message]] = {}
        self.status = "initializing"

    def register_handler(
        self, msg_type: MessageType, handler: Callable[[Message], Message]
    ):
        """Register a message handler."""
        self.handlers[msg_type] = handler

    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message."""
        handler = self.handlers.get(message.msg_type)
        if handler:
            return handler(message)
        return None

    async def send(self, message: Message) -> Optional[Message]:
        """Send a message through the bus."""
        if self.bus:
            return await self.bus.send(message)
        return None

    async def broadcast(
        self, payload: Dict[str, Any], msg_type: MessageType = MessageType.UPDATE
    ):
        """Broadcast a message to all components."""
        message = Message(
            source=self.component_type,
            target=None,  # Broadcast
            msg_type=msg_type,
            payload=payload,
        )
        return await self.send(message)

    def get_info(self) -> ComponentInfo:
        """Get component information."""
        return ComponentInfo(
            component_type=self.component_type,
            instance_id=self.instance_id,
            endpoint=f"component://{self.component_type.value}/{self.instance_id}",
            capabilities=self.capabilities,
            status=self.status,
        )


# Specialized adapters for each component type


class LRSAgentAdapter(ComponentAdapter):
    """Adapter for LRS Agents (Learning/Reasoning/Skill)."""

    def __init__(self, agent_type: str, skills: List[str], bus=None):
        super().__init__(
            ComponentType.LRS_AGENT, f"lrs_{agent_type}_{uuid4().hex[:8]}", skills, bus
        )
        self.agent_type = agent_type
        self.skills = skills


class OpenCodeAdapter(ComponentAdapter):
    """Adapter for OpenCode."""

    def __init__(self, workspace_id: str, languages: List[str], bus=None):
        super().__init__(
            ComponentType.OPENCODE,
            f"opencode_{workspace_id}",
            ["code_generation", "code_review", "refactoring", "debugging"],
            bus,
        )
        self.workspace_id = workspace_id
        self.languages = languages


class AdvancedResearchAdapter(ComponentAdapter):
    """Adapter for Advanced Research."""

    def __init__(self, research_domain: str, bus=None):
        super().__init__(
            ComponentType.ADVANCED_RESEARCH,
            f"research_{research_domain}_{uuid4().hex[:8]}",
            [
                "paper_analysis",
                "hypothesis_generation",
                "experiment_design",
                "data_analysis",
            ],
            bus,
        )
        self.research_domain = research_domain


class ComputationalAxiomsAdapter(ComponentAdapter):
    """Adapter for Computational Axioms."""

    def __init__(self, logic_system: str, bus=None):
        super().__init__(
            ComponentType.COMPUTATIONAL_AXIOMS,
            f"axioms_{logic_system}_{uuid4().hex[:8]}",
            [
                "theorem_proving",
                "formal_verification",
                "constraint_solving",
                "proof_generation",
            ],
            bus,
        )
        self.logic_system = logic_system


class EmergentPromptAdapter(ComponentAdapter):
    """Adapter for Emergent Prompt Architecture."""

    def __init__(self, model_provider: str, bus=None):
        super().__init__(
            ComponentType.EMERGENT_PROMPT,
            f"epa_{model_provider}_{uuid4().hex[:8]}",
            [
                "prompt_optimization",
                "chain_of_thought",
                "prompt_templates",
                "auto_prompting",
            ],
            bus,
        )
        self.model_provider = model_provider


class NeuralBlitzAdapter(ComponentAdapter):
    """Adapter for NeuralBlitz V50."""

    def __init__(self, engine_instance: "MinimalCognitiveEngine", bus=None):
        super().__init__(
            ComponentType.NEURALBLITZ,
            f"nb_{uuid4().hex[:8]}",
            [
                "consciousness_processing",
                "intent_analysis",
                "pattern_recognition",
                "coherence_tracking",
            ],
            bus,
        )
        self.engine = engine_instance


__all__ = [
    "ComponentType",
    "MessageType",
    "Priority",
    "Message",
    "ComponentInfo",
    "Protocol",
    "ComponentAdapter",
    "LRSAgentAdapter",
    "OpenCodeAdapter",
    "AdvancedResearchAdapter",
    "ComputationalAxiomsAdapter",
    "EmergentPromptAdapter",
    "NeuralBlitzAdapter",
]
