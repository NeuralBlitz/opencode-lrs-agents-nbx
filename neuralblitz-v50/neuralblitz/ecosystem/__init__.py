"""
NeuralBlitz V50 - Ecosystem Module
Bidirectional communication system for distributed components.
"""

# Protocol exports
from .protocol import (
    ComponentType,
    MessageType,
    Priority,
    Message,
    ComponentInfo,
    Protocol,
    ComponentAdapter,
    LRSAgentAdapter,
    OpenCodeAdapter as StubOpenCodeAdapter,
    AdvancedResearchAdapter as StubAdvancedResearchAdapter,
    ComputationalAxiomsAdapter as StubComputationalAxiomsAdapter,
    EmergentPromptAdapter as StubEmergentPromptAdapter,
    NeuralBlitzAdapter,
)

# Service Bus exports
from .service_bus import (
    ServiceBus,
    ServiceRegistry,
    SubscriptionManager,
    BidirectionalStream,
)

# Orchestrator exports
from .orchestrator import EcosystemOrchestrator, Workflow, WorkflowTemplates

# Real adapters with business logic
from .real_adapters import (
    OpenCodeAdapter,
    LRSAgentsAdapter,
    CodeSolution,
    ResearchFinding,
    MathematicalProof,
)

from .additional_adapters import (
    AdvancedResearchAdapter,
    ComputationalAxiomsAdapter,
    EmergentPromptArchitectureAdapter,
    ResearchReport,
    MathResult,
    PromptOptimization,
)

try:
    from .api_gateway import EcosystemAPI

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

__all__ = [
    # Protocol base classes
    "ComponentType",
    "MessageType",
    "Priority",
    "Message",
    "ComponentInfo",
    "Protocol",
    "ComponentAdapter",
    "NeuralBlitzAdapter",
    # Stub adapters (for compatibility)
    "LRSAgentAdapter",
    "StubOpenCodeAdapter",
    "StubAdvancedResearchAdapter",
    "StubComputationalAxiomsAdapter",
    "StubEmergentPromptAdapter",
    # Real business logic adapters
    "OpenCodeAdapter",
    "LRSAgentsAdapter",
    "AdvancedResearchAdapter",
    "ComputationalAxiomsAdapter",
    "EmergentPromptArchitectureAdapter",
    # Data classes
    "CodeSolution",
    "ResearchFinding",
    "MathematicalProof",
    "ResearchReport",
    "MathResult",
    "PromptOptimization",
    # Service Bus
    "ServiceBus",
    "ServiceRegistry",
    "SubscriptionManager",
    "BidirectionalStream",
    # Orchestrator
    "EcosystemOrchestrator",
    "Workflow",
    "WorkflowTemplates",
]

if API_AVAILABLE:
    __all__.append("EcosystemAPI")
