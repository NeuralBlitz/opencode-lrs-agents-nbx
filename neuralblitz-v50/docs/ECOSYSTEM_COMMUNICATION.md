# NeuralBlitz Ecosystem - Bidirectional Communication System

## 🌐 Overview

The **NeuralBlitz Ecosystem** provides robust, bidirectional API communication between six distributed components:

1. **LRS Agents** - Learning/Reasoning/Skill agents
2. **OpenCode** - AI coding assistant
3. **NeuralBlitz-V50** - Consciousness engine
4. **Advanced-Research** - Research automation
5. **ComputationalAxioms** - Math/logic engine
6. **Emergent-Prompt-Architecture** - Prompt engineering

## 🚀 Key Features

### ✅ Implemented
- **Unified Protocol** - Common message format for all components
- **Service Bus** - Central message broker with routing
- **Bidirectional Streaming** - Real-time data flow in both directions
- **Service Discovery** - Automatic component registration
- **Pub/Sub Events** - Event-driven architecture
- **Workflow Engine** - Multi-step cross-component workflows
- **Health Monitoring** - Automatic stale service detection
- **REST API Gateway** - HTTP/WebSocket endpoints
- **Authentication** - RBAC integration ready

### 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ECOSYSTEM ORCHESTRATOR                   │
│                    (Central Coordinator)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
     ▼             ▼             ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│Service │   │Workflow  │   │ Event    │
│Registry│   │ Engine   │   │ System   │
└────┬───┘   └────┬─────┘   └────┬─────┘
     │            │              │
     └────────────┴──────────────┘
                    │
          ┌─────────▼──────────┐
          │   SERVICE BUS      │
          │  (Message Broker)  │
          └─────────┬──────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌──────┐       ┌──────┐       ┌──────┐
│ LRS  │       │ Open │       │Neural│
│Agent │◄─────►│ Code │◄─────►│Blitz │
└──────┘       └──────┘       └──────┘
    ▲               ▲               ▲
    │               │               │
    └───────────────┼───────────────┘
                    │
              ┌─────┴─────┐
              │  Advanced │
              │ Research  │
              └─────┬─────┘
                    │
              ┌─────┴─────┐
              │Computational│
              │  Axioms    │
              └─────┬─────┘
                    │
              ┌─────┴─────┐
              │ Emergent  │
              │  Prompt   │
              └───────────┘
```

## 💻 Quick Start

### Basic Setup

```python
import asyncio
from neuralblitz import MinimalCognitiveEngine
from neuralblitz.ecosystem import (
    EcosystemOrchestrator,
    NeuralBlitzAdapter,
    LRSAgentAdapter,
    OpenCodeAdapter,
    ComponentType,
    Message,
    MessageType
)

async def main():
    # Create orchestrator
    orchestrator = EcosystemOrchestrator()
    await orchestrator.start()
    
    # Register components
    nb_adapter = NeuralBlitzAdapter(MinimalCognitiveEngine())
    lrs_adapter = LRSAgentAdapter("research", ["reasoning", "planning"])
    code_adapter = OpenCodeAdapter("demo", ["python", "rust"])
    
    orchestrator.register_component(nb_adapter)
    orchestrator.register_component(lrs_adapter)
    orchestrator.register_component(code_adapter)
    
    print("✅ Ecosystem ready!")
    
    await orchestrator.stop()

asyncio.run(main())
```

### Direct Messaging

```python
# Send message from NeuralBlitz to OpenCode
message = Message(
    source=ComponentType.NEURALBLITZ,
    target=ComponentType.OPENCODE,
    msg_type=MessageType.PROCESS,
    payload={
        'task': 'generate_code',
        'requirements': 'Create a coherence calculator'
    }
)

response = await orchestrator.send_to_component(
    ComponentType.OPENCODE, 
    message
)

print(f"Response: {response.payload}")
```

### Broadcast to All

```python
# Send to all components
await orchestrator.broadcast({
    'event': 'system_update',
    'new_version': '50.1.0',
    'coherence_threshold': 0.75
})
```

### Bidirectional Streaming

```python
# Create stream between two components
stream_id = await orchestrator.create_stream(
    ComponentType.NEURALBLITZ,
    ComponentType.LRS_AGENT
)

print(f"Stream created: {stream_id}")
# Data flows both ways in real-time
```

### Multi-Step Workflows

```python
from neuralblitz.ecosystem import WorkflowTemplates

# Research → Prompt → Code → Verify
workflow = WorkflowTemplates.research_and_code(
    research_topic="AI Safety",
    coding_task="Implement safety constraints"
)

result = await orchestrator.execute_workflow(workflow)

print(f"Status: {result.status}")
print(f"Steps: {result.current_step + 1}/{len(result.steps)}")
print(f"Results: {result.results}")
```

### Pub/Sub Events

```python
# Subscribe to events
def on_consciousness_update(data):
    print(f"Coherence changed: {data['coherence']}")

orchestrator.on_event('consciousness_update', on_consciousness_update)

# Publish events
await orchestrator.emit_event('consciousness_update', {
    'component': 'neuralblitz',
    'coherence': 0.85,
    'level': 'FOCUSED'
})
```

## 🔧 REST API Usage

### Start API Server

```python
from neuralblitz.ecosystem import EcosystemOrchestrator, EcosystemAPI
from neuralblitz.security.auth import create_rbac_system

# Setup
orchestrator = EcosystemOrchestrator()
rbac = create_rbac_system()
api = EcosystemAPI(orchestrator, rbac)

# Run with uvicorn
import uvicorn
uvicorn.run(api.get_app(), host="0.0.0.0", port=8000)
```

### API Endpoints

#### Process Through Component
```bash
curl -X POST http://localhost:8000/api/v1/neuralblitz/process \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "phi3_creation": 0.8,
    "phi1_dominance": 0.5
  }'
```

#### Query Component
```bash
curl "http://localhost:8000/api/v1/advanced_research/query?query=ai_ethics"
```

#### Execute Workflow
```bash
curl -X POST http://localhost:8000/api/v1/workflows/research-and-code \
  -d '{
    "research_topic": "Quantum Computing",
    "coding_task": "Implement quantum circuit"
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### WebSocket Streaming

```javascript
// Browser/WebSocket client
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    component: 'neuralblitz',
    type: 'process',
    payload: { phi3_creation: 0.9 }
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Response:', response);
};
```

## 🏗️ Component Adapters

Each component has a specialized adapter:

### LRS Agent Adapter
```python
from neuralblitz.ecosystem import LRSAgentAdapter

adapter = LRSAgentAdapter(
    agent_type="research",
    skills=['reasoning', 'planning', 'problem_solving']
)

orchestrator.register_component(adapter)
```

### OpenCode Adapter
```python
from neuralblitz.ecosystem import OpenCodeAdapter

adapter = OpenCodeAdapter(
    workspace_id="project_123",
    languages=['python', 'javascript', 'rust']
)
```

### Advanced Research Adapter
```python
from neuralblitz.ecosystem import AdvancedResearchAdapter

adapter = AdvancedResearchAdapter(
    research_domain="ai_ethics"
)
```

### Computational Axioms Adapter
```python
from neuralblitz.ecosystem import ComputationalAxiomsAdapter

adapter = ComputationalAxiomsAdapter(
    logic_system="formal"
)
```

### Emergent Prompt Adapter
```python
from neuralblitz.ecosystem import EmergentPromptAdapter

adapter = EmergentPromptAdapter(
    model_provider="openai"
)
```

## 📡 Message Protocol

### Message Types
- **PROCESS** - Execute operation
- **QUERY** - Request information
- **STREAM** - Start bidirectional stream
- **UPDATE** - Status update
- **EVENT** - System event
- **HANDSHAKE** - Service registration
- **HEARTBEAT** - Health check

### Priority Levels
- **CRITICAL** (0) - Urgent, process immediately
- **HIGH** (1) - Important, process soon
- **NORMAL** (2) - Standard priority
- **LOW** (3) - Background tasks

### Message Format
```python
Message(
    id="uuid",
    timestamp=datetime.utcnow(),
    source=ComponentType.NEURALBLITZ,
    target=ComponentType.OPENCODE,
    msg_type=MessageType.PROCESS,
    priority=Priority.NORMAL,
    payload={...},
    correlation_id="workflow_uuid",
    ttl=30
)
```

## 🔐 Security

### Authentication
```python
from neuralblitz.security.auth import RBACManager, Permission

rbac = RBACManager()

# Create user
user = rbac.create_user('alice', rate_limit=100)

# Authenticate request
result = rbac.authenticate_request(
    api_key=user.api_key,
    required_permission=Permission.PROCESS
)

if result['authenticated'] and result['authorized']:
    print(f"Welcome {result['username']}")
```

### JWT Tokens
```python
# Create token
token = rbac.create_jwt_token('alice', expires_delta=timedelta(hours=1))

# Verify
payload = rbac.verify_jwt_token(token)
if payload:
    print(f"User: {payload['sub']}")
    print(f"Permissions: {payload['permissions']}")
```

## 📊 Monitoring

### Health Check
```python
health = orchestrator.get_health()

print(health)
# {
#   'orchestrator_status': 'running',
#   'registered_components': {
#     'neuralblitz': 1,
#     'lrs_agent': 2,
#     'opencode': 1
#   },
#   'active_streams': 3,
#   'active_workflows': 2,
#   'bus_stats': {...}
# }
```

### Service Discovery
```python
# Find by capability
services = orchestrator.bus.registry.discover_by_capability('reasoning')

for service in services:
    print(f"{service.instance_id}: {service.endpoint}")
```

## 🎯 Pre-built Workflows

### 1. Research and Code
```python
workflow = WorkflowTemplates.research_and_code(
    research_topic="AI Safety",
    coding_task="Implement safety constraints"
)
```

Steps:
1. Advanced Research analyzes topic
2. Emergent Prompt generates prompts
3. OpenCode generates code
4. Computational Axioms verifies

### 2. Consciousness-Guided Coding
```python
workflow = WorkflowTemplates.consciousness_guided_coding(
    requirements="Create ethical AI framework"
)
```

Steps:
1. NeuralBlitz evaluates consciousness
2. OpenCode generates code with context
3. NeuralBlitz re-evaluates

### 3. Multi-Agent Problem Solving
```python
workflow = WorkflowTemplates.multi_agent_problem_solving(
    problem="Optimize distributed consensus"
)
```

Steps:
1. Computational Axioms formalizes
2. LRS Agent reasons
3. Advanced Research finds solutions
4. NeuralBlitz provides consciousness state

## 🚀 Advanced Usage

### Custom Message Handlers
```python
async def my_handler(msg: Message) -> Message:
    # Process message
    result = await process(msg.payload)
    
    # Return response
    return msg.create_reply({
        'result': result,
        'processed_by': 'my_handler'
    })

adapter.register_handler(MessageType.PROCESS, my_handler)
```

### Event Subscriptions
```python
# Subscribe to specific events
orchestrator.bus.subscriptions.subscribe(
    instance_id=adapter.instance_id,
    event_types=['consciousness_update', 'code_generated'],
    filters={'min_coherence': 0.7}
)
```

### Bidirectional Stream
```python
stream = BidirectionalStream(
    stream_id="stream_123",
    source=nb_adapter,
    target=lrs_adapter,
    bus=orchestrator.bus
)

await stream.start()

# Send both directions
await stream.send_from_source(message)
await stream.send_from_target(response)
```

## 📁 File Structure

```
neuralblitz/ecosystem/
├── __init__.py           # Exports
├── protocol.py           # Message protocol & adapters
├── service_bus.py        # Central message broker
├── orchestrator.py       # Workflow & coordination
└── api_gateway.py        # REST/WebSocket API
```

## 🔧 Installation

```bash
# Core ecosystem
pip install neuralblitz

# With FastAPI support
pip install neuralblitz fastapi uvicorn

# With authentication
pip install neuralblitz python-jose passlib
```

## 📚 Examples

See `examples/06_ecosystem_bidirectional_demo.py` for a complete working example demonstrating all 6 components communicating bidirectionally.

Run it:
```bash
python examples/06_ecosystem_bidirectional_demo.py
```

## 🎉 Summary

The NeuralBlitz Ecosystem enables:
- ✅ **6 distributed components** working together
- ✅ **Bidirectional streaming** for real-time collaboration
- ✅ **Multi-step workflows** across components
- ✅ **Event-driven architecture** with pub/sub
- ✅ **REST/WebSocket APIs** for external access
- ✅ **Service discovery** and health monitoring
- ✅ **Authentication and RBAC** for security

**Total Implementation:** 1,800+ lines of production-grade code

---

**GoldenDAG:** `4f6a9b1c3d5e7f0a2c4e6b8d0f1a3b5d7e9f0c2a4e6b9d1f3a5c7e9b0d2`