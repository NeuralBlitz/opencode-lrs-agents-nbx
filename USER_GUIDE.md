# Go-LRS User Guide

## Complete Multi-Agent Active Inference System in Go

This guide covers everything you need to know to use Go-LRS effectively.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Building Your First Agent](#building-your-first-agent)
4. [Multi-Agent Coordination](#multi-agent-coordination)
5. [API Usage](#api-usage)
6. [Integration Examples](#integration-examples)
7. [Monitoring & Dashboard](#monitoring--dashboard)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/go-lrs.git
cd go-lrs

# Install dependencies
go mod tidy

# Build the server
make build

# Run tests
make test
```

### Running the Server

```bash
# Basic usage
./bin/server

# With configuration
./bin/server --config configs/production.yaml

# Specify ports
./bin/server --http-port 8080 --grpc-port 9090
```

### Hello World Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/neuralblitz/go-lrs/internal/math"
    "github.com/neuralblitz/go-lrs/internal/state"
    "github.com/neuralblitz/go-lrs/pkg/core"
)

func main() {
    // Create a new agent state
    agentState := state.NewLRSState("my-first-agent")
    
    // Create a simple tool
    searchTool := core.NewSimpleTool(
        "search",
        "Search for information",
        []string{"search"},
        func(ctx context.Context, s *state.LRSState) (interface{}, error) {
            return "Search results found!", nil
        },
        nil,
    )
    
    // Execute the tool
    result, err := searchTool.Execute(context.Background(), agentState)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Tool execution: %v\n", result.Output)
    fmt.Printf("Success: %v\n", result.Success)
    fmt.Printf("Prediction Error: %.3f\n", result.PredictionError)
}
```

## Core Concepts

### Active Inference

Go-LRS implements the Active Inference framework from computational neuroscience. Key concepts:

**Precision (γ)**: Represents confidence in predictions. Updated via Bayesian belief revision:
- γ = α/(α+β) where α and β are Beta distribution parameters
- Asymmetric learning: faster loss of confidence than gain

**Free Energy (G)**: Measures surprise of outcomes:
- G(π) = Epistemic Value - Pragmatic Value
- Lower free energy = better policy

**Policy Selection**: Boltzmann distribution over free energies:
- P(π) ∝ exp(-β × G(π))
- β = 1/γ (precision as inverse temperature)

### Hierarchical Precision

Three levels of precision tracking:
- **Abstract**: Long-term goals and strategies
- **Planning**: Policy sequences and intermediate steps
- **Execution**: Individual tool calls and actions

```go
precision := math.NewHierarchicalPrecision(0.1, 0.2)
precision.Update(math.LevelPlanning, 0.2) // Low error = success
```

### ToolLens Pattern

Bidirectional tool abstraction with automatic error handling:

```go
// Define a tool
tool := core.NewSimpleTool(
    "my-tool",
    "Description",
    []string{"capability1", "capability2"},
    executeFunc,
    updateFunc,
)

// Execute forward
tool.Execute(ctx, state)

// Update backward
newState, _ := tool.Update(state, result)

// Compose tools
pipeline := tool1 >> tool2 >> tool3
```

## Building Your First Agent

### Step 1: Create Agent State

```go
import (
    "github.com/neuralblitz/go-lrs/internal/state"
    "github.com/neuralblitz/go-lrs/internal/math"
)

// Create initial state
agent := state.NewLRSState("agent-001")

// Add preferences
agent = agent.WithPreference("accuracy", 0.9, 1.0, 1)
agent = agent.WithPreference("speed", 0.7, 0.5, 2)

// Set initial belief
agent = agent.WithBeliefState("task", "research project")
```

### Step 2: Define Tools

```go
searchTool := core.NewSimpleTool(
    "web-search",
    "Search the web",
    []string{"search", "web"},
    func(ctx context.Context, s *state.LRSState) (interface{}, error) {
        query := s.GetLastMessage().Content
        results := performSearch(query)
        return results, nil
    },
    func(s *state.LRSState, obs *core.ExecutionResult) (*state.LRSState, error) {
        // Update state with observation
        newState := s.WithBeliefState("search_results", obs.Output)
        return newState, nil
    },
)
```

### Step 3: Execute Policy

```go
// Create policy calculator
precision := math.NewPrecisionParameters(0.1, 0.2)
calculator := math.NewFreeEnergyCalculator(precision)

// Generate candidate policies
generator := math.NewPolicyGenerator(
    []string{"search", "filter", "analyze"},
    math.DefaultPolicyGenerationParams(),
)
policies := generator.GeneratePolicies()

// Evaluate policies
evaluations := calculator.EvaluatePolicies(policies, agent.Preferences)

// Select best policy
selected := calculator.SelectPolicy(evaluations)
```

### Step 4: Update Precision

```go
// After execution, update precision based on results
if result.Success {
    agent, _ = agent.WithPrecision(math.LevelExecution, 0.1) // Low error
} else {
    agent, _ = agent.WithPrecision(math.LevelExecution, 0.8) // High error
}
```

## Multi-Agent Coordination

### Creating a Coordinator

```go
coordinator := multiagent.NewCoordinator(&multiagent.CoordinatorConfig{
    MaxConcurrentTasks:    5,
    TaskTimeout:          30 * time.Minute,
    CollaborationThreshold: 0.6,
    MinTrustLevel:         0.4,
})
```

### Registering Agents

```go
// Create specialized agents
searchAgent := multiagent.NewAgent(
    "search-agent",
    "Research Specialist",
    []string{"search", "retrieval", "research"},
)

analysisAgent := multiagent.NewAgent(
    "analysis-agent",
    "Data Analyst",
    []string{"analysis", "processing", "statistics"},
)

// Register with coordinator
coordinator.RegisterAgent(searchAgent)
coordinator.RegisterAgent(analysisAgent)
```

### Building Social Trust

```go
// Simulate successful collaboration
searchAgent.UpdateSocialPrecision(
    "analysis-agent",
    "collaborate",      // Predicted action
    "collaborate",      // Observed action
    1.0,                // Success outcome
)

// Check trust level
trust := searchAgent.GetTrustLevel("analysis-agent")
fmt.Printf("Trust level: %.3f\n", trust)

// Check if collaboration is advisable
if searchAgent.CanCooperateWith("analysis-agent", task) {
    fmt.Println("Safe to collaborate!")
}
```

### Creating Collaborative Tasks

```go
// Create complex task requiring multiple agents
task := coordinator.CreateTask(
    "market-analysis",
    "Comprehensive market research and analysis",
    0.8, // High complexity
    []string{"search", "analysis"},
    time.Now().Add(2*time.Hour),
)

// Coordinator automatically assigns agents based on:
// - Capability matching
// - Social trust levels
// - Availability
```

### Inter-Agent Communication

```go
// Send message to another agent
searchAgent.SendMessage(
    "analysis-agent",
    "Found 50 relevant articles",
    "resource_share",
    map[string]interface{}{
        "resource_type": "search_results",
        "count": 50,
    },
)

// Receive messages
messages := searchAgent.GetMessages()
for _, msg := range messages {
    fmt.Printf("From %s: %s\n", msg.From, msg.Content)
}
```

## API Usage

### REST API

```bash
# Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-agent",
    "description": "Research agent",
    "config": {},
    "metadata": {"department": "research"}
  }'

# Execute policy
curl -X POST http://localhost:8080/api/v1/agents/agent-001/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Search for AI trends",
    "context": {"domain": "machine learning"},
    "preferences": {"accuracy": 0.9}
  }'

# Get state
curl http://localhost:8080/api/v1/states/agent-001

# Update precision
curl -X PUT http://localhost:8080/api/v1/precision/agent-001 \
  -H "Content-Type: application/json" \
  -d '{
    "level": "execution",
    "prediction_error": 0.2
  }'
```

### gRPC Streaming

```go
import (
    pb "github.com/neuralblitz/go-lrs/pkg/api/proto/v1"
    "google.golang.org/grpc"
)

// Connect to gRPC server
conn, _ := grpc.Dial("localhost:9090", grpc.WithInsecure())
defer conn.Close()

client := pb.NewLRSServiceClient(conn)

// Execute policy with streaming
stream, _ := client.ExecutePolicy(ctx, &pb.ExecutePolicyRequest{
    AgentId: "agent-001",
    Task:    "Search for trends",
})

// Process streaming results
for {
    response, err := stream.Recv()
    if err == io.EOF {
        break
    }
    
    switch result := response.Result.(type) {
    case *pb.ExecutePolicyResponse_Start:
        fmt.Printf("Started: %s\n", result.Start.ExecutionId)
    case *pb.ExecutePolicyResponse_Step:
        fmt.Printf("Step %d: %v\n", result.Step.StepNumber, result.Step.ToolExecution)
    case *pb.ExecutePolicyResponse_Complete:
        fmt.Printf("Completed: %v\n", result.Complete.Result)
    }
}

// Stream state changes
stateStream, _ := client.StreamStateChanges(ctx, &pb.StreamStateChangesRequest{
    AgentId:     "agent-001",
    FromVersion: 0,
})

for {
    stateChange, _ := stateStream.Recv()
    fmt.Printf("State updated: version %d\n", stateChange.State.Version)
}
```

## Integration Examples

### LangChain Integration

```go
// Wrap LangChain tool
config := integration.AdapterConfig{
    Name:            "duckduckgo-search",
    Description:     "Web search using DuckDuckGo",
    Capabilities:    []string{"search", "web"},
    DefaultTimeout:  30 * time.Second,
    MaxRetries:      3,
}

langchainTool := integration.NewLangChainAdapter(
    config,
    integration.LangChainConfig{
        ToolType:   "search",
        Parameters: map[string]interface{}{"engine": "duckduckgo"},
    },
    func(ctx context.Context, input interface{}) (interface{}, error) {
        // Execute LangChain tool
        return searchEngine.Search(input.(string)), nil
    },
)

// Register globally
core.RegisterTool(langchainTool)
```

### OpenAI Integration

```go
// Create OpenAI adapter
openaiAdapter := integration.NewOpenAIAdapter(
    integration.AdapterConfig{
        Name:         "gpt-assistant",
        Description:  "GPT-3.5 assistant",
        Capabilities: []string{"generation", "analysis"},
    },
    integration.OpenAIConfig{
        APIKey:      os.Getenv("OPENAI_API_KEY"),
        Model:       "gpt-3.5-turbo",
        Temperature: 0.7,
        MaxTokens:   1000,
    },
)

// Register
core.RegisterTool(openaiAdapter)
```

### Custom Tool Integration

```go
// Create custom adapter
adapter := integration.NewGenericAdapter(
    integration.AdapterConfig{
        Name:            "my-api",
        Description:     "My custom API",
        Capabilities:    []string{"custom"},
        DefaultTimeout:  60 * time.Second,
        ErrorThreshold:  0.5,
    },
    func(ctx context.Context, input interface{}) (interface{}, error) {
        // Custom execution logic
        return myAPI.Call(input), nil
    },
)

// Custom error function
adapter.ErrorFunc = func(input, output interface{}, err error) float64 {
    if err != nil {
        if isRateLimit(err) {
            return 0.3 // Low error for rate limits
        }
        return 0.8 // High error for other failures
    }
    return 0.1 // Low error for success
}

core.RegisterTool(adapter)
```

## Monitoring & Dashboard

### Starting the Dashboard

```go
import "github.com/neuralblitz/go-lrs/pkg/monitoring"

// Create coordinator
coordinator := multiagent.NewCoordinator(config)

// Create dashboard
dashboard := monitoring.NewDashboard(coordinator, 5*time.Second)

// Start dashboard on port 8081
err := dashboard.Start(8081)
```

### Dashboard Features

The dashboard provides:

1. **System Overview**: Real-time metrics
   - Active agents
   - Active collaborations
   - Success rate
   - Average trust level

2. **Agent Management**: Detailed agent information
   - Status and activity
   - Precision levels
   - Trust levels
   - Cooperation history

3. **Task Tracking**: Monitor task execution
   - Task status
   - Progress indicators
   - Assigned agents
   - Duration tracking

4. **Collaboration View**: Track multi-agent work
   - Active collaborations
   - Progress tracking
   - Participant roles
   - Shared resources

5. **Social Metrics**: Analyze trust networks
   - Trust level trends
   - Cooperation patterns
   - Success rates
   - Network visualization

### Accessing Metrics via API

```bash
# Get all metrics
curl http://localhost:8081/api/v1/metrics

# Get agent-specific metrics
curl http://localhost:8081/api/v1/agents

# Get social metrics
curl http://localhost:8081/api/v1/social-metrics
```

### Real-time Updates

The dashboard uses WebSockets for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8081/ws');

ws.onmessage = function(event) {
    const metrics = JSON.parse(event.data);
    updateDashboard(metrics);
};
```

## Performance Tuning

### Concurrency Settings

```yaml
# configs/production.yaml
server:
  grpc:
    max_connections: 1000
    max_recv_msg_size: 4194304  # 4MB
    max_send_msg_size: 4194304  # 4MB
  
  http:
    max_header_bytes: 1048576   # 1MB

tools:
  registry:
    max_concurrent: 10
  
  execution:
    parallel_execution: true
    max_parallel: 5
```

### Memory Optimization

```go
// Limit state history
stateManager := state.NewStateManager(initialState, 5) // Max 5 checkpoints

// Clear old history periodically
if len(state.ToolHistory) > 100 {
    state.ClearHistory()
}
```

### Precision Optimization

```go
// Adjust learning rates for your domain
precision := math.NewPrecisionParameters(
    0.05, // Slower gain (stable environments)
    0.15, // Moderate loss
)

// Set thresholds
precision.HighThreshold = 0.85
precision.LowThreshold = 0.15
```

### Caching Strategies

```go
adapter := integration.NewGenericAdapter(config, executeFunc)
adapter.Config.EnableCaching = true
adapter.Config.CacheTimeout = 5 * time.Minute
```

## Troubleshooting

### Common Issues

**High Prediction Errors**
```go
// Check precision levels
precision := agent.Precision.GetAllPrecision()
fmt.Printf("Abstract: %.3f, Planning: %.3f, Execution: %.3f\n",
    precision[math.LevelAbstract],
    precision[math.LevelPlanning],
    precision[math.LevelExecution])

// If execution precision is low, reduce tool complexity
// or increase retry count
```

**Agent Not Collaborating**
```go
// Check trust levels
trust := agent.GetTrustLevel(otherAgentID)
if trust < coordinator.Config.MinTrustLevel {
    // Build trust through successful interactions
    agent.UpdateSocialPrecision(otherAgentID, action, action, 1.0)
}
```

**Slow Policy Generation**
```go
// Reduce max policies
params := math.DefaultPolicyGenerationParams()
params.MaxPolicies = 5  // Default is 10
params.MaxPolicyLength = 3  // Default is 5

generator := math.NewPolicyGenerator(tools, params)
```

**Memory Issues**
```go
// Reduce state history size
state := state.NewLRSState("agent-001")
state = state.ClearHistory()

// Clear old messages
if len(state.Messages) > 50 {
    state.Messages = state.Messages[len(state.Messages)-50:]
}
```

### Debug Mode

```bash
# Enable debug logging
export LRS_LOG_LEVEL=debug

# Run with pprof
./bin/server --debug

# Access pprof
# http://localhost:6060/debug/pprof/
```

### Metrics and Logging

```go
// Enable detailed metrics
dashboard := monitoring.NewDashboard(coordinator, 1*time.Second)

// Log agent decisions
log.Printf("Agent %s selected policy %s with G=%.3f",
    agent.ID, selected.Policy.ID, selected.FreeEnergy.Value)

// Monitor precision changes
log.Printf("Precision updated: execution=%.3f -> %.3f",
    oldPrecision, agent.Precision.GetPrecision(math.LevelExecution).Value())
```

## Best Practices

1. **Start Simple**: Begin with single-agent scenarios before adding coordination
2. **Monitor Precision**: Keep an eye on precision levels to ensure learning
3. **Build Trust Gradually**: Don't expect high collaboration rates immediately
4. **Use Appropriate Timeouts**: Set realistic timeouts based on tool complexity
5. **Handle Errors Gracefully**: Always implement fallback mechanisms
6. **Test Thoroughly**: Use provided test suite and add your own integration tests
7. **Monitor Performance**: Use dashboard to identify bottlenecks
8. **Document Capabilities**: Clearly define agent capabilities for better matching
9. **Version Control**: Track state versions for debugging and rollback
10. **Secure APIs**: Use TLS and authentication in production

## Additional Resources

- **API Documentation**: See `/api/v1/` endpoints
- **Protobuf Definitions**: See `pkg/api/proto/v1/lrs.proto`
- **Example Applications**: See `examples/` directory
- **Benchmarks**: Run `make benchmark` for performance metrics
- **Contributing**: See CONTRIBUTING.md for guidelines

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourorg/go-lrs/issues
- Documentation: https://docs.go-lrs.io
- Community: https://discord.gg/go-lrs

---

**Happy Agent Building!** 🤖✨