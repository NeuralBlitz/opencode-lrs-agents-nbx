# Go-LRS Implementation - Complete Documentation

## Executive Summary

Go-LRS is a production-ready implementation of an Active Inference-based agent system using the Free Energy Principle from neuroscience. It implements resilient AI agents with precision tracking, hierarchical state management, and multi-agent coordination.

## Key Features

### Core Capabilities
- **Active Inference Mathematics**: Precision tracking (Beta distribution), Free Energy calculations, Policy selection (Boltzmann distribution)
- **Hierarchical Precision**: Three levels (Abstract → Planning → Execution) with error propagation
- **ToolLens Pattern**: Bidirectional tool abstraction with composition support
- **Multi-Agent Coordination**: Social trust tracking, collaboration management, task assignment

### Production Infrastructure
- **APIs**: gRPC (Protocol Buffers), HTTP/REST (Gin), WebSocket streaming, CLI (Cobra)
- **Storage**: PostgreSQL for persistence, Redis for caching and Pub/Sub
- **Observability**: Prometheus metrics, Grafana dashboards, OpenTelemetry tracing
- **Security**: JWT authentication, API key management, AES/RSA encryption
- **Resilience**: Circuit breaker, retry logic, rate limiting, key rotation

### Deployment
- **Container**: Multi-stage Docker build
- **Orchestration**: Kubernetes manifests, Helm charts
- **CI/CD**: GitHub Actions workflows

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                              │
│                  (gRPC + HTTP + WebSocket)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Agent Manager                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Precision   │ │ Free Energy │ │ Policy Selector     │   │
│  │ Tracker     │ │ Calculator  │ │                     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Immutable State Store                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Tool Registry                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────────┐    │
│  │ Search │ │  Calc  │ │ Format │ │ Custom Tools     │    │
│  └────────┘ └────────┘ └────────┘ └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. Request → API Gateway
2. Agent Manager processes request
3. Policy selection using Free Energy
4. Tool execution with precision update
5. State immutably updated
6. Results returned

## Mathematical Foundations

### Precision (γ)

Precision represents confidence in predictions:

```
γ = α / (α + β)
```

Where:
- α = successes + prior (default: 1.0)
- β = failures + prior (default: 1.0)

Update rules (asymmetric learning):
```
# Success (prediction error = 0)
α' = α + 0.1 × (1 - δ)
β' = β

# Failure (prediction error = 1)
α' = α
β' = β + 0.2 × δ
```

### Expected Free Energy (G)

```
G(π) = Epistemic Value - Pragmatic Value

Epistemic = Σ H[Tool_t]  # Information gain
Pragmatic = Σ γ^t × R      # Expected reward
```

### Policy Selection

```
P(π) ∝ exp(-β × G(π))
```

Where β = 1/γ (inverse temperature)

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /api/v1/agents | Create agent |
| GET | /api/v1/agents | List agents |
| GET | /api/v1/agents/{id} | Get agent |
| DELETE | /api/v1/agents/{id} | Delete agent |
| POST | /api/v1/tasks | Create task |
| GET | /api/v1/tasks/{id} | Get task |

### gRPC Services

```protobuf
service LRSService {
    rpc CreateAgent(CreateAgentRequest) returns (CreateAgentResponse);
    rpc GetAgent(GetAgentRequest) returns (GetAgentResponse);
    rpc ExecutePolicy(ExecutePolicyRequest) returns (ExecutePolicyResponse);
    rpc StreamExecution(ExecutionRequest) returns (stream ExecutionResponse);
}
```

## Installation

### Quick Install

```bash
# Using Go
go install github.com/neuralblitz/go-lrs/cmd/server@latest

# Using Docker
docker pull ghcr.io/neuralblitz/go-lrs:latest

# Using Helm
helm install lrs neuralblitz/go-lrs
```

### Building from Source

```bash
git clone https://github.com/neuralblitz/go-lrs.git
cd go-lrs
make build
```

## Configuration

### Environment Variables

```bash
LRS_HTTP_PORT=8080      # HTTP server port
LRS_GRPC_PORT=9090      # gRPC server port
LRS_DB_HOST=localhost   # PostgreSQL host
LRS_DB_PORT=5432       # PostgreSQL port
LRS_REDIS_HOST=localhost # Redis host
LRS_REDIS_PORT=6379    # Redis port
```

### Config File

```yaml
server:
  http_port: 8080
  grpc_port: 9090

database:
  host: localhost
  port: 5432
  username: lrs_user
  password: ${DB_PASSWORD}
  name: lrs_db

redis:
  host: localhost
  port: 6379
```

## Usage

### Creating an Agent

```go
manager := api.NewAgentManager()

agent, err := manager.CreateAgent(
    "my-agent",
    "A demonstration agent",
    map[string]interface{}{
        "capabilities": []string{"search", "calculate"},
    },
    nil,
)
```

### Executing Tasks

```go
result, err := tool.Execute(ctx, agent.State)
agent.UpdatePrecision(math.LevelExecution, result.PredictionError)
```

## Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Agent Creation | ~50μs | 20K/s |
| State Update | ~1μs | 1M/s |
| Precision Update | ~1μs | 1M/s |
| Free Energy Calc | ~100μs | 10K/s |
| Policy Selection | ~500μs | 2K/s |
| Max Concurrent Agents | | 10,000+ |

## Deployment Options

### Docker

```bash
docker run -p 8080:8080 ghcr.io/neuralblitz/go-lrs:latest
```

### Kubernetes

```bash
kubectl apply -f deploy/k8s/
helm install lrs deploy/helm/lrs/
```

### Docker Compose

```bash
docker-compose up -d
```

## Monitoring

### Prometheus Metrics

```bash
curl http://localhost:8080/metrics
```

### Grafana Dashboards

Import from `deploy/grafana/`

### Health Checks

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

## Security

### JWT Authentication

```bash
# Generate token
lrs-cli auth generate --user admin

# Use token
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/v1/agents
```

### Rate Limiting

| Plan | Requests/Second | Burst |
|------|----------------|-------|
| Free | 10 | 20 |
| Pro | 100 | 200 |
| Enterprise | 1000 | 2000 |

## File Structure

```
go-lrs/
├── cmd/
│   ├── server/main.go       # Server entry point
│   └── cli/main.go         # CLI tool
├── pkg/
│   ├── api/                # API handlers
│   ├── core/               # Core interfaces
│   ├── integration/        # LLM adapters
│   ├── multiagent/         # Coordination
│   ├── monitoring/        # Dashboard
│   ├── security/           # Auth & encryption
│   ├── storage/            # Database
│   └── resilience/         # Patterns
├── internal/
│   ├── math/               # Mathematics
│   ├── state/              # State management
│   └── registry/           # Tool registry
├── deploy/
│   ├── k8s/               # Kubernetes manifests
│   ├── helm/               # Helm charts
│   ├── prometheus/        # Metrics
│   └── grafana/           # Dashboards
├── examples/               # Example code
├── test/                  # Tests
├── docs/                  # Documentation
└── Makefile               # Build automation
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

## License

MIT License

## Support

- GitHub Issues
- Documentation: /docs
- Examples: /examples
