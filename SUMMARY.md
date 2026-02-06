# Go-LRS Documentation Summary

## Complete Documentation Index

### Getting Started
1. [Installation](docs/getting-started/01-installation.md)
2. [Quick Start](docs/getting-started/02-quickstart.md)
3. [Architecture](docs/getting-started/03-architecture.md)
4. [README](docs/getting-started/README.md)

### Architecture
1. [Active Inference](docs/architecture/active_inference.md)
2. [Mathematical Foundations](docs/architecture/mathematical_foundations.md)
3. [Performance Tuning](docs/architecture/performance_tuning.md)

### API
1. [REST API Reference](docs/api/reference.md)
2. [gRPC API](docs/api/grpc_api.md)

### Deployment
1. [Docker](docs/deployment/docker.md)
2. [Kubernetes](docs/deployment/kubernetes.md)
3. [Helm](docs/deployment/helm.md)
4. [Options](docs/deployment/options.md)

### Guides
1. [Configuration](docs/configuration.md)
2. [Performance](docs/performance.md)
3. [Security Best Practices](docs/security/best_practices.md)
4. [Contributing](docs/contributing/CONTRIBUTING.md)

## Quick Links

| Topic | Link |
|-------|------|
| Installation | [Quick Start](docs/getting-started/02-quickstart.md) |
| API Docs | [REST Reference](docs/api/reference.md) |
| Deployment | [Docker](docs/deployment/docker.md) |
| Configuration | [Config Options](docs/configuration.md) |
| Performance | [Benchmarks](docs/performance.md) |

## Examples

| Example | Location |
|---------|----------|
| Basic Agent | [examples/basic/](examples/basic/) |
| Custom Tools | [examples/custom_tools/](examples/custom_tools/) |
| Multi-Agent | [examples/advanced/](examples/advanced/) |
| Production | [examples/production/](examples/production/) |

## API Endpoints

### Agents
- `POST /api/v1/agents` - Create agent
- `GET /api/v1/agents` - List agents
- `GET /api/v1/agents/{id}` - Get agent
- `DELETE /api/v1/agents/{id}` - Delete agent

### Tasks
- `POST /api/v1/tasks` - Create task
- `GET /api/v1/tasks/{id}` - Get task

### Collaboration
- `POST /api/v1/collaborations` - Create collaboration
- `GET /api/v1/collaborations/{id}` - Get collaboration

## Configuration

See [Configuration Reference](docs/configuration.md) for all options.

## Performance

See [Performance Benchmarks](docs/performance.md) for metrics.
