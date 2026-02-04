# NeuralBlitz v50.0 - Kubernetes Deployment

This directory contains the complete Kubernetes manifests for deploying NeuralBlitz v50.0 Omega Singularity Architecture.

## Architecture Overview

The deployment includes three microservices implementing the same API in different languages:
- **Python API** (Port 8080) - FastAPI implementation
- **Rust API** (Port 8081) - Actix-web implementation  
- **Go API** (Port 8082) - Gin framework implementation

## Files

### Core Deployments
- `namespace-and-config.yaml` - Namespace, ConfigMaps, Secrets, Ingress, HPA
- `python-deployment.yaml` - Python FastAPI service deployment
- `rust-deployment.yaml` - Rust Actix-web service deployment
- `go-deployment.yaml` - Go Gin service deployment
- `monitoring-and-rbac.yaml` - Service monitoring, RBAC, and PDBs

### Deployment Options
Each service supports the 6 deployment options (A-F) with different resource allocations:
- **Option A**: Minimal deployment (50MB, 1 core)
- **Option B**: Standard deployment (100MB, 2 cores)
- **Option C**: Enhanced deployment (200MB, 4 cores)
- **Option D**: Production deployment (500MB, 8 cores)
- **Option E**: Enterprise deployment (1000MB, 16 cores)
- **Option F**: Cosmic deployment (2000MB, 32 cores)

## Quick Start

1. **Create namespace and configurations:**
   ```bash
   kubectl apply -f namespace-and-config.yaml
   ```

2. **Deploy all services:**
   ```bash
   kubectl apply -f python-deployment.yaml
   kubectl apply -f rust-deployment.yaml
   kubectl apply -f go-deployment.yaml
   kubectl apply -f monitoring-and-rbac.yaml
   ```

3. **Verify deployment:**
   ```bash
   kubectl get pods -n neuralblitz
   kubectl get services -n neuralblitz
   kubectl get ingress -n neuralblitz
   ```

## API Endpoints

All services implement the same REST API:

- `GET /status` - System status check
- `POST /intent` - Submit intent for processing
- `POST /verify` - Verify GoldenDAG coherence
- `POST /nbcl/interpret` - Interpret NBCL commands
- `GET /attestation` - Get system attestation
- `GET /symbiosis` - Symbiosis field status
- `GET /synthesis` - Synthesis status
- `GET /options/{option}` - Deployment option info

## Monitoring and Observability

- **Prometheus**: Service monitoring configured
- **Grafana**: Dashboards available for metrics
- **Horizontal Pod Autoscaling**: Auto-scaling based on CPU/Memory
- **Pod Disruption Budgets**: High availability guarantees
- **Health Checks**: Liveness and readiness probes

## Configuration

### Environment Variables
- `GOLDEN_DAG_SEED`: Core seed for GoldenDAG operations
- `COHERENCE_TARGET`: Target coherence value (always 1.0)
- `SEPARATION_TARGET`: Target separation value (always 0.0)
- `VERSION`: Application version (v50.0.0)

### Secrets
- `ATTETSTATION_KEY`: Key for Omega attestation protocol
- `SYNTHESIS_KEY`: Key for synthesis operations

## Security

- **TLS/SSL**: Encrypted communication via Ingress
- **RBAC**: Role-based access control
- **Network Policies**: Isolated namespace
- **Secrets Management**: Encrypted sensitive data

## High Availability

- **3 Replicas** per service (minimum)
- **Horizontal Pod Autoscaling** up to 10 replicas
- **Pod Disruption Budgets** ensure minimum availability
- **Health Checks** for automatic failover

## Scaling

The deployment supports:
- **Horizontal Scaling**: Via HPA based on resource usage
- **Vertical Scaling**: By adjusting resource requests/limits
- **Multi-zone**: Can be deployed across availability zones

## Troubleshooting

### Common Issues

1. **Pods not starting:**
   ```bash
   kubectl describe pod -n neuralblitz
   kubectl logs -n neuralblitz <pod-name>
   ```

2. **Service not accessible:**
   ```bash
   kubectl get svc -n neuralblitz
   kubectl describe svc <service-name> -n neuralblitz
   ```

3. **Ingress issues:**
   ```bash
   kubectl describe ingress neuralblitz-ingress -n neuralblitz
   ```

## Cleanup

To remove the entire deployment:
```bash
kubectl delete namespace neuralblitz
```

## Advanced Configuration

### Custom Resource Limits
Edit the `resources` section in each deployment YAML to adjust CPU and memory allocations.

### Custom Ingress Rules
Modify the `neuralblitz-ingress` to add custom routing rules or domain configurations.

### Monitoring Integration
The ServiceMonitor resources assume Prometheus Operator is installed. Adjust as needed for your monitoring stack.