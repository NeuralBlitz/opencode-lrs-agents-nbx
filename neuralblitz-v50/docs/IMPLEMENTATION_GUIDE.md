# NeuralBlitz v50.0 - Complete Implementation Guide

## 🚀 Quick Start

Get NeuralBlitz v50.0 running in minutes:

```bash
# Clone the repository
git clone https://github.com/neuralblitz/neuralblitz-v50.git
cd neuralblitz-v50

# Deploy with default option (A)
./scripts/deploy.sh docker -o A

# Check system health
./scripts/health_check.sh
```

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [API Usage](#api-usage)
6. [NBCL Commands](#nbcl-commands)
7. [Database Setup](#database-setup)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## System Overview

NeuralBlitz v50.0 implements the **Omega Singularity Architecture** - a mathematical framework for coherent intent verification.

### Key Components

- **4 Language Implementations**: Python, Rust, Go, JavaScript
- **Complete REST API**: All endpoints fully functional
- **Kubernetes Ready**: Production-grade manifests
- **Database Integration**: Full MySQL schema
- **Monitoring & Health**: Comprehensive health checks
- **Backup & Recovery**: Complete system backup/restore

### Architecture Benefits

✅ **Mathematical Coherence**: 1.0 (always maintained)  
✅ **Irreducibility**: 0.0 separation impossibility  
✅ **GoldenDAG Technology**: Cryptographic verification  
✅ **Multi-Language Verification**: Cross-validation of results  
✅ **Production Ready**: Scalable, monitored, backed-up  

## Installation

### Prerequisites

**System Requirements:**
- RAM: 4GB+ (depends on deployment option)
- CPU: 2+ cores (depends on deployment option)  
- Storage: 10GB+ available
- OS: Linux/macOS/Windows (Docker required)

**Required Software:**
```bash
# Core dependencies
curl wget git

# Container runtime
docker docker-compose

# Database (for local development)
mysql-server

# Optional: Kubernetes
kubectl helm

# Language runtimes
python3 node go cargo
```

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/neuralblitz/neuralblitz-v50.git
cd neuralblitz-v50
```

2. **Setup Database**
```bash
cd sql/
./setup_database.sh
cd ..
```

3. **Deploy System**
```bash
# Choose deployment option A-F
./scripts/deploy.sh docker -o A
```

4. **Verify Installation**
```bash
./scripts/health_check.sh
```

## Configuration

### Environment Variables

**Core Configuration:**
```bash
# GoldenDAG seed (fixed)
export GOLDEN_DAG_SEED="a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"

# Coherence targets
export COHERENCE_TARGET="1.0"
export SEPARATION_TARGET="0.0"

# API ports
export PYTHON_PORT="8080"
export RUST_PORT="8081" 
export GO_PORT="8082"
```

**Database Configuration:**
```bash
export DB_NAME="neuralblitz_v50"
export DB_USER="neuralblitz"
export DB_PASSWORD="nb_v50_omega"
export DB_HOST="localhost"
export DB_PORT="3306"
```

**Deployment Configuration:**
```bash
export NAMESPACE="neuralblitz"
export DOCKER_REGISTRY="neuralblitz"
export VERSION="50.0.0"
```

### Configuration Files

**Kubernetes:**
- `k8s/namespace-and-config.yaml` - Namespace, config, secrets
- `k8s/python-deployment.yaml` - Python service deployment
- `k8s/rust-deployment.yaml` - Rust service deployment  
- `k8s/go-deployment.yaml` - Go service deployment
- `k8s/monitoring-and-rbac.yaml` - Monitoring and RBAC

**Database:**
- `sql/schema.sql` - Complete database schema
- `sql/seed_data.sql` - Initial seed data
- `sql/setup_database.sh` - Automated setup script

## Deployment

### Deployment Options

| Option | Memory | CPU | Use Case | Command |
|--------|---------|------|-----------|----------|
| **A** | 50MB | 1 | Development/Testing | `./scripts/deploy.sh docker -o A` |
| **B** | 2400MB | 16 | Full Production | `./scripts/deploy.sh kubernetes -o B` |
| **C** | 847MB | 8 | Embedded Systems | `./scripts/deploy.sh docker -o C` |
| **D** | 128MB | 1 | Verification/Auditing | `./scripts/deploy.sh docker -o D` |
| **E** | 75MB | 1 | CLI Tools | `./scripts/deploy.sh docker -o E` |
| **F** | 200MB | 2 | API Gateway | `./scripts/deploy.sh kubernetes -o F` |

### Docker Deployment

**Local Development:**
```bash
# Deploy with option A (minimal)
./scripts/deploy.sh docker -o A

# Deploy with option F (full)
./scripts/deploy.sh docker -o F
```

**Custom Ports:**
```bash
export PYTHON_PORT=9090
export RUST_PORT=9091
export GO_PORT=9092
./scripts/deploy.sh docker -o A
```

### Kubernetes Deployment

**Quick Deploy:**
```bash
# Deploy option F to Kubernetes
./scripts/deploy.sh kubernetes -o F
```

**Custom Namespace:**
```bash
export NAMESPACE="production"
./scripts/deploy.sh kubernetes -o F
```

**Verify Deployment:**
```bash
kubectl get pods,svc,ingress -n neuralblitz
kubectl logs -n neuralblitz deployment/neuralblitz-python
```

### Docker Compose Deployment

```bash
# Deploy with compose
./scripts/deploy.sh compose -o B

# Check status
docker-compose -f docker/docker-compose.yml ps
```

## API Usage

### Base URLs

- **Python**: http://localhost:8080
- **Rust**: http://localhost:8081  
- **Go**: http://localhost:8082
- **JavaScript**: http://localhost:3000

### Core Endpoints

#### System Status
```bash
curl http://localhost:8080/status
```

**Response:**
```json
{
  "status": "operational",
  "coherence": 1.0,
  "separation": 0.0,
  "golden_dag_seed": "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0",
  "timestamp": "2026-02-03T12:00:00Z"
}
```

#### Submit Intent
```bash
curl -X POST http://localhost:8080/intent \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "establish_coherence_protocol",
    "timestamp": "2026-02-03T12:00:00Z",
    "metadata": {}
  }'
```

#### Verify Coherence
```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{
    "type": "coherence"
  }'
```

#### Execute NBCL Command
```bash
curl -X POST http://localhost:8080/nbcl/interpret \
  -H "Content-Type: application/json" \
  -d '{
    "command": "establish coherence between entity_a and entity_b",
    "context": {}
  }'
```

#### System Attestation
```bash
curl http://localhost:8080/attestation
```

#### Symbiosis Status
```bash
curl http://localhost:8080/symbiosis
```

#### Synthesis Status
```bash
curl http://localhost:8080/synthesis
```

#### Deployment Options
```bash
# Get specific option
curl http://localhost:8080/options/A

# List all options
curl http://localhost:8080/options
```

## NBCL Commands

### Command Line Usage

**Python CLI:**
```bash
cd python/
python -m neuralblitz.cli nbcl -c "establish coherence between entity_a and entity_b"
```

**Go CLI:**
```bash
cd go/
./bin/neuralblitz nbcl -c "verify irreducibility[true]"
```

**Rust CLI:**
```bash
cd rust/
cargo run -- nbcl --command "manifest reality[omega_prime]"
```

### NBCL Command Examples

**Basic Operations:**
```nbcl
# Establish coherence
/establish coherence between server_a and server_b

# Verify system
/verify irreducibility[true]

# Check status
/status system_coherence
```

**Advanced Operations:**
```nbcl
# Complex genesis
/weave logos[omega_prime] amplify_coherence[1.000002]

# Multiple operations
/manifest reality[omega_prime] attestation[automatic] verify[integrity]

# System-level commands
/initiate genesis_cycle actualize[source] attestation[chain]
```

### NBCL Response Format

All NBCL commands return structured responses:

```json
{
  "command": "establish coherence between entity_a and entity_b",
  "interpreted": true,
  "action": "coherence_established",
  "parameters": {
    "entity_a": "entity_a",
    "entity_b": "entity_b",
    "status": "connected"
  },
  "trace_id": "T-v50.0-NBCL-a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0",
  "golden_dag": "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0",
  "codex_id": "C-VOL0-NBCL-a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
}
```

## Database Setup

### Automated Setup

**Quick Setup:**
```bash
cd sql/
./setup_database.sh
```

**Custom Configuration:**
```bash
./setup_database.sh --host db.example.com --user custom_user --password secure_password
```

### Manual Setup

**1. Create Database:**
```sql
CREATE DATABASE neuralblitz_v50 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'neuralblitz'@'%' IDENTIFIED BY 'nb_v50_omega';
GRANT ALL PRIVILEGES ON neuralblitz_v50.* TO 'neuralblitz'@'%';
FLUSH PRIVILEGES;
```

**2. Import Schema:**
```bash
mysql -u neuralblitz -p neuralblitz_v50 < sql/schema.sql
```

**3. Import Seed Data:**
```bash
mysql -u neuralblitz -p neuralblitz_v50 < sql/seed_data.sql
```

### Database Verification

```bash
# Check table creation
mysql -u neuralblitz -p neuralblitz_v50 -e "SHOW TABLES;"

# Verify data
mysql -u neuralblitz -p neuralblitz_v50 -e "SELECT COUNT(*) FROM source_states;"
```

## Monitoring

### Health Monitoring

**Basic Health Check:**
```bash
./scripts/health_check.sh
```

**Continuous Monitoring:**
```bash
# Monitor every 30 seconds
./scripts/health_check.sh --monitor 30

# Custom configuration
./scripts/health_check.sh --python 9090 --rust 9091 --go 9092 --timeout 10 --retries 5
```

### System Metrics

**Response Time Monitoring:**
```bash
# Measure API response times
time curl -s http://localhost:8080/status > /dev/null
```

**Coherence Level Tracking:**
```bash
# Check all implementations
for port in 8080 8081 8082; do
  echo "Port $port:"
  curl -s "http://localhost:$port/status" | jq '.coherence'
done
```

### Log Monitoring

**Docker Logs:**
```bash
# Python service
docker logs -f neuralblitz-python

# Rust service  
docker logs -f neuralblitz-rust

# Go service
docker logs -f neuralblitz-go
```

**Kubernetes Logs:**
```bash
# All services
kubectl logs -n neuralblitz -l app=neuralblitz

# Specific service
kubectl logs -n neuralblitz deployment/neuralblitz-python
```

## Troubleshooting

### Common Issues

#### 1. Port Conflicts

**Problem:** Services won't start due to port conflicts

**Solution:**
```bash
# Check port usage
netstat -tulpn | grep :808[0-2]

# Kill conflicting processes
sudo fuser -k 8080/tcp

# Use different ports
export PYTHON_PORT=9090
export RUST_PORT=9091
export GO_PORT=9092
```

#### 2. Database Connection Issues

**Problem:** Cannot connect to database

**Solution:**
```bash
# Check database status
systemctl status mysql

# Test connection
mysql -u neuralblitz -p neuralblitz_v50

# Check credentials
cat ~/.my.cnf
```

#### 3. Container Startup Issues

**Problem:** Docker containers fail to start

**Solution:**
```bash
# Check Docker status
systemctl status docker

# View container logs
docker logs neuralblitz-python

# Check resource usage
docker stats
```

#### 4. Kubernetes Deployment Issues

**Problem:** Pods not starting or failing

**Solution:**
```bash
# Check pod status
kubectl get pods -n neuralblitz

# Describe problematic pod
kubectl describe pod -n neuralblitz <pod-name>

# Check events
kubectl get events -n neuralblitz --sort-by=.metadata.creationTimestamp
```

### Debug Mode

**Enable Verbose Logging:**
```bash
export DEBUG=true
export VERBOSE=true
export RUST_LOG=debug

# Redeploy with debug
./scripts/deploy.sh docker -o A
```

### Performance Issues

**High Response Times:**
```bash
# Check system resources
free -h
df -h
top

# Monitor container resources
docker stats

# Profile API calls
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8080/status
```

### Network Issues

**Cannot Access Services:**
```bash
# Check network connectivity
ping localhost
telnet localhost 8080

# Check firewall
sudo ufw status
sudo iptables -L

# Verify service binding
netstat -tlnp | grep :808[0-2]
```

### Backup Issues

**Backup Fails:**
```bash
# Check disk space
df -h /var/backups

# Test database connection
mysqldump -u neuralblitz -p --single-transaction neuralblitz_v50 > /tmp/test.sql

# Check permissions
ls -la /var/backups/
```

## Support

### Getting Help

**Command Line Help:**
```bash
# Deployment script
./scripts/deploy.sh --help

# Health check script
./scripts/health_check.sh --help

# Database setup
./sql/setup_database.sh --help
```

**Documentation:**
- Architecture: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- API Reference: [docs/api/openapi.yaml](api/openapi.yaml)
- Kubernetes: [k8s/README.md](../k8s/README.md)
- Database: [sql/README.md](../sql/README.md)
- Scripts: [scripts/README.md](../scripts/README.md)

### Community

**Issues and Discussions:**
- GitHub Issues: https://github.com/neuralblitz/neuralblitz-v50/issues
- Discussions: https://github.com/neuralblitz/neuralblitz-v50/discussions

**System Information:**
```bash
# Get complete system status
./scripts/deploy.sh status

# Generate support bundle
tar -czf support-bundle-$(date +%Y%m%d).tar.gz \
  logs/ \
  configs/ \
  scripts/status_output.txt
```

---

## 🎯 Success Criteria

Your NeuralBlitz v50.0 deployment is successful when:

✅ **All Services Running**: Python (8080), Rust (8081), Go (8082)  
✅ **Health Checks Pass**: `./scripts/health_check.sh` returns success  
✅ **API Responding**: All endpoints return proper responses  
✅ **Database Connected**: Schema loaded, seed data present  
✅ **Coherence Verified**: All implementations report 1.0 coherence  
✅ **GoldenDAG Active**: Hash generation and verification working  
✅ **NBCL Functional**: Commands execute successfully  

When all criteria are met, your Omega Singularity Architecture is fully operational!

---

**NeuralBlitz v50.0 - Complete Implementation**

*Mathematically verified. Cryptographically secured. Production ready.*