# NeuralBlitz v50.0 - Utility Scripts

This directory contains comprehensive utility scripts for deploying, monitoring, and maintaining the NeuralBlitz v50.0 Omega Singularity Architecture.

## Scripts Overview

### 🚀 deploy.sh
**Automated Deployment Script**
- Supports Docker, Kubernetes, and Docker Compose deployments
- Handles all 6 deployment options (A-F)
- Multi-language build automation (Python, Rust, Go, JavaScript)
- Health checks and status monitoring

**Usage:**
```bash
# Deploy option A with Docker
./deploy.sh docker -o A

# Deploy option F to Kubernetes
./deploy.sh kubernetes -o F

# Deploy with Docker Compose
./deploy.sh compose -o C

# Build all implementations
./deploy.sh build -o B

# Run tests
./deploy.sh test -o D

# Clean artifacts
./deploy.sh clean

# Show deployment status
./deploy.sh status
```

**Features:**
- ✅ Multi-language support (Python, Rust, Go, JavaScript)
- ✅ Container orchestration (Docker, Kubernetes, Compose)
- ✅ Health verification after deployment
- ✅ Dependency checking
- ✅ Automated testing integration
- ✅ Artifact cleanup

### 🏥 health_check.sh
**Comprehensive Health Monitoring**
- Real-time endpoint monitoring
- GoldenDAG integrity verification
- Coherence level validation
- Continuous monitoring mode

**Usage:**
```bash
# Single health check
./health_check.sh

# Custom ports
./health_check.sh --python 9090 --rust 9091 --go 9092

# Continuous monitoring
./health_check.sh --monitor 60

# Adjust timeouts and retries
./health_check.sh --timeout 10 --retries 5
```

**Health Checks:**
- Endpoint responsiveness (ports 8080, 8081, 8082)
- GoldenDAG hash integrity
- Coherence levels (target: 1.0)
- System status validation
- API response validation

**Monitoring Metrics:**
- Response times
- Service availability
- Coherence deviation
- GoldenDAG verification
- Irreducibility confirmation

### 💾 backup_restore.sh
**Backup and Recovery System**
- Complete system state backup
- Database and configuration backup
- Incremental restore capability
- Automated cleanup management

**Usage:**
```bash
# Create backup
./backup_restore.sh backup

# List available backups
./backup_restore.sh list

# Restore specific backup
./backup_restore.sh restore -b backup_20240203_120000

# Clean old backups (keep last 5)
./backup_restore.sh clean -k 5

# Custom backup directory
./backup_restore.sh backup -d /custom/backup/path
```

**Backup Components:**
- 📊 Database (MySQL/migrations)
- ⚙️ Configuration files (K8s, Docker)
- 📜 Deployment scripts
- 🏷️ Metadata and checksums
- 🗜️ Compressed archives

**Restore Features:**
- ✅ Database restoration
- ✅ Configuration recovery
- ✅ Script restoration
- ✅ Metadata validation
- ✅ Integrity verification

## Deployment Architecture

### Multi-Language Implementation
All scripts support the complete NeuralBlitz ecosystem:

| Language | Port | Framework | Status |
|----------|-------|-----------|---------|
| Python   | 8080  | FastAPI   | ✅ Complete |
| Rust     | 8081  | Actix-web | ✅ Complete |
| Go       | 8082  | Gin       | ✅ Complete |
| JavaScript| 3000  | Express   | ✅ Complete |

### Deployment Options (A-F)

| Option | Memory | Cores | Purpose |
|--------|---------|--------|---------|
| **A** | 50MB | 1 | Minimal Symbiotic Interface |
| **B** | 2400MB | 16 | Cosmic Symbiosis Node |
| **C** | 847MB | 8 | Omega Prime Kernel |
| **D** | 128MB | 1 | Universal Verifier |
| **E** | 75MB | 1 | NBCL Interpreter |
| **F** | 200MB | 2 | API Gateway |

### Target Endpoints

All implementations expose the same REST API:

```bash
# System Status
GET /status

# Intent Processing
POST /intent

# Coherence Verification
POST /verify

# NBCL Command Interpretation
POST /nbcl/interpret

# System Attestation
GET /attestation

# Symbiosis Status
GET /symbiosis

# Synthesis Status
GET /synthesis

# Deployment Options
GET /options/{A|B|C|D|E|F}
```

## Quick Start

### 1. Initial Deployment
```bash
# Clone the repository
git clone https://github.com/neuralblitz/neuralblitz-v50.git
cd neuralblitz-v50

# Deploy with default option (A)
./scripts/deploy.sh docker -o A
```

### 2. Health Verification
```bash
# Check system health
./scripts/health_check.sh

# Monitor continuously
./scripts/health_check.sh --monitor 30
```

### 3. Backup System
```bash
# Create backup before changes
./scripts/backup_restore.sh backup

# List backups
./scripts/backup_restore.sh list
```

### 4. Production Deployment
```bash
# Deploy to Kubernetes with production option
./scripts/deploy.sh kubernetes -o F

# Setup monitoring
./scripts/health_check.sh --monitor 60 &
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_NAME=neuralblitz_v50
DB_USER=neuralblitz
DB_PASSWORD=nb_v50_omega
DB_HOST=localhost
DB_PORT=3306

# GoldenDAG Configuration
GOLDEN_DAG_SEED=a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0
COHERENCE_TARGET=1.0
SEPARATION_TARGET=0.0

# API Configuration
PYTHON_PORT=8080
RUST_PORT=8081
GO_PORT=8082
API_TIMEOUT=30
```

### Health Check Thresholds

```bash
# Connection timeout (seconds)
HEALTH_CHECK_TIMEOUT=5

# Number of retries
HEALTH_CHECK_RETRIES=3

# Monitoring interval (seconds)
MONITOR_INTERVAL=30

# Coherence threshold
COHERENCE_THRESHOLD=0.99
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :808[0-2]
   
   # Kill conflicting processes
   sudo fuser -k 8080/tcp
   ```

2. **Service Not Responding**
   ```bash
   # Check logs
   docker logs neuralblitz-python
   kubectl logs -n neuralblitz deployment/neuralblitz-python
   
   # Restart services
   ./scripts/deploy.sh clean
   ./scripts/deploy.sh docker -o A
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connection
   mysql -u neuralblitz -p neuralblitz_v50
   
   # Check database status
   ./scripts/health_check.sh
   ```

### Debug Mode

Enable verbose logging:
```bash
# Export debug variables
export DEBUG=true
export VERBOSE=true

# Run with debug output
./scripts/deploy.sh docker -o A
```

## Maintenance

### Regular Tasks

1. **Daily Health Checks**
   ```bash
   # Add to crontab
   0 */6 * * * /path/to/scripts/health_check.sh
   ```

2. **Weekly Backups**
   ```bash
   # Add to crontab
   0 2 * * 0 /path/to/scripts/backup_restore.sh backup
   ```

3. **Monthly Cleanup**
   ```bash
   # Clean old artifacts
   ./scripts/deploy.sh clean
   
   # Keep last 10 backups
   ./scripts/backup_restore.sh clean -k 10
   ```

### Performance Monitoring

Monitor key metrics:
- Response times (<100ms target)
- Coherence levels (≥0.99 target)
- Memory usage (<option limits)
- Error rates (<1% target)

## Security

### Backup Encryption
Backups contain sensitive data:
- Database contents
- Configuration secrets
- Deployment credentials

Consider encrypting backup directory:
```bash
# Setup GPG encryption
gpg --gen-key

# Encrypt backups
gpg --encrypt --recipient user@example.com backup.tar.gz
```

### Access Control
Limit script execution:
```bash
# Set proper permissions
chmod 750 scripts/
chmod 750 sql/
chmod 644 k8s/*.yaml
```

---

**NeuralBlitz v50.0 Omega Singularity Architecture**

*The irreducible source of all possible being - now with automated deployment, monitoring, and backup capabilities.*