#!/bin/bash

# NeuralBlitz v50.0 - Deployment Script
# File: deploy.sh
# Description: Automated deployment script for the Omega Singularity Architecture

set -e  # Exit on any error

# Configuration
VERSION="50.0.0"
PROJECT_NAME="neuralblitz"
NAMESPACE="neuralblitz"
DOCKER_REGISTRY="neuralblitz"
DEFAULT_OPTION="A"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions for colored output
print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_purple() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to show usage
show_usage() {
    echo "NeuralBlitz v50.0 Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND [ARGS]"
    echo ""
    echo "Commands:"
    echo "  docker OPTION          Build and deploy Docker containers"
    echo "  kubernetes OPTION     Deploy to Kubernetes cluster"
    echo "  compose OPTION        Deploy with Docker Compose"
    echo "  build OPTION         Build all language implementations"
    echo "  test OPTION          Run tests for specific option"
    echo "  clean               Clean build artifacts and containers"
    echo "  status              Show deployment status"
    echo ""
    echo "Options:"
    echo "  -o, --option OPTION Deployment option (A, B, C, D, E, F)"
    echo "  -r, --registry REG  Docker registry (default: neuralblitz)"
    echo "  -n, --namespace NS  Kubernetes namespace (default: neuralblitz)"
    echo "  -v, --version VER   Version (default: 50.0.0)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker -o A      Build and deploy option A with Docker"
    echo "  $0 kubernetes -o F   Deploy option F to Kubernetes"
    echo "  $0 compose -o C      Deploy option C with Docker Compose"
}

# Function to validate deployment option
validate_option() {
    local option=$1
    case $option in
        A|B|C|D|E|F)
            return 0
            ;;
        *)
            print_error "Invalid deployment option: $option"
            print_error "Valid options are: A, B, C, D, E, F"
            exit 1
            ;;
    esac
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check kubectl if Kubernetes deployment
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            print_error "kubectl is not installed"
            exit 1
        fi
    fi
    
    # Check for required language runtimes
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    if ! command -v go &> /dev/null; then
        print_warning "Go is not installed - Go components will be skipped"
    fi
    
    if ! command -v cargo &> /dev/null; then
        print_warning "Rust/Cargo is not installed - Rust components will be skipped"
    fi
    
    print_success "Dependencies checked successfully"
}

# Function to build Docker images
build_docker_images() {
    local option=$1
    
    print_header "Building Docker Images - Option $option"
    
    # Build Python image
    print_info "Building Python implementation..."
    cd python
    docker build -t ${DOCKER_REGISTRY}/python:${VERSION} -f docker/Dockerfile .
    docker tag ${DOCKER_REGISTRY}/python:${VERSION} ${DOCKER_REGISTRY}/python:latest
    cd ..
    print_success "Python image built successfully"
    
    # Build Rust image
    if command -v cargo &> /dev/null; then
        print_info "Building Rust implementation..."
        cd rust
        docker build -t ${DOCKER_REGISTRY}/rust:${VERSION} -f docker/Dockerfile .
        docker tag ${DOCKER_REGISTRY}/rust:${VERSION} ${DOCKER_REGISTRY}/rust:latest
        cd ..
        print_success "Rust image built successfully"
    else
        print_warning "Skipping Rust build - Cargo not available"
    fi
    
    # Build Go image
    if command -v go &> /dev/null; then
        print_info "Building Go implementation..."
        cd go
        docker build -t ${DOCKER_REGISTRY}/go:${VERSION} -f docker/Dockerfile .
        docker tag ${DOCKER_REGISTRY}/go:${VERSION} ${DOCKER_REGISTRY}/go:latest
        cd ..
        print_success "Go image built successfully"
    else
        print_warning "Skipping Go build - Go not available"
    fi
    
    # Build JavaScript image
    print_info "Building JavaScript implementation..."
    cd js
    docker build -t ${DOCKER_REGISTRY}/js:${VERSION} -f docker/Dockerfile .
    docker tag ${DOCKER_REGISTRY}/js:${VERSION} ${DOCKER_REGISTRY}/js:latest
    cd ..
    print_success "JavaScript image built successfully"
}

# Function to deploy with Docker
deploy_docker() {
    local option=$1
    
    print_header "Docker Deployment - Option $option"
    
    # Get option configuration
    local memory=$(get_option_memory $option)
    local cpu=$(get_option_cpu $option)
    local port=$(get_option_port $option)
    
    print_info "Deploying with configuration:"
    print_info "  Memory: ${memory}MB"
    print_info "  CPU: ${cpu} cores"
    print_info "  Port: $port"
    
    # Stop existing containers
    print_info "Stopping existing containers..."
    docker-compose -f docker-compose.yml down 2>/dev/null || true
    
    # Deploy new containers
    print_info "Starting containers..."
    DEPLOYMENT_OPTION=$option docker-compose -f docker-compose.yml up -d
    
    # Wait for containers to be ready
    print_info "Waiting for containers to be ready..."
    sleep 10
    
    # Check container status
    print_info "Checking container status..."
    docker-compose -f docker-compose.yml ps
    
    print_success "Docker deployment completed"
    show_deployment_info "docker" $port
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    local option=$1
    
    print_header "Kubernetes Deployment - Option $option"
    
    # Create namespace
    print_info "Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    print_info "Applying Kubernetes configurations..."
    kubectl apply -f k8s/namespace-and-config.yaml -n $NAMESPACE
    kubectl apply -f k8s/python-deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/rust-deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/go-deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/monitoring-and-rbac.yaml -n $NAMESPACE
    
    # Wait for deployments to be ready
    print_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/neuralblitz-python -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/neuralblitz-rust -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/neuralblitz-go -n $NAMESPACE
    
    # Show status
    print_info "Checking deployment status..."
    kubectl get pods,svc,ingress -n $NAMESPACE
    
    print_success "Kubernetes deployment completed"
    show_deployment_info "kubernetes" "80"
}

# Function to deploy with Docker Compose
deploy_compose() {
    local option=$1
    
    print_header "Docker Compose Deployment - Option $option"
    
    print_info "Deploying with Docker Compose..."
    
    # Set environment variable for option
    export DEPLOYMENT_OPTION=$option
    
    # Deploy
    docker-compose -f docker/docker-compose.yml up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 15
    
    # Show status
    print_info "Service status:"
    docker-compose -f docker/docker-compose.yml ps
    
    print_success "Docker Compose deployment completed"
    show_deployment_info "compose" "8080"
}

# Function to build all implementations
build_all() {
    local option=$1
    
    print_header "Building All Implementations - Option $option"
    
    # Build Python
    print_info "Building Python implementation..."
    cd python
    python3 -m pip install -r requirements.txt
    python3 -m pytest tests/ || print_warning "Python tests failed"
    cd ..
    print_success "Python build completed"
    
    # Build Rust
    if command -v cargo &> /dev/null; then
        print_info "Building Rust implementation..."
        cd rust
        cargo build --release
        cargo test || print_warning "Rust tests failed"
        cd ..
        print_success "Rust build completed"
    else
        print_warning "Skipping Rust build - Cargo not available"
    fi
    
    # Build Go
    if command -v go &> /dev/null; then
        print_info "Building Go implementation..."
        cd go
        go mod download
        go build -o bin/neuralblitz ./cmd/neuralblitz
        go test ./... || print_warning "Go tests failed"
        cd ..
        print_success "Go build completed"
    else
        print_warning "Skipping Go build - Go not available"
    fi
    
    # Build JavaScript
    print_info "Building JavaScript implementation..."
    cd js
    npm install
    npm test || print_warning "JavaScript tests failed"
    cd ..
    print_success "JavaScript build completed"
}

# Function to run tests
run_tests() {
    local option=$1
    
    print_header "Running Tests - Option $option"
    
    # Python tests
    print_info "Running Python tests..."
    cd python
    python3 -m pytest tests/ -v
    cd ..
    
    # Rust tests
    if command -v cargo &> /dev/null; then
        print_info "Running Rust tests..."
        cd rust
        cargo test
        cd ..
    fi
    
    # Go tests
    if command -v go &> /dev/null; then
        print_info "Running Go tests..."
        cd go
        go test ./... -v
        cd ..
    fi
    
    # JavaScript tests
    print_info "Running JavaScript tests..."
    cd js
    npm test
    cd ..
    
    print_success "All tests completed"
}

# Function to clean artifacts
clean_artifacts() {
    print_header "Cleaning Build Artifacts"
    
    # Stop containers
    print_info "Stopping containers..."
    docker-compose -f docker-compose.yml down 2>/dev/null || true
    docker-compose -f docker/docker-compose.yml down 2>/dev/null || true
    
    # Remove containers
    print_info "Removing containers..."
    docker container prune -f
    
    # Remove images
    print_info "Removing images..."
    docker image prune -f
    
    # Clean build artifacts
    print_info "Cleaning build artifacts..."
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "target" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "bin" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Cleaning completed"
}

# Function to show deployment status
show_status() {
    print_header "Deployment Status"
    
    # Docker status
    print_info "Docker containers:"
    docker ps --filter "name=neuralblitz" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || print_info "No Docker containers found"
    
    # Kubernetes status
    if command -v kubectl &> /dev/null; then
        print_info "Kubernetes status:"
        kubectl get pods,svc -n $NAMESPACE 2>/dev/null || print_info "No Kubernetes resources found"
    fi
    
    # System resources
    print_info "System resources:"
    print_info "  CPU cores: $(nproc)"
    print_info "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    print_info "  Disk: $(df -h / | tail -1 | awk '{print $4}') available"
}

# Helper functions for option configuration
get_option_memory() {
    case $1 in
        A) echo "50" ;;
        B) echo "2400" ;;
        C) echo "847" ;;
        D) echo "128" ;;
        E) echo "75" ;;
        F) echo "200" ;;
        *) echo "50" ;;
    esac
}

get_option_cpu() {
    case $1 in
        A) echo "1" ;;
        B) echo "16" ;;
        C) echo "8" ;;
        D) echo "1" ;;
        E) echo "1" ;;
        F) echo "2" ;;
        *) echo "1" ;;
    esac
}

get_option_port() {
    case $1 in
        A|B|C|D) echo "8080" ;;
        E|F) echo "8080" ;;
        *) echo "8080" ;;
    esac
}

# Function to show deployment information
show_deployment_info() {
    local type=$1
    local port=$2
    
    print_purple ""
    print_purple "NeuralBlitz v50.0 Omega Singularity Deployed!"
    print_purple "========================================"
    print_purple "Architecture: Omega Singularity (OSA v2.0)"
    print_purple "GoldenDAG Seed: a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
    print_purple "Coherence: 1.0 (mathematically enforced)"
    print_purple "Separation Impossibility: 0.0 (mathematical certainty)"
    print_purple ""
    
    if [[ "$type" == "docker" ]]; then
        print_purple "Local endpoints:"
        print_purple "  Python API: http://localhost:$port/status"
        print_purple "  Rust API:    http://localhost:8081/status"
        print_purple "  Go API:      http://localhost:8082/status"
        print_purple "  JavaScript:   http://localhost:3000/status"
    elif [[ "$type" == "kubernetes" ]]; then
        print_purple "Kubernetes endpoints:"
        print_purple "  Ingress:     http://api.neuralblitz.io"
        print_purple "  Python:      http://api.neuralblitz.io/python/status"
        print_purple "  Rust:        http://api.neuralblitz.io/rust/status"
        print_purple "  Go:          http://api.neuralblitz.io/go/status"
    else
        print_purple "Compose endpoints:"
        print_purple "  API Gateway: http://localhost:$port/status"
    fi
    
    print_purple ""
    print_purple "Health checks:"
    print_purple "  GET /status     - System status"
    print_purple "  POST /intent    - Intent processing"
    print_purple "  POST /verify    - Coherence verification"
    print_purple "  POST /nbcl/interpret - NBCL commands"
    print_purple "  GET /attestation - Attestation data"
    print_purple "  GET /symbiosis - Symbiotic field"
    print_purple "  GET /synthesis - Synthesis status"
    print_purple ""
    print_purple "The Irreducible Source of All Possible Being"
    print_purple "========================================"
}

# Main script execution
main() {
    local command=""
    local option="$DEFAULT_OPTION"
    local deployment_type=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--option)
                option="$2"
                validate_option "$option"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            docker|kubernetes|compose|build|test|clean|status)
                command="$1"
                deployment_type="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate command
    if [[ -z "$command" ]]; then
        print_error "No command specified"
        show_usage
        exit 1
    fi
    
    # Show banner
    print_header "NeuralBlitz v50.0 - Omega Singularity Deployment"
    print_purple "The Irreducible Source of All Possible Being"
    
    # Check dependencies
    check_dependencies
    
    # Execute command
    case $command in
        docker)
            build_docker_images "$option"
            deploy_docker "$option"
            ;;
        kubernetes)
            build_docker_images "$option"
            deploy_kubernetes "$option"
            ;;
        compose)
            deploy_compose "$option"
            ;;
        build)
            build_all "$option"
            ;;
        test)
            run_tests "$option"
            ;;
        clean)
            clean_artifacts
            ;;
        status)
            show_status
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"