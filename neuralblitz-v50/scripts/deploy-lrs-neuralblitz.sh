#!/bin/bash

# NeuralBlitz v50.0 - LRS-NeuralBlitz Deployment Script
# Deploy complete bidirectional communication system

set -e

# Configuration
COMPOSE_FILE="docker-compose.lrs-fixed.yml"
LRS_AGENT_PORT="9000"
PYTHON_LRS_PORT="8083"
RUST_LRS_PORT="8084"
GO_LRS_PORT="8085"
JS_LRS_PORT="8086"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

show_usage() {
    echo "NeuralBlitz v50.0 - LRS-NeuralBlitz Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy          Deploy complete communication system"
    echo "  start-lrs       Start LRS agent only"
    echo "  start-bridge    Start all NeuralBlitz services with LRS bridges"
    echo "  stop           Stop all services"
    echo "  status          Show system status"
    echo "  logs           Show service logs"
    echo "  test-communication  Test bidirectional communication"
    echo ""
    echo "Options:"
    echo "  --env-file FILE    Use custom compose file (default: docker-compose.lrs-fixed.yml)"
    echo "  --no-monitoring    Skip monitoring services"
    echo "  --development     Development mode with verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                      # Deploy complete system"
    echo "  $0 start-lrs                  # Start only LRS agent"
    echo "  $0 start-bridge               # Start all services with bridges"
    echo "  $0 test-communication       # Test communication between systems"
}

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
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

deploy_system() {
    print_info "Deploying LRS-NeuralBlitz communication system..."
    
    # Stop any existing services
    docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    
    # Start the system
    docker-compose -f "$COMPOSE_FILE" up -d
    
    print_success "System deployment started"
    print_info "Services starting:"
    print_info "  - LRS Agent: http://localhost:$LRS_AGENT_PORT"
    print_info "  - Python + LRS Bridge: http://localhost:$PYTHON_LRS_PORT (main) + http://localhost:$PYTHON_LRS_PORT (bridge)"
    print_info "  - Rust + LRS Bridge: http://localhost:$RUST_LRS_PORT (main) + http://localhost:$RUST_LRS_PORT (bridge)"
    print_info "  - Go + LRS Bridge: http://localhost:$GO_LRS_PORT (main) + http://localhost:$GO_LRS_PORT (bridge)"
    print_info "  - JavaScript + LRS Bridge: http://localhost:$JS_LRS_PORT (main) + http://localhost:$JS_LRS_PORT (bridge)"
    
    if [[ "$1" != "--no-monitoring" ]]; then
        print_info "Monitoring services:"
        print_info "  - Grafana: http://localhost:3001"
        print_info "  - Prometheus: http://localhost:9090"
    fi
    
    print_success "System deployment completed"
}

start_lrs_only() {
    print_info "Starting LRS agent only..."
    
    docker-compose -f "$COMPOSE_FILE" up -d lrs-agent
    
    print_success "LRS agent started on http://localhost:$LRS_AGENT_PORT"
}

start_bridges() {
    print_info "Starting NeuralBlitz services with LRS bridges..."
    
    docker-compose -f "$COMPOSE_FILE" up -d neuralblitz-python-lrs neuralblitz-rust-lrs neuralblitz-go-lrs neuralblitz-js-lrs
    
    print_success "All services with LRS bridges started"
}

stop_services() {
    print_info "Stopping all services..."
    
    docker-compose -f "$COMPOSE_FILE" down
    
    print_success "All services stopped"
}

show_status() {
    print_info "System status:"
    
    docker-compose -f "$COMPOSE_FILE" ps
    
    print_info ""
    print_info "Health checks:"
    
    # Check LRS Agent
    if curl -s http://localhost:$LRS_AGENT_PORT/health &>/dev/null; then
        print_success "  ✓ LRS Agent: Healthy"
    else
        print_error "  ✗ LRS Agent: Unhealthy"
    fi
    
    # Check bridges
    for port in $PYTHON_LRS_PORT $RUST_LRS_PORT $GO_LRS_PORT $JS_LRS_PORT; do
        if curl -s http://localhost:$port/lrs_bridge/status &>/dev/null; then
            print_success "  ✓ Bridge (port $port): Healthy"
        else
            print_error "  ✗ Bridge (port $port): Unhealthy"
        fi
    done
    
    if [[ "$1" != "--no-monitoring" ]]; then
        print_info "Monitoring services:"
        if curl -s http://localhost:3001 &>/dev/null; then
            print_success "  ✓ Grafana: http://localhost:3001"
        else
            print_error "  ✗ Grafana: Unavailable"
        fi
    fi
}

show_logs() {
    print_info "Service logs:"
    
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    
    if [[ "$1" != "--all" ]]; then
        echo ""
        echo "Show all logs with: $0 logs --all"
        echo "Follow logs with: $0 logs -f [service-name]"
    fi
}

test_communication() {
    print_info "Testing LRS-NeuralBlitz bidirectional communication..."
    
    # Test LRS Agent
    print_info "Testing LRS Agent..."
    lrs_response=$(curl -s -X POST http://localhost:$LRS_AGENT_PORT/neuralblitz/bridge/health \
        -H "Content-Type: application/json" \
        -H "X-System-ID: NEURALBLITZ_V50" \
        -d '{"system_id": "NEURALBLITZ_V50", "metrics": {}}' 2>/dev/null)
    
    if [[ $? -eq 0 && "$lrs_response" == *"healthy"* ]]; then
        print_success "  ✓ LRS Agent communication working"
    else
        print_error "  ✗ LRS Agent communication failed"
        echo "Response: $lrs_response"
    fi
    
    # Test Python bridge
    print_info "Testing Python LRS Bridge..."
    python_response=$(curl -s -X POST http://localhost:$PYTHON_LRS_PORT/lrs_bridge/status 2>/dev/null)
    
    if [[ $? -eq 0 ]]; then
        print_success "  ✓ Python LRS Bridge communication working"
    else
        print_error "  ✗ Python LRS Bridge communication failed"
    fi
    
    # Test intent submission (Python -> LRS)
    print_info "Testing intent submission (Python -> LRS)..."
    intent_response=$(curl -s -X POST http://localhost:$PYTHON_LRS_PORT/lrs_bridge/intent/submit \
        -H "Content-Type: application/json" \
        -H "X-System-ID: NEURALBLITZ_V50" \
        -d '{
            "phi_1": 1.0,
            "phi_22": 1.0,
            "phi_omega": 1.0,
            "metadata": {"source": "NEURALBLITZ_V50", "test": "bidirectional"}
        }' 2>/dev/null)
    
    if [[ $? -eq 0 ]]; then
        print_success "  ✓ Intent submission working"
    else
        print_error "  ✗ Intent submission failed"
    fi
    
    print_info "Communication test completed"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        check_dependencies
        deploy_system "$@"
        ;;
    start-lrs)
        check_dependencies
        start_lrs_only "$@"
        ;;
    start-bridge)
        check_dependencies
        start_bridges "$@"
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status "$@"
        ;;
    logs)
        show_logs "$@"
        ;;
    test-communication)
        test_communication "$@"
        ;;
    --help|*)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac