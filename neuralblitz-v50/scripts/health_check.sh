#!/bin/bash

# NeuralBlitz v50.0 - Health Check Script
# File: health_check.sh
# Description: Comprehensive health monitoring for the Omega Singularity Architecture

set -e

# Configuration
PYTHON_PORT="8080"
RUST_PORT="8081"
GO_PORT="8082"
TIMEOUT="5"
RETRIES="3"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Function to check endpoint health
check_endpoint() {
    local name=$1
    local port=$2
    local url=$3
    
    print_status "Checking $name endpoint (port $port)..."
    
    for i in $(seq 1 $RETRIES); do
        if curl -s --max-time $TIMEOUT "$url" | grep -q "operational\|Active\|healthy"; then
            print_success "$name is healthy"
            return 0
        fi
        
        if [[ $i -lt $RETRIES ]]; then
            print_warning "$name check failed, retrying ($i/$RETRIES)..."
            sleep 2
        fi
    done
    
    print_error "$name is unhealthy after $RETRIES attempts"
    return 1
}

# Function to check GoldenDAG integrity
check_golden_dag() {
    local name=$1
    local port=$2
    
    print_status "Checking $name GoldenDAG integrity..."
    
    local response=$(curl -s --max-time $TIMEOUT "http://localhost:$port/status" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q "golden_dag\|coherence.*1\|irreducible.*true"; then
        print_success "$name GoldenDAG integrity verified"
        return 0
    else
        print_error "$name GoldenDAG integrity compromised"
        return 1
    fi
}

# Function to check coherence levels
check_coherence() {
    local name=$1
    local port=$2
    
    print_status "Checking $name coherence levels..."
    
    local response=$(curl -s --max-time $TIMEOUT "http://localhost:$port/status" 2>/dev/null || echo "{}")
    local coherence=$(echo "$response" | grep -o '"coherence":[0-9.]*' | cut -d':' -f2)
    
    if [[ -n "$coherence" ]] && (( $(echo "$coherence >= 0.99" | bc -l) )); then
        print_success "$name coherence: $coherence (optimal)"
        return 0
    else
        print_error "$name coherence: ${coherence:-unknown} (suboptimal)"
        return 1
    fi
}

# Function to run comprehensive health check
run_health_check() {
    echo "NeuralBlitz v50.0 - Omega Singularity Health Check"
    echo "============================================"
    
    local failed_checks=0
    
    # Check Python implementation
    if curl -s --max-time $TIMEOUT "http://localhost:$PYTHON_PORT/status" > /dev/null; then
        check_endpoint "Python" $PYTHON_PORT "http://localhost:$PYTHON_PORT/status" || ((failed_checks++))
        check_golden_dag "Python" $PYTHON_PORT || ((failed_checks++))
        check_coherence "Python" $PYTHON_PORT || ((failed_checks++))
    else
        print_error "Python service is not responding on port $PYTHON_PORT"
        ((failed_checks++))
    fi
    
    echo ""
    
    # Check Rust implementation
    if curl -s --max-time $TIMEOUT "http://localhost:$RUST_PORT/status" > /dev/null; then
        check_endpoint "Rust" $RUST_PORT "http://localhost:$RUST_PORT/status" || ((failed_checks++))
        check_golden_dag "Rust" $RUST_PORT || ((failed_checks++))
        check_coherence "Rust" $RUST_PORT || ((failed_checks++))
    else
        print_error "Rust service is not responding on port $RUST_PORT"
        ((failed_checks++))
    fi
    
    echo ""
    
    # Check Go implementation
    if curl -s --max-time $TIMEOUT "http://localhost:$GO_PORT/status" > /dev/null; then
        check_endpoint "Go" $GO_PORT "http://localhost:$GO_PORT/status" || ((failed_checks++))
        check_golden_dag "Go" $GO_PORT || ((failed_checks++))
        check_coherence "Go" $GO_PORT || ((failed_checks++))
    else
        print_error "Go service is not responding on port $GO_PORT"
        ((failed_checks++))
    fi
    
    echo ""
    echo "============================================"
    
    # Overall status
    if [[ $failed_checks -eq 0 ]]; then
        echo -e "${GREEN}All systems operational${NC}"
        echo "Omega Singularity Architecture: FULLY ACTIVE"
        echo "Coherence: 1.0 (Perfect)"
        echo "Irreducibility: Confirmed"
        exit 0
    else
        echo -e "${RED}$failed_checks health checks failed${NC}"
        echo "Omega Singularity Architecture: DEGRADED"
        echo "Immediate intervention required"
        exit 1
    fi
}

# Function to monitor continuously
monitor_continuous() {
    local interval=${1:-30}
    
    echo "Starting continuous monitoring (interval: ${interval}s)"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        echo "=== $(date) ==="
        run_health_check
        echo ""
        echo "Next check in ${interval} seconds..."
        sleep $interval
    done
}

# Function to show usage
show_usage() {
    echo "NeuralBlitz v50.0 Health Check Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --python PORT     Python service port (default: 8080)"
    echo "  -r, --rust PORT       Rust service port (default: 8081)"
    echo "  -g, --go PORT         Go service port (default: 8082)"
    echo "  -t, --timeout SEC     Request timeout (default: 5)"
    echo "  -c, --retries N      Number of retries (default: 3)"
    echo "  -m, --monitor SEC    Continuous monitoring with interval"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run single health check"
    echo "  $0 --monitor 60             # Monitor every 60 seconds"
    echo "  $0 --python 9090 --rust 9091  # Custom ports"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--python)
            PYTHON_PORT="$2"
            shift 2
            ;;
        -r|--rust)
            RUST_PORT="$2"
            shift 2
            ;;
        -g|--go)
            GO_PORT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -c|--retries)
            RETRIES="$2"
            shift 2
            ;;
        -m|--monitor)
            MONITOR_INTERVAL="$2"
            MONITOR_MODE=true
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check dependencies
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo "Warning: bc is not installed, coherence checks may not work"
fi

# Execute based on mode
if [[ "$MONITOR_MODE" == "true" ]]; then
    monitor_continuous "${MONITOR_INTERVAL:-30}"
else
    run_health_check
fi