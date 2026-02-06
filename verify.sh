#!/bin/bash
# Go-LRS Project Verification Script

set -e

echo "=========================================="
echo "  Go-LRS Project Verification"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ERRORS=0
WARNINGS=0

check_file() {
    local file="$1"
    local desc="$2"
    if [ -f "$file" ]; then
        echo "✅ $desc"
    else
        echo "❌ MISSING: $desc ($file)"
        ((ERRORS++))
    fi
}

check_dir() {
    local dir="$1"
    local desc="$2"
    if [ -d "$dir" ]; then
        echo "✅ $desc"
    else
        echo "❌ MISSING: $desc ($dir)"
        ((ERRORS++))
    fi
}

echo "📦 Core Packages..."
check_file "pkg/core/toollens.go" "ToolLens implementation"
check_file "pkg/api/agent_manager.go" "Agent manager"
check_file "pkg/api/grpc_server.go" "gRPC server"
check_file "pkg/api/state_manager.go" "State manager"
check_file "internal/math/precision.go" "Precision tracking"
check_file "internal/math/free_energy.go" "Free Energy calculator"
check_file "internal/math/hierarchical_precision.go" "Hierarchical precision"
check_file "internal/state/lrs_state.go" "LRS State management"
echo ""

echo "🔧 Deployment..."
check_file "Dockerfile" "Dockerfile"
check_file "docker-compose.yml" "Docker Compose"
check_file "Makefile" "Makefile"
check_file "go.mod" "Go module"
echo ""

echo "☸️  Kubernetes..."
check_file "deploy/k8s/deployment.yaml" "K8s Deployment (includes Service)"
check_file "deploy/k8s/configmap.yaml" "K8s ConfigMap"
check_file "deploy/k8s/secrets.yaml" "K8s Secrets"
echo ""

echo "⚓ Helm..."
check_file "deploy/helm/lrs/Chart.yaml" "Helm Chart.yaml"
check_file "deploy/helm/lrs/values.yaml" "Helm values"
echo ""

echo "🔄 CI/CD..."
check_file ".github/workflows/ci.yml" "CI workflow"
check_file ".github/workflows/cd.yml" "CD workflow"
echo ""

echo "📚 Documentation..."
check_file "README.md" "README"
check_file "USER_GUIDE.md" "User Guide"
echo ""

echo "🧪 Testing..."
check_file "internal/math/math_test.go" "Math tests"
check_file "pkg/multiagent/multiagent_test.go" "Multi-agent tests"
echo ""

echo "📊 Metrics..."
check_file "deploy/prometheus/prometheus.yml" "Prometheus config"
echo ""

echo "💻 Examples..."
check_file "examples/test_basic.go" "Basic example"
check_file "examples/multiagent_demo.go" "Multi-agent demo"
echo ""

# Count Go files
GO_FILES=$(find . -name "*.go" -type f | wc -l)
echo "📈 Statistics:"
echo "   - Go files: $GO_FILES"
echo ""

# Total lines of code
GO_LINES=$(find . -name "*.go" -type f -exec wc -l {} + | tail -1)
echo "   - Go lines: $GO_LINES"
echo ""

# Check for proto files
PROTO_COUNT=$(find . -name "*.proto" -type f 2>/dev/null | wc -l)
if [ "$PROTO_COUNT" -gt 0 ]; then
    echo "📝 Protobuf files: $PROTO_COUNT"
else
    echo "⚠️  No .proto files found (proto generation may be needed)"
    ((WARNINGS++))
fi
echo ""

# Summary
echo "=========================================="
echo "  Summary"
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All critical checks passed!"
    echo "⚠️  Warnings: $WARNINGS"
    echo ""
    echo "To build and test:"
    echo "  cd go-lrs"
    echo "  make build"
    echo "  make test"
    echo ""
    echo "To run with Docker:"
    echo "  docker-compose up"
    exit 0
else
    echo "❌ Errors found: $ERRORS"
    echo "⚠️  Warnings: $WARNINGS"
    exit 1
fi
