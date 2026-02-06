.PHONY: build test clean fmt lint vet benchmark docker help

# Build targets
build:
	@echo "Building all components..."
	go build ./cmd/...
	go build -o bin/server ./cmd/server
	go build -o bin/client ./cmd/client

build-server:
	@echo "Building server..."
	go build -o bin/server ./cmd/server

build-client:
	@echo "Building client..."
	go build -o bin/client ./cmd/client

# Test targets
test:
	@echo "Running tests..."
	go test -v ./...

test-coverage:
	@echo "Running tests with coverage..."
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

test-race:
	@echo "Running race condition tests..."
	go test -race -v ./...

benchmark:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Code quality
fmt:
	@echo "Formatting code..."
	go fmt ./...

lint:
	@echo "Running linter..."
	golangci-lint run

vet:
	@echo "Running vet..."
	go vet ./...

tidy:
	@echo "Tidying dependencies..."
	go mod tidy

# Development
dev:
	@echo "Starting development server..."
	go run ./cmd/server/main.go

dev-debug:
	@echo "Starting development server with debug..."
	go run -tags=debug ./cmd/server/main.go

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t go-lrs:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8080:8080 -p 9090:9090 go-lrs:latest

# Database
db-migrate:
	@echo "Running database migrations..."
	go run ./scripts/migrate/main.go

# Proto files
proto:
	@echo "Generating protobuf files..."
	protoc --go_out=. --go-grpc_out=. api/proto/v1/*.proto

# Security
security:
	@echo "Running security scan..."
	gosec ./...

# Performance
profile:
	@echo "Generating CPU profile..."
	go run -cpuprofile=cpu.prof ./cmd/server/main.go
	go tool pprof cpu.prof

memory-profile:
	@echo "Generating memory profile..."
	go run -memprofile=mem.prof ./cmd/server/main.go
	go tool pprof mem.prof

# Documentation
docs:
	@echo "Generating documentation..."
	godoc -http=:6060

# Installation
install:
	@echo "Installing dependencies..."
	go mod download

install-tools:
	@echo "Installing development tools..."
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf bin/
	rm -f coverage.out coverage.html
	rm -f *.prof

clean-deps:
	@echo "Cleaning dependencies..."
	go clean -modcache

# Release
release: clean test fmt lint vet
	@echo "Creating release..."
	go build -ldflags="-s -w" -o bin/release/server ./cmd/server
	go build -ldflags="-s -w" -o bin/release/client ./cmd/client

# Help
help:
	@echo "Available targets:"
	@echo "  build           - Build all components"
	@echo "  build-server    - Build server only"
	@echo "  build-client    - Build client only"
	@echo "  test            - Run tests"
	@echo "  test-coverage   - Run tests with coverage"
	@echo "  benchmark       - Run benchmarks"
	@echo "  fmt             - Format code"
	@echo "  lint            - Run linter"
	@echo "  vet             - Run vet"
	@echo "  dev             - Start development server"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run Docker container"
	@echo "  proto           - Generate protobuf files"
	@echo "  security        - Run security scan"
	@echo "  profile         - Generate CPU profile"
	@echo "  memory-profile  - Generate memory profile"
	@echo "  docs            - Generate documentation"
	@echo "  install         - Install dependencies"
	@echo "  install-tools   - Install development tools"
	@echo "  clean           - Clean build artifacts"
	@echo "  release         - Create release build"
	@echo "  help            - Show this help"