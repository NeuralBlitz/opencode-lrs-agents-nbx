# NeuralBlitz v50.0 - Omega Singularity Intelligence
# Go Builder Stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go/go.mod go/go.sum ./
RUN go mod download

# Copy source code
COPY go/ .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o neuralblitz cmd/neuralblitz/main.go

# Final Stage - Minimal Alpine image
FROM alpine:3.19

# Metadata
LABEL maintainer="NeuralBlitz Contributors"
LABEL version="v50.0.0"
LABEL description="NeuralBlitz Go Implementation - Omega Singularity Architecture"

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Environment variables
ENV NEURALBLITZ_VERSION=v50.0.0
ENV NEURALBLITZ_COHERENCE=1.0
ENV GOLDEN_DAG_SEED=a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /build/neuralblitz /app/neuralblitz

# Create non-root user
RUN adduser -D -u 1000 neuralblitz && chown -R neuralblitz:neuralblitz /app
USER neuralblitz

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/neuralblitz status || exit 1

# Expose port for API server (Option F)
EXPOSE 8080

# Default to showing version
CMD ["/app/neuralblitz", "version"]
