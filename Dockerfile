FROM golang:1.21-alpine AS builder

ARG VERSION="dev"
ARG BUILD_TIME="unknown"

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME}" \
    -o bin/server ./cmd/server && \
    CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME}" \
    -o bin/cli ./cmd/cli

FROM alpine:3.19 AS runner

RUN addgroup -g 1000 appgroup && \
    adduser -u 1000 -G appgroup -s /bin/sh -D appuser

WORKDIR /home/runner/workspace

COPY --from=builder /app/bin/ /usr/local/bin/

COPY --from=builder /app/configs/default.yaml /etc/go-lrs/default.yaml
COPY --from=builder /app/web/ /web/

USER appuser

ENV LRS_CONFIG=/etc/go-lrs/default.yaml
ENV LRS_HTTP_PORT=8080
ENV LRS_GRPC_PORT=9090

EXPOSE 8080 9090 8081

ENTRYPOINT ["server"]
CMD ["--config", "/etc/go-lrs/default.yaml"]
