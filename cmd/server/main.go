package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/neuralblitz/go-lrs/internal/math"
	"github.com/neuralblitz/go-lrs/internal/state"
	"github.com/neuralblitz/go-lrs/pkg/api"
	"github.com/neuralblitz/go-lrs/pkg/core"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

var (
	Version    string = "1.0.0"
	BuildTime  string = time.Now().Format(time.RFC3339)
	GitCommit  string = "unknown"
	VersionCmd = &cobra.Command{
		Use:     "version",
		Short:   "Print version information",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Go-LRS Version: %s\n", Version)
			fmt.Printf("Build Time: %s\n", BuildTime)
			fmt.Printf("Git Commit: %s\n", GitCommit)
		},
	}
)

var (
	cfgFile      string
	logLevel     string
	grpcPort     int
	httpPort     int
	metricsPort  int
	enableDebug  bool
)

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		viper.SetConfigName("config")
		viper.SetConfigType("yaml")
		viper.AddConfigPath("/etc/go-lrs/")
		viper.AddConfigPath("$HOME/.go-lrs")
		viper.AddConfigPath(".")
	}

	viper.SetEnvPrefix("GOLRS")
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: Could not read config file: %v", err)
	}

	if grpcPort == 0 {
		grpcPort = viper.GetInt("grpc.port")
	}
	if httpPort == 0 {
		httpPort = viper.GetInt("http.port")
	}
	if metricsPort == 0 {
		metricsPort = viper.GetInt("metrics.port")
	}
	if logLevel == "" {
		logLevel = viper.GetString("log.level")
	}
}

type Config struct {
	GRPC    GRPCConfig    `mapstructure:"grpc"`
	HTTP    HTTPConfig    `mapstructure:"http"`
	Metrics MetricsConfig `mapstructure:"metrics"`
	Log     LogConfig     `mapstructure:"log"`
}

type GRPCConfig struct {
	Port    int    `mapstructure:"port"`
	MaxConn int    `mapstructure:"max_connections"`
	CERT    string `mapstructure:"cert_file"`
	Key     string `mapstructure:"key_file"`
}

type HTTPConfig struct {
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	StaticDir    string `mapstructure:"static_dir"`
}

type MetricsConfig struct {
	Port     int    `mapstructure:"port"`
	Path     string `mapstructure:"path"`
	Disabled bool   `mapstructure:"disabled"`
}

type LogConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
}

var cfg Config

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the LRS server",
	Long:  `Start the LRS server with gRPC and HTTP endpoints`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runServer()
	},
}

func init() {
	cobra.OnInitialize(initConfig)

	startCmd.Flags().StringVarP(&cfgFile, "config", "c", "", "config file path")
	startCmd.Flags().StringVar(&logLevel, "log-level", "info", "log level (debug, info, warn, error)")
	startCmd.Flags().IntVarP(&grpcPort, "grpc-port", "g", 9090, "gRPC server port")
	startCmd.Flags().IntVarP(&httpPort, "http-port", "h", 8080, "HTTP server port")
	startCmd.Flags().IntVarP(&metricsPort, "metrics-port", "m", 9091, "Metrics server port")
	startCmd.Flags().BoolVar(&enableDebug, "debug", false, "enable debug mode")

	rootCmd.AddCommand(VersionCmd)
	rootCmd.AddCommand(startCmd)
}

var rootCmd = &cobra.Command{
	Use:   "go-lrs",
	Short: "Go-LRS: Resilient AI Agents using Active Inference",
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

type LRSServer struct {
	agentManager *api.AgentManager
	logger       *zap.Logger
	server       *api.Server
	httpServer   *http.Server
	metricsServer *http.Server
	upgrader     websocket.Upgrader
	registry     *prometheus.Registry
}

func NewLRSServer() *LRSServer {
	logger, _ := zap.NewProduction()
	if enableDebug {
		logger, _ = zap.NewDevelopment()
	}

	registry := prometheus.NewRegistry()

	return &LRSServer{
		agentManager: api.NewAgentManager(logger),
		logger:       logger,
		server:       api.NewServer(logger),
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
		registry: registry,
	}
}

func (s *LRSServer) setupMiddleware(router *gin.Engine) {
	router.Use(gin.Recovery())
	router.Use(s.loggingMiddleware())
	router.Use(s.metricsMiddleware())
	router.Use(corsMiddleware())
}

func (s *LRSServer) loggingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path

		c.Next()

		s.logger.Info("HTTP request",
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.Int("status", c.Writer.Status()),
			zap.Duration("latency", time.Since(start)),
			zap.String("client_ip", c.ClientIP()),
		)
	}
}

func (s *LRSServer) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()

		duration := time.Since(start).Seconds()
		httpRequestsTotal.WithLabelValues(c.Request.Method, c.FullPath(), fmt.Sprintf("%d", c.Writer.Status())).Inc()
		httpRequestDuration.WithLabelValues(c.Request.Method, c.FullPath()).Observe(duration)
	}
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func (s *LRSServer) setupRoutes(router *gin.Engine) {
	api := router.Group("/api/v1")
	{
		api.GET("/health", s.healthCheck)
		api.GET("/ready", s.readinessCheck)

		agents := api.Group("/agents")
		{
			agents.POST("", s.createAgent)
			agents.GET("", s.listAgents)
			agents.GET("/:id", s.getAgent)
			agents.DELETE("/:id", s.deleteAgent)
			agents.POST("/:id/execute", s.executePolicy)
			agents.GET("/:id/state", s.getAgentState)
			agents.POST("/:id/state", s.updateAgentState)
			agents.GET("/:id/precision", s.getAgentPrecision)
			agents.GET("/:id/policies", s.listPolicies)
			agents.POST("/:id/checkpoint", s.createCheckpoint)
			agents.POST("/:id/rollback", s.rollbackToCheckpoint)
		}

		tools := api.Group("/tools")
		{
			tools.GET("", s.listTools)
			tools.POST("/register", s.registerTool)
			tools.GET("/:name", s.getTool)
		}

		ws := router.Group("/ws")
		{
			ws.GET("/agent/:id", s.handleWebSocket)
		}
	}

	router.GET("/metrics", gin.WrapH(promhttp.HandlerFor(s.registry, promhttp.HandlerOpts{})))
	router.GET("/version", s.versionHandler)
}

func (s *LRSServer) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now().UnixMilli(),
		"version":   Version,
	})
}

func (s *LRSServer) readinessCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"ready":     true,
		"timestamp": time.Now().UnixMilli(),
	})
}

func (s *LRSServer) versionHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"version":   Version,
		"buildTime": BuildTime,
		"gitCommit": GitCommit,
	})
}

func (s *LRSServer) createAgent(c *gin.Context) {
	var req struct {
		Name        string `json:"name" binding:"required"`
		Description string `json:"description"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	agent, err := s.agentManager.CreateAgent(req.Name, req.Description)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"agent":     agent.ToProto(),
		"createdAt": time.UnixMilli(agent.CreatedAt).Format(time.RFC3339),
	})
}

func (s *LRSServer) listAgents(c *gin.Context) {
	agents := s.agentManager.ListAgents()
	protos := make([]*api.AgentProto, len(agents))
	for i, agent := range agents {
		protos[i] = agent.ToProto()
	}

	c.JSON(http.StatusOK, gin.H{
		"agents": protos,
		"total":  len(protos),
	})
}

func (s *LRSServer) getAgent(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"agent": agent.ToProto(),
	})
}

func (s *LRSServer) deleteAgent(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	if err := s.agentManager.DeleteAgent(id); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "agent deleted"})
}

func (s *LRSServer) executePolicy(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	var context map[string]interface{}
	if err := c.ShouldBindJSON(&context); err != nil {
		context = make(map[string]interface{})
	}

	policy, err := agent.PolicyCalc.SelectPolicyWithInfo(context)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"selectedPolicy": policy.SelectedPolicy,
		"allPolicies":    policy.AllPolicies,
		"selectionProbs": policy.SelectionProbs,
		"policyEntropy":  policy.PolicyEntropy,
	})
}

func (s *LRSServer) getAgentState(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	stateJSON, _ := agent.State.ToJSON()
	var stateMap map[string]interface{}
	json.Unmarshal(stateJSON, &stateMap)

	c.JSON(http.StatusOK, gin.H{
		"state":     stateMap,
		"version":   agent.State.Version,
		"updatedAt": time.UnixMilli(agent.State.UpdatedAt).Format(time.RFC3339),
	})
}

func (s *LRSServer) updateAgentState(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	var updates map[string]interface{}
	if err := c.ShouldBindJSON(&updates); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	if messages, ok := updates["messages"].([]interface{}); ok {
		for _, m := range messages {
			if msgMap, ok := m.(map[string]interface{}); ok {
				msg := state.NewMessage(msgMap["role"].(string), msgMap["content"].(string))
				agent.State = agent.State.WithMessage(msg)
			}
		}
	}

	if precision, ok := updates["precision"].(map[string]interface{}); ok {
		for level, val := range precision {
			if fval, ok := val.(float64); ok {
				agent.State = agent.State.WithPrecision(level, fval)
			}
		}
	}

	if context, ok := updates["context"].(map[string]interface{}); ok {
		for key, val := range context {
			agent.State = agent.State.WithContext(key, val)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"message":  "state updated",
		"version": agent.State.Version,
	})
}

func (s *LRSServer) getAgentPrecision(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	precision := agent.Precision.GetAllPrecision()
	c.JSON(http.StatusOK, gin.H{
		"precision":       precision,
		"average":          agent.Precision.GetAveragePrecision(),
		"weighted":         agent.Precision.GetWeightedPrecision(),
		"variance":         agent.Precision.GetPrecisionVariance(),
		"dominantLevel":    agent.Precision.GetDominantLevel().String(),
		"coherence":        agent.Precision.CalculateCoherence(),
		"topLevelMetrics":  agent.Precision.GetTopLevelMetrics(),
	})
}

func (s *LRSServer) listPolicies(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	policies := agent.PolicyCalc.GetPolicyHistory()
	c.JSON(http.StatusOK, gin.H{
		"policies": policies,
		"count":    len(policies),
	})
}

func (s *LRSServer) createCheckpoint(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	var req struct {
		CheckpointID string `json:"checkpoint_id" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":       "checkpoint created",
		"checkpointId": req.CheckpointID,
		"timestamp":    time.Now().Format(time.RFC3339),
	})
}

func (s *LRSServer) rollbackToCheckpoint(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	var req struct {
		CheckpointID string `json:"checkpoint_id" binding:"required"`
		TargetVersion int   `json:"target_version"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":        "rollback successful",
		"checkpointId":  req.CheckpointID,
		"targetVersion": req.TargetVersion,
		"timestamp":     time.Now().Format(time.RFC3339),
	})
}

func (s *LRSServer) listTools(c *gin.Context) {
	registry := s.agentManager.GetRegistry()
	tools := registry.List()

	toolList := make([]map[string]interface{}, len(tools))
	for i, tool := range tools {
		toolList[i] = map[string]interface{}{
			"name":         tool.GetName(),
			"description":  tool.GetDescription(),
			"version":      tool.GetVersion(),
			"parameters":   tool.GetParameters(),
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"tools": toolList,
		"count": len(toolList),
	})
}

func (s *LRSServer) registerTool(c *gin.Context) {
	var tool core.ToolLens
	if err := c.ShouldBindJSON(&tool); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := s.agentManager.RegisterTool(tool); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"message": "tool registered",
		"name":    tool.GetName(),
	})
}

func (s *LRSServer) getTool(c *gin.Context) {
	name := c.Param("name")
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "tool name required"})
		return
	}

	tool, err := s.agentManager.GetRegistry().Get(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"name":        tool.GetName(),
		"description": tool.GetDescription(),
		"version":     tool.GetVersion(),
		"parameters": tool.GetParameters(),
	})
}

func (s *LRSServer) handleWebSocket(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID required"})
		return
	}

	conn, err := s.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		s.logger.Error("WebSocket upgrade failed", zap.Error(err))
		return
	}
	defer conn.Close()

	s.logger.Info("WebSocket connection established", zap.String("agent_id", id))

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				s.logger.Error("WebSocket error", zap.Error(err))
			}
			break
		}

		var msg struct {
			Type    string          `json:"type"`
			Payload json.RawMessage `json:"payload"`
		}
		if err := json.Unmarshal(message, &msg); err != nil {
			s.logger.Error("Failed to parse message", zap.Error(err))
			continue
		}

		switch msg.Type {
		case "ping":
			s.handleWebSocketPing(conn)
		case "getState":
			s.handleWebSocketGetState(conn, id)
		case "execute":
			s.handleWebSocketExecute(conn, id, msg.Payload)
		case "updatePrecision":
			s.handleWebSocketUpdatePrecision(conn, id, msg.Payload)
		default:
			s.sendError(conn, fmt.Sprintf("unknown message type: %s", msg.Type))
		}
	}
}

func (s *LRSServer) handleWebSocketPing(conn *websocket.Conn) {
	response := map[string]interface{}{
		"type":      "pong",
		"timestamp": time.Now().UnixMilli(),
	}
	if err := conn.WriteJSON(response); err != nil {
		s.logger.Error("Failed to send pong", zap.Error(err))
	}
}

func (s *LRSServer) handleWebSocketGetState(conn *websocket.Conn, id string) {
	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		s.sendError(conn, err.Error())
		return
	}

	stateJSON, _ := agent.State.ToJSON()
	response := map[string]interface{}{
		"type":  "state",
		"state": string(stateJSON),
	}
	if err := conn.WriteJSON(response); err != nil {
		s.logger.Error("Failed to send state", zap.Error(err))
	}
}

func (s *LRSServer) handleWebSocketExecute(conn *websocket.Conn, id string, payload []byte) {
	var req struct {
		Context map[string]interface{} `json:"context"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(conn, "invalid payload")
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		s.sendError(conn, err.Error())
		return
	}

	policy, err := agent.PolicyCalc.SelectPolicyWithInfo(req.Context)
	if err != nil {
		s.sendError(conn, err.Error())
		return
	}

	response := map[string]interface{}{
		"type":   "execution",
		"policy": policy.SelectedPolicy,
	}
	if err := conn.WriteJSON(response); err != nil {
		s.logger.Error("Failed to send execution result", zap.Error(err))
	}
}

func (s *LRSServer) handleWebSocketUpdatePrecision(conn *websocket.Conn, id string, payload []byte) {
	var req struct {
		Level          string  `json:"level"`
		PredictionError float64 `json:"prediction_error"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(conn, "invalid payload")
		return
	}

	var level math.HierarchyLevel
	switch req.Level {
	case "abstract":
		level = math.LevelAbstract
	case "planning":
		level = math.LevelPlanning
	case "execution":
		level = math.LevelExecution
	default:
		s.sendError(conn, "invalid level")
		return
	}

	agent, err := s.agentManager.GetAgent(id)
	if err != nil {
		s.sendError(conn, err.Error())
		return
	}

	if err := agent.Precision.Update(level, req.PredictionError); err != nil {
		s.sendError(conn, err.Error())
		return
	}

	response := map[string]interface{}{
		"type":       "precision_updated",
		"level":      req.Level,
		"newValue":   agent.Precision.GetAllPrecision()[level],
	}
	if err := conn.WriteJSON(response); err != nil {
		s.logger.Error("Failed to send precision update", zap.Error(err))
	}
}

func (s *LRSServer) sendError(conn *websocket.Conn, message string) {
	response := map[string]interface{}{
		"type":    "error",
		"message": message,
	}
	if err := conn.WriteJSON(response); err != nil {
		s.logger.Error("Failed to send error", zap.Error(err))
	}
}

var (
	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "path", "status"},
	)

	httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "path"},
	)

	agentOperationsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "agent_operations_total",
			Help: "Total number of agent operations",
		},
		[]string{"operation", "status"},
	)

	precisionValue = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "agent_precision_value",
			Help: "Current precision values for agents",
		},
		[]string{"agent_id", "level"},
	)

	freeEnergyValue = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "agent_free_energy_value",
			Help: "Current free energy values for agents",
		},
		[]string{"agent_id"},
	)
)

func init() {
	prometheus.MustRegister(httpRequestsTotal)
	prometheus.MustRegister(httpRequestDuration)
	prometheus.MustRegister(agentOperationsTotal)
	prometheus.MustRegister(precisionValue)
	prometheus.MustRegister(freeEnergyValue)
}

func runServer() error {
	logrus.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: "2006-01-02T15:04:05.000Z",
	})

	server := NewLRSServer()
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.httpServer.Shutdown(ctx); err != nil {
			logrus.Errorf("HTTP server shutdown error: %v", err)
		}
	}()

	router := gin.New()
	router.Use(gin.Recovery())
	server.setupMiddleware(router)
	server.setupRoutes(router)

	server.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", httpPort),
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	server.metricsServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", metricsPort),
		Handler: gin.WrapH(promhttp.Handler()),
	}

	go func() {
		logrus.Infof("Starting HTTP server on port %d", httpPort)
		if err := server.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logrus.Fatalf("HTTP server error: %v", err)
		}
	}()

	go func() {
		logrus.Infof("Starting metrics server on port %d", metricsPort)
		if err := server.metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logrus.Fatalf("Metrics server error: %v", err)
		}
	}()

	if err := server.server.Start(grpcPort, httpPort); err != nil {
		logrus.Fatalf("Failed to start gRPC server: %v", err)
	}

	logrus.Infof("Go-LRS server started successfully")
	logrus.Infof("Version: %s", Version)
	logrus.Infof("HTTP: http://localhost:%d", httpPort)
	logrus.Infof("gRPC: localhost:%d", grpcPort)
	logrus.Infof("Metrics: http://localhost:%d/metrics", metricsPort)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logrus.Info("Shutting down servers...")

	return nil
}
