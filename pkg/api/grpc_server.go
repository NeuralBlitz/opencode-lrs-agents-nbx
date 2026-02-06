package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neuralblitz/go-lrs/internal/math"
	"github.com/neuralblitz/go-lrs/internal/state"
	"github.com/neuralblitz/go-lrs/pkg/core"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Agent struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	State        *state.LRSState       `json:"state"`
	Precision    *math.HierarchicalPrecision `json:"precision"`
	PolicyCalc   *math.FreeEnergyCalculator `json:"policy_calculator"`
	CreatedAt    int64                  `json:"created_at"`
	UpdatedAt    int64                  `json:"updated_at"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	mu           sync.RWMutex
}

func NewAgent(name, description string) (*Agent, error) {
	if name == "" {
		return nil, errors.New("agent name cannot be empty")
	}

	state := state.NewLRSState()
	precision := math.NewHierarchicalPrecision()
	policyCalc := math.NewFreeEnergyCalculator(mustNewPrecisionTracker(2.0, 2.0))

	return &Agent{
		ID:           uuid.New().String(),
		Name:         name,
		Description:  description,
		State:        state,
		Precision:    precision,
		PolicyCalc:   policyCalc,
		CreatedAt:    time.Now().UnixMilli(),
		UpdatedAt:    time.Now().UnixMilli(),
		Metadata:     make(map[string]interface{}),
	}, nil
}

func mustNewPrecisionTracker(alpha, beta float64) *math.PrecisionTracker {
	tracker, err := math.NewPrecisionTracker(alpha, beta)
	if err != nil {
		panic(err)
	}
	return tracker
}

type AgentManager struct {
	agents      map[string]*Agent
	registry    *core.ToolRegistry
	logger      *zap.Logger
	mu          sync.RWMutex
}

func NewAgentManager(logger *zap.Logger) *AgentManager {
	return &AgentManager{
		agents:   make(map[string]*Agent),
		registry: core.NewToolRegistry(),
		logger:   logger,
	}
}

func (m *AgentManager) CreateAgent(name, description string) (*Agent, error) {
	agent, err := NewAgent(name, description)
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.agents[agent.ID]; exists {
		return nil, fmt.Errorf("agent ID collision: %s", agent.ID)
	}

	m.agents[agent.ID] = agent
	m.logger.Info("created agent", zap.String("id", agent.ID), zap.String("name", agent.Name))

	return agent, nil
}

func (m *AgentManager) GetAgent(id string) (*Agent, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	agent, exists := m.agents[id]
	if !exists {
		return nil, fmt.Errorf("agent '%s' not found", id)
	}
	return agent.Clone(), nil
}

func (m *AgentManager) ListAgents() []*Agent {
	m.mu.RLock()
	defer m.mu.RUnlock()

	agents := make([]*Agent, 0, len(m.agents))
	for _, agent := range m.agents {
		agents = append(agents, agent.Clone())
	}
	return agents
}

func (m *AgentManager) DeleteAgent(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.agents[id]; !exists {
		return fmt.Errorf("agent '%s' not found", id)
	}

	delete(m.agents, id)
	m.logger.Info("deleted agent", zap.String("id", id))

	return nil
}

func (m *AgentManager) RegisterTool(tool core.ToolLens) error {
	return m.registry.Register(tool)
}

func (m *AgentManager) GetRegistry() *core.ToolRegistry {
	return m.registry
}

type Server struct {
	agentManager *AgentManager
	grpcServer   *grpc.Server
	logger       *zap.Logger
	mu           sync.RWMutex
	wg           sync.WaitGroup
}

func NewServer(logger *zap.Logger) *Server {
	return &Server{
		agentManager: NewAgentManager(logger),
		logger:       logger,
	}
}

func (s *Server) CreateAgent(ctx context.Context, req *CreateAgentRequest) (*AgentResponse, error) {
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	agent, err := s.agentManager.CreateAgent(req.Name, req.Description)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create agent: %v", err)
	}

	return &AgentResponse{
		Agent:       agent.ToProto(),
		CreatedAt:  timestamppb.New(time.UnixMilli(agent.CreatedAt)),
		UpdatedAt:  timestamppb.New(time.UnixMilli(agent.UpdatedAt)),
	}, nil
}

func (s *Server) GetAgent(ctx context.Context, req *GetAgentRequest) (*AgentResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}

	agent, err := s.agentManager.GetAgent(req.Id)
	if err != nil {
		return nil, status.Errorf(codes.NotFound, "agent not found: %v", err)
	}

	return &AgentResponse{
		Agent:       agent.ToProto(),
		CreatedAt:  timestamppb.New(time.UnixMilli(agent.CreatedAt)),
		UpdatedAt:  timestamppb.New(time.UnixMilli(agent.UpdatedAt)),
	}, nil
}

func (s *Server) ListAgents(ctx context.Context, req *ListAgentsRequest) (*ListAgentsResponse, error) {
	agents := s.agentManager.ListAgents()
	protos := make([]*AgentProto, len(agents))
	for i, agent := range agents {
		protos[i] = agent.ToProto()
	}

	return &ListAgentsResponse{
		Agents: protos,
		Total:  int32(len(agents)),
	}, nil
}

func (s *Server) DeleteAgent(ctx context.Context, req *DeleteAgentRequest) (*emptypb.Empty, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}

	if err := s.agentManager.DeleteAgent(req.Id); err != nil {
		return nil, status.Errorf(codes.NotFound, "failed to delete agent: %v", err)
	}

	return &emptypb.Empty{}, nil
}

func (s *Server) ExecutePolicy(ctx context.Context, req *ExecutePolicyRequest) (*ExecutePolicyResponse, error) {
	if req.AgentId == "" {
		return nil, status.Error(codes.InvalidArgument, "agent_id is required")
	}

	agent, err := s.agentManager.GetAgent(req.AgentId)
	if err != nil {
		return nil, status.Errorf(codes.NotFound, "agent not found: %v", err)
	}

	context := make(map[string]interface{})
	for k, v := range req.Context {
		context[k] = v
	}

	policy, err := agent.PolicyCalc.SelectPolicyWithInfo(context)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to select policy: %v", err)
	}

	execution := state.NewPolicyExecution(policy.SelectedPolicy.ID, policy.SelectedPolicy.Name)

	for _, toolName := range policy.SelectedPolicy.Tools {
		toolExec := state.NewToolExecution(toolName)
		tool, err := s.agentManager.registry.Get(toolName)
		if err != nil {
			toolExec.SetSuccess(false).SetPredictionError(1.0).SetError(err.Error()).Complete()
			execution.AddToolExecution(toolExec)
			continue
		}

		result, err := tool.Execute(agent.State)
		if err != nil {
			toolExec.SetSuccess(false).SetPredictionError(1.0).SetError(err.Error()).Complete()
			execution.AddToolExecution(toolExec)
			continue
		}

		toolExec.SetOutput(result.Observation).SetPredictionError(result.GetPredictionError()).SetSuccess(result.Success).Complete()
		execution.AddToolExecution(toolExec)

		if err := agent.Precision.Update(math.LevelExecution, result.GetPredictionError()); err != nil {
			log.Printf("failed to update precision: %v", err)
		}
	}

	execution.Complete()
	agent.State = agent.State.WithPolicy(execution)
	agent.Precision.Update(math.LevelPlanning, execution.PredictionError)

	return &ExecutePolicyResponse{
		Policy: &PolicyProto{
			Id:            policy.SelectedPolicy.ID,
			Name:          policy.SelectedPolicy.Name,
			FreeEnergy:   policy.SelectedPolicy.FreeEnergy,
			Confidence:   policy.SelectedPolicy.Confidence,
			EpistemicVal: policy.SelectedPolicy.EpistemicVal,
			PragmaticVal: policy.SelectedPolicy.PragmaticVal,
		},
		Execution: &ExecutionProto{
			PolicyId:        execution.PolicyID,
			Duration:        execution.Duration,
			Success:        execution.Success,
			PredictionError: execution.PredictionError,
			ToolCount:      int32(len(execution.Tools)),
		},
		Precision: &PrecisionProto{
			Abstract: agent.Precision.GetAllPrecision()[math.LevelAbstract],
			Planning: agent.Precision.GetAllPrecision()[math.LevelPlanning],
			Execution: agent.Precision.GetAllPrecision()[math.LevelExecution],
		},
	}, nil
}

func (s *Server) StreamStateChanges(req *StateChangeRequest, stream grpc.ServerStreamingServer[*StateChangeResponse) error {
	agent, err := s.agentManager.GetAgent(req.AgentId)
	if err != nil {
		return status.Errorf(codes.NotFound, "agent not found: %v", err)
	}

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	versions := make(map[string]int32)

	for {
		select {
		case <-stream.Context().Done():
			return stream.Context().Err()
		case <-ticker.C:
			currentAgent, err := s.agentManager.GetAgent(req.AgentId)
			if err != nil {
				return err
			}

			lastVersion := versions[req.AgentId]
			if currentAgent.State.Version != lastVersion {
				stateJSON, _ := currentAgent.State.ToJSON()
				var stateMap map[string]interface{}
				json.Unmarshal(stateJSON, &stateMap)

				response := &StateChangeResponse{
					AgentId:   req.AgentId,
					Version:   int32(currentAgent.State.Version),
					Timestamp: timestamppb.Now(),
					Changes: &StateChangesProto{
						MessageCount: int32(len(currentAgent.State.GetMessages())),
						Precision:    currentAgent.Precision.GetAllPrecision(),
						ContextKeys:  getMapKeys(currentAgent.State.GetContext()),
					},
					StateJson: string(stateJSON),
				}

				if err := stream.Send(response); err != nil {
					return err
				}

				versions[req.AgentId] = int32(currentAgent.State.Version)
			}
		}
	}
}

func (s *Server) StreamPrecisionUpdates(req *PrecisionUpdateRequest, stream grpc.ServerStreamingServer[*PrecisionUpdateResponse]) error {
	agent, err := s.agentManager.GetAgent(req.AgentId)
	if err != nil {
		return status.Errorf(codes.NotFound, "agent not found: %v", err)
	}

	ticker := time.NewTicker(time.Duration(req.IntervalMs) * time.Millisecond)
	defer ticker.Stop()

	versions := make(map[math.HierarchyLevel]int)

	for {
		select {
		case <-stream.Context().Done():
			return stream.Context().Err()
		case <-ticker.C:
			currentAgent, err := s.agentManager.GetAgent(req.AgentId)
			if err != nil {
				return err
			}

			allPrecision := currentAgent.Precision.GetAllPrecision()
			response := &PrecisionUpdateResponse{
				AgentId:   req.AgentId,
				Timestamp: timestamppb.Now(),
				Values:    make(map[string]float64),
			}

			for level, value := range allPrecision {
				if int(level) != versions[level] {
					response.Values[level.String()] = value
					response.Updates = append(response.Updates, &PrecisionUpdateProto{
						Level:    level.String(),
						Value:    value,
						Variance: currentAgent.Precision.GetPrecisionVariance(),
					})
					versions[level] = int(level)
				}
			}

			if len(response.Updates) > 0 {
				if err := stream.Send(response); err != nil {
					return err
				}
			}
		}
	}
}

func (s *Server) Start(grpcPort, httpPort int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	kaParams := keepalive.ServerParameters{
		MaxConnectionIdle:     15 * time.Minute,
		MaxConnectionAge:      30 * time.Minute,
		MaxConnectionAgeGrace: 5 * time.Second,
		Time:                  5 * time.Second,
		Timeout:               1 * time.Second,
	}

	s.grpcServer = grpc.NewServer(
		grpc.KeepaliveParams(kaParams),
	)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", grpcPort))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", grpcPort, err)
	}

	go func() {
		s.logger.Info("starting gRPC server", zap.Int("port", grpcPort))
		if err := s.grpcServer.Serve(lis); err != nil {
			s.logger.Error("gRPC server error", zap.Error(err))
		}
	}()

	return nil
}

func (s *Server) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.grpcServer != nil {
		s.grpcServer.GracefulStop()
	}
}

func (a *Agent) Clone() *Agent {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return &Agent{
		ID:           a.ID,
		Name:         a.Name,
		Description:  a.Description,
		State:        a.State.Clone(),
		Precision:    a.Precision.Clone(),
		PolicyCalc:   nil,
		CreatedAt:    a.CreatedAt,
		UpdatedAt:    a.UpdatedAt,
		Metadata:     copyMetadata(a.Metadata),
	}
}

func copyMetadata(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{})
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func (a *Agent) ToProto() *AgentProto {
	a.mu.RLock()
	defer a.mu.RUnlock()

	stateJSON, _ := a.State.ToJSON()

	return &AgentProto{
		Id:          a.ID,
		Name:        a.Name,
		Description: a.Description,
		State:       string(stateJSON),
		Precision: &PrecisionProto{
			Abstract:  a.Precision.GetAllPrecision()[math.LevelAbstract],
			Planning:  a.Precision.GetAllPrecision()[math.LevelPlanning],
			Execution: a.Precision.GetAllPrecision()[math.LevelExecution],
		},
		Version: int32(a.State.Version),
	}
}

type AgentProto struct {
	Id          string       `json:"id"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	State       string       `json:"state"`
	Precision   *PrecisionProto `json:"precision"`
	Version     int32        `json:"version"`
}

type PrecisionProto struct {
	Abstract  float64 `json:"abstract"`
	Planning  float64 `json:"planning"`
	Execution float64 `json:"execution"`
}

type PolicyProto struct {
	Id            string  `json:"id"`
	Name          string  `json:"name"`
	FreeEnergy    float64 `json:"free_energy"`
	Confidence    float64 `json:"confidence"`
	EpistemicVal  float64 `json:"epistemic_value"`
	PragmaticVal  float64 `json:"pragmatic_value"`
}

type ExecutionProto struct {
	PolicyId        string  `json:"policy_id"`
	Duration        float64 `json:"duration_ms"`
	Success        bool    `json:"success"`
	PredictionError float64 `json:"prediction_error"`
	ToolCount      int32   `json:"tool_count"`
}

type StateChangesProto struct {
	MessageCount int32              `json:"message_count"`
	Precision    map[string]float64 `json:"precision"`
	ContextKeys  []string          `json:"context_keys"`
}

type PrecisionUpdateProto struct {
	Level    string  `json:"level"`
	Value    float64 `json:"value"`
	Variance float64 `json:"variance"`
}

type CreateAgentRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type GetAgentRequest struct {
	Id string `json:"id"`
}

type ListAgentsRequest struct {
	Limit  int32 `json:"limit"`
	Offset int32 `json:"offset"`
}

type DeleteAgentRequest struct {
	Id string `json:"id"`
}

type ExecutePolicyRequest struct {
	AgentId string                 `json:"agent_id"`
	Context map[string]interface{} `json:"context"`
}

type StateChangeRequest struct {
	AgentId string `json:"agent_id"`
}

type PrecisionUpdateRequest struct {
	AgentId     string `json:"agent_id"`
	IntervalMs int64  `json:"interval_ms"`
}

type AgentResponse struct {
	Agent     *AgentProto     `json:"agent"`
	CreatedAt *timestamppb.Timestamp `json:"created_at"`
	UpdatedAt *timestamppb.Timestamp `json:"updated_at"`
}

type ListAgentsResponse struct {
	Agents []*AgentProto `json:"agents"`
	Total  int32          `json:"total"`
}

type ExecutePolicyResponse struct {
	Policy    *PolicyProto    `json:"policy"`
	Execution *ExecutionProto `json:"execution"`
	Precision *PrecisionProto `json:"precision"`
}

type StateChangeResponse struct {
	AgentId   string           `json:"agent_id"`
	Version   int32            `json:"version"`
	Timestamp *timestamppb.Timestamp `json:"timestamp"`
	Changes   *StateChangesProto `json:"changes"`
	StateJson string           `json:"state_json"`
}

type PrecisionUpdateResponse struct {
	AgentId   string                 `json:"agent_id"`
	Timestamp *timestamppb.Timestamp `json:"timestamp"`
	Values    map[string]float64     `json:"values"`
	Updates   []*PrecisionUpdateProto `json:"updates"`
}

func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

type AgentServiceServer interface {
	CreateAgent(ctx context.Context, req *CreateAgentRequest) (*AgentResponse, error)
	GetAgent(ctx context.Context, req *GetAgentRequest) (*AgentResponse, error)
	ListAgents(ctx context.Context, req *ListAgentsRequest) (*ListAgentsResponse, error)
	DeleteAgent(ctx context.Context, req *DeleteAgentRequest) (*emptypb.Empty, error)
	ExecutePolicy(ctx context.Context, req *ExecutePolicyRequest) (*ExecutePolicyResponse, error)
	StreamStateChanges(req *StateChangeRequest, stream grpc.ServerStreamingServer[*StateChangeResponse]) error
	StreamPrecisionUpdates(req *PrecisionUpdateRequest, stream grpc.ServerStreamingServer[*PrecisionUpdateResponse]) error
}

type grpcServerStream[T any] interface {
	Send(T) error
	Context() context.Context
}
