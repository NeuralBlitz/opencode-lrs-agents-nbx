package api

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/neuralblitz/go-lrs/internal/math"
	"github.com/neuralblitz/go-lrs/internal/state"
	"github.com/neuralblitz/go-lrs/pkg/core"
	"go.uber.org/zap"
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
		ID:           generateID(),
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
	}
	return tracker
}

func generate		panic(err)
ID() string {
	return fmt.Sprintf("agent-%d", time.Now().UnixNano())
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

	return &AgentProto{
		Id:          a.ID,
		Name:        a.Name,
		Description: a.Description,
		Version:     int32(a.State.Version),
	}
}

type AgentProto struct {
	Id          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Version     int32  `json:"version"`
}

type AgentManager struct {
	agents    map[string]*Agent
	registry  *core.ToolRegistry
	logger    *zap.Logger
	mu        sync.RWMutex
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

func (m *AgentManager) GetAgentCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.agents)
}

func (m *AgentManager) UpdateAgentState(id string, updater func(*Agent) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	agent, exists := m.agents[id]
	if !exists {
		return fmt.Errorf("agent '%s' not found", id)
	}

	if err := updater(agent); err != nil {
		return err
	}

	agent.UpdatedAt = time.Now().UnixMilli()
	return nil
}

func (m *AgentManager) UpdateAgentPrecision(id string, level math.HierarchyLevel, predictionError float64) error {
	return m.UpdateAgentState(id, func(agent *Agent) error {
		return agent.Precision.Update(level, predictionError)
	})
}

func (m *AgentManager) ExecutePolicyForAgent(id string, context map[string]interface{}) (*math.PolicySelection, error) {
	agent, err := m.GetAgent(id)
	if err != nil {
		return nil, err
	}

	policy, err := agent.PolicyCalc.SelectPolicyWithInfo(context)
	if err != nil {
		return nil, err
	}

	return policy, nil
}

func (m *AgentManager) GetAgentMetrics(id string) (map[string]float64, error) {
	agent, err := m.GetAgent(id)
	if err != nil {
		return nil, err
	}

	precision := agent.Precision.GetAllPrecision()
	return map[string]float64{
		"precision_abstract":  precision[math.LevelAbstract],
		"precision_planning":  precision[math.LevelPlanning],
		"precision_execution": precision[math.LevelExecution],
		"average_precision":    agent.Precision.GetAveragePrecision(),
		"weighted_precision":   agent.Precision.GetWeightedPrecision(),
		"coherence":           agent.Precision.CalculateCoherence(),
	}, nil
}

type AgentEventType int

const (
	AgentCreated AgentEventType = iota
	AgentUpdated
	AgentDeleted
	PolicyExecuted
	PrecisionUpdated
)

type AgentEvent struct {
	Type      AgentEventType
	AgentID   string
	Timestamp int64
	Data      map[string]interface{}
}

type AgentEventHandler func(event AgentEvent)

type EventBus struct {
	handlers []AgentEventHandler
	mu       sync.RWMutex
}

func NewEventBus() *EventBus {
	return &EventBus{
		handlers: make([]AgentEventHandler, 0),
	}
}

func (b *EventBus) Subscribe(handler AgentEventHandler) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.handlers = append(b.handlers, handler)
}

func (b *EventBus) Publish(event AgentEvent) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for _, handler := range b.handlers {
		go handler(event)
	}
}

func (m *AgentManager) GetEventBus() *EventBus {
	return NewEventBus()
}

func (m *AgentManager) WithTransaction(fn func(*AgentManager) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return fn(m)
}
