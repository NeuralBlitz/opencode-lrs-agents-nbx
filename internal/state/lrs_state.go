package state

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/neuralblitz/go-lrs/internal/math"
)

type Message struct {
	Role       string                 `json:"role"`
	Content    string                 `json:"content"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Timestamp  int64                  `json:"timestamp"`
	Weight     float64                `json:"weight"`
	SenderID   string                 `json:"sender_id,omitempty"`
	ReceiverID string                 `json:"receiver_id,omitempty"`
}

func NewMessage(role, content string) *Message {
	return &Message{
		Role:      role,
		Content:   content,
		Metadata:  make(map[string]interface{}),
		Timestamp: time.Now().UnixMilli(),
		Weight:    1.0,
	}
}

func (m *Message) WithMetadata(key string, value interface{}) *Message {
	if m.Metadata == nil {
		m.Metadata = make(map[string]interface{})
	}
	m.Metadata[key] = value
	return m
}

func (m *Message) SetTimestamp(ts int64) *Message {
	m.Timestamp = ts
	return m
}

func (m *Message) SetWeight(weight float64) *Message {
	m.Weight = weight
	return m
}

type ToolExecution struct {
	ToolName       string                 `json:"tool_name"`
	Input          map[string]interface{} `json:"input,omitempty"`
	Output         interface{}            `json:"output,omitempty"`
	StartTime      int64                  `json:"start_time"`
	EndTime        int64                  `json:"end_time"`
	Duration       float64                `json:"duration_ms"`
	PredictionError float64               `json:"prediction_error"`
	Success        bool                   `json:"success"`
	Error          string                 `json:"error,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

func NewToolExecution(toolName string) *ToolExecution {
	return &ToolExecution{
		ToolName:  toolName,
		Input:     make(map[string]interface{}),
		Metadata:  make(map[string]interface{}),
		StartTime: time.Now().UnixMilli(),
		Success:   true,
	}
}

func (e *ToolExecution) WithInput(key string, value interface{}) *ToolExecution {
	e.Input[key] = value
	return e
}

func (e *ToolExecution) SetOutput(output interface{}) *ToolExecution {
	e.Output = output
	return e
}

func (e *ToolExecution) SetSuccess(success bool) *ToolExecution {
	e.Success = success
	return e
}

func (e *ToolExecution) SetPredictionError(err float64) *ToolExecution {
	e.PredictionError = err
	return e
}

func (e *ToolExecution) Complete() *ToolExecution {
	e.EndTime = time.Now().UnixMilli()
	e.Duration = float64(e.EndTime - e.StartTime)
	return e
}

type PolicyExecution struct {
	PolicyID       string                 `json:"policy_id"`
	PolicyName     string                 `json:"policy_name"`
	Tools          []string               `json:"tools"`
	StartTime      int64                  `json:"start_time"`
	EndTime        int64                  `json:"end_time"`
	Duration       float64                `json:"duration_ms"`
	Success        bool                   `json:"success"`
	FreeEnergy     float64                `json:"free_energy"`
	PredictionError float64               `json:"prediction_error"`
	ToolExecutions []*ToolExecution       `json:"tool_executions"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

func NewPolicyExecution(policyID, policyName string) *PolicyExecution {
	return &PolicyExecution{
		PolicyID:        policyID,
		PolicyName:      policyName,
		Tools:           make([]string, 0),
		StartTime:       time.Now().UnixMilli(),
		Success:         true,
		ToolExecutions:  make([]*ToolExecution, 0),
		Metadata:        make(map[string]interface{}),
	}
}

func (e *PolicyExecution) AddToolExecution(exec *ToolExecution) *PolicyExecution {
	e.ToolExecutions = append(e.ToolExecutions, exec)
	e.Tools = append(e.Tools, exec.ToolName)
	return e
}

func (e *PolicyExecution) Complete() *PolicyExecution {
	e.EndTime = time.Now().UnixMilli()
	e.Duration = float64(e.EndTime - e.StartTime)

	var totalPredError float64
	successCount := 0
	for _, exec := range e.ToolExecutions {
		totalPredError += exec.PredictionError
		if exec.Success {
			successCount++
		}
	}

	if len(e.ToolExecutions) > 0 {
		e.PredictionError = totalPredError / float64(len(e.ToolExecutions))
	}
	e.Success = successCount == len(e.ToolExecutions)

	return e
}

type LRSState struct {
	Messages         []*Message           `json:"messages"`
	Precision        map[string]float64   `json:"precision"`
	CurrentPolicy    *PolicyExecution     `json:"current_policy,omitempty"`
	PolicyHistory    []*PolicyExecution   `json:"policy_history"`
	ExecutionStack  []*PolicyExecution   `json:"execution_stack"`
	Context          map[string]interface{} `json:"context"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
	Checkpoint       *Checkpoint          `json:"-"`
	Version          int                  `json:"version"`
	CreatedAt        int64                `json:"created_at"`
	UpdatedAt        int64                `json:"updated_at"`
	mu               sync.RWMutex
}

type Checkpoint struct {
	State    *LRSState
	Snapshot []byte
	Created  int64
}

func NewLRSState() *LRSState {
	now := time.Now().UnixMilli()
	return &LRSState{
		Messages:        make([]*Message, 0),
		Precision:       make(map[string]float64),
		PolicyHistory:   make([]*PolicyExecution, 0),
		ExecutionStack:  make([]*PolicyExecution, 0),
		Context:         make(map[string]interface{}),
		Metadata:        make(map[string]interface{}),
		Version:         1,
		CreatedAt:       now,
		UpdatedAt:       now,
	}
}

func (s *LRSState) Clone() *LRSState {
	s.mu.RLock()
	defer s.mu.RUnlock()

	clonedMessages := make([]*Message, len(s.Messages))
	for i, msg := range s.Messages {
		msgCopy := *msg
		msgCopy.Metadata = copyMetadata(msg.Metadata)
		clonedMessages[i] = &msgCopy
	}

	clonedPrecision := make(map[string]float64)
	for k, v := range s.Precision {
		clonedPrecision[k] = v
	}

	clonedPolicyHistory := make([]*PolicyExecution, len(s.PolicyHistory))
	for i, pe := range s.PolicyHistory {
		peCopy := *pe
		peCopy.Metadata = copyMetadata(pe.Metadata)
		clonedPolicyHistory[i] = &peCopy
	}

	clonedContext := copyMetadata(s.Context)

	return &LRSState{
		Messages:        clonedMessages,
		Precision:       clonedPrecision,
		CurrentPolicy:   s.CurrentPolicy,
		PolicyHistory:   clonedPolicyHistory,
		ExecutionStack:  s.ExecutionStack,
		Context:         clonedContext,
		Metadata:        copyMetadata(s.Metadata),
		Version:         s.Version,
		CreatedAt:       s.CreatedAt,
		UpdatedAt:       s.UpdatedAt,
	}
}

func copyMetadata(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{})
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func (s *LRSState) WithMessage(msg *Message) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	newState.Messages = append(newState.Messages, msg)
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) WithMessages(messages []*Message) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	newState.Messages = append(newState.Messages, messages...)
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) WithPrecision(level string, value float64) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	newState.Precision[level] = value
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) WithContext(key string, value interface{}) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	newState.Context[key] = value
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) WithPolicy(policy *PolicyExecution) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	newState.PolicyHistory = append(newState.PolicyHistory, policy)
	newState.CurrentPolicy = policy
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) PushPolicy(policy *PolicyExecution) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	newState.ExecutionStack = append(newState.ExecutionStack, policy)
	newState.CurrentPolicy = policy
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) PopPolicy() (*PolicyExecution, *LRSState) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.ExecutionStack) == 0 {
		return nil, s
	}

	newState := s.Clone()
	newState.ExecutionStack = newState.ExecutionStack[:len(newState.ExecutionStack)-1]

	if len(newState.ExecutionStack) > 0 {
		newState.CurrentPolicy = newState.ExecutionStack[len(newState.ExecutionStack)-1]
	} else {
		newState.CurrentPolicy = nil
	}

	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	popped := s.ExecutionStack[len(s.ExecutionStack)-1]
	return popped, newState
}

func (s *LRSState) GetMessages() []*Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Messages
}

func (s *LRSState) GetPrecision() map[string]float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	precision := make(map[string]float64)
	for k, v := range s.Precision {
		precision[k] = v
	}
	return precision
}

func (s *LRSState) GetContext() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return copyMetadata(s.Context)
}

func (s *LRSState) GetPolicyHistory() []*PolicyExecution {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.PolicyHistory
}

func (s *LRSState) GetCurrentPolicy() *PolicyExecution {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.CurrentPolicy
}

func (s *LRSState) GetExecutionStack() []*PolicyExecution {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ExecutionStack
}

func (s *LRSState) AddMessage(role, content string) *LRSState {
	msg := NewMessage(role, content)
	return s.WithMessage(msg)
}

func (s *LRSState) AddUserMessage(content string) *LRSState {
	return s.AddMessage("user", content)
}

func (s *LRSState) AddAssistantMessage(content string) *LRSState {
	return s.AddMessage("assistant", content)
}

func (s *LRSState) AddSystemMessage(content string) *LRSState {
	return s.AddMessage("system", content)
}

func (s *LRSState) SetPrecisionFromTracker(tracker *math.PrecisionTracker) *LRSState {
	return s.WithPrecision("default", tracker.CurrentValue())
}

func (s *LRSState) SetHierarchicalPrecision(hp *math.HierarchicalPrecision) *LRSState {
	s.mu.Lock()
	defer s.mu.Unlock()

	newState := s.Clone()
	for level, value := range hp.GetAllPrecision() {
		newState.Precision[level.String()] = value
	}
	newState.UpdatedAt = time.Now().UnixMilli()
	newState.Version++

	return newState
}

func (s *LRSState) ToJSON() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return json.MarshalIndent(s, "", "  ")
}

func (s *LRSState) FromJSON(data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return json.Unmarshal(data, s)
}

func (s *LRSState) String() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return fmt.Sprintf("LRSState{messages=%d, precision=%v, version=%d}", len(s.Messages), s.Precision, s.Version)
}

type StateManager struct {
	states        map[string]*LRSState
	checkpoints   map[string]*Checkpoint
	maxHistory    int
	maxCheckpoints int
	mu            sync.RWMutex
}

func NewStateManager() *StateManager {
	return &StateManager{
		states:        make(map[string]*LRSState),
		checkpoints:   make(map[string]*Checkpoint),
		maxHistory:    100,
		maxCheckpoints: 50,
	}
}

func (m *StateManager) CreateState(id string) (*LRSState, error) {
	if id == "" {
		return nil, errors.New("state ID cannot be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.states[id]; exists {
		return nil, fmt.Errorf("state '%s' already exists", id)
	}

	state := NewLRSState()
	m.states[id] = state
	return state, nil
}

func (m *StateManager) GetState(id string) (*LRSState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	state, exists := m.states[id]
	if !exists {
		return nil, fmt.Errorf("state '%s' not found", id)
	}
	return state.Clone(), nil
}

func (m *StateManager) UpdateState(id string, updater func(*LRSState) *LRSState) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	state, exists := m.states[id]
	if !exists {
		return fmt.Errorf("state '%s' not found", id)
	}

	updatedState := updater(state.Clone())
	m.states[id] = updatedState

	return nil
}

func (m *StateManager) DeleteState(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.states[id]; !exists {
		return fmt.Errorf("state '%s' not found", id)
	}

	delete(m.states, id)
	delete(m.checkpoints, id)

	return nil
}

func (m *StateManager) ListStates() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ids := make([]string, 0, len(m.states))
	for id := range m.states {
		ids = append(ids, id)
	}
	return ids
}

func (m *StateManager) CreateCheckpoint(stateID, checkpointID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	state, exists := m.states[stateID]
	if !exists {
		return fmt.Errorf("state '%s' not found", stateID)
	}

	snapshot, err := state.ToJSON()
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %w", err)
	}

	checkpoint := &Checkpoint{
		State:    state.Clone(),
		Snapshot: snapshot,
		Created:  time.Now().UnixMilli(),
	}

	m.checkpoints[fmt.Sprintf("%s:%s", stateID, checkpointID)] = checkpoint

	if len(m.checkpoints) > m.maxCheckpoints {
		m.cleanupOldestCheckpoint()
	}

	return nil
}

func (m *StateManager) RestoreCheckpoint(stateID, checkpointID string) (*LRSState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	key := fmt.Sprintf("%s:%s", stateID, checkpointID)
	checkpoint, exists := m.checkpoints[key]
	if !exists {
		return nil, fmt.Errorf("checkpoint '%s' not found for state '%s'", checkpointID, stateID)
	}

	return checkpoint.State.Clone(), nil
}

func (m *StateManager) ListCheckpoints(stateID string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	prefix := fmt.Sprintf("%s:", stateID)
	checkpoints := make([]string, 0)
	for key := range m.checkpoints {
		if len(key) > len(prefix) && key[:len(prefix)] == prefix {
			checkpoints = append(checkpoints, key[len(prefix):])
		}
	}
	return checkpoints, nil
}

func (m *StateManager) DeleteCheckpoint(stateID, checkpointID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := fmt.Sprintf("%s:%s", stateID, checkpointID)
	if _, exists := m.checkpoints[key]; !exists {
		return fmt.Errorf("checkpoint '%s' not found for state '%s'", checkpointID, stateID)
	}

	delete(m.checkpoints, key)
	return nil
}

func (m *StateManager) cleanupOldestCheckpoint() {
	var oldestKey string
	var oldestTime int64 = -1

	for key, checkpoint := range m.checkpoints {
		if oldestTime == -1 || checkpoint.Created < oldestTime {
			oldestTime = checkpoint.Created
			oldestKey = key
		}
	}

	if oldestKey != "" {
		delete(m.checkpoints, oldestKey)
	}
}

func (m *StateManager) Rollback(stateID string, targetVersion int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	state, exists := m.states[stateID]
	if !exists {
		return fmt.Errorf("state '%s' not found", stateID)
	}

	if targetVersion < 1 || targetVersion >= state.Version {
		return fmt.Errorf("invalid target version %d (current: %d)", targetVersion, state.Version)
	}

	for _, checkpoint := range m.checkpoints {
		if checkpoint.State.Version == targetVersion {
			m.states[stateID] = checkpoint.State.Clone()
			return nil
		}
	}

	return fmt.Errorf("no checkpoint found for version %d", targetVersion)
}

func (m *StateManager) GetStateCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.states)
}

func (m *StateManager) GetCheckpointCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.checkpoints)
}

type StateDiff struct {
	AddedMessages   []*Message          `json:"added_messages"`
	RemovedMessages int                `json:"removed_messages"`
	PrecisionChanges map[string]float64 `json:"precision_changes"`
	ContextChanges  map[string]interface{} `json:"context_changes"`
	VersionChange   int                `json:"version_change"`
}

func (s *LRSState) Diff(other *LRSState) *StateDiff {
	s.mu.RLock()
	other.mu.RLock()
	defer s.mu.RUnlock()
	defer other.mu.RUnlock()

	diff := &StateDiff{
		AddedMessages:    make([]*Message, 0),
		PrecisionChanges: make(map[string]float64),
		ContextChanges:   make(map[string]interface{}),
	}

	for _, msg := range other.Messages {
		found := false
		for _, existingMsg := range s.Messages {
			if msg.Content == existingMsg.Content && msg.Role == existingMsg.Role {
				found = true
				break
			}
		}
		if !found {
			diff.AddedMessages = append(diff.AddedMessages, msg)
		}
	}

	diff.RemovedMessages = len(s.Messages) - len(other.Messages)

	for k, v := range other.Precision {
		if s.Precision[k] != v {
			diff.PrecisionChanges[k] = v
		}
	}

	for k, v := range other.Context {
		if s.Context[k] != v {
			diff.ContextChanges[k] = v
		}
	}

	diff.VersionChange = other.Version - s.Version

	return diff
}

func (m *StateManager) GetStateDiff(stateID1, stateID2 string) (*StateDiff, error) {
	state1, err := m.GetState(stateID1)
	if err != nil {
		return nil, err
	}

	state2, err := m.GetState(stateID2)
	if err != nil {
		return nil, err
	}

	return state1.Diff(state2), nil
}
