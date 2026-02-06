package api

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/neuralblitz/go-lrs/internal/state"
)

type StateManager struct {
	manager    *state.StateManager
	logger     interface{}
	mu         sync.RWMutex
}

func NewStateManager() *StateManager {
	return &StateManager{
		manager: state.NewStateManager(),
	}
}

func (m *StateManager) CreateState(id string) (*state.LRSState, error) {
	return m.manager.CreateState(id)
}

func (m *StateManager) GetState(id string) (*state.LRSState, error) {
	return m.manager.GetState(id)
}

func (m *StateManager) UpdateState(id string, updater func(*state.LRSState) *state.LRSState) error {
	return m.manager.UpdateState(id, updater)
}

func (m *StateManager) DeleteState(id string) error {
	return m.manager.DeleteState(id)
}

func (m *StateManager) ListStates() []string {
	return m.manager.ListStates()
}

func (m *StateManager) CreateCheckpoint(stateID, checkpointID string) error {
	return m.manager.CreateCheckpoint(stateID, checkpointID)
}

func (m *StateManager) RestoreCheckpoint(stateID, checkpointID string) (*state.LRSState, error) {
	return m.manager.RestoreCheckpoint(stateID, checkpointID)
}

func (m *StateManager) ListCheckpoints(stateID string) ([]string, error) {
	return m.manager.ListCheckpoints(stateID)
}

func (m *StateManager) DeleteCheckpoint(stateID, checkpointID string) error {
	return m.manager.DeleteCheckpoint(stateID, checkpointID)
}

func (m *StateManager) GetStateCount() int {
	return m.manager.GetStateCount()
}

func (m *StateManager) GetCheckpointCount() int {
	return m.manager.GetCheckpointCount()
}

func (m *StateManager) GetStateDiff(stateID1, stateID2 string) (*state.StateDiff, error) {
	return m.manager.GetStateDiff(stateID1, stateID2)
}

type StateSnapshot struct {
	ID        string                 `json:"id"`
	State     map[string]interface{} `json:"state"`
	Version   int                   `json:"version"`
	CreatedAt int64                 `json:"created_at"`
}

func (s *LRSState) ToSnapshot() (*StateSnapshot, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stateJSON, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal state: %w", err)
	}

	var stateMap map[string]interface{}
	if err := json.Unmarshal(stateJSON, &stateMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal state: %w", err)
	}

	return &StateSnapshot{
		ID:        fmt.Sprintf("snapshot-%d", time.Now().UnixNano()),
		State:     stateMap,
		Version:   s.Version,
		CreatedAt: time.Now().UnixMilli(),
	}, nil
}

func (m *StateManager) CreateSnapshot(stateID string) (*StateSnapshot, error) {
	state, err := m.manager.GetState(stateID)
	if err != nil {
		return nil, err
	}
	return state.ToSnapshot()
}

type StateMerger struct {
	mu sync.RWMutex
}

func NewStateMerger() *StateMerger {
	return &StateMerger{}
}

func (m *StateMerger) Merge(base, overlay *state.LRSState) (*state.LRSState, error) {
	base.mu.RLock()
	overlay.mu.RLock()
	defer base.mu.RUnlock()
	defer overlay.mu.RUnlock()

	merged := base.Clone()

	for _, msg := range overlay.GetMessages() {
		merged = merged.WithMessage(msg)
	}

	for level, val := range overlay.GetPrecision() {
		merged = merged.WithPrecision(level, val)
	}

	for k, v := range overlay.GetContext() {
		merged = merged.WithContext(k, v)
	}

	return merged, nil
}

func (m *StateMerger) DeepMerge(state1, state2 *state.LRSState) (*state.LRSState, error) {
	state1.mu.RLock()
	state2.mu.RLock()
	defer state1.mu.RUnlock()
	defer state2.mu.RUnlock()

	merged := state1.Clone()

	messages1 := state1.GetMessages()
	messages2 := state2.GetMessages()
	mergedMessages := append(messages1, messages2...)

	precision1 := state1.GetPrecision()
	precision2 := state2.GetPrecision()
	mergedPrecision := make(map[string]float64)
	for k, v := range precision1 {
		mergedPrecision[k] = v
	}
	for k, v := range precision2 {
		mergedPrecision[k] = v
	}

	context1 := state1.GetContext()
	context2 := state2.GetContext()
	mergedContext := make(map[string]interface{})
	for k, v := range context1 {
		mergedContext[k] = v
	}
	for k, v := range context2 {
		mergedContext[k] = v
	}

	state := state.NewLRSState()
	state = state.WithMessages(mergedMessages)
	for k, v := range mergedPrecision {
		state = state.WithPrecision(k, v)
	}
	for k, v := range mergedContext {
		state = state.WithContext(k, v)
	}

	return state, nil
}

type StateValidator struct {
	mu sync.RWMutex
}

func NewStateValidator() *StateValidator {
	return &StateValidator{}
}

func (v *StateValidator) Validate(state *state.LRSState) error {
	if state == nil {
		return fmt.Errorf("state cannot be nil")
	}

	state.mu.RLock()
	defer state.mu.RUnlock()

	if state.Version < 0 {
		return fmt.Errorf("state version cannot be negative")
	}

	for _, msg := range state.GetMessages() {
		if msg.Role == "" {
			return fmt.Errorf("message role cannot be empty")
		}
		if msg.Content == "" {
			return fmt.Errorf("message content cannot be empty")
		}
	}

	for _, pe := range state.GetPolicyHistory() {
		if pe.PolicyID == "" {
			return fmt.Errorf("policy execution ID cannot be empty")
		}
		if pe.PredictionError < 0 || pe.PredictionError > 1 {
			return fmt.Errorf("policy prediction error must be in [0, 1]")
		}
	}

	return nil
}

func (v *StateValidator) ValidateCheckpoint(checkpoint *state.Checkpoint) error {
	if checkpoint == nil {
		return fmt.Errorf("checkpoint cannot be nil")
	}

	if checkpoint.State == nil {
		return fmt.Errorf("checkpoint state cannot be nil")
	}

	if err := v.Validate(checkpoint.State); err != nil {
		return fmt.Errorf("invalid checkpoint state: %w", err)
	}

	return nil
}

type StateObserver struct {
	observers map[string][]chan *state.LRSState
	mu        sync.RWMutex
}

func NewStateObserver() *StateObserver {
	return &StateObserver{
		observers: make(map[string][]chan *state.LRSState),
	}
}

func (o *StateObserver) Subscribe(stateID string, ch chan *state.LRSState) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.observers[stateID] = append(o.observers[stateID], ch)
}

func (o *StateObserver) Unsubscribe(stateID string, ch chan *state.LRSState) {
	o.mu.Lock()
	defer o.mu.Unlock()

	channels := o.observers[stateID]
	for i, c := range channels {
		if c == ch {
			o.observers[stateID] = append(channels[:i], channels[i+1:]...)
			break
		}
	}
}

func (o *StateObserver) Notify(stateID string, newState *state.LRSState) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	channels := o.observers[stateID]
	for _, ch := range channels {
		go func(c chan *state.LRSState) {
			c <- newState
		}(ch)
	}
}

func (m *StateManager) GetStateObserver() *StateObserver {
	return NewStateObserver()
}

type StateCache struct {
	cache    map[string]*state.LRSState
	maxSize  int
	mu       sync.RWMutex
}

func NewStateCache(maxSize int) *StateCache {
	if maxSize <= 0 {
		maxSize = 100
	}
	return &StateCache{
		cache:   make(map[string]*state.LRSState),
		maxSize: maxSize,
	}
}

func (c *StateCache) Get(id string) (*state.LRSState, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	state, exists := c.cache[id]
	if !exists {
		return nil, false
	}
	return state.Clone(), true
}

func (c *StateCache) Put(id string, state *state.LRSState) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.cache) >= c.maxSize {
		c.evictOldest()
	}

	c.cache[id] = state.Clone()
}

func (c *StateCache) Delete(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.cache, id)
}

func (c *StateCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[string]*state.LRSState)
}

func (c *StateCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache)
}

func (c *StateCache) evictOldest() {
	var oldestID string
	var oldestTime int64 = -1

	for id, state := range c.cache {
		if oldestTime == -1 || state.UpdatedAt < oldestTime {
			oldestTime = state.UpdatedAt
			oldestID = id
		}
	}

	if oldestID != "" {
		delete(c.cache, oldestID)
	}
}

func (c *StateCache) Contains(id string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	_, exists := c.cache[id]
	return exists
}
