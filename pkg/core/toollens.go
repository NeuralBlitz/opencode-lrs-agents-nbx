package core

import (
	"errors"
	"fmt"
	"sync"
)

type ExecutionResult struct {
	Success          bool
	PredictionError  float64
	Observation      interface{}
	Metadata         map[string]interface{}
	ExecutionTime    float64
	ToolName         string
	Error            error
	mu               sync.RWMutex
}

func NewExecutionResult() *ExecutionResult {
	return &ExecutionResult{
		Success:   true,
		Metadata:  make(map[string]interface{}),
	}
}

func (r *ExecutionResult) SetPredictionError(predictionError float64) *ExecutionResult {
	if predictionError < 0 {
		predictionError = 0
	} else if predictionError > 1 {
		predictionError = 1
	}
	r.mu.Lock()
	r.PredictionError = predictionError
	r.mu.Unlock()
	return r
}

func (r *ExecutionResult) GetPredictionError() float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.PredictionError
}

func (r *ExecutionResult) SetSuccess(success bool) *ExecutionResult {
	r.mu.Lock()
	r.Success = success
	if !success {
		r.PredictionError = 1.0
	}
	r.mu.Unlock()
	return r
}

func (r *ExecutionResult) SetObservation(observation interface{}) *ExecutionResult {
	r.mu.Lock()
	r.Observation = observation
	r.mu.Unlock()
	return r
}

func (r *ExecutionResult) AddMetadata(key string, value interface{}) *ExecutionResult {
	r.mu.Lock()
	r.Metadata[key] = value
	r.mu.Unlock()
	return r
}

func (r *ExecutionResult) SetExecutionTime(time float64) *ExecutionResult {
	r.mu.Lock()
	r.ExecutionTime = time
	r.mu.Unlock()
	return r
}

func (r *ExecutionResult) SetError(err error) *ExecutionResult {
	r.mu.Lock()
	r.Error = err
	if err != nil {
		r.Success = false
		r.PredictionError = 1.0
	}
	r.mu.Unlock()
	return r
}

func (r *ExecutionResult) Clone() *ExecutionResult {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return &ExecutionResult{
		Success:         r.Success,
		PredictionError: r.PredictionError,
		Observation:     r.Observation,
		Metadata:        copyMetadata(r.Metadata),
		ExecutionTime:  r.ExecutionTime,
		ToolName:       r.ToolName,
		Error:           r.Error,
	}
}

func copyMetadata(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{})
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func (r *ExecutionResult) String() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return fmt.Sprintf("ExecutionResult{success=%v, prediction_error=%.4f, tool=%s}",
		r.Success, r.PredictionError, r.ToolName)
}

type ToolLens interface {
	Execute(state interface{}) (*ExecutionResult, error)
	Update(result *ExecutionResult) error
	GetName() string
	GetDescription() string
	GetParameters() map[string]interface{}
	GetVersion() string
	ValidateParameters(params map[string]interface{}) error
}

type BaseTool struct {
	Name        string
	Description  string
	Version     string
	Parameters   map[string]interface{}
	mu          sync.RWMutex
}

func (t *BaseTool) GetName() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.Name
}

func (t *BaseTool) SetName(name string) {
	t.mu.Lock()
	t.Name = name
	t.mu.Unlock()
}

func (t *BaseTool) GetDescription() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.Description
}

func (t *BaseTool) SetDescription(description string) {
	t.mu.Lock()
	t.Description = description
	t.mu.Unlock()
}

func (t *BaseTool) GetVersion() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.Version
}

func (t *BaseTool) SetVersion(version string) {
	t.mu.Lock()
	t.Version = version
	t.mu.Unlock()
}

func (t *BaseTool) GetParameters() map[string]interface{} {
	t.mu.RLock()
	defer t.mu.RUnlock()
	params := make(map[string]interface{})
	for k, v := range t.Parameters {
		params[k] = v
	}
	return params
}

func (t *BaseTool) SetParameters(params map[string]interface{}) {
	t.mu.Lock()
	t.Parameters = params
	t.mu.Unlock()
}

func (t *BaseTool) ValidateParameters(params map[string]interface{}) error {
	return nil
}

type SimpleTool struct {
	BaseTool
	ExecuteFunc func(state interface{}) (*ExecutionResult, error)
	UpdateFunc  func(result *ExecutionResult) error
}

func NewSimpleTool(name, description string) *SimpleTool {
	return &SimpleTool{
		BaseTool: BaseTool{
			Name:        name,
			Description: description,
			Version:     "1.0.0",
			Parameters:  make(map[string]interface{}),
		},
	}
}

func (t *SimpleTool) Execute(state interface{}) (*ExecutionResult, error) {
	result := NewExecutionResult()
	result.ToolName = t.GetName()

	if t.ExecuteFunc != nil {
		execResult, err := t.ExecuteFunc(state)
		if err != nil {
			return result.SetError(err), err
		}
		return execResult, nil
	}

	return result.SetSuccess(true), nil
}

func (t *SimpleTool) Update(result *ExecutionResult) error {
	if t.UpdateFunc != nil {
		return t.UpdateFunc(result)
	}
	return nil
}

func (t *SimpleTool) WithExecuteFn(fn func(state interface{}) (*ExecutionResult, error)) *SimpleTool {
	t.ExecuteFunc = fn
	return t
}

func (t *SimpleTool) WithUpdateFn(fn func(result *ExecutionResult) error) *SimpleTool {
	t.UpdateFunc = fn
	return t
}

type CompositeTool struct {
	BaseTool
	Tools      []ToolLens
	Strategy   CompositeStrategy
	mu         sync.RWMutex
}

type CompositeStrategy int

const (
	StrategySequential CompositeStrategy = iota
	StrategyParallel
	StrategyFirstSuccess
	StrategyAll
)

func (t *CompositeTool) Execute(state interface{}) (*ExecutionResult, error) {
	result := NewExecutionResult()
	result.ToolName = t.GetName()

	t.mu.RLock()
	tools := t.Tools
	strategy := t.Strategy
	t.mu.RUnlock()

	switch strategy {
	case StrategySequential:
		return t.executeSequential(state, tools)
	case StrategyParallel:
		return t.executeParallel(state, tools)
	case StrategyFirstSuccess:
		return t.executeFirstSuccess(state, tools)
	case StrategyAll:
		return t.executeAll(state, tools)
	default:
		return t.executeSequential(state, tools)
	}
}

func (t *CompositeTool) executeSequential(state interface{}, tools []ToolLens) (*ExecutionResult, error) {
	currentState := state
	var lastResult *ExecutionResult

	for _, tool := range tools {
		result, err := tool.Execute(currentState)
		if err != nil {
			return result, err
		}

		if err := tool.Update(result); err != nil {
			return result, err
		}

		lastResult = result

		if observation, ok := result.Observation.(interface{}); ok {
			currentState = observation
		}
	}

	return lastResult, nil
}

func (t *CompositeTool) executeParallel(state interface{}, tools []ToolLens) (*ExecutionResult, error) {
	type resultChan struct {
		result *ExecutionResult
		err    error
	}

	results := make(chan resultChan, len(tools))

	for _, tool := range tools {
		go func(t ToolLens) {
			result, err := t.Execute(state)
			results <- resultChan{result: result, err: err}
		}(tool)
	}

	combinedResult := NewExecutionResult()
	combinedResult.ToolName = t.GetName()

	var sumPredictionError float64
	successCount := 0

	for i := 0; i < len(tools); i++ {
		res := <-results
		if res.err != nil {
			return res.result, res.err
		}

		if err := tools[i].Update(res.result); err != nil {
			return res.result, err
		}

		sumPredictionError += res.result.GetPredictionError()
		if res.result.Success {
			successCount++
		}
	}

	combinedResult.SetPredictionError(sumPredictionError / float64(len(tools)))
	combinedResult.SetSuccess(successCount > 0)

	return combinedResult, nil
}

func (t *CompositeTool) executeFirstSuccess(state interface{}, tools []ToolLens) (*ExecutionResult, error) {
	for _, tool := range tools {
		result, err := tool.Execute(state)
		if err != nil {
			continue
		}

		if err := tool.Update(result); err != nil {
			continue
		}

		if result.Success {
			return result, nil
		}
	}

	return NewExecutionResult().SetSuccess(false).SetPredictionError(1.0), errors.New("all tools failed")
}

func (t *CompositeTool) executeAll(state interface{}, tools []ToolLens) (*ExecutionResult, error) {
	results := make([]*ExecutionResult, len(tools))

	for i, tool := range tools {
		result, err := tool.Execute(state)
		if err != nil {
			return result, err
		}

		if err := tool.Update(result); err != nil {
			return result, err
		}

		results[i] = result
	}

	combinedResult := NewExecutionResult()
	combinedResult.ToolName = t.GetName()

	allSuccess := true
	var sumPredictionError float64
	var allObservations []interface{}

	for _, r := range results {
		if !r.Success {
			allSuccess = false
		}
		sumPredictionError += r.GetPredictionError()
		if r.Observation != nil {
			allObservations = append(allObservations, r.Observation)
		}
	}

	combinedResult.SetPredictionError(sumPredictionError / float64(len(results)))
	combinedResult.SetSuccess(allSuccess)
	combinedResult.SetObservation(allObservations)

	return combinedResult, nil
}

func (t *CompositeTool) Update(result *ExecutionResult) error {
	return nil
}

func (t *CompositeTool) AddTool(tool ToolLens) {
	t.mu.Lock()
	t.Tools = append(t.Tools, tool)
	t.mu.Unlock()
}

func (t *CompositeTool) RemoveTool(name string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	for i, tool := range t.Tools {
		if tool.GetName() == name {
			t.Tools = append(t.Tools[:i], t.Tools[i+1:]...)
			return
		}
	}
}

func (t *CompositeTool) GetTools() []ToolLens {
	t.mu.RLock()
	defer t.mu.RUnlock()
	tools := make([]ToolLens, len(t.Tools))
	copy(tools, t.Tools)
	return tools
}

func (t *CompositeTool) SetStrategy(strategy CompositeStrategy) {
	t.mu.Lock()
	t.Strategy = strategy
	t.mu.Unlock()
}

func NewCompositeTool(name, description string, strategy CompositeStrategy) *CompositeTool {
	return &CompositeTool{
		BaseTool: BaseTool{
			Name:        name,
			Description: description,
			Version:     "1.0.0",
			Parameters:  make(map[string]interface{}),
		},
		Tools:    make([]ToolLens, 0),
		Strategy: strategy,
	}
}

type ToolRegistry struct {
	tools      map[string]ToolLens
	aliases    map[string]string
	categories map[string][]string
	mu         sync.RWMutex
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools:      make(map[string]ToolLens),
		aliases:    make(map[string]string),
		categories: make(map[string][]string),
	}
}

func (r *ToolRegistry) Register(tool ToolLens, alternatives ...ToolLens) error {
	if tool == nil {
		return errors.New("cannot register nil tool")
	}

	name := tool.GetName()
	if name == "" {
		return errors.New("tool name cannot be empty")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tool '%s' already registered", name)
	}

	r.tools[name] = tool

	for _, alt := range alternatives {
		if alt != nil {
			r.tools[alt.GetName()] = alt
		}
	}

	return nil
}

func (r *ToolRegistry) RegisterAlias(name, alias string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; !exists {
		return fmt.Errorf("tool '%s' not found", name)
	}

	r.aliases[alias] = name
	return nil
}

func (r *ToolRegistry) RegisterCategory(category string, tools []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, toolName := range tools {
		if _, exists := r.tools[toolName]; !exists {
			return fmt.Errorf("tool '%s' not found", toolName)
		}
	}

	r.categories[category] = append(r.categories[category], tools...)
	return nil
}

func (r *ToolRegistry) Get(name string) (ToolLens, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if tool, exists := r.tools[name]; exists {
		return tool, nil
	}

	if canonicalName, exists := r.aliases[name]; exists {
		if tool, exists := r.tools[canonicalName]; exists {
			return tool, nil
		}
	}

	return nil, fmt.Errorf("tool '%s' not found", name)
}

func (r *ToolRegistry) GetByCategory(category string) ([]ToolLens, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	toolNames, exists := r.categories[category]
	if !exists {
		return nil, fmt.Errorf("category '%s' not found", category)
	}

	tools := make([]ToolLens, 0, len(toolNames))
	for _, name := range toolNames {
		if tool, exists := r.tools[name]; exists {
			tools = append(tools, tool)
		}
	}

	return tools, nil
}

func (r *ToolRegistry) List() []ToolLens {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tools := make([]ToolLens, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}
	return tools
}

func (r *ToolRegistry) ListNames() []string {
	r.mu.RLock()
	defer mu.RUnlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

func (r *ToolRegistry) Remove(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; !exists {
		return fmt.Errorf("tool '%s' not found", name)
	}

	delete(r.tools, name)

	for alias, canonical := range r.aliases {
		if canonical == name {
			delete(r.aliases, alias)
		}
	}

	return nil
}

func (r *ToolRegistry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, exists := r.tools[name]
	return exists
}

func (r *ToolRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}

func (r *ToolRegistry) FindByCapability(capability string) []ToolLens {
	r.mu.RLock()
	defer r.mu.RUnlock()

	matching := make([]ToolLens, 0)
	for _, tool := range r.tools {
		if hasCapability(tool, capability) {
			matching = append(matching, tool)
		}
	}
	return matching
}

func hasCapability(tool ToolLens, capability string) bool {
	params := tool.GetParameters()
	if caps, ok := params["capabilities"].([]string); ok {
		for _, cap := range caps {
			if cap == capability {
				return true
			}
		}
	}
	return false
}

type ToolLensBuilder struct {
	BaseTool
	executeFn func(state interface{}) (*ExecutionResult, error)
	updateFn  func(result *ExecutionResult) error
}

func NewToolLensBuilder() *ToolLensBuilder {
	return &ToolLensBuilder{
		BaseTool: BaseTool{
			Parameters: make(map[string]interface{}),
		},
	}
}

func (b *ToolLensBuilder) Name(name string) *ToolLensBuilder {
	b.BaseTool.SetName(name)
	return b
}

func (b *ToolLensBuilder) Description(desc string) *ToolLensBuilder {
	b.BaseTool.SetDescription(desc)
	return b
}

func (b *ToolLensBuilder) Version(version string) *ToolLensBuilder {
	b.BaseTool.SetVersion(version)
	return b
}

func (b *ToolLensBuilder) Parameter(key string, value interface{}) *ToolLensBuilder {
	b.BaseTool.Parameters[key] = value
	return b
}

func (b *ToolLensBuilder) Execute(fn func(state interface{}) (*ExecutionResult, error)) *ToolLensBuilder {
	b.executeFn = fn
	return b
}

func (b *ToolLensBuilder) Update(fn func(result *ExecutionResult) error) *ToolLensBuilder {
	b.updateFn = fn
	return b
}

func (b *ToolLensBuilder) Build() *SimpleTool {
	tool := &SimpleTool{
		BaseTool: b.BaseTool,
		ExecuteFunc: b.executeFn,
		UpdateFunc:  b.updateFn,
	}
	return tool
}

type ToolMetadata struct {
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	Version      string            `json:"version"`
	Parameters   map[string]string `json:"parameters"`
	Capabilities []string          `json:"capabilities"`
}

func (tool *SimpleTool) GetMetadata() ToolMetadata {
	return ToolMetadata{
		Name:         tool.GetName(),
		Description:  tool.GetDescription(),
		Version:      tool.GetVersion(),
		Parameters:   tool.GetParameters(),
		Capabilities: []string{},
	}
}

func (tool *SimpleTool) ToJSON() (string, error) {
	metadata := tool.GetMetadata()
	return fmt.Sprintf("%+v", metadata), nil
}
