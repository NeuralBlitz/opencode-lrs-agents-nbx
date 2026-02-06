package main

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/neuralblitz/go-lrs/internal/math"
	"github.com/neuralblitz/go-lrs/internal/state"
	"github.com/neuralblitz/go-lrs/pkg/api"
	"github.com/neuralblitz/go-lrs/pkg/core"
)

func main() {
	log.Println("Running Go-LRS Basic Tests")
	log.Println("============================")

	if err := testPrecisionTracking(); err != nil {
		log.Fatalf("Precision tracking test failed: %v", err)
	}

	if err := testFreeEnergyCalculation(); err != nil {
		log.Fatalf("Free energy calculation test failed: %v", err)
	}

	if err := testToolLensPattern(); err != nil {
		log.Fatalf("ToolLens pattern test failed: %v", err)
	}

	if err := testStateManagement(); err != nil {
		log.Fatalf("State management test failed: %v", err)
	}

	log.Println("============================")
	log.Println("All tests passed!")
}

func testPrecisionTracking() error {
	log.Println("\n--- Testing Precision Tracking ---")

	tracker, err := math.NewPrecisionTracker(2.0, 2.0)
	if err != nil {
		return fmt.Errorf("failed to create precision tracker: %w", err)
	}

	initialValue := tracker.CurrentValue()
	log.Printf("Initial precision value: %.4f", initialValue)

	if initialValue < 0.4 || initialValue > 0.6 {
		return fmt.Errorf("unexpected initial precision: %.4f", initialValue)
	}

	testErrors := []float64{0.1, 0.3, 0.5, 0.7, 0.9}
	for i, pe := range testErrors {
		if err := tracker.Update(pe); err != nil {
			return fmt.Errorf("failed to update precision: %w", err)
		}
		current := tracker.CurrentValue()
		log.Printf("Update %d (PE=%.2f): precision=%.4f", i+1, pe, current)
	}

	history := tracker.GetHistory()
	if len(history) == 0 {
		return fmt.Errorf("expected history to be non-empty")
	}
	log.Printf("Precision history length: %d", len(history))

	avgPrecision := tracker.AveragePrecision()
	log.Printf("Average precision: %.4f", avgPrecision)

	tracker.Reset()
	resetValue := tracker.CurrentValue()
	if resetValue < 0.4 || resetValue > 0.6 {
		return fmt.Errorf("unexpected reset precision: %.4f", resetValue)
	}

	log.Println("✓ Precision tracking test passed")
	return nil
}

func testFreeEnergyCalculation() error {
	log.Println("\n--- Testing Free Energy Calculation ---")

	precisionTracker, err := math.NewPrecisionTracker(2.0, 2.0)
	if err != nil {
		return fmt.Errorf("failed to create precision tracker: %w", err)
	}

	calculator := math.NewFreeEnergyCalculator(precisionTracker)

	policy := math.Policy{
		ID:           "test-policy",
		Name:         "Test Policy",
		Tools:        []string{"search", "retrieve", "format"},
		Confidence:   0.75,
		EpistemicVal: 0.6,
		PragmaticVal: 0.8,
	}

	preferences := map[string]float64{
		"search":   0.9,
		"retrieve": 0.7,
		"format":   0.85,
	}

	freeEnergy := calculator.CalculateExpectedFreeEnergy(policy, preferences)
	log.Printf("Free energy for test policy: %.4f", freeEnergy)

	if freeEnergy < 0 || freeEnergy > 1 {
		return fmt.Errorf("free energy out of range: %.4f", freeEnergy)
	}

	calculator.AddPreference("quality", 0.8, 0.9)
	calculator.AddPreference("speed", 0.6, 0.7)

	context := map[string]interface{}{
		"task_type": "retrieval",
		"priority":  "high",
	}

	policies := calculator.EvaluatePolicies(context)
	log.Printf("Evaluated %d policies", len(policies))

	if len(policies) == 0 {
		return fmt.Errorf("expected at least one policy")
	}

	selectedPolicy, err := calculator.SelectPolicy(context)
	if err != nil {
		return fmt.Errorf("failed to select policy: %w", err)
	}
	log.Printf("Selected policy: %s (free energy: %.4f)", selectedPolicy.Name, selectedPolicy.FreeEnergy)

	selection, err := calculator.SelectPolicyWithInfo(context)
	if err != nil {
		return fmt.Errorf("failed to select policy with info: %w", err)
	}
	log.Printf("Selection entropy: %.4f", selection.PolicyEntropy)

	calculator.SetTemperature(2.0)
	newTemp := calculator.GetTemperature()
	if newTemp != 2.0 {
		return fmt.Errorf("unexpected temperature: %.4f", newTemp)
	}

	log.Println("✓ Free energy calculation test passed")
	return nil
}

func testToolLensPattern() error {
	log.Println("\n--- Testing ToolLens Pattern ---")

	searchTool := core.NewSimpleTool("search", "Search for information")
	searchTool.ExecuteFunc = func(state interface{}) (*core.ExecutionResult, error) {
		result := core.NewExecutionResult()
		result.ToolName = "search"
		result.SetSuccess(true)
		result.SetPredictionError(0.15)
		result.SetObservation(map[string]interface{}{
			"results": []string{"result1", "result2"},
		})
		return result, nil
	}

	retrieveTool := core.NewSimpleTool("retrieve", "Retrieve detailed information")
	retrieveTool.ExecuteFunc = func(state interface{}) (*core.ExecutionResult, error) {
		result := core.NewExecutionResult()
		result.ToolName = "retrieve"
		result.SetSuccess(true)
		result.SetPredictionError(0.2)
		result.SetObservation(map[string]interface{}{
			"data": "retrieved content",
		})
		return result, nil
	}

	formatTool := core.NewSimpleTool("format", "Format the output")
	formatTool.ExecuteFunc = func(state interface{}) (*core.ExecutionResult, error) {
		result := core.NewExecutionResult()
		result.ToolName = "format"
		result.SetSuccess(true)
		result.SetPredictionError(0.1)
		result.SetObservation("formatted output")
		return result, nil
	}

	result, err := searchTool.Execute(nil)
	if err != nil {
		return fmt.Errorf("search tool execution failed: %w", err)
	}
	log.Printf("Search tool result: success=%v, prediction_error=%.4f",
		result.Success, result.GetPredictionError())

	compositeTool := core.NewCompositeTool("pipeline", "Search, retrieve, and format", core.StrategySequential)
	compositeTool.AddTool(searchTool)
	compositeTool.AddTool(retrieveTool)
	compositeTool.AddTool(formatTool)

	pipelineResult, err := compositeTool.Execute(nil)
	if err != nil {
		return fmt.Errorf("pipeline execution failed: %w", err)
	}
	log.Printf("Pipeline result: success=%v, prediction_error=%.4f",
		pipelineResult.Success, pipelineResult.GetPredictionError())

	registry := core.NewToolRegistry()
	if err := registry.Register(searchTool); err != nil {
		return fmt.Errorf("failed to register search tool: %w", err)
	}
	if err := registry.Register(retrieveTool); err != nil {
		return fmt.Errorf("failed to register retrieve tool: %w", err)
	}
	if err := registry.Register(formatTool); err != nil {
		return fmt.Errorf("failed to register format tool: %w", err)
	}

	registry.RegisterAlias("search", "find")
	registry.RegisterAlias("retrieve", "get")

	tool, err := registry.Get("find")
	if err != nil {
		return fmt.Errorf("failed to get tool by alias: %w", err)
	}
	log.Printf("Retrieved tool by alias: %s", tool.GetName())

	toolNames := registry.ListNames()
	log.Printf("Registered tools: %v", toolNames)

	if registry.Count() != 3 {
		return fmt.Errorf("unexpected tool count: %d", registry.Count())
	}

	log.Println("✓ ToolLens pattern test passed")
	return nil
}

func testStateManagement() error {
	log.Println("\n--- Testing State Management ---")

	stateManager := state.NewStateManager()

	state1, err := stateManager.CreateState("test-agent-1")
	if err != nil {
		return fmt.Errorf("failed to create state: %w", err)
	}

	state1 = state1.AddUserMessage("Hello, I need help with math")
	state1 = state1.AddAssistantMessage("Of course! What math problem can I help you with?")
	state1 = state1.AddUserMessage("What's 2+2?")
	state1 = state1.WithContext("topic", "math")
	state1 = state1.WithContext("difficulty", "easy")

	state1JSON, _ := state1.ToJSON()
	log.Printf("State JSON length: %d bytes", len(state1JSON))

	messages := state1.GetMessages()
	if len(messages) != 3 {
		return fmt.Errorf("expected 3 messages, got %d", len(messages))
	}
	log.Printf("Message count: %d", len(messages))

	context := state1.GetContext()
	if context["topic"] != "math" {
		return fmt.Errorf("unexpected context topic: %v", context["topic"])
	}
	log.Printf("Context topic: %v", context["topic"])

	clonedState := state1.Clone()
	clonedState = clonedState.AddMessage("system", "This is a clone")

	if len(state1.GetMessages()) != 3 {
		return fmt.Errorf("clone should not affect original state")
	}
	if len(clonedState.GetMessages()) != 4 {
		return fmt.Errorf("clone should have 4 messages")
	}
	log.Println("State immutability verified")

	execution := state.NewPolicyExecution("policy-1", "Test Policy")
	execution.AddToolExecution(state.NewToolExecution("tool1").SetPredictionError(0.15).Complete())
	execution.AddToolExecution(state.NewToolExecution("tool2").SetPredictionError(0.25).Complete())
	execution.Complete()

	state2 := state1.WithPolicy(execution)
	policyHistory := state2.GetPolicyHistory()
	if len(policyHistory) != 1 {
		return fmt.Errorf("expected 1 policy in history, got %d", len(policyHistory))
	}
	log.Printf("Policy history count: %d", len(policyHistory))
	log.Printf("Policy prediction error: %.4f", policyHistory[0].PredictionError)

	agent, err := api.NewAgent("TestAgent", "A test agent")
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	agent.State = agent.State.AddUserMessage("Test message")
	agent.State = agent.State.WithPrecision("execution", 0.8)
	agent.State = agent.State.WithContext("test_key", "test_value")

	stateJSON, _ := agent.State.ToJSON()
	var stateMap map[string]interface{}
	json.Unmarshal(stateJSON, &stateMap)
	log.Printf("Agent state: %d messages, %d precision entries, %d context keys",
		len(agent.State.GetMessages()),
		len(agent.State.GetPrecision()),
		len(agent.State.GetContext()))

	precision := agent.Precision.GetAllPrecision()
	log.Printf("Hierarchical precision: abstract=%.4f, planning=%.4f, execution=%.4f",
		precision[math.LevelAbstract],
		precision[math.LevelPlanning],
		precision[math.LevelExecution])

	err = agent.Precision.Update(math.LevelExecution, 0.3)
	if err != nil {
		return fmt.Errorf("failed to update precision: %w", err)
	}

	newPrecision := agent.Precision.GetAllPrecision()
	log.Printf("Updated precision: execution=%.4f", newPrecision[math.LevelExecution])

	dominantLevel := agent.Precision.GetDominantLevel()
	log.Printf("Dominant level: %s", dominantLevel.String())

	coherence := agent.Precision.CalculateCoherence()
	log.Printf("Coherence: %.4f", coherence)

	log.Println("✓ State management test passed")
	return nil
}

func printHeader(title string) {
	fmt.Println("\n" + "=".repeat(50))
	fmt.Println(" " + title)
	fmt.Println("=".repeat(50))
}

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.SetPrefix("[Go-LRS] ")
}

type testResult struct {
	Name    string
	Passed  bool
	Message string
}

func (r *testResult) String() string {
	if r.Passed {
		return fmt.Sprintf("✓ %s", r.Name)
	}
	return fmt.Sprintf("✗ %s: %s", r.Name, r.Message)
}

func runAllTests() []testResult {
	results := []testResult{}

	tests := []struct {
		name string
		fn   func() error
	}{
		{"Precision Tracking", testPrecisionTracking},
		{"Free Energy Calculation", testFreeEnergyCalculation},
		{"ToolLens Pattern", testToolLensPattern},
		{"State Management", testStateManagement},
	}

	for _, test := range tests {
		result := testResult{Name: test.name}
		if err := test.fn(); err != nil {
			result.Passed = false
			result.Message = err.Error()
		} else {
			result.Passed = true
		}
		results = append(results, result)
	}

	return results
}

func benchmarkTest(name string, iterations int, fn func() error) (time.Duration, error) {
	start := time.Now()
	for i := 0; i < iterations; i++ {
		if err := fn(); err != nil {
			return time.Since(start), err
		}
	}
	return time.Since(start), nil
}
