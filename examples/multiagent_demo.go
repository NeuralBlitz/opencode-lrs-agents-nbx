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
	log.Println("Multi-Agent Demo")
	log.Println("=================")

	manager := api.NewAgentManager(nil)

	agents := []struct {
		name        string
		description string
	}{
		{"Coordinator", "Coordinates other agents and manages global state"},
		{"Researcher", "Specializes in information gathering and analysis"},
		{"Planner", "Creates and evaluates action policies"},
		{"Executor", "Implements policies through tool execution"},
	}

	for _, a := range agents {
		agent, err := manager.CreateAgent(a.name, a.description)
		if err != nil {
			log.Fatalf("Failed to create agent %s: %v", a.name, err)
		}
		log.Printf("Created agent: %s (%s)", agent.Name, agent.ID)
	}

	log.Println("\n--- Agent Interaction Demo ---")

	coordinator, _ := manager.GetAgent("coordinator")
	researcher, _ := manager.GetAgent("researcher")
	planner, _ := manager.GetAgent("planner")
	executor, _ := manager.GetAgent("executor")

	taskContext := map[string]interface{}{
		"task_type":    "information_gathering",
		"priority":     "high",
		"deadline":     time.Now().Add(1 * time.Hour).UnixMilli(),
		"requirements": []string{"accuracy", "speed"},
	}

	log.Println("1. Coordinator receives task")
	coordinator.State = coordinator.State.AddMessage("system", "New task: Research quantum computing applications in medicine")
	coordinator.State = coordinator.State.WithContext("current_task", "quantum_medical_research")

	log.Println("2. Coordinator distributes to Researcher")
	researcher.State = researcher.State.AddMessage("coordinator", "Please research quantum computing in medicine")
	researcher.State = researcher.State.WithContext("task", "quantum_medical_research")

	log.Println("3. Researcher gathers information")
	researcher.Precision.Update(math.LevelExecution, 0.2)
	researcher.State = researcher.State.AddMessage("system", "Found 15 relevant papers on quantum computing in drug discovery")

	log.Println("4. Planner evaluates options")
	planner.Precision.Update(math.LevelPlanning, 0.3)
	policy, _ := planner.PolicyCalc.SelectPolicyWithInfo(taskContext)
	log.Printf("Planner selected policy: %s (free energy: %.4f)", policy.SelectedPolicy.Name, policy.SelectedPolicy.FreeEnergy)

	log.Println("5. Executor implements policy")
	executor.Precision.Update(math.LevelExecution, 0.15)
	executor.State = executor.State.WithContext("policies_executed", 3)

	log.Println("\n--- Hierarchical Precision Demo ---")

	hp := math.NewHierarchicalPrecision()

	updates := []struct {
		level     math.HierarchyLevel
		predError float64
	}{
		{math.LevelExecution, 0.1},
		{math.LevelExecution, 0.2},
		{math.LevelPlanning, 0.3},
		{math.LevelAbstract, 0.4},
		{math.LevelExecution, 0.15},
	}

	for _, u := range updates {
		if err := hp.Update(u.level, u.predError); err != nil {
			log.Printf("Error updating precision: %v", err)
			continue
		}
		all := hp.GetAllPrecision()
		log.Printf("After update (%s, PE=%.2f): γ=%.3f (abstract=%.3f, planning=%.3f, execution=%.3f)",
			u.level.String(), u.predError, hp.GetWeightedPrecision(),
			all[math.LevelAbstract], all[math.LevelPlanning], all[math.LevelExecution])
	}

	log.Printf("\nDominant level: %s", hp.GetDominantLevel().String())
	log.Printf("Coherence: %.4f", hp.CalculateCoherence())

	log.Println("\n--- Tool Registry Demo ---")

	registry := core.NewToolRegistry()

	tools := []struct {
		name        string
		description string
	}{
		{"search", "Search for information"},
		{"retrieve", "Retrieve detailed information"},
		{"analyze", "Analyze data and extract insights"},
		{"summarize", "Generate concise summaries"},
		{"format", "Format output for presentation"},
		{"validate", "Validate results for accuracy"},
	}

	for _, t := range tools {
		tool := core.NewSimpleTool(t.name, t.description)
		registry.Register(tool)
		log.Printf("Registered tool: %s", t.name)
	}

	searchTool, _ := registry.Get("search")
	log.Printf("Retrieved tool: %s", searchTool.GetName())

	category := []string{"search", "retrieve"}
	registry.RegisterCategory("data_gathering", category)
	gatheringTools, _ := registry.GetByCategory("data_gathering")
	log.Printf("Category 'data_gathering' has %d tools", len(gatheringTools))

	log.Println("\n--- State Management Demo ---")

	stateManager := state.NewStateManager()

	testState, _ := stateManager.CreateState("test-state")
	testState = testState.AddUserMessage("I need help with machine learning")
	testState = testState.AddAssistantMessage("Of course! What aspect of machine learning interests you?")
	testState = testState.AddUserMessage("How do neural networks work?")

	checkpointID := "checkpoint-1"
	if err := stateManager.CreateCheckpoint("test-state", checkpointID); err != nil {
		log.Printf("Failed to create checkpoint: %v", err)
	} else {
		log.Printf("Created checkpoint: %s", checkpointID)
	}

	checkpoints, _ := stateManager.ListCheckpoints("test-state")
	log.Printf("Available checkpoints: %v", checkpoints)

	restoredState, _ := stateManager.RestoreCheckpoint("test-state", checkpointID)
	log.Printf("Restored state has %d messages", len(restoredState.GetMessages()))

	log.Println("\n--- Policy Ensemble Demo ---")

	tracker, _ := math.NewPrecisionTracker(2.0, 2.0)
	calc := math.NewFreeEnergyCalculator(tracker)

	ensemble := math.NewPolicyEnsemble(calc)

	policies := []math.Policy{
		{ID: "policy-1", Name: "Aggressive", FreeEnergy: 0.2, Confidence: 0.9},
		{ID: "policy-2", Name: "Balanced", FreeEnergy: 0.5, Confidence: 0.7},
		{ID: "policy-3", Name: "Conservative", FreeEnergy: 0.8, Confidence: 0.5},
	}

	for _, p := range policies {
		ensemble.AddPolicy(p, p.Confidence)
		log.Printf("Added policy: %s (weight: %.2f)", p.Name, p.Confidence)
	}

	ensemblePolicy := ensemble.GetEnsemblePolicy()
	log.Printf("Ensemble policy: free_energy=%.4f, confidence=%.4f",
		ensemblePolicy.FreeEnergy, ensemblePolicy.Confidence)

	disagreement := ensemble.CalculateDisagreement()
	log.Printf("Ensemble disagreement: %.4f", disagreement)

	log.Println("\n--- Complete Summary ---")

	agentList := manager.ListAgents()
	log.Printf("Total agents: %d", len(agentList))
	for _, a := range agentList {
		metrics, _ := manager.GetAgentMetrics(a.ID)
		log.Printf("  - %s: precision=%.3f, coherence=%.3f",
			a.Name, metrics["average_precision"], metrics["coherence"])
	}

	log.Println("\nDemo completed successfully!")
}

type DemoSummary struct {
	Timestamp     time.Time              `json:"timestamp"`
	AgentCount    int                    `json:"agent_count"`
	TotalMessages int                    `json:"total_messages"`
	Precision     map[string]float64     `json:"precision"`
	Metrics       map[string]interface{} `json:"metrics"`
}

func (s *DemoSummary) ToJSON() string {
	data, _ := json.MarshalIndent(s, "", "  ")
	return string(data)
}
