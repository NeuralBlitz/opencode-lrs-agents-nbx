package multiagent

import (
	"testing"

	"github.com/neuralblitz/go-lrs/internal/math"
	"github.com/neuralblitz/go-lrs/pkg/api"
)

func TestMultiAgent(t *testing.T) {
	t.Run("AgentCreation", func(t *testing.T) {
		agent, err := api.NewAgent("TestAgent", "A test agent")
		if err != nil {
			t.Fatalf("failed to create agent: %v", err)
		}
		if agent.Name != "TestAgent" {
			t.Errorf("expected name TestAgent, got %s", agent.Name)
		}
	})

	t.Run("AgentManager", func(t *testing.T) {
		manager := api.NewAgentManager(nil)
		agent, err := manager.CreateAgent("TestAgent", "Description")
		if err != nil {
			t.Fatalf("failed to create agent: %v", err)
		}
		if manager.GetAgentCount() != 1 {
			t.Errorf("expected 1 agent, got %d", manager.GetAgentCount())
		}
		retrieved, err := manager.GetAgent(agent.ID)
		if err != nil {
			t.Fatalf("failed to get agent: %v", err)
		}
		if retrieved.ID != agent.ID {
			t.Errorf("agent IDs don't match")
		}
	})

	t.Run("AgentUpdate", func(t *testing.T) {
		manager := api.NewAgentManager(nil)
		agent, _ := manager.CreateAgent("TestAgent", "Description")

		err := manager.UpdateAgentPrecision(agent.ID, math.LevelExecution, 0.7)
		if err != nil {
			t.Errorf("failed to update precision: %v", err)
		}
	})

	t.Run("AgentMetrics", func(t *testing.T) {
		manager := api.NewAgentManager(nil)
		agent, _ := manager.CreateAgent("TestAgent", "Description")

		metrics, err := manager.GetAgentMetrics(agent.ID)
		if err != nil {
			t.Fatalf("failed to get metrics: %v", err)
		}
		if metrics == nil {
			t.Error("metrics should not be nil")
		}
	})

	t.Run("EventBus", func(t *testing.T) {
		bus := api.NewEventBus()
		if bus == nil {
			t.Error("event bus should not be nil")
		}

		handlerCalled := false
		bus.Subscribe(func(event api.AgentEvent) {
			handlerCalled = true
		})

		bus.Publish(api.AgentEvent{
			Type:      api.AgentCreated,
			AgentID:   "test-agent",
			Timestamp: 1234567890,
		})

		if !handlerCalled {
			t.Error("handler should have been called")
		}
	})
}

func BenchmarkAgentCreation(b *testing.B) {
	manager := api.NewAgentManager(nil)
	for i := 0; i < b.N; i++ {
		manager.CreateAgent("TestAgent", "Description")
	}
}
