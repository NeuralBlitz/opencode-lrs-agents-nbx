package math

import (
	"math"
	"testing"
)

func TestPrecisionParameters(t *testing.T) {
	t.Run("NewPrecisionParameters", func(t *testing.T) {
		pp, err := NewPrecisionParameters(2.0, 2.0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pp.Alpha != 2.0 || pp.Beta != 2.0 {
			t.Errorf("expected alpha=2.0, beta=2.0, got alpha=%.2f, beta=%.2f", pp.Alpha, pp.Beta)
		}
	})

	t.Run("NewPrecisionParameters_Invalid", func(t *testing.T) {
		_, err := NewPrecisionParameters(0, 2.0)
		if err == nil {
			t.Error("expected error for invalid parameters")
		}
	})

	t.Run("Value", func(t *testing.T) {
		pp, _ := NewPrecisionParameters(2.0, 2.0)
		value := pp.Value()
		if math.Abs(value-0.5) > 0.01 {
			t.Errorf("expected value ~0.5, got %.4f", value)
		}
	})

	t.Run("Update", func(t *testing.T) {
		pp, _ := NewPrecisionParameters(2.0, 2.0)
		initial := pp.Value()
		if err := pp.Update(0.1); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pp.Value() == initial {
			t.Error("expected precision to change after update")
		}
	})

	t.Run("Update_Invalid", func(t *testing.T) {
		pp, _ := NewPrecisionParameters(2.0, 2.0)
		if err := pp.Update(1.5); err == nil {
			t.Error("expected error for invalid prediction error")
		}
	})
}

func TestPrecisionTracker(t *testing.T) {
	tracker, err := NewPrecisionTracker(2.0, 2.0)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}

	t.Run("CurrentValue", func(t *testing.T) {
		value := tracker.CurrentValue()
		if math.Abs(value-0.5) > 0.01 {
			t.Errorf("expected ~0.5, got %.4f", value)
		}
	})

	t.Run("Update", func(t *testing.T) {
		if err := tracker.Update(0.2); err != nil {
			t.Errorf("update failed: %v", err)
		}
		if len(tracker.GetHistory()) != 1 {
			t.Error("expected history length 1")
		}
	})

	t.Run("AveragePrecision", func(t *testing.T) {
		tracker.Reset()
		tracker.Update(0.3)
		tracker.Update(0.7)
		avg := tracker.AveragePrecision()
		if avg < 0 || avg > 1 {
			t.Errorf("average precision out of range: %.4f", avg)
		}
	})
}

func TestBetaDistribution(t *testing.T) {
	bd, err := NewBetaDistribution(2.0, 2.0)
	if err != nil {
		t.Fatalf("failed to create beta distribution: %v", err)
	}

	t.Run("Mean", func(t *testing.T) {
		mean := bd.Mean()
		if math.Abs(mean-0.5) > 0.01 {
			t.Errorf("expected mean ~0.5, got %.4f", mean)
		}
	})

	t.Run("Sample", func(t *testing.T) {
		sample := bd.Sample()
		if sample < 0 || sample > 1 {
			t.Errorf("sample out of range: %.4f", sample)
		}
	})

	t.Run("PDF", func(t *testing.T) {
		pdf := bd.PDF(0.5)
		if pdf < 0 {
			t.Errorf("PDF should be non-negative, got %.4f", pdf)
		}
	})

	t.Run("Entropy", func(t *testing.T) {
		entropy := bd.Entropy()
		if entropy < 0 {
			t.Errorf("entropy should be non-negative, got %.4f", entropy)
		}
	})
}

func TestFreeEnergyCalculator(t *testing.T) {
	tracker, _ := NewPrecisionTracker(2.0, 2.0)
	calc := NewFreeEnergyCalculator(tracker)

	t.Run("CalculateExpectedFreeEnergy", func(t *testing.T) {
		policy := Policy{
			ID:           "test",
			Name:         "Test Policy",
			Tools:        []string{"search", "retrieve"},
			Confidence:   0.7,
			EpistemicVal: 0.6,
			PragmaticVal: 0.8,
		}
		preferences := map[string]float64{"search": 0.9, "retrieve": 0.7}
		fe := calc.CalculateExpectedFreeEnergy(policy, preferences)
		if fe < 0 || fe > 1 {
			t.Errorf("free energy out of range: %.4f", fe)
		}
	})

	t.Run("AddPreference", func(t *testing.T) {
		if err := calc.AddPreference("test", 0.5, 0.5); err != nil {
			t.Errorf("failed to add preference: %v", err)
		}
	})

	t.Run("EvaluatePolicies", func(t *testing.T) {
		policies := calc.EvaluatePolicies(map[string]interface{}{"task": "test"})
		if len(policies) == 0 {
			t.Error("expected at least one policy")
		}
	})

	t.Run("SelectPolicy", func(t *testing.T) {
		policy, err := calc.SelectPolicy(map[string]interface{}{"task": "test"})
		if err != nil {
			t.Errorf("failed to select policy: %v", err)
		}
		if policy == nil {
			t.Error("expected non-nil policy")
		}
	})
}

func TestPolicyGenerator(t *testing.T) {
	gen := NewPolicyGenerator()

	t.Run("GeneratePolicies", func(t *testing.T) {
		policies := gen.GeneratePolicies(map[string]interface{}{}, 0.5)
		if len(policies) == 0 {
			t.Error("expected at least one policy")
		}
	})

	t.Run("RegisterGenerator", func(t *testing.T) {
		customGen := func(ctx map[string]interface{}, prec float64) []Policy {
			return []Policy{
				{
					ID:         "custom",
					Name:       "Custom Policy",
					FreeEnergy: 0.3,
				},
			}
		}
		if err := gen.RegisterGenerator("custom", customGen); err != nil {
			t.Errorf("failed to register generator: %v", err)
		}
		policies := gen.GeneratePolicies(map[string]interface{}{}, 0.5)
		found := false
		for _, p := range policies {
			if p.ID == "custom" {
				found = true
				break
			}
		}
		if !found {
			t.Error("custom policy not found")
		}
	})
}

func TestHierarchicalPrecision(t *testing.T) {
	hp := NewHierarchicalPrecision()

	t.Run("Update", func(t *testing.T) {
		if err := hp.Update(LevelExecution, 0.3); err != nil {
			t.Errorf("failed to update: %v", err)
		}
	})

	t.Run("GetPrecision", func(t *testing.T) {
		prec, err := hp.GetPrecision(LevelExecution)
		if err != nil {
			t.Errorf("failed to get precision: %v", err)
		}
		if prec < 0 || prec > 1 {
			t.Errorf("precision out of range: %.4f", prec)
		}
	})

	t.Run("GetAllPrecision", func(t *testing.T) {
		all := hp.GetAllPrecision()
		if len(all) != 3 {
			t.Errorf("expected 3 levels, got %d", len(all))
		}
	})

	t.Run("GetDominantLevel", func(t *testing.T) {
		dominant := hp.GetDominantLevel()
		if !dominant.IsValid() {
			t.Error("invalid dominant level")
		}
	})

	t.Run("GetAveragePrecision", func(t *testing.T) {
		avg := hp.GetAveragePrecision()
		if avg < 0 || avg > 1 {
			t.Errorf("average precision out of range: %.4f", avg)
		}
	})

	t.Run("CalculateCoherence", func(t *testing.T) {
		coherence := hp.CalculateCoherence()
		if coherence < 0 || coherence > 1 {
			t.Errorf("coherence out of range: %.4f", coherence)
		}
	})

	t.Run("GetPrecisionTrend", func(t *testing.T) {
		trend := hp.GetPrecisionTrend(LevelExecution)
		if math.IsNaN(trend) {
			t.Error("trend should not be NaN")
		}
	})

	t.Run("Clone", func(t *testing.T) {
		clone := hp.Clone()
		if clone == nil {
			t.Error("clone should not be nil")
		}
	})
}

func TestHierarchyLevel(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		if LevelAbstract.String() != "abstract" {
			t.Errorf("unexpected string: %s", LevelAbstract.String())
		}
		if LevelPlanning.String() != "planning" {
			t.Errorf("unexpected string: %s", LevelPlanning.String())
		}
		if LevelExecution.String() != "execution" {
			t.Errorf("unexpected string: %s", LevelExecution.String())
		}
	})

	t.Run("Parent", func(t *testing.T) {
		if LevelPlanning.Parent() != LevelAbstract {
			t.Error("planning parent should be abstract")
		}
		if LevelExecution.Parent() != LevelPlanning {
			t.Error("execution parent should be planning")
		}
	})

	t.Run("Depth", func(t *testing.T) {
		if LevelAbstract.Depth() != 0 {
			t.Error("abstract depth should be 0")
		}
		if LevelPlanning.Depth() != 1 {
			t.Error("planning depth should be 1")
		}
		if LevelExecution.Depth() != 2 {
			t.Error("execution depth should be 2")
		}
	})
}
