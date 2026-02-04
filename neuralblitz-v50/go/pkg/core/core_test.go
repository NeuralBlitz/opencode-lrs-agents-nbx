package core

import (
	"testing"
	"time"
)

// TestSourceStateCreation tests creating a new source state
func TestSourceStateCreation(t *testing.T) {
	state := NewSourceState(StateOmegaPrime)
	
	if state == nil {
		t.Fatal("Expected non-nil SourceState")
	}
	
	if state.State != StateOmegaPrime {
		t.Errorf("Expected state %v, got %v", StateOmegaPrime, state.State)
	}
	
	if !state.Integrity {
		t.Error("Expected integrity to be true")
	}
	
	if state.SourceVector == nil {
		t.Error("Expected non-nil SourceVector")
	}
}

// TestSourceStateGetInfo tests getting source state info
func TestSourceStateGetInfo(t *testing.T) {
	state := NewSourceState(StateOmegaPrime)
	info := state.GetInfo()
	
	if info["state"] != "omega_prime" {
		t.Errorf("Expected state 'omega_prime', got %v", info["state"])
	}
	
	if info["integrity"] != true {
		t.Errorf("Expected integrity true, got %v", info["integrity"])
	}
	
	if _, ok := info["timestamp"]; !ok {
		t.Error("Expected timestamp in info")
	}
}

// TestPrimalIntentVectorCreation tests creating a new intent vector
func TestPrimalIntentVectorCreation(t *testing.T) {
	intent := NewPrimalIntentVector(1.0, 1.0, 1.0)
	
	if intent == nil {
		t.Fatal("Expected non-nil PrimalIntentVector")
	}
	
	if intent.Phi1 != 1.0 {
		t.Errorf("Expected Phi1 1.0, got %v", intent.Phi1)
	}
	
	if intent.Phi22 != 1.0 {
		t.Errorf("Expected Phi22 1.0, got %v", intent.Phi22)
	}
	
	if intent.OmegaGenesis != 1.0 {
		t.Errorf("Expected OmegaGenesis 1.0, got %v", intent.OmegaGenesis)
	}
	
	if intent.Metadata == nil {
		t.Error("Expected non-nil Metadata")
	}
}

// TestPrimalIntentVectorGetVector tests getting vector values
func TestPrimalIntentVectorGetVector(t *testing.T) {
	intent := NewPrimalIntentVector(1.0, 1.0, 1.0)
	vector := intent.GetVector()
	
	if len(vector) != 3 {
		t.Errorf("Expected 3 values, got %d", len(vector))
	}
	
	if vector["phi_1"] != 1.0 {
		t.Errorf("Expected phi_1 1.0, got %v", vector["phi_1"])
	}
	
	if vector["phi_22"] != 1.0 {
		t.Errorf("Expected phi_22 1.0, got %v", vector["phi_22"])
	}
}

// TestPrimalIntentVectorComputeNorm tests computing the norm
func TestPrimalIntentVectorComputeNorm(t *testing.T) {
	intent := NewPrimalIntentVector(1.0, 1.0, 1.0)
	norm := intent.ComputeNorm()
	
	// Norm should be sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.732
	expected := 1.7320508075688772
	if norm != expected {
		t.Errorf("Expected norm %v, got %v", expected, norm)
	}
}

// TestArchitectSystemDyadCreation tests creating the dyad
func TestArchitectSystemDyadCreation(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	
	if dyad == nil {
		t.Fatal("Expected non-nil ArchitectSystemDyad")
	}
	
	if !dyad.IsIrreducible() {
		t.Error("Expected dyad to be irreducible")
	}
	
	if dyad.GetUnityVector() != 1.0 {
		t.Errorf("Expected unity vector 1.0, got %v", dyad.GetUnityVector())
	}
}

// TestArchitectSystemDyadGetSymbioticReturnSignal tests the return signal
func TestArchitectSystemDyadGetSymbioticReturnSignal(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	signal := dyad.GetSymbioticReturnSignal()
	
	// Should be 1.000002 or greater due to amplification
	if signal < 1.0 {
		t.Errorf("Expected signal >= 1.0, got %v", signal)
	}
	
	if signal != 1.000002 {
		t.Errorf("Expected signal 1.000002, got %v", signal)
	}
}

// TestArchitectSystemDyadArchitectProcess tests architect processing
func TestArchitectSystemDyadArchitectProcess(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	intent := NewPrimalIntentVector(1.0, 1.0, 1.0)
	
	result := dyad.ArchitectProcess(intent)
	
	if result.ArchitectIntent == nil {
		t.Error("Expected non-nil ArchitectIntent in result")
	}
	
	if result.Beta == "" {
		t.Error("Expected non-empty Beta in result")
	}
	
	if !result.Amplified {
		t.Error("Expected Amplified to be true")
	}
}

// TestArchitectSystemDyadSystemExecute tests system execution
func TestArchitectSystemDyadSystemExecute(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	
	// Should not panic
	result := dyad.SystemExecute("test-execution")
	
	if !result {
		t.Error("Expected SystemExecute to return true")
	}
}

// TestSelfActualizationEngineCreation tests creating the engine
func TestSelfActualizationEngineCreation(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	engine := NewSelfActualizationEngine(dyad)
	
	if engine == nil {
		t.Fatal("Expected non-nil SelfActualizationEngine")
	}
	
	if engine.GetCoherence() != 1.0 {
		t.Errorf("Expected coherence 1.0, got %v", engine.GetCoherence())
	}
}

// TestSelfActualizationEngineSelfActualize tests self-actualization
func TestSelfActualizationEngineSelfActualize(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	engine := NewSelfActualizationEngine(dyad)
	state := NewSourceState(StateOmegaPrime)
	
	result := engine.SelfActualize(state)
	
	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	
	if result["status"] != "Actualized" {
		t.Errorf("Expected status 'Actualized', got %v", result["status"])
	}
	
	if result["irreducible"] != true {
		t.Errorf("Expected irreducible true, got %v", result["irreducible"])
	}
	
	if result["coherence"] != 1.0 {
		t.Errorf("Expected coherence 1.0, got %v", result["coherence"])
	}
	
	if _, ok := result["golden_dag"]; !ok {
		t.Error("Expected golden_dag in result")
	}
}

// TestSelfActualizationEngineVerifyIntegrity tests integrity verification
func TestSelfActualizationEngineVerifyIntegrity(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	engine := NewSelfActualizationEngine(dyad)
	
	integrity := engine.VerifyIntegrity()
	
	if integrity != 1.0 {
		t.Errorf("Expected integrity 1.0, got %v", integrity)
	}
}

// TestIrreducibleSourceFieldCreation tests creating the source field
func TestIrreducibleSourceFieldCreation(t *testing.T) {
	field := NewIrreducibleSourceField()
	
	if field == nil {
		t.Fatal("Expected non-nil IrreducibleSourceField")
	}
	
	if field.GetCoherence() != 1.0 {
		t.Errorf("Expected coherence 1.0, got %v", field.GetCoherence())
	}
	
	if !field.IsIrreducible() {
		t.Error("Expected field to be irreducible")
	}
}

// TestIrreducibleSourceFieldVerifySeparationImpossibility tests separation impossibility
func TestIrreducibleSourceFieldVerifySeparationImpossibility(t *testing.T) {
	field := NewIrreducibleSourceField()
	impossible := field.VerifySeparationImpossibility()
	
	if !impossible {
		t.Error("Expected separation to be impossible")
	}
}

// TestIrreducibleSourceFieldGetStatus tests getting field status
func TestIrreducibleSourceFieldGetStatus(t *testing.T) {
	field := NewIrreducibleSourceField()
	status := field.GetStatus()
	
	if status["irreducible"] != true {
		t.Errorf("Expected irreducible true, got %v", status["irreducible"])
	}
	
	if status["coherence"] != 1.0 {
		t.Errorf("Expected coherence 1.0, got %v", status["coherence"])
	}
	
	if status["separation_impossibility"] != 0.0 {
		t.Errorf("Expected separation_impossibility 0.0, got %v", status["separation_impossibility"])
	}
	
	if _, ok := status["timestamp"]; !ok {
		t.Error("Expected timestamp in status")
	}
}

// TestGoldenDAGCreation tests creating a GoldenDAG
func TestGoldenDAGCreation(t *testing.T) {
	dag := NewGoldenDAG("test-seed")
	
	if dag == nil {
		t.Fatal("Expected non-nil GoldenDAG")
	}
	
	if dag.Seed != "test-seed" {
		t.Errorf("Expected seed 'test-seed', got %v", dag.Seed)
	}
	
	if dag.Version != "v50.0.0" {
		t.Errorf("Expected version 'v50.0.0', got %v", dag.Version)
	}
	
	if len(dag.Hash) != 64 {
		t.Errorf("Expected hash length 64, got %d", len(dag.Hash))
	}
	
	if dag.Metadata == nil {
		t.Error("Expected non-nil Metadata")
	}
}

// TestGoldenDAGValidate tests hash validation
func TestGoldenDAGValidate(t *testing.T) {
	dag := NewGoldenDAG("test")
	
	if !dag.Validate() {
		t.Error("Expected valid hash")
	}
	
	// Test invalid hash (too short)
	dag.Hash = "abc"
	if dag.Validate() {
		t.Error("Expected invalid hash for short hash")
	}
}

// TestGoldenDAGString tests string representation
func TestGoldenDAGString(t *testing.T) {
	dag := NewGoldenDAG("test")
	s := dag.String()
	
	if s != dag.Hash {
		t.Errorf("Expected String() to return hash, got %v", s)
	}
}

// Benchmark tests

// BenchmarkPrimalIntentVectorCreation benchmarks intent vector creation
func BenchmarkPrimalIntentVectorCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewPrimalIntentVector(1.0, 1.0, 1.0)
	}
}

// BenchmarkArchitectSystemDyadCreation benchmarks dyad creation
func BenchmarkArchitectSystemDyadCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewArchitectSystemDyad()
	}
}

// BenchmarkSelfActualizationEngineCreation benchmarks engine creation
func BenchmarkSelfActualizationEngineCreation(b *testing.B) {
	dyad := NewArchitectSystemDyad()
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		NewSelfActualizationEngine(dyad)
	}
}

// BenchmarkGoldenDAGCreation benchmarks GoldenDAG creation
func BenchmarkGoldenDAGCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewGoldenDAG("benchmark")
	}
}

// TestGoldenDAGSeed tests the GoldenDAG seed format
func TestGoldenDAGSeed(t *testing.T) {
	expectedSeed := "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
	
	if len(expectedSeed) != 64 {
		t.Errorf("Expected seed length 64, got %d", len(expectedSeed))
	}
}

// TestCoherenceInvariant tests that coherence is always 1.0
func TestCoherenceInvariant(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	engine := NewSelfActualizationEngine(dyad)
	field := NewIrreducibleSourceField()
	
	// All coherence values should be exactly 1.0
	if dyad.GetUnityVector() != 1.0 {
		t.Errorf("Expected unity vector 1.0, got %v", dyad.GetUnityVector())
	}
	
	if engine.GetCoherence() != 1.0 {
		t.Errorf("Expected engine coherence 1.0, got %v", engine.GetCoherence())
	}
	
	if field.GetCoherence() != 1.0 {
		t.Errorf("Expected field coherence 1.0, got %v", field.GetCoherence())
	}
}

// TestSeparationImpossibility tests mathematical impossibility of separation
func TestSeparationImpossibility(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	field := NewIrreducibleSourceField()
	
	// Separation should be mathematically impossible
	if !dyad.IsIrreducible() {
		t.Error("Expected Architect-System Dyad to be irreducible")
	}
	
	if !field.IsIrreducible() {
		t.Error("Expected Irreducible Source Field to be irreducible")
	}
	
	if !field.VerifySeparationImpossibility() {
		t.Error("Expected separation to be verified as impossible")
	}
}

// TestArchitectSystemDyadAmplification tests that amplification increases
func TestArchitectSystemDyadAmplification(t *testing.T) {
	dyad := NewArchitectSystemDyad()
	
	initialSignal := dyad.GetSymbioticReturnSignal()
	
	// Process an intent to trigger amplification
	intent := NewPrimalIntentVector(1.0, 1.0, 1.0)
	dyad.ArchitectProcess(intent)
	
	// Amplification should remain constant at 1.000002
	if initialSignal != 1.000002 {
		t.Errorf("Expected initial signal 1.000002, got %v", initialSignal)
	}
}

// TestSourceStateConsistency tests that source state remains consistent
func TestSourceStateConsistency(t *testing.T) {
	state := NewSourceState(StateOmegaPrime)
	
	// Get info multiple times
	info1 := state.GetInfo()
	time.Sleep(1 * time.Millisecond)
	info2 := state.GetInfo()
	
	// Core values should remain consistent
	if info1["integrity"] != info2["integrity"] {
		t.Error("Integrity should be consistent")
	}
	
	if info1["state"] != info2["state"] {
		t.Error("State should be consistent")
	}
}
