package math

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type Policy struct {
	ID           string
	Name         string
	Tools        []string
	Confidence   float64
	EpistemicVal float64
	PragmaticVal float64
	FreeEnergy   float64
	Priority     int
	Metadata     map[string]interface{}
}

type PolicyGenerator struct {
	mu           sync.RWMutex
	generators   []PolicyGeneratorFunc
	maxPolicies  int
	minConfidence float64
}

type PolicyGeneratorFunc func(context map[string]interface{}, precision float64) []Policy

type Preference struct {
	Key       string
	Weight    float64
	TargetVal float64
}

type FreeEnergyCalculator struct {
	precisionCalculator *PrecisionTracker
	policyGenerator     *PolicyGenerator
	preferences         []Preference
	gamma               float64
	beta                float64
	mu                  sync.RWMutex
}

func NewFreeEnergyCalculator(precisionTracker *PrecisionTracker) *FreeEnergyCalculator {
	return &FreeEnergyCalculator{
		precisionCalculator: precisionTracker,
		policyGenerator:     NewPolicyGenerator(),
		preferences:         make([]Preference, 0),
		gamma:               0.5,
		beta:                1.0,
	}
}

func NewPolicyGenerator() *PolicyGenerator {
	return &PolicyGenerator{
		generators:    make([]PolicyGeneratorFunc, 0),
		maxPolicies:   10,
		minConfidence: 0.3,
	}
}

func (g *PolicyGenerator) RegisterGenerator(name string, fn PolicyGeneratorFunc) error {
	if fn == nil {
		return fmt.Errorf("policy generator function cannot be nil")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.generators = append(g.generators, fn)
	return nil
}

func (g *PolicyGenerator) GeneratePolicies(context map[string]interface{}, precision float64) []Policy {
	g.mu.RLock()
	generators := g.generators
	g.mu.RUnlock()

	allPolicies := make([]Policy, 0)
	for _, generator := range generators {
		policies := generator(context, precision)
		allPolicies = append(allPolicies, policies...)
	}

	if len(allPolicies) == 0 {
		allPolicies = g.generateDefaultPolicies(context, precision)
	}

	sort.Slice(allPolicies, func(i, j int) bool {
		return allPolicies[i].FreeEnergy < allPolicies[j].FreeEnergy
	})

	if len(allPolicies) > g.maxPolicies {
		filteredPolicies := make([]Policy, 0)
		for _, p := range allPolicies[:g.maxPolicies] {
			if p.Confidence >= g.minConfidence {
				filteredPolicies = append(filteredPolicies, p)
			}
		}
		if len(filteredPolicies) == 0 && len(allPolicies) > 0 {
			filteredPolicies = append(filteredPolicies, allPolicies[0])
		}
		allPolicies = filteredPolicies
	}

	return allPolicies
}

func (g *PolicyGenerator) generateDefaultPolicies(context map[string]interface{}, precision float64) []Policy {
	return []Policy{
		{
			ID:           "default_exploit",
			Name:         "Exploit Current Knowledge",
			Tools:        []string{"search", "retrieve"},
			Confidence:   0.7,
			EpistemicVal: 0.2,
			PragmaticVal: 0.8,
			FreeEnergy:   0.5,
			Priority:     1,
			Metadata:     map[string]interface{}{"strategy": "exploitation"},
		},
		{
			ID:           "default_explore",
			Name:         "Explore New Information",
			Tools:        []string{"search", "analyze", "discover"},
			Confidence:   0.5,
			EpistemicVal: 0.8,
			PragmaticVal: 0.2,
			FreeEnergy:   0.6,
			Priority:     2,
			Metadata:     map[string]interface{}{"strategy": "exploration"},
		},
	}
}

func (c *FreeEnergyCalculator) AddPreference(key string, weight, targetVal float64) error {
	if weight <= 0 {
		return fmt.Errorf("weight must be positive, got %.4f", weight)
	}
	if targetVal < 0 || targetVal > 1 {
		return fmt.Errorf("targetVal must be in [0, 1], got %.4f", targetVal)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.preferences = append(c.preferences, Preference{
		Key:       key,
		Weight:    weight,
		TargetVal: targetVal,
	})
	return nil
}

func (c *FreeEnergyCalculator) ClearPreferences() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.preferences = make([]Preference, 0)
}

func (c *FreeEnergyCalculator) CalculateExpectedFreeEnergy(policy Policy, preferences map[string]float64) float64 {
	epistemicValue := c.calculateEpistemicValue(policy, preferences)
	pragmaticValue := c.calculatePragmaticValue(policy, preferences)
	precision := c.precisionCalculator.CurrentValue()

	freeEnergy := epistemicValue - pragmaticValue
	freeEnergy = freeEnergy * (1 + precision*0.5)
	freeEnergy = math.Max(0, math.Min(1, freeEnergy))

	return freeEnergy
}

func (c *FreeEnergyCalculator) calculateEpistemicValue(policy Policy, preferences map[string]float64) float64 {
	baseEpistemic := policy.EpistemicVal
	if baseEpistemic == 0 {
		baseEpistemic = 0.5
	}

	noveltyBonus := 0.0
	for tool := range map[string]bool{} {
		if _, exists := preferences[tool]; !exists {
			noveltyBonus += 0.1
		}
	}

	uncertaintyReduction := 0.0
	for _, pref := range c.preferences {
		if currentVal, exists := preferences[pref.Key]; exists {
			uncertaintyReduction += pref.Weight * math.Abs(pref.TargetVal-currentVal)
		}
	}

	epistemicValue := baseEpistemic + noveltyBonus + uncertaintyReduction*0.1
	return math.Min(1.0, math.Max(0.0, epistemicValue))
}

func (c *FreeEnergyCalculator) calculatePragmaticValue(policy Policy, preferences map[string]float64) float64 {
	basePragmatic := policy.PragmaticVal
	if basePragmatic == 0 {
		basePragmatic = 0.5
	}

	rewardExpectation := 0.0
	toolMatchBonus := 0.0

	for i, tool := range policy.Tools {
		if preferredVal, exists := preferences[tool]; exists {
			rewardExpectation += preferredVal * float64(len(policy.Tools)-i) / float64(len(policy.Tools))
			toolMatchBonus += 0.1
		}
	}

	pragmaticValue := basePragmatic * 0.6 + rewardExpectation*0.3 + toolMatchBonus
	return math.Min(1.0, math.Max(0.0, pragmaticValue))
}

func (c *FreeEnergyCalculator) EvaluatePolicies(context map[string]interface{}) []Policy {
	c.mu.RLock()
	preferences := make(map[string]float64)
	for _, pref := range c.preferences {
		preferences[pref.Key] = pref.TargetVal
	}
	gamma := c.gamma
	c.mu.RUnlock()

	precision := c.precisionCalculator.CurrentValue()
	policies := c.policyGenerator.GeneratePolicies(context, precision)

	evaluatedPolicies := make([]Policy, 0, len(policies))
	for _, policy := range policies {
		freeEnergy := c.CalculateExpectedFreeEnergy(policy, preferences)
		evaluatedPolicy := policy
		evaluatedPolicy.FreeEnergy = freeEnergy
		evaluatedPolicies = append(evaluatedPolicies, evaluatedPolicy)
	}

	c.mu.Lock()
	c.gamma = gamma
	c.mu.Unlock()

	return evaluatedPolicies
}

func (c *FreeEnergyCalculator) SelectPolicy(context map[string]interface{}) (*Policy, error) {
	policies := c.EvaluatePolicies(context)

	if len(policies) == 0 {
		return nil, fmt.Errorf("no policies available for selection")
	}

	sort.Slice(policies, func(i, j int) bool {
		return policies[i].FreeEnergy < policies[j].FreeEnergy
	})

	temperature := c.beta
	normalizedProbs := make([]float64, len(policies))
	sumExp := 0.0
	for i, policy := range policies {
		normalizedProbs[i] = math.Exp(-temperature * policy.FreeEnergy)
		sumExp += normalizedProbs[i]
	}

	if sumExp == 0 {
		return &policies[0], nil
	}

	for i := range normalizedProbs {
		normalizedProbs[i] /= sumExp
	}

	cumulative := 0.0
	random := uniformRandom()
	for i, prob := range normalizedProbs {
		cumulative += prob
		if random <= cumulative {
			selectedPolicy := policies[i]
			return &selectedPolicy, nil
		}
	}

	return &policies[len(policies)-1], nil
}

func uniformRandom() float64 {
	return float64(uint64(0x1BD10A33F12D93B9)>>32) / float64(uint64(1)<<32)
}

func (c *FreeEnergyCalculator) UpdatePrecision(predictionError float64) error {
	return c.precisionCalculator.Update(predictionError)
}

func (c *FreeEnergyCalculator) GetCurrentPrecision() float64 {
	return c.precisionCalculator.CurrentValue()
}

func (c *FreeEnergyCalculator) GetPolicyHistory() []Policy {
	c.mu.RLock()
	defer c.mu.RUnlock()
	policies := c.policyGenerator.GeneratePolicies(map[string]interface{}{}, c.precisionCalculator.CurrentValue())
	return policies
}

func (c *FreeEnergyCalculator) SetTemperature(beta float64) {
	if beta <= 0 {
		beta = 1.0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.beta = beta
}

func (c *FreeEnergyCalculator) GetTemperature() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.beta
}

func (c *FreeEnergyCalculator) CalculateDivergence(policy1, policy2 Policy) float64 {
	if len(policy1.Tools) == 0 || len(policy2.Tools) == 0 {
		return 0.5
	}

	toolSet1 := make(map[string]bool)
	for _, t := range policy1.Tools {
		toolSet1[t] = true
	}

	toolSet2 := make(map[string]bool)
	for _, t := range policy2.Tools {
		toolSet2[t] = true
	}

	intersection := 0
	for t := range toolSet1 {
		if toolSet2[t] {
			intersection++
		}
	}

	union := len(toolSet1) + len(toolSet2) - intersection
	if union == 0 {
		return 0.0
	}

	jaccardDistance := 1.0 - float64(intersection)/float64(union)
	freeEnergyDiff := math.Abs(policy1.FreeEnergy - policy2.FreeEnergy)

	divergence := jaccardDistance*0.6 + freeEnergyDiff*0.4
	return math.Min(1.0, divergence)
}

func (c *FreeEnergyCalculator) CalculateExpectedValueOfInformation(policy Policy, preferences map[string]float64) float64 {
	currentValue := c.calculatePragmaticValue(policy, preferences)
	epistemicGain := c.calculateEpistemicValue(policy, preferences)

	expectedImprovement := 0.0
	uncertainty := 1.0 - policy.Confidence

	expectedImprovement = epistemicGain * uncertainty * 0.5

	return math.Min(0.5, expectedImprovement)
}

func (c *FreeEnergyCalculator) CalculatePolicyEntropy(policies []Policy) float64 {
	if len(policies) == 0 {
		return 0
	}

	sort.Slice(policies, func(i, j int) bool {
		return policies[i].FreeEnergy < policies[j].FreeEnergy
	})

	temperature := c.beta
	normalizedProbs := make([]float64, len(policies))
	sumExp := 0.0
	for i, policy := range policies {
		normalizedProbs[i] = math.Exp(-temperature * policy.FreeEnergy)
		sumExp += normalizedProbs[i]
	}

	if sumExp == 0 {
		return 0
	}

	for i := range normalizedProbs {
		normalizedProbs[i] /= sumExp
	}

	entropy := 0.0
	for _, prob := range normalizedProbs {
		if prob > 0 {
			entropy -= prob * math.Log(prob)
		}
	}

	maxEntropy := math.Log(float64(len(policies)))
	if maxEntropy > 0 {
		entropy /= maxEntropy
	}

	return entropy
}

type PolicySelection struct {
	SelectedPolicy   Policy
	AllPolicies      []Policy
	SelectionProbs   []float64
	PolicyEntropy    float64
	Timestamp        int64
}

func (c *FreeEnergyCalculator) SelectPolicyWithInfo(context map[string]interface{}) (*PolicySelection, error) {
	policies := c.EvaluatePolicies(context)

	if len(policies) == 0 {
		return nil, fmt.Errorf("no policies available for selection")
	}

	sort.Slice(policies, func(i, j int) bool {
		return policies[i].FreeEnergy < policies[j].FreeEnergy
	})

	temperature := c.beta
	normalizedProbs := make([]float64, len(policies))
	sumExp := 0.0
	for i, policy := range policies {
		normalizedProbs[i] = math.Exp(-temperature * policy.FreeEnergy)
		sumExp += normalizedProbs[i]
	}

	if sumExp == 0 {
		for i := range normalizedProbs {
			normalizedProbs[i] = 1.0 / float64(len(normalizedProbs))
		}
	} else {
		for i := range normalizedProbs {
			normalizedProbs[i] /= sumExp
		}
	}

	cumulative := 0.0
	random := uniformRandom()
	var selectedPolicy Policy
	for i, prob := range normalizedProbs {
		cumulative += prob
		if random <= cumulative {
			selectedPolicy = policies[i]
			break
		}
	}
	if (Policy{}) == selectedPolicy {
		selectedPolicy = policies[len(policies)-1]
	}

	policyEntropy := c.CalculatePolicyEntropy(policies)

	return &PolicySelection{
		SelectedPolicy:  selectedPolicy,
		AllPolicies:     policies,
bs:  normalizedProbs,
			SelectionPro	PolicyEntropy:   policyEntropy,
		Timestamp:       makeTimestamp(),
	}, nil
}

func makeTimestamp() int64 {
	return int64(uniformRandom() * 1e12)
}

func (c *FreeEnergyCalculator) SimulatePolicyExecution(policy Policy, steps int) []float64 {
	precision := c.precisionCalculator.CurrentValue()
	simulation := make([]float64, steps)

	for i := range simulation {
		baseValue := policy.FreeEnergy
		noise := (uniformRandom() - 0.5) * 0.2 * (1 - precision)
		decay := float64(i) / float64(steps) * 0.1

		simulation[i] = baseValue + noise - decay
		simulation[i] = math.Max(0, math.Min(1, simulation[i]))
	}

	return simulation
}

func (c *FreeEnergyCalculator) CalculateRegret(policy Policy, optimalPolicy Policy) float64 {
	if optimalPolicy.FreeEnergy == 0 {
		return 0
	}

	freeEnergyRegret := (optimalPolicy.FreeEnergy - policy.FreeEnergy) / optimalPolicy.FreeEnergy
	toolRegret := 0.0

	if len(optimalPolicy.Tools) > 0 && len(policy.Tools) > 0 {
		optimalTools := make(map[string]bool)
		for _, t := range optimalPolicy.Tools {
			optimalTools[t] = true
		}

		matchingTools := 0
		for _, t := range policy.Tools {
			if optimalTools[t] {
				matchingTools++
			}
		}
		toolRegret = 1.0 - float64(matchingTools)/float64(len(policy.Tools))
	}

	totalRegret := freeEnergyRegret*0.6 + toolRegret*0.4
	return math.Min(1.0, totalRegret)
}

type PolicyEnsemble struct {
	policies     []Policy
	weights      []float64
	calculator   *FreeEnergyCalculator
	mu           sync.RWMutex
}

func NewPolicyEnsemble(calculator *FreeEnergyCalculator) *PolicyEnsemble {
	return &PolicyEnsemble{
		policies:   make([]Policy, 0),
		weights:    make([]float64, 0),
		calculator: calculator,
	}
}

func (e *PolicyEnsemble) AddPolicy(policy Policy, weight float64) error {
	if weight <= 0 {
		return fmt.Errorf("weight must be positive, got %.4f", weight)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.policies = append(e.policies, policy)
	e.weights = append(e.weights, weight)
	return nil
}

func (e *PolicyEnsemble) GetEnsemblePolicy() Policy {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.policies) == 0 {
		return Policy{}
	}

	sumWeights := 0.0
	for _, w := range e.weights {
		sumWeights += w
	}

	normalizedWeights := make([]float64, len(e.weights))
	for i, w := range e.weights {
		normalizedWeights[i] = w / sumWeights
	}

	weightedFreeEnergy := 0.0
	weightedConfidence := 0.0
	weightedEpistemic := 0.0
	weightedPragmatic := 0.0

	for i, policy := range e.policies {
		weightedFreeEnergy += policy.FreeEnergy * normalizedWeights[i]
		weightedConfidence += policy.Confidence * normalizedWeights[i]
		weightedEpistemic += policy.EpistemicVal * normalizedWeights[i]
		weightedPragmatic += policy.PragmaticVal * normalizedWeights[i]
	}

	return Policy{
		FreeEnergy:   weightedFreeEnergy,
		Confidence:   weightedConfidence,
		EpistemicVal: weightedEpistemic,
		PragmaticVal: weightedPragmatic,
		Metadata: map[string]interface{}{
			"ensemble_size":  len(e.policies),
			"num_tools":     len(e.policies[0].Tools),
		},
	}
}

func (e *PolicyEnsemble) CalculateDisagreement() float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.policies) < 2 {
		return 0.0
	}

	sumDiff := 0.0
	count := 0
	for i := 0; i < len(e.policies); i++ {
		for j := i + 1; j < len(e.policies); j++ {
			diff := math.Abs(e.policies[i].FreeEnergy - e.policies[j].FreeEnergy)
			sumDiff += diff
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return sumDiff / float64(count)
}
