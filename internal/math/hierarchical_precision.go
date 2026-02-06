package math

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type HierarchyLevel int

const (
	LevelAbstract HierarchyLevel = iota
	LevelPlanning
	LevelExecution
)

func (l HierarchyLevel) String() string {
	switch l {
	case LevelAbstract:
		return "abstract"
	case LevelPlanning:
		return "planning"
	case LevelExecution:
		return "execution"
	default:
		return "unknown"
	}
}

func (l HierarchyLevel) IsValid() bool {
	return l >= LevelAbstract && l <= LevelExecution
}

func (l HierarchyLevel) Parent() HierarchyLevel {
	switch l {
	case LevelPlanning:
		return LevelAbstract
	case LevelExecution:
		return LevelPlanning
	default:
		return LevelAbstract
	}
}

func (l HierarchyLevel) Children() []HierarchyLevel {
	switch l {
	case LevelAbstract:
		return []HierarchyLevel{LevelPlanning}
	case LevelPlanning:
		return []HierarchyLevel{LevelExecution}
	default:
		return []HierarchyLevel{}
	}
}

func (l HierarchyLevel) Depth() int {
	switch l {
	case LevelAbstract:
		return 0
	case LevelPlanning:
		return 1
	case LevelExecution:
		return 2
	default:
		return -1
	}
}

type HierarchicalPrecision struct {
	levels          map[HierarchyLevel]*PrecisionTracker
	propagationRate float64
	decayFactor     float64
	mu              sync.RWMutex
}

func NewHierarchicalPrecision() *HierarchicalPrecision {
	hp := &HierarchicalPrecision{
		levels:          make(map[HierarchyLevel]*PrecisionTracker),
		propagationRate: 0.3,
		decayFactor:     0.1,
	}

	for _, level := range []HierarchyLevel{LevelAbstract, LevelPlanning, LevelExecution} {
		hp.levels[level] = mustNewPrecisionTracker(2.0, 2.0)
	}

	return hp
}

func mustNewPrecisionTracker(alpha, beta float64) *PrecisionTracker {
	tracker, err := NewPrecisionTracker(alpha, beta)
	if err != nil {
		panic(err)
	}
	return tracker
}

func (hp *HierarchicalPrecision) Update(level HierarchyLevel, predictionError float64) error {
	if !level.IsValid() {
		return fmt.Errorf("invalid hierarchy level: %v", level)
	}
	if predictionError < 0 || predictionError > 1 {
		return fmt.Errorf("predictionError must be in [0, 1], got %.4f", predictionError)
	}

	hp.mu.Lock()
	defer hp.mu.Unlock()

	if tracker, exists := hp.levels[level]; exists {
		if err := tracker.Update(predictionError); err != nil {
			return err
		}
	}

	hp.propagateChanges(level)

	return nil
}

func (hp *HierarchicalPrecision) propagateChanges(changedLevel HierarchyLevel) {
	hp.mu.Lock()
	defer hp.mu.Unlock()

	children := changedLevel.Children()
	changedValue := hp.levels[changedLevel].CurrentValue()

	for _, child := range children {
		if childTracker, exists := hp.levels[child]; exists {
			childValue := childTracker.CurrentValue()
			diff := (changedValue - childValue) * hp.propagationRate
			childTracker.Update(math.Abs(diff))
		}
	}

	if parent := changedLevel.Parent(); parent != changedLevel {
		if parentTracker, exists := hp.levels[parent]; exists {
			parentValue := parentTracker.CurrentValue()
			childValue := hp.levels[changedLevel].CurrentValue()
			diff := (childValue - parentValue) * hp.propagationRate * hp.decayFactor
			parentTracker.Update(math.Abs(diff))
		}
	}

	hp.applyDecay()
}

func (hp *HierarchicalPrecision) applyDecay() {
	for level := range hp.levels {
		hp.levels[level].Update(hp.decayFactor * 0.1)
	}
}

func (hp *HierarchicalPrecision) GetPrecision(level HierarchyLevel) (float64, error) {
	if !level.IsValid() {
		return 0, fmt.Errorf("invalid hierarchy level: %v", level)
	}

	hp.mu.RLock()
	defer hp.mu.RUnlock()

	if tracker, exists := hp.levels[level]; exists {
		return tracker.CurrentValue(), nil
	}

	return 0.5, nil
}

func (hp *HierarchicalPrecision) GetAllPrecision() map[HierarchyLevel]float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	result := make(map[HierarchyLevel]float64)
	for level, tracker := range hp.levels {
		result[level] = tracker.CurrentValue()
	}
	return result
}

func (hp *HierarchicalPrecision) GetDominantLevel() HierarchyLevel {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	var dominantLevel HierarchyLevel = LevelAbstract
	var maxPrecision float64 = -1

	for level, tracker := range hp.levels {
		precision := tracker.CurrentValue()
		if precision > maxPrecision {
			maxPrecision = precision
			dominantLevel = level
		}
	}

	return dominantLevel
}

func (hp *HierarchicalPrecision) GetAveragePrecision() float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	if len(hp.levels) == 0 {
		return 0.5
	}

	sum := 0.0
	for _, tracker := range hp.levels {
		sum += tracker.CurrentValue()
	}
	return sum / float64(len(hp.levels))
}

func (hp *HierarchicalPrecision) GetWeightedPrecision() float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	weights := []float64{0.2, 0.3, 0.5}
	levels := []HierarchyLevel{LevelAbstract, LevelPlanning, LevelExecution}

	weightedSum := 0.0
	for i, level := range levels {
		if tracker, exists := hp.levels[level]; exists {
			weightedSum += tracker.CurrentValue() * weights[i]
		}
	}

	return weightedSum
}

func (hp *HierarchicalPrecision) GetPrecisionVariance() float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	levels := []HierarchyLevel{LevelAbstract, LevelPlanning, LevelExecution}
	values := make([]float64, 0, len(levels))
	for _, level := range levels {
		if tracker, exists := hp.levels[level]; exists {
			values = append(values, tracker.CurrentValue())
		}
	}

	if len(values) == 0 {
		return 0
	}

	mean := hp.GetAveragePrecision()
	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	return variance / float64(len(values))
}

func (hp *HierarchicalPrecision) GetPrecisionHistory(level HierarchyLevel) []float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	if tracker, exists := hp.levels[level]; exists {
		return tracker.GetHistory()
	}
	return nil
}

func (hp *HierarchicalPrecision) Clone() *HierarchicalPrecision {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	clone := &HierarchicalPrecision{
		levels:          make(map[HierarchyLevel]*PrecisionTracker),
		propagationRate: hp.propagationRate,
		decayFactor:     hp.decayFactor,
	}

	for level, tracker := range hp.levels {
		clone.levels[level] = &PrecisionTracker{
			parameters: tracker.parameters.Clone(),
			history:    append([]float64{}, tracker.GetHistory()...),
		}
	}

	return clone
}

func (hp *HierarchicalPrecision) Reset() {
	hp.mu.Lock()
	defer hp.mu.Unlock()

	for level := range hp.levels {
		hp.levels[level].Reset()
	}
}

func (hp *HierarchicalPrecision) ResetLevel(level HierarchyLevel) error {
	if !level.IsValid() {
		return fmt.Errorf("invalid hierarchy level: %v", level)
	}

	hp.mu.Lock()
	defer hp.mu.Unlock()

	if tracker, exists := hp.levels[level]; exists {
		tracker.Reset()
	}

	return nil
}

func (hp *HierarchicalPrecision) SetPropagationRate(rate float64) error {
	if rate < 0 || rate > 1 {
		return fmt.Errorf("propagationRate must be in [0, 1], got %.4f", rate)
	}

	hp.mu.Lock()
	defer hp.mu.Unlock()
	hp.propagationRate = rate
	return nil
}

func (hp *HierarchicalPrecision) GetPropagationRate() float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()
	return hp.propagationRate
}

func (hp *HierarchicalPrecision) SetDecayFactor(factor float64) error {
	if factor < 0 || factor > 1 {
		return fmt.Errorf("decayFactor must be in [0, 1], got %.4f", factor)
	}

	hp.mu.Lock()
	defer hp.mu.Unlock()
	hp.decayFactor = factor
	return nil
}

func (hp *HierarchicalPrecision) GetDecayFactor() float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()
	return hp.decayFactor
}

type PrecisionNode struct {
	Level       HierarchyLevel
	Precision   float64
	Confidence  float64
	Children    []*PrecisionNode
	Parent      *PrecisionNode
	mu          sync.RWMutex
}

func (hp *HierarchicalPrecision) BuildHierarchyTree() *PrecisionNode {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	abstract := &PrecisionNode{
		Level:      LevelAbstract,
		Precision:  hp.levels[LevelAbstract].CurrentValue(),
		Confidence: hp.levels[LevelAbstract].AveragePrecision(),
	}

	planning := &PrecisionNode{
		Level:      LevelPlanning,
		Precision:  hp.levels[LevelPlanning].CurrentValue(),
		Confidence: hp.levels[LevelPlanning].AveragePrecision(),
	}

	execution := &PrecisionNode{
		Level:      LevelExecution,
		Precision:  hp.levels[LevelExecution].CurrentValue(),
		Confidence: hp.levels[LevelExecution].AveragePrecision(),
	}

	abstract.Children = []*PrecisionNode{planning}
	planning.Children = []*PrecisionNode{execution}
	planning.Parent = abstract
	execution.Parent = planning

	return abstract
}

func (n *PrecisionNode) GetDepth() int {
	if n.Children == nil || len(n.Children) == 0 {
		return 0
	}
	return 1 + n.Children[0].GetDepth()
}

func (n *PrecisionNode) GetMaxPrecisionPath() []*PrecisionNode {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if len(n.Children) == 0 {
		return []*PrecisionNode{n}
	}

	childPath := n.Children[0].GetMaxPrecisionPath()
	return append([]*PrecisionNode{n}, childPath...)
}

func (n *PrecisionNode) CalculateInformationGain() float64 {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if len(n.Children) == 0 {
		return 0
	}

	parentPrecision := n.Precision
	childInfoGain := 0.0
	for _, child := range n.Children {
		childInfoGain += math.Abs(child.Precision - parentPrecision)
	}

	return childInfoGain / float64(len(n.Children))
}

type HierarchicalState struct {
	Hierarchy    *HierarchicalPrecision
	CurrentLevel HierarchyLevel
	State        map[string]interface{}
	Timestamp    int64
}

func (hp *HierarchicalPrecision) CreateState(context map[string]interface{}) *HierarchicalState {
	hp.mu.RLock()
	dominantLevel := hp.GetDominantLevel()
	precisionValues := hp.GetAllPrecision()
	hp.mu.RUnlock()

	return &HierarchicalState{
		Hierarchy:    hp.Clone(),
		CurrentLevel: dominantLevel,
		State:        context,
		Timestamp:    makeTimestamp(),
	}
}

func (hp *HierarchicalPrecision) UpdateFromState(state *HierarchicalState) error {
	hp.mu.Lock()
	defer hp.mu.Unlock()

	if state.Hierarchy != nil {
		for level, precision := range state.Hierarchy.GetAllPrecision() {
			if tracker, exists := hp.levels[level]; exists {
				diff := math.Abs(precision - tracker.CurrentValue())
				if diff > 0.01 {
					tracker.Update(diff)
				}
			}
		}
	}

	return nil
}

type PrecisionChange struct {
	Level        HierarchyLevel
	OldValue     float64
	NewValue     float64
	Change       float64
	Timestamp    int64
}

func (hp *HierarchicalPrecision) GetChangesSince(sinceTimestamp int64) []PrecisionChange {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	changes := make([]PrecisionChange, 0)
	for level, tracker := range hp.levels {
		history := tracker.GetHistory()
		for _, value := range history {
			changes = append(changes, PrecisionChange{
				Level:    level,
				NewValue: value,
				Timestamp: makeTimestamp(),
			})
		}
	}

	sort.Slice(changes, func(i, j int) bool {
		return changes[i].Timestamp < changes[j].Timestamp
	})

	return changes
}

func (hp *HierarchicalPrecision) GetPrecisionTrend(level HierarchyLevel) float64 {
	hp.mu.RLock()
	history := hp.levels[level].GetHistory()
	hp.mu.RUnlock()

	if len(history) < 2 {
		return 0
	}

	recent := history[len(history)-10:]
	if len(recent) < 2 {
		return 0
	}

	sumXY := 0.0
	sumX := 0.0
	sumY := 0.0
	sumX2 := 0.0

	for i, y := range recent {
		x := float64(i)
		sumXY += x * y
		sumX += x
		sumY += y
		sumX2 += x * x
	}

	n := float64(len(recent))
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	return math.Tanh(slope)
}

func (hp *HierarchicalPrecision) PredictFuturePrecision(level HierarchyLevel, steps int) []float64 {
	hp.mu.RLock()
	history := hp.levels[level].GetHistory()
	hp.mu.RUnlock()

	predictions := make([]float64, steps)

	currentValue := hp.levels[level].CurrentValue()
	trend := hp.GetPrecisionTrend(level)

	for i := 0; i < steps; i++ {
		noise := (uniformRandom() - 0.5) * 0.05
		predictions[i] = currentValue + trend*float64(i+1)*0.1 + noise
		predictions[i] = math.Max(0, math.Min(1, predictions[i]))
	}

	return predictions
}

func (hp *HierarchicalPrecision) GetLevelConfidence() map[HierarchyLevel]float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	confidence := make(map[HierarchyLevel]float64)
	for level, tracker := range hp.levels {
		confidence[level] = tracker.AveragePrecision()
	}
	return confidence
}

func (hp *HierarchicalPrecision) CalculateCoherence() float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	levels := []HierarchyLevel{LevelAbstract, LevelPlanning, LevelExecution}
	values := make([]float64, 0, len(levels))
	for _, level := range levels {
		if tracker, exists := hp.levels[level]; exists {
			values = append(values, tracker.CurrentValue())
		}
	}

	if len(values) < 2 {
		return 1.0
	}

	variance := hp.GetPrecisionVariance()
	coherence := math.Exp(-variance * 2)
	return coherence
}

func (hp *HierarchicalPrecision) GetTopLevelMetrics() map[string]float64 {
	hp.mu.RLock()
	defer hp.mu.RUnlock()

	return map[string]float64{
		"abstract_precision":    hp.levels[LevelAbstract].CurrentValue(),
		"planning_precision":    hp.levels[LevelPlanning].CurrentValue(),
		"execution_precision":   hp.levels[LevelExecution].CurrentValue(),
		"average_precision":     hp.GetAveragePrecision(),
		"weighted_precision":    hp.GetWeightedPrecision(),
		"precision_variance":    hp.GetPrecisionVariance(),
		"dominant_level":        float64(hp.GetDominantLevel()),
		"coherence":             hp.CalculateCoherence(),
		"propagation_rate":      hp.propagationRate,
		"decay_factor":          hp.decayFactor,
	}
}
