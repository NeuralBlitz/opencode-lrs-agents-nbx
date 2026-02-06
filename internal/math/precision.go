package precision

import (
	"fmt"
	"math"
	"sync"
)

type PrecisionParameters struct {
	Alpha float64
	Beta  float64
	mu    sync.RWMutex
}

func NewPrecisionParameters(alpha, beta float64) (*PrecisionParameters, error) {
	if alpha <= 0 || beta <= 0 {
		return nil, fmt.Errorf("alpha and beta must be positive, got alpha=%.2f, beta=%.2f", alpha, beta)
	}
	return &PrecisionParameters{
		Alpha: alpha,
		Beta:  beta,
	}, nil
}

func NewDefaultPrecisionParameters() *PrecisionParameters {
	return &PrecisionParameters{
		Alpha: 2.0,
		Beta:  2.0,
	}
}

func (p *PrecisionParameters) Value() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.Alpha+p.Beta == 0 {
		return 0.5
	}
	return p.Alpha / (p.Alpha + p.Beta)
}

func (p *PrecisionParameters) Variance() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.Alpha+p.Beta <= 0 {
		return 0
	}
	return (p.Alpha * p.Beta) / ((p.Alpha + p.Beta) * (p.Alpha + p.Beta) * (p.Alpha + p.Beta + 1))
}

func (p *PrecisionParameters) Update(predictionError float64) error {
	if predictionError < 0 || predictionError > 1 {
		return fmt.Errorf("predictionError must be in [0, 1], got %.4f", predictionError)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	learningRate := 0.1
	surprise := predictionError

	if surprise < 0.5 {
		p.Alpha += learningRate * (1 - surprise)
	} else {
		p.Alpha += learningRate * surprise * 0.5
	}

	if surprise > 0.5 {
		p.Beta += learningRate * surprise
	} else {
		p.Beta += learningRate * (1 - surprise) * 0.5
	}

	p.Alpha = math.Max(0.1, p.Alpha)
	p.Beta = math.Max(0.1, p.Beta)

	return nil
}

func (p *PrecisionParameters) Clone() *PrecisionParameters {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return &PrecisionParameters{
		Alpha: p.Alpha,
		Beta:  p.Beta,
	}
}

func (p *PrecisionParameters) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Alpha = 2.0
	p.Beta = 2.0
}

func (p *PrecisionParameters) String() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return fmt.Sprintf("PrecisionParameters{γ=%.4f, α=%.4f, β=%.4f}", p.Value(), p.Alpha, p.Beta)
}

type BetaDistribution struct {
	Alpha float64
	Beta  float64
	mu    sync.RWMutex
}

func NewBetaDistribution(alpha, beta float64) (*BetaDistribution, error) {
	if alpha <= 0 || beta <= 0 {
		return nil, fmt.Errorf("alpha and beta must be positive, got alpha=%.2f, beta=%.2f", alpha, beta)
	}
	return &BetaDistribution{
		Alpha: alpha,
		Beta:  beta,
	}, nil
}

func (b *BetaDistribution) Mean() float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.Alpha+b.Beta == 0 {
		return 0.5
	}
	return b.Alpha / (b.Alpha + b.Beta)
}

func (b *BetaDistribution) Mode() float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.Alpha > 1 && b.Beta > 1 {
		return (b.Alpha - 1) / (b.Alpha + b.Beta - 2)
	}
	return b.Mean()
}

func (b *BetaDistribution) Sample() float64 {
	b.mu.RLock()
	alpha, beta := b.Alpha, b.Beta
	b.mu.RUnlock()

	gammaSample := sampleGamma(alpha)
	deltaSample := sampleGamma(beta)
	return gammaSample / (gammaSample + deltaSample)
}

func sampleGamma(shape float64) float64 {
	if shape < 1 {
		shape += 1
	}
	scale := 1.0
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	v := 0.0

	for v <= 0 {
		x := normalSample()
		v = 1.0 + c*x
		v = v * v * v
	}

	u := uniformSample()
	if u < 1.0-0.0331*(x*x)*(x*x) {
		return d * v * scale
	}

	if math.Log(u) < 0.5*x*x + d*(1.0-v+math.Log(v)) {
		return d * v * scale
	}

	return sampleGamma(shape)
}

func normalSample() float64 {
	u1 := uniformSample()
	u2 := uniformSample()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

func uniformSample() float64 {
	return float64(uint64(0x1BD10A33F12D93B9)>>32) / float64(uint64(1)<<32)
}

func (b *BetaDistribution) UpdateWithObservation(observation float64) error {
	if observation < 0 || observation > 1 {
		return fmt.Errorf("observation must be in [0, 1], got %.4f", observation)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	learningRate := 0.1
	expectedPrecision := b.Mean()
	predictionError := math.Abs(observation - expectedPrecision)

	if expectedPrecision > observation {
		b.Alpha += learningRate * (1 - predictionError)
	} else {
		b.Alpha += learningRate * predictionError * 0.5
	}

	if predictionError > 0.5 {
		b.Beta += learningRate * predictionError
	} else {
		b.Beta += learningRate * (1 - predictionError) * 0.5
	}

	b.Alpha = math.Max(0.1, b.Alpha)
	b.Beta = math.Max(0.1, b.Beta)

	return nil
}

func (b *BetaDistribution) PDF(x float64) float64 {
	if x < 0 || x > 1 {
		return 0
	}
	b.mu.RLock()
	alpha, beta := b.Alpha, b.Beta
	b.mu.RUnlock()

	return math.Pow(x, alpha-1) * math.Pow(1-x, beta-1) / betaFunction(alpha, beta)
}

func betaFunction(alpha, beta float64) float64 {
	return math.Gamma(alpha) * math.Gamma(beta) / math.Gamma(alpha+beta)
}

func (b *BetaDistribution) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	b.mu.RLock()
	alpha, beta := b.Alpha, b.Beta
	b.mu.RUnlock()

	return incompleteBeta(alpha, beta, x)
}

func incompleteBeta(alpha, beta, x float64) float64 {
	bt := math.Exp(logBeta(alpha, beta) + alpha*math.Log(x) + beta*math.Log(1-x))
	if x < (alpha+1)/(alpha+beta+2) {
		return bt * betaRegularized(alpha, beta, x) / alpha
	}
	return 1 - bt*betaRegularized(beta, alpha, 1-x)/beta
}

func logBeta(alpha, beta float64) float64 {
	return math.Lgamma(alpha) + math.Lgamma(beta) - math.Lgamma(alpha+beta)
}

func betaRegularized(a, b, x float64) float64 {
	bt := math.Exp(logBeta(a, b) + a*math.Log(x) + b*math.Log(1-x))
	if x == 0 || x == 1 {
		return 0
	}

	var result float64
	if x < (a+1)/(a+b+2) {
		result = bt * betaCF(a, b, x) / a
	} else {
		result = 1 - bt*betaCF(b, a, 1-x)/b
	}
	return result
}

func betaCF(a, b, x float64) float64 {
	maxIterations := 100
	epsilon := 1e-10

	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1.0 - qab*x/qap
	if math.Abs(d) < epsilon {
		d = epsilon
	}
	d = 1.0 / d
	h := d

	for m := 1; m <= maxIterations; m++ {
		m2 := 2 * float64(m)

		aa := float64(m) * (b - float64(m)) * x / ((qam + m2) * (a + m2))
		d = 1.0 + aa*d
		if math.Abs(d) < epsilon {
			d = epsilon
		}
		c = 1.0 + aa/c
		if math.Abs(c) < epsilon {
			c = epsilon
		}
		d = 1.0 / d
		h *= d * c

		aa = -(a + float64(m)) * (qab + float64(m)) * x / ((a + m2) * (qap + m2))
		d = 1.0 + aa*d
		if math.Abs(d) < epsilon {
			d = epsilon
		}
		c = 1.0 + aa/c
		if math.Abs(c) < epsilon {
			c = epsilon
		}
		d = 1.0 / d
		delta := d * c
		h *= delta

		if math.Abs(delta-1.0) < epsilon {
			break
		}
	}

	return h
}

func (b *BetaDistribution) Entropy() float64 {
	b.mu.RLock()
	alpha, beta := b.Alpha, b.Beta
	b.mu.RUnlock()

	logBetaAB := logBeta(alpha, beta)
	entropy := logBetaAB - (alpha-1)*math.Digamma(alpha) - (beta-1)*math.Digamma(beta) + (alpha+beta-2)*math.Digamma(alpha+beta)
	return entropy
}

func (b *BetaDistribution) KL_divergence(other *BetaDistribution) float64 {
	b.mu.RLock()
	alpha1, beta1 := b.Alpha, b.Beta
	b.mu.RUnlock()

	other.mu.RLock()
	alpha2, beta2 := other.Alpha, other.Beta
	other.mu.RUnlock()

	logBeta1 := logBeta(alpha1, beta1)
	logBeta2 := logBeta(alpha2, beta2)

	kl := (alpha1-alpha2)*math.Digamma(alpha1) + (beta1-beta2)*math.Digamma(beta1) +
		(math.Gamma(alpha1+beta1) - math.Gamma(alpha2+beta2)) / math.Gamma(alpha1+beta1) +
		(alpha2+beta2-1)*(math.Digamma(alpha2+beta2)-math.Digamma(alpha1+beta1)) +
		logBeta2 - logBeta1

	return kl
}

type PrecisionTracker struct {
	parameters *PrecisionParameters
	history    []float64
	mu         sync.RWMutex
}

func NewPrecisionTracker(alpha, beta float64) (*PrecisionTracker, error) {
	params, err := NewPrecisionParameters(alpha, beta)
	if err != nil {
		return nil, err
	}
	return &PrecisionTracker{
		parameters: params,
		history:    make([]float64, 0),
	}, nil
}

func (t *PrecisionTracker) CurrentValue() float64 {
	return t.parameters.Value()
}

func (t *PrecisionTracker) Update(predictionError float64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.parameters.Update(predictionError); err != nil {
		return err
	}

	t.history = append(t.history, t.parameters.Value())
	if len(t.history) > 1000 {
		t.history = t.history[len(t.history)-1000:]
	}

	return nil
}

func (t *PrecisionTracker) GetHistory() []float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	historyCopy := make([]float64, len(t.history))
	copy(historyCopy, t.history)
	return historyCopy
}

func (t *PrecisionTracker) AveragePrecision() float64 {
	t.mu.RLock()
	history := t.history
	t.mu.RUnlock()

	if len(history) == 0 {
		return 0.5
	}

	sum := 0.0
	for _, v := range history {
		sum += v
	}
	return sum / float64(len(history))
}

func (t *PrecisionTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.parameters.Reset()
	t.history = make([]float64, 0)
}
