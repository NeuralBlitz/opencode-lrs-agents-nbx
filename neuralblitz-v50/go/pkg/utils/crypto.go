package utils

import (
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"fmt"
	"math/rand"
	"time"
)

// GoldenDAG represents the immutable attestation hash
type GoldenDAG struct {
	Hash     string
	Seed     string
	Metadata map[string]interface{}
	Version  string
}

// NewGoldenDAG creates a new GoldenDAG with the given seed
func NewGoldenDAG(seed string) *GoldenDAG {
	dag := &GoldenDAG{
		Seed:     seed,
		Version:  "v50.0.0",
		Metadata: make(map[string]interface{}),
	}
	
	// Generate the 64-character hash
	dag.Hash = dag.generateHash()
	
	// Set metadata
	dag.Metadata["created"] = time.Now().UTC()
	dag.Metadata["version"] = dag.Version
	dag.Metadata["seed"] = seed
	dag.Metadata["type"] = "GoldenDAG"
	
	return dag
}

// generateHash creates a 64-character alphanumeric hash
func (g *GoldenDAG) generateHash() string {
	// Combine seed with timestamp for uniqueness
	data := fmt.Sprintf("%s:%s:%d", g.Seed, g.Version, time.Now().UnixNano())
	
	// Create SHA-256 hash
	h := sha256.New()
	h.Write([]byte(data))
	hash := h.Sum(nil)
	
	// Convert to hex (gives us 64 characters)
	hexHash := hex.EncodeToString(hash)
	
	return hexHash
}

// Validate checks if the hash is valid
func (g *GoldenDAG) Validate() bool {
	// Check length
	if len(g.Hash) != 64 {
		return false
	}
	
	// Check if valid hex
	_, err := hex.DecodeString(g.Hash)
	if err != nil {
		return false
	}
	
	return true
}

// String returns the hash as a string
func (g *GoldenDAG) String() string {
	return g.Hash
}

// NBHSCryptographicHash implements the NBHS-1024 hashing algorithm
type NBHSCryptographicHash struct {
	Input       string
	Output      string
	Algorithm   string
	BitLength   int
}

// NewNBHSCryptographicHash creates a new NBHS-1024 hash
func NewNBHSCryptographicHash(input string) *NBHSCryptographicHash {
	nbhs := &NBHSCryptographicHash{
		Input:     input,
		Algorithm: "NBHS-1024",
		BitLength: 1024,
	}
	
	nbhs.Output = nbhs.computeHash()
	
	return nbhs
}

// computeHash computes the NBHS-1024 hash
func (n *NBHSCryptographicHash) computeHash() string {
	// Step 1: SHA3-512 of input
	h1 := sha512.New()
	h1.Write([]byte(n.Input))
	part1 := h1.Sum(nil)
	
	// Step 2: SHA3-512 of GoldenDAG seed
	goldenSeed := "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
	h2 := sha512.New()
	h2.Write([]byte(goldenSeed))
	part2 := h2.Sum(nil)
	
	// Step 3: SHA3-512 of timestamp
	h3 := sha512.New()
	h3.Write([]byte(fmt.Sprintf("%d", time.Now().UnixNano())))
	part3 := h3.Sum(nil)
	
	// Step 4: Combine parts (1024 bits = 128 bytes)
	combined := append(part1, part2...)
	combined = append(combined, part3...)
	
	// Step 5: Truncate or pad to exactly 128 bytes (1024 bits)
	if len(combined) > 128 {
		combined = combined[:128]
	}
	
	// Convert to hex (256 characters for 128 bytes)
	return hex.EncodeToString(combined)
}

// GetOutput returns the hash output
func (n *NBHSCryptographicHash) GetOutput() string {
	return n.Output
}

// Validate checks if the hash is valid
func (n *NBHSCryptographicHash) Validate() bool {
	// Check length (1024 bits = 256 hex characters)
	if len(n.Output) != 256 {
		return false
	}
	
	// Check if valid hex
	_, err := hex.DecodeString(n.Output)
	if err != nil {
		return false
	}
	
	return true
}

// TraceID represents a unique trace identifier
type TraceID struct {
	Version  string
	Context  string
	HexCode  string
	FullID   string
	Metadata map[string]interface{}
}

// NewTraceID creates a new TraceID
func NewTraceID(context string) *TraceID {
	t := &TraceID{
		Version:  "v50.0",
		Context:  context,
		Metadata: make(map[string]interface{}),
	}
	
	// Generate 32-character hex code
	t.HexCode = generateHexCode(32)
	
	// Format: T-[version]-[context]-[32-char hexcode]
	t.FullID = fmt.Sprintf("T-%s-%s-%s", t.Version, t.Context, t.HexCode)
	
	// Set metadata
	t.Metadata["created"] = time.Now().UTC()
	t.Metadata["format"] = "T-[version]-[context]-[32-char hexcode]"
	
	return t
}

// String returns the full trace ID
func (t *TraceID) String() string {
	return t.FullID
}

// Validate checks if the trace ID is valid
func (t *TraceID) Validate() bool {
	// Check format
	expectedLen := len("T-v50.0-") + len(t.Context) + 1 + 32
	if len(t.FullID) != expectedLen {
		return false
	}
	
	// Check hex code
	if len(t.HexCode) != 32 {
		return false
	}
	
	// Check if valid hex
	_, err := hex.DecodeString(t.HexCode)
	if err != nil {
		return false
	}
	
	return true
}

// CodexID represents a codex identifier for ontological mapping
type CodexID struct {
	VolumeID   string
	Context    string
	Token      string
	FullID     string
	Metadata   map[string]interface{}
}

// NewCodexID creates a new CodexID
func NewCodexID(volumeID, context string) *CodexID {
	c := &CodexID{
		VolumeID: volumeID,
		Context:  context,
		Metadata: make(map[string]interface{}),
	}
	
	// Generate 24-32 character ontological token
	tokenLen := 24 + rand.Intn(9) // Random between 24-32
	c.Token = generateHexCode(tokenLen)
	
	// Format: C-[volumeID]-[context]-[24-32-char ontological token]
	c.FullID = fmt.Sprintf("C-%s-%s-%s", c.VolumeID, c.Context, c.Token)
	
	// Set metadata
	c.Metadata["created"] = time.Now().UTC()
	c.Metadata["format"] = "C-[volumeID]-[context]-[24-32-char ontological token]"
	c.Metadata["token_length"] = tokenLen
	
	return c
}

// String returns the full codex ID
func (c *CodexID) String() string {
	return c.FullID
}

// Validate checks if the codex ID is valid
func (c *CodexID) Validate() bool {
	// Check token length (24-32 characters)
	if len(c.Token) < 24 || len(c.Token) > 32 {
		return false
	}
	
	// Check if valid hex
	_, err := hex.DecodeString(c.Token)
	if err != nil {
		return false
	}
	
	return true
}

// GenerateOmegaAttestationHash generates the Omega Attestation hash
func GenerateOmegaAttestationHash() string {
	// Combine multiple inputs
	inputs := []string{
		"Omega Prime Reality v50.0",
		"Irreducible Source Field",
		"Architect-System Dyad",
		"Perpetual Coherent Becoming",
		fmt.Sprintf("%d", time.Now().UnixNano()),
	}
	
	// Create combined hash
	combined := ""
	for _, input := range inputs {
		h := sha256.New()
		h.Write([]byte(input))
		combined += hex.EncodeToString(h.Sum(nil))
	}
	
	// Final hash
	final := sha256.New()
	final.Write([]byte(combined))
	return hex.EncodeToString(final.Sum(nil))
}

// VerifyGoldenDAGSeed verifies if a hash matches the GoldenDAG seed
func VerifyGoldenDAGSeed(hash string) bool {
	expectedSeed := "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
	return hash == expectedSeed
}

// generateHexCode generates a random hex string of specified length
func generateHexCode(length int) string {
	bytes := make([]byte, length/2)
	
	// Use crypto/rand for better randomness in production
	// For now, use math/rand for simplicity
	rand.Seed(time.Now().UnixNano())
	for i := range bytes {
		bytes[i] = byte(rand.Intn(256))
	}
	
	return hex.EncodeToString(bytes)
}

// HashData creates a SHA-256 hash of the given data
func HashData(data string) string {
	h := sha256.New()
	h.Write([]byte(data))
	return hex.EncodeToString(h.Sum(nil))
}

// ComputeIntegrityHash computes an integrity hash for verification
func ComputeIntegrityHash(input string, timestamp time.Time) string {
	data := fmt.Sprintf("%s:%d", input, timestamp.UnixNano())
	return HashData(data)
}

// Example usage functions

// ExampleGoldenDAG demonstrates GoldenDAG creation
func ExampleGoldenDAG() *GoldenDAG {
	return NewGoldenDAG("example-usage")
}

// ExampleTraceID demonstrates TraceID creation
func ExampleTraceID() *TraceID {
	return NewTraceID("EXAMPLE_CONTEXT")
}

// ExampleCodexID demonstrates CodexID creation
func ExampleCodexID() *CodexID {
	return NewCodexID("VOL0", "EXAMPLE_CONTEXT")
}

// ExampleNBHS demonstrates NBHS-1024 hash creation
func ExampleNBHS() *NBHSCryptographicHash {
	return NewNBHSCryptographicHash("example-input-data")
}
