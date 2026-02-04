"""
NeuralBlitz V50 - LangChain Integration
Custom tool for LangChain consciousness processing.
"""

try:
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import CallbackManagerForToolRun

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object

from typing import Optional, Type
from pydantic import BaseModel, Field

from ..minimal import MinimalCognitiveEngine, IntentVector


class ConsciousnessInput(BaseModel):
    """Input schema for consciousness tool."""

    intent_description: str = Field(
        description="Description of the intent in natural language"
    )
    dominance: float = Field(
        default=0.5,
        description="Level of control/authority needed (-1 to 1)",
        ge=-1,
        le=1,
    )
    creativity: float = Field(
        default=0.5,
        description="Level of creativity/innovation needed (-1 to 1)",
        ge=-1,
        le=1,
    )
    harmony: float = Field(
        default=0.5,
        description="Level of cooperation/balance needed (-1 to 1)",
        ge=-1,
        le=1,
    )
    analytical: float = Field(
        default=0.5,
        description="Level of analysis/knowledge needed (-1 to 1)",
        ge=-1,
        le=1,
    )


class NeuralBlitzConsciousnessTool(BaseTool):
    """
    LangChain tool for NeuralBlitz consciousness processing.

    This tool allows LLMs to:
    1. Process intents through the consciousness engine
    2. Analyze consciousness states
    3. Get recommendations for optimal intent formulation

    Example:
        >>> tool = NeuralBlitzConsciousnessTool()
        >>> result = tool.run({
        ...     "intent_description": "I need a creative solution",
        ...     "creativity": 0.9,
        ...     "dominance": 0.3
        ... })
    """

    name = "neuralblitz_consciousness"
    description = """
    Process an intent through the NeuralBlitz consciousness engine.
    
    This tool analyzes cognitive intent and returns consciousness level,
    coherence, and processing results. Use it to:
    - Analyze the cognitive state of an intent
    - Get consciousness metrics (coherence, level, complexity)
    - Understand how an intent affects the system
    - Receive recommendations for optimal intent formulation
    
    Input parameters:
    - intent_description: What you want to accomplish (text)
    - dominance: Control/authority level (-1 to 1, default 0.5)
    - creativity: Innovation level (-1 to 1, default 0.5)
    - harmony: Cooperation level (-1 to 1, default 0.5)
    - analytical: Analysis level (-1 to 1, default 0.5)
    """

    args_schema: Type[BaseModel] = ConsciousnessInput

    def __init__(self, engine: Optional[MinimalCognitiveEngine] = None):
        super().__init__()
        self.engine = engine or MinimalCognitiveEngine()

    def _run(
        self,
        intent_description: str,
        dominance: float = 0.5,
        creativity: float = 0.5,
        harmony: float = 0.5,
        analytical: float = 0.5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the consciousness processing tool.

        Args:
            intent_description: Natural language description
            dominance: Control/authority level
            creativity: Innovation level
            harmony: Cooperation level
            analytical: Analysis level

        Returns:
            Formatted string with consciousness analysis
        """
        # Create intent vector from parameters
        intent = IntentVector(
            phi1_dominance=dominance,
            phi2_harmony=harmony,
            phi3_creation=creativity,
            phi4_preservation=0.5,  # Neutral
            phi5_transformation=0.5,  # Neutral
            phi6_knowledge=analytical,
            phi7_connection=harmony,  # Map connection to harmony
        )

        # Process through engine
        result = self.engine.process_intent(intent)

        # Format response
        response = f"""
Consciousness Analysis for: "{intent_description}"

🧠 Consciousness State:
   Level: {result["consciousness_level"]}
   Coherence: {result["coherence"]:.3f}
   Confidence: {result["confidence"]:.2%}

⚡ Processing:
   Time: {result["processing_time_ms"]:.2f}ms
   Patterns in Memory: {result["patterns_stored"]}

📊 Output Vector: [{", ".join(f"{x:.2f}" for x in result["output_vector"])}]

💡 Interpretation:
   This intent demonstrates {self._interpret_level(result["consciousness_level"])} 
   cognitive engagement with {self._interpret_coherence(result["coherence"])} coherence.
"""

        return response.strip()

    def _interpret_level(self, level: str) -> str:
        """Human-friendly interpretation of consciousness level."""
        interpretations = {
            "DORMANT": "minimal",
            "AWARE": "basic",
            "FOCUSED": "moderate",
            "TRANSCENDENT": "high",
            "SINGULARITY": "peak",
        }
        return interpretations.get(level, "unknown")

    def _interpret_coherence(self, coherence: float) -> str:
        """Human-friendly interpretation of coherence."""
        if coherence > 0.8:
            return "excellent"
        elif coherence > 0.6:
            return "good"
        elif coherence > 0.4:
            return "moderate"
        else:
            return "low"

    async def _arun(self, *args, **kwargs):
        """Async version - not implemented."""
        raise NotImplementedError("Async not supported")


class ConsciousnessAnalysisChain:
    """
    Helper class to create LangChain chains with consciousness analysis.

    Example:
        >>> chain = ConsciousnessAnalysisChain()
        >>> result = chain.analyze("I want to create something innovative")
    """

    def __init__(self, engine: Optional[MinimalCognitiveEngine] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Run: pip install langchain")

        self.engine = engine or MinimalCognitiveEngine()
        self.tool = NeuralBlitzConsciousnessTool(engine)

    def analyze_intent(self, description: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze an intent and return structured data.

        Args:
            description: Natural language intent description
            **kwargs: Additional intent parameters

        Returns:
            Dictionary with analysis results
        """
        # Run the tool
        result_str = self.tool._run(intent_description=description, **kwargs)

        # Also get raw result from engine
        intent = IntentVector(
            phi1_dominance=kwargs.get("dominance", 0.5),
            phi2_harmony=kwargs.get("harmony", 0.5),
            phi3_creation=kwargs.get("creativity", 0.5),
            phi4_preservation=0.5,
            phi5_transformation=0.5,
            phi6_knowledge=kwargs.get("analytical", 0.5),
            phi7_connection=kwargs.get("harmony", 0.5),
        )

        result = self.engine.process_intent(intent)

        return {
            "description": description,
            "text_analysis": result_str,
            "consciousness_level": result["consciousness_level"],
            "coherence": result["coherence"],
            "confidence": result["confidence"],
            "processing_time_ms": result["processing_time_ms"],
            "output_vector": result["output_vector"],
        }

    def get_recommendation(self, target_level: str = "FOCUSED") -> Dict[str, Any]:
        """
        Get recommended intent parameters to achieve a target consciousness level.

        Args:
            target_level: Desired consciousness level (DORMANT, AWARE, FOCUSED, TRANSCENDENT, SINGULARITY)

        Returns:
            Recommended intent parameters
        """
        # Level-specific recommendations
        recommendations = {
            "DORMANT": {
                "dominance": 0.1,
                "harmony": 0.5,
                "creativity": 0.1,
                "analytical": 0.1,
                "description": "Low activity state",
            },
            "AWARE": {
                "dominance": 0.3,
                "harmony": 0.6,
                "creativity": 0.3,
                "analytical": 0.4,
                "description": "Basic engagement",
            },
            "FOCUSED": {
                "dominance": 0.5,
                "harmony": 0.7,
                "creativity": 0.6,
                "analytical": 0.7,
                "description": "Optimal for most tasks",
            },
            "TRANSCENDENT": {
                "dominance": 0.7,
                "harmony": 0.8,
                "creativity": 0.9,
                "analytical": 0.8,
                "description": "High performance state",
            },
            "SINGULARITY": {
                "dominance": 0.9,
                "harmony": 0.9,
                "creativity": 0.95,
                "analytical": 0.9,
                "description": "Peak consciousness",
            },
        }

        return recommendations.get(target_level, recommendations["FOCUSED"])


def create_langchain_tool(
    engine: Optional[MinimalCognitiveEngine] = None,
) -> NeuralBlitzConsciousnessTool:
    """
    Factory function to create a LangChain tool.

    Args:
        engine: Optional engine instance

    Returns:
        Configured NeuralBlitzConsciousnessTool

    Example:
        >>> from langchain import OpenAI, initialize_agent
        >>> tool = create_langchain_tool()
        >>> llm = OpenAI()
        >>> agent = initialize_agent([tool], llm, agent="zero-shot-react-description")
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain not installed. Install with: pip install langchain\n"
            "For OpenAI integration: pip install openai"
        )

    return NeuralBlitzConsciousnessTool(engine)


# Export
__all__ = [
    "NeuralBlitzConsciousnessTool",
    "ConsciousnessAnalysisChain",
    "ConsciousnessInput",
    "create_langchain_tool",
    "LANGCHAIN_AVAILABLE",
]
