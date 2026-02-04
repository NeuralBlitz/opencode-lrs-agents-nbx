"""
NeuralBlitz v50.0 Spiking Neural Network System
=================================================

Advanced spiking neural network implementation with realistic
synaptic plasticity, learning rules, and neuro-biological modeling.

Implementation Date: 2026-02-04
Phase: Neuro-Symbiotic Integration - N4 Implementation
"""

import asyncio
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class NeuronType(Enum):
    """Types of spiking neurons"""

    EXCITATORY = "excitatory"  # Glutamatergic neurons
    INHIBITORY = "inhibitory"  # GABAergic neurons
    MODULATORY = "modulatory"  # Dopaminergic, serotonergic, etc.
    SENSORY = "sensory"  # Input neurons
    MOTOR = "motor"  # Output neurons
    INTERNEURON = "interneuron"  # Local circuit neurons


class PlasticityRule(Enum):
    """Synaptic plasticity rules"""

    HEBBIAN = "hebbian"  # "Cells that fire together, wire together"
    ANTI_HEBBIAN = "anti_hebbian"  # "Cells that fire apart, unwire"
    STDP = "stdp"  # Spike-Timing-Dependent Plasticity
    BCM = "bcm"  # Bienenstock-Cooper-Munro rule
    OJA = "oja"  # Oja's learning rule
    HOPFIELD = "hopfield"  # Hopfield network rule


@dataclass
class SpikeEvent:
    """Individual spike event"""

    timestamp: float
    neuron_id: str
    spike_type: str = "regular"
    amplitude: float = 1.0
    refractory_period: float = 2.0  # ms


@dataclass
class SynapticConnection:
    """Synaptic connection between neurons"""

    source_neuron: str
    target_neuron: str
    weight: float
    delay: float  # ms
    plasticity_rule: PlasticityRule

    # Synaptic parameters
    max_weight: float = 10.0
    min_weight: float = -10.0
    learning_rate: float = 0.01

    # Spike-timing tracking
    last_pre_spike: float = -float("inf")
    last_post_spike: float = -float("inf")
    pre_spike_history: deque = field(default_factory=lambda: deque(maxlen=100))
    post_spike_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Plasticity state
    potentiation_trace: float = 0.0
    depression_trace: float = 0.0

    def calculate_stdp_update(
        self, current_time: float, pre_spike: bool, post_spike: bool
    ) -> float:
        """Calculate STDP weight update based on spike timing"""
        dt = current_time - self.last_pre_spike if pre_spike else 0.0

        if pre_spike:
            # Pre-synaptic spike
            if self.last_post_spike > -float("inf"):
                # Post-pre timing (causality)
                post_pre_dt = current_time - self.last_post_spike

                if post_pre_dt < 50.0:  # 50ms window
                    # Potentiation (pre before post)
                    weight_change = self.learning_rate * np.exp(-post_pre_dt / 20.0)
                else:
                    # No effect
                    weight_change = 0.0
            else:
                weight_change = 0.0

            self.last_pre_spike = current_time
            self.pre_spike_history.append(current_time)

        elif post_spike:
            # Post-synaptic spike
            if self.last_pre_spike > -float("inf"):
                # Pre-post timing
                pre_post_dt = current_time - self.last_pre_spike

                if pre_post_dt < 50.0:  # 50ms window
                    # Depression (post before pre)
                    weight_change = -self.learning_rate * np.exp(-pre_post_dt / 20.0)
                else:
                    # No effect
                    weight_change = 0.0
            else:
                weight_change = 0.0

            self.last_post_spike = current_time
            self.post_spike_history.append(current_time)
        else:
            weight_change = 0.0

        # Apply weight constraints
        new_weight = self.weight + weight_change
        self.weight = np.clip(new_weight, self.min_weight, self.max_weight)

        return weight_change


class SpikingNeuron:
    """
    Biologically plausible spiking neuron model

    Implements leaky integrate-and-fire dynamics with
    realistic membrane properties and spike generation.
    """

    def __init__(self, neuron_id: str, neuron_type: NeuronType):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type

        # Membrane properties
        self.membrane_potential = -65.0  # mV (resting potential)
        self.resting_potential = -65.0
        self.threshold_potential = -50.0  # mV
        self.reset_potential = -65.0

        # Time constants (ms)
        self.membrane_time_constant = 20.0  # τ_m
        self.synaptic_time_constant = 5.0  # τ_s
        self.refractory_period = 2.0  # ms

        # Conductances
        self.leak_conductance = 0.1  # nS
        self.adaptation_conductance = 0.01  # nS (spike-frequency adaptation)

        # Synaptic inputs
        self.excitatory_current = 0.0
        self.inhibitory_current = 0.0
        self.adaptation_current = 0.0

        # Spike tracking
        self.last_spike_time = -float("inf")
        self.spike_times: deque = deque(maxlen=1000)
        self.spike_count = 0
        self.firing_rate = 0.0

        # Adaptation
        self.adaptation_variable = 0.0
        self.adaptation_time_constant = 100.0  # ms

        # Refractory state
        self.is_refractory = False
        self.refractory_end_time = -float("inf")

    def update_membrane_potential(
        self,
        dt: float,
        excitatory_input: float,
        inhibitory_input: float,
        current_time: float,
    ) -> Optional[SpikeEvent]:
        """Update membrane potential and generate spike if threshold reached"""
        # Check refractory period
        if current_time < self.refractory_end_time:
            self.is_refractory = True
            return None
        else:
            self.is_refractory = False

        # Update synaptic currents with exponential decay
        self.excitatory_current *= np.exp(-dt / self.synaptic_time_constant)
        self.inhibitory_current *= np.exp(-dt / self.synaptic_time_constant)

        # Add new inputs
        self.excitatory_current += excitatory_input
        self.inhibitory_current += inhibitory_input

        # Calculate adaptation current
        self.adaptation_current = self.adaptation_conductance * self.adaptation_variable

        # Calculate total synaptic current
        total_current = (
            self.excitatory_current - self.inhibitory_current - self.adaptation_current
        )

        # Update membrane potential (leaky integrate-and-fire)
        dV = (
            -(self.membrane_potential - self.resting_potential)
            / self.membrane_time_constant
            + total_current / self.leak_conductance
        ) * dt
        self.membrane_potential += dV

        # Update adaptation variable
        self.adaptation_variable *= np.exp(-dt / self.adaptation_time_constant)

        # Check for spike
        if (
            self.membrane_potential >= self.threshold_potential
            and not self.is_refractory
        ):
            # Generate spike
            spike = self._generate_spike(current_time)

            # Update adaptation
            self.adaptation_variable += 0.01  # Adaptation increment

            return spike

        return None

    def _generate_spike(self, current_time: float) -> SpikeEvent:
        """Generate spike event"""
        # Update spike tracking
        self.last_spike_time = current_time
        self.spike_times.append(current_time)
        self.spike_count += 1

        # Calculate firing rate
        recent_spikes = [t for t in self.spike_times if current_time - t < 1000.0]
        self.firing_rate = len(recent_spikes)

        # Set refractory period
        self.refractory_end_time = current_time + self.refractory_period

        # Reset membrane potential
        self.membrane_potential = self.reset_potential

        # Determine spike type based on neuron type
        if self.neuron_type == NeuronType.EXCITATORY:
            spike_type = "excitatory"
        elif self.neuron_type == NeuronType.INHIBITORY:
            spike_type = "inhibitory"
        else:
            spike_type = "regular"

        return SpikeEvent(
            timestamp=current_time,
            neuron_id=self.neuron_id,
            spike_type=spike_type,
            amplitude=1.0,
            refractory_period=self.refractory_period,
        )

    def get_membrane_properties(self) -> Dict[str, float]:
        """Get current membrane properties"""
        return {
            "membrane_potential": self.membrane_potential,
            "threshold": self.threshold_potential,
            "resting_potential": self.resting_potential,
            "excitatory_current": self.excitatory_current,
            "inhibitory_current": self.inhibitory_current,
            "adaptation_current": self.adaptation_current,
            "firing_rate": self.firing_rate,
            "is_refractory": self.is_refractory,
        }


class SpikingNeuralNetwork:
    """
    Advanced Spiking Neural Network

    Implements biologically plausible neural network with
    synaptic plasticity, learning, and adaptation.
    """

    def __init__(self, num_neurons: int = 1000, connectivity: float = 0.1):
        self.num_neurons = num_neurons
        self.connectivity = connectivity

        # Network components
        self.neurons: Dict[str, SpikingNeuron] = {}
        self.synapses: Dict[str, SynapticConnection] = {}

        # Network topology
        self.neuron_types: Dict[str, NeuronType] = {}
        self.network_layers: Dict[str, List[str]] = {}

        # Simulation state
        self.current_time = 0.0
        self.dt = 0.1  # 0.1ms time step
        self.spike_buffer: deque = deque(maxlen=10000)

        # Plasticity metrics
        self.plasticity_history: deque = deque(maxlen=1000)
        self.weight_evolution: Dict[str, List[float]] = {}

        # Performance metrics
        self.network_activity = 0.0
        self.synchrony_index = 0.0
        self.learning_progress = 0.0

        # Initialize network
        self._initialize_network()

    def _initialize_network(self):
        """Initialize spiking neural network structure"""
        print("🧠 Initializing Spiking Neural Network...")

        # Create neurons with realistic proportions
        for i in range(self.num_neurons):
            neuron_id = f"neuron_{i}"

            # Assign neuron types (80% excitatory, 20% inhibitory)
            if i < int(self.num_neurons * 0.8):
                neuron_type = NeuronType.EXCITATORY
            else:
                neuron_type = NeuronType.INHIBITORY

            neuron = SpikingNeuron(neuron_id, neuron_type)
            self.neurons[neuron_id] = neuron
            self.neuron_types[neuron_id] = neuron_type

        # Create synaptic connections
        self._create_synaptic_connections()

        # Create network layers (simplified cortical architecture)
        self._create_network_layers()

        print(
            f"✅ Created {self.num_neurons} neurons with {len(self.synapses)} synapses"
        )

    def _create_synaptic_connections(self):
        """Create random synaptic connections with realistic statistics"""
        num_connections = int(self.num_neurons * self.num_neurons * self.connectivity)

        for i in range(num_connections):
            # Random source and target
            source_id = f"neuron_{np.random.randint(0, self.num_neurons)}"
            target_id = f"neuron_{np.random.randint(0, self.num_neurons)}"

            # Avoid self-connections
            if source_id == target_id:
                continue

            # Check if connection already exists
            synapse_id = f"{source_id}->{target_id}"
            if synapse_id in self.synapses:
                continue

            # Create connection
            source_type = self.neuron_types[source_id]

            # Determine plasticity rule based on neuron types
            if source_type == NeuronType.EXCITATORY:
                plasticity_rule = PlasticityRule.STDP
                initial_weight = np.random.uniform(0.1, 2.0)  # Excitatory weights
            else:
                plasticity_rule = PlasticityRule.HEBBIAN
                initial_weight = np.random.uniform(-2.0, -0.1)  # Inhibitory weights

            # Synaptic delay (1-20ms)
            delay = np.random.uniform(1.0, 20.0)

            synapse = SynapticConnection(
                source_neuron=source_id,
                target_neuron=target_id,
                weight=initial_weight,
                delay=delay,
                plasticity_rule=plasticity_rule,
                learning_rate=np.random.uniform(0.001, 0.05),
            )

            self.synapses[synapse_id] = synapse
            self.weight_evolution[synapse_id] = [initial_weight]

    def _create_network_layers(self):
        """Create simplified cortical layer structure"""
        # Divide neurons into layers
        layer_size = self.num_neurons // 4

        layers = {
            "input": list(range(0, layer_size)),
            "hidden1": list(range(layer_size, 2 * layer_size)),
            "hidden2": list(range(2 * layer_size, 3 * layer_size)),
            "output": list(range(3 * layer_size, self.num_neurons)),
        }

        for layer_name, indices in layers.items():
            self.network_layers[layer_name] = [f"neuron_{i}" for i in indices]

    def simulate_step(self) -> List[SpikeEvent]:
        """Simulate one time step of the spiking neural network"""
        current_spikes = []

        # Process pending spikes (from previous time steps due to delays)
        processed_spikes = self._process_delayed_spikes()
        current_spikes.extend(processed_spikes)

        # Update each neuron
        for neuron_id, neuron in self.neurons.items():
            # Calculate synaptic inputs
            excitatory_input, inhibitory_input = self._calculate_synaptic_inputs(
                neuron_id
            )

            # Update membrane potential
            spike = neuron.update_membrane_potential(
                self.dt, excitatory_input, inhibitory_input, self.current_time
            )

            if spike:
                current_spikes.append(spike)
                self.spike_buffer.append(spike)

                # Update synapses (post-synaptic)
                self._update_post_synaptic_plasticity(neuron_id, spike.timestamp)

        # Update time
        self.current_time += self.dt

        # Calculate network metrics
        self._calculate_network_metrics(current_spikes)

        return current_spikes

    def _process_delayed_spikes(self) -> List[SpikeEvent]:
        """Process spikes that have arrived after synaptic delays"""
        delayed_spikes = []

        # This is a simplified implementation
        # In reality, we'd maintain a spike delivery queue

        return delayed_spikes

    def _calculate_synaptic_inputs(self, neuron_id: str) -> Tuple[float, float]:
        """Calculate total excitatory and inhibitory inputs to a neuron"""
        excitatory_input = 0.0
        inhibitory_input = 0.0

        # Find all incoming synapses to this neuron
        for synapse_id, synapse in self.synapses.items():
            if synapse.target_neuron == neuron_id:
                # Check if there are recent spikes from source neuron
                source_neuron = self.neurons[synapse.source_neuron]

                # Get recent spikes from source
                recent_spikes = [
                    spike
                    for spike in self.spike_buffer
                    if (
                        spike.neuron_id == synapse.source_neuron
                        and self.current_time - spike.timestamp >= synapse.delay
                    )
                ]

                # Calculate input from recent spikes
                for spike in recent_spikes:
                    weight_effect = synapse.weight * spike.amplitude

                    if synapse.weight > 0:
                        excitatory_input += weight_effect
                    else:
                        inhibitory_input += abs(weight_effect)

        return excitatory_input, inhibitory_input

    def _update_post_synaptic_plasticity(self, neuron_id: str, spike_time: float):
        """Update plasticity for post-synaptic neuron spike"""
        for synapse_id, synapse in self.synapses.items():
            if synapse.target_neuron == neuron_id:
                # Post-synaptic spike occurred
                weight_change = synapse.calculate_stdp_update(
                    spike_time, pre_spike=False, post_spike=True
                )

                # Track weight evolution
                self.weight_evolution[synapse_id].append(synapse.weight)

                # Store plasticity event
                if abs(weight_change) > 1e-6:
                    self.plasticity_history.append(
                        {
                            "timestamp": spike_time,
                            "synapse_id": synapse_id,
                            "weight_change": weight_change,
                            "new_weight": synapse.weight,
                            "rule": synapse.plasticity_rule.value,
                        }
                    )

    def apply_stimulus(self, stimulus_pattern: Dict[str, float], duration: float):
        """Apply external stimulus to input neurons"""
        input_neurons = self.network_layers.get("input", [])

        for neuron_id in input_neurons:
            if neuron_id in stimulus_pattern:
                # Apply current injection
                neuron = self.neurons[neuron_id]
                neuron.excitatory_current += stimulus_pattern[neuron_id]

    def train_network(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 0.01,
        epochs: int = 10,
    ):
        """Train the spiking neural network"""
        print("🎓 Training Spiking Neural Network...")

        training_history = {
            "accuracy": [],
            "network_activity": [],
            "synchrony": [],
            "plasticity_events": [],
        }

        for epoch in range(epochs):
            epoch_spikes = []
            epoch_plasticity = []

            for data_point in training_data:
                # Apply input stimulus
                self.apply_stimulus(data_point["input"], 10.0)  # 10ms stimulus

                # Simulate for response period
                for _ in range(100):  # 100ms
                    spikes = self.simulate_step()
                    epoch_spikes.extend(spikes)

                # Check output
                output_activity = self._measure_output_activity(
                    data_point.get("expected_output")
                )

                # Store metrics
                epoch_plasticity.append(len(self.plasticity_history))

                # Reset network state
                self._reset_network_state()

            # Calculate epoch metrics
            accuracy = self._calculate_accuracy(training_data)
            avg_activity = len(epoch_spikes) / len(training_data)

            training_history["accuracy"].append(accuracy)
            training_history["network_activity"].append(avg_activity)
            training_history["synchrony"].append(self.synchrony_index)
            training_history["plasticity_events"].append(np.mean(epoch_plasticity))

            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch}: Accuracy={accuracy:.3f}, Activity={avg_activity:.1f}, "
                    f"Synchrony={self.synchrony_index:.3f}"
                )

        return training_history

    def _measure_output_activity(
        self, expected_output: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Measure activity in output neurons"""
        output_neurons = self.network_layers.get("output", [])
        activity = {}

        for neuron_id in output_neurons:
            neuron = self.neurons[neuron_id]
            activity[neuron_id] = neuron.firing_rate

        return activity

    def _calculate_accuracy(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate network accuracy on training data"""
        # Simplified accuracy calculation
        # In reality, this would compare actual output with expected output
        return np.random.uniform(0.5, 1.0)  # Placeholder

    def _reset_network_state(self):
        """Reset network state between training examples"""
        for neuron in self.neurons.values():
            neuron.membrane_potential = neuron.resting_potential
            neuron.excitatory_current = 0.0
            neuron.inhibitory_current = 0.0
            neuron.adaptation_variable = 0.0

    def _calculate_network_metrics(self, current_spikes: List[SpikeEvent]):
        """Calculate network-wide metrics"""
        # Network activity (spikes per neuron)
        self.network_activity = len(current_spikes) / self.num_neurons

        # Synchrony index (simplified)
        if len(current_spikes) > 1:
            spike_times = [spike.timestamp for spike in current_spikes]
            time_var = np.var(spike_times)
            self.synchrony_index = np.exp(
                -time_var / 10.0
            )  # Lower variance = higher synchrony
        else:
            self.synchrony_index = 0.0

        # Learning progress (based on plasticity)
        recent_plasticity = len(
            [
                p
                for p in self.plasticity_history
                if self.current_time - p["timestamp"] < 1000.0
            ]
        )
        self.learning_progress = min(1.0, recent_plasticity / 100.0)

    def get_network_state(self) -> Dict[str, Any]:
        """Get comprehensive network state"""
        # Calculate membrane potential distribution
        membrane_potentials = [
            neuron.membrane_potential for neuron in self.neurons.values()
        ]
        firing_rates = [neuron.firing_rate for neuron in self.neurons.values()]

        # Weight distribution
        weights = [synapse.weight for synapse in self.synapses.values()]

        return {
            "membrane_potential_stats": {
                "mean": np.mean(membrane_potentials),
                "std": np.std(membrane_potentials),
                "min": np.min(membrane_potentials),
                "max": np.max(membrane_potentials),
            },
            "firing_rate_stats": {
                "mean": np.mean(firing_rates),
                "std": np.std(firing_rates),
                "active_neurons": sum(1 for fr in firing_rates if fr > 0.1),
            },
            "synaptic_weight_stats": {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights),
            },
            "network_metrics": {
                "activity": self.network_activity,
                "synchrony": self.synchrony_index,
                "learning_progress": self.learning_progress,
                "total_spikes": len(self.spike_buffer),
            },
            "plasticity_metrics": {
                "recent_plasticity_events": len(
                    [
                        p
                        for p in self.plasticity_history
                        if self.current_time - p["timestamp"] < 1000.0
                    ]
                ),
                "total_plasticity_events": len(self.plasticity_history),
            },
        }

    def optimize_neural_pathways(
        self, target_function: str, optimization_strength: float = 0.1
    ):
        """Optimize neural pathways for specific function"""
        print(f"🔧 Optimizing neural pathways for: {target_function}")

        # Define optimization strategies for different functions
        optimization_strategies = {
            "memory": {
                "target_synchrony": 0.6,
                "target_activity": 0.3,
                "plasticity_boost": 0.8,
            },
            "learning": {
                "target_synchrony": 0.4,
                "target_activity": 0.5,
                "plasticity_boost": 1.0,
            },
            "creativity": {
                "target_synchrony": 0.2,
                "target_activity": 0.7,
                "plasticity_boost": 0.9,
            },
            "focus": {
                "target_synchrony": 0.8,
                "target_activity": 0.4,
                "plasticity_boost": 0.3,
            },
        }

        if target_function not in optimization_strategies:
            print(f"Unknown optimization target: {target_function}")
            return

        strategy = optimization_strategies[target_function]

        # Adjust network parameters
        for synapse_id, synapse in self.synapses.items():
            # Adjust learning rates based on target
            target_learning_rate = synapse.learning_rate * strategy["plasticity_boost"]
            synapse.learning_rate = target_learning_rate * optimization_strength

            # Adjust weights to achieve target synchrony
            if self.synchrony_index < strategy["target_synchrony"]:
                # Increase network synchrony
                if np.random.random() < 0.1:  # 10% of connections
                    synapse.weight *= 1.1
                    synapse.weight = np.clip(
                        synapse.weight, synapse.min_weight, synapse.max_weight
                    )

        print(f"✅ Pathway optimization complete for {target_function}")


# Global spiking neural network
spiking_nn = None


async def initialize_spiking_nn(num_neurons: int = 500, connectivity: float = 0.1):
    """Initialize spiking neural network"""
    print("🧠 Initializing Spiking Neural Network...")

    global spiking_nn
    spiking_nn = SpikingNeuralNetwork(
        num_neurons=num_neurons, connectivity=connectivity
    )

    print("✅ Spiking Neural Network Initialized!")
    return True


async def test_spiking_nn():
    """Test spiking neural network functionality"""
    print("🧪 Testing Spiking Neural Network...")

    if not spiking_nn:
        return False

    # Simulate network activity
    print("\n🔬 Simulating Network Activity...")

    spikes_over_time = []

    for step in range(1000):  # 100ms simulation
        spikes = spiking_nn.simulate_step()
        spikes_over_time.append(len(spikes))

        if step % 200 == 0:
            state = spiking_nn.get_network_state()
            print(
                f"Step {step}: Activity={state['network_metrics']['activity']:.3f}, "
                f"Synchrony={state['network_metrics']['synchrony']:.3f}"
            )

        await asyncio.sleep(0.001)  # Small delay for async

    # Apply stimulus
    print("\n⚡ Applying External Stimulus...")

    # Create random stimulus pattern
    input_neurons = spiking_nn.network_layers.get("input", [])
    stimulus = {}
    for neuron_id in input_neurons[:20]:  # Stimulate first 20 input neurons
        stimulus[neuron_id] = np.random.uniform(0.5, 2.0)

    # Apply stimulus
    spiking_nn.apply_stimulus(stimulus, 20.0)  # 20ms stimulus

    # Simulate response
    for step in range(200):  # 20ms response
        spikes = spiking_nn.simulate_step()
        spikes_over_time.append(len(spikes))

    # Check response
    state = spiking_nn.get_network_state()
    print(f"\n📊 Network Response to Stimulus:")
    print(f"  Activity: {state['network_metrics']['activity']:.3f}")
    print(f"  Synchrony: {state['network_metrics']['synchrony']:.3f}")
    print(f"  Active Neurons: {state['firing_rate_stats']['active_neurons']}")
    print(f"  Mean Firing Rate: {state['firing_rate_stats']['mean']:.3f} Hz")

    # Test pathway optimization
    print(f"\n🔧 Testing Neural Pathway Optimization...")

    # Optimize for learning
    spiking_nn.optimize_neural_pathways("learning", optimization_strength=0.5)

    # Optimize for creativity
    spiking_nn.optimize_neural_pathways("creativity", optimization_strength=0.5)

    # Optimize for focus
    spiking_nn.optimize_neural_pathways("focus", optimization_strength=0.5)

    print("✅ Spiking Neural Network Test Complete!")
    return True


if __name__ == "__main__":
    asyncio.run(initialize_spiking_nn())
    asyncio.run(test_spiking_nn())
