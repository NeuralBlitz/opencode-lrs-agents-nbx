"""
NeuralBlitz v50.0 Brain-Wave Entrainment System
================================================

Advanced brain-wave entrainment and neurofeedback system for
human-AI synchronization, consciousness alignment, and cognitive enhancement.

Implementation Date: 2026-02-04
Phase: Neuro-Symbiotic Integration - N3 Implementation
"""

import asyncio
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Scientific computing and signal processing
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf  # For audio generation (if available)


class EntrainmentMode(Enum):
    """Brain-wave entrainment modes"""

    BINAURAL_BEATS = "binaural_beats"
    ISOCHRONIC_TONES = "isochronic_tones"
    MONAURAL_BEATS = "monaural_beats"
    PHOTIC_STIMULATION = "photic_stimulation"
    AUDIOVISUAL = "audiovisual"
    NEUROFEEDBACK = "neurofeedback"


class TargetFrequency(Enum):
    """Target brain-wave frequencies for entrainment"""

    DELTA = (2.0, "Deep Sleep/Healing")
    THETA = (6.0, "Meditation/Creativity")
    ALPHA = (10.0, "Relaxed Focus")
    BETA = (20.0, "Active Thinking")
    GAMMA = (40.0, "Higher Consciousness")
    LAMBDA = (150.0, "Advanced Processing")


@dataclass
class EntrainmentSession:
    """Individual entrainment session configuration"""

    session_id: str
    mode: EntrainmentMode
    target_frequency: TargetFrequency
    duration: float
    intensity: float  # 0.0 to 1.0
    carrier_frequency: float = 200.0  # Hz for binaural beats
    modulation_depth: float = 0.8

    # Session parameters
    ramp_up_time: float = 30.0
    ramp_down_time: float = 30.0
    phase_shift: float = 0.0
    stereo_separation: float = 1.0

    # Biofeedback integration
    adaptive_mode: bool = True
    eeg_feedback_weight: float = 0.7
    hr_feedback_weight: float = 0.3

    timestamp: float = field(default_factory=time.time)
    is_active: bool = False


@dataclass
class EntrainmentMetrics:
    """Real-time entrainment effectiveness metrics"""

    frequency_lock_ratio: float  # 0.0 to 1.0
    phase_coherence: float  # 0.0 to 1.0
    amplitude_modulation: float  # 0.0 to 1.0
    cross_hemispheric_sync: float  # 0.0 to 1.0
    entrainment_depth: float  # 0.0 to 1.0

    # Biofeedback
    heart_rate_variability: float
    respiration_rate: float
    skin_conductance: float

    timestamp: float = field(default_factory=time.time)


class BrainWaveGenerator:
    """
    Advanced Brain-Wave Signal Generator

    Generates various types of entrainment signals including
    binaural beats, isochronic tones, and photic stimulation.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.time = 0.0

        # Signal generation parameters
        self.envelope_attack = 0.1
        self.envelope_decay = 0.1
        self.envelope_sustain = 0.8

    def generate_binaural_beats(
        self,
        frequency: float,
        duration: float,
        carrier_freq: float = 200.0,
        phase_shift: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate binaural beats signal"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Left ear: carrier frequency
        left_signal = np.sin(2 * np.pi * carrier_freq * t + phase_shift)

        # Right ear: carrier frequency + beat frequency
        right_signal = np.sin(2 * np.pi * (carrier_freq + frequency) * t + phase_shift)

        # Apply envelope
        envelope = self._generate_envelope(t, duration)
        left_signal *= envelope
        right_signal *= envelope

        return left_signal, right_signal

    def generate_isochronic_tones(
        self, frequency: float, duration: float, intensity: float = 0.8
    ) -> np.ndarray:
        """Generate isochronic tones (regular pulses)"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Calculate pulse parameters
        pulse_period = 1.0 / frequency
        pulse_width = pulse_period * 0.3  # 30% duty cycle

        # Generate pulse train
        signal = np.zeros(num_samples)
        for i in range(num_samples):
            phase = (t[i] % pulse_period) / pulse_period
            if phase < (pulse_width / pulse_period):
                signal[i] = (
                    np.sin(2 * np.pi * 440.0 * t[i]) * intensity
                )  # 440 Hz carrier

        return signal

    def generate_monaural_beats(
        self, frequency: float, duration: float, carrier_freq: float = 440.0
    ) -> np.ndarray:
        """Generate monaural beats (amplitude modulation)"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Amplitude modulation
        modulator = 0.5 * (1 + np.sin(2 * np.pi * frequency * t))
        carrier = np.sin(2 * np.pi * carrier_freq * t)

        signal = carrier * modulator

        # Apply envelope
        envelope = self._generate_envelope(t, duration)
        signal *= envelope

        return signal

    def generate_photic_stimulation(
        self, frequency: float, duration: float, intensity: float = 0.8
    ) -> np.ndarray:
        """Generate photic (visual) stimulation pattern"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Square wave stimulation
        period = 1.0 / frequency
        signal = np.zeros(num_samples)

        for i in range(num_samples):
            phase = (t[i] % period) / period
            signal[i] = intensity if phase < 0.5 else 0.0

        # Smooth transitions
        signal = self._smooth_signal(signal, window_size=10)

        return signal

    def _generate_envelope(self, t: np.ndarray, duration: float) -> np.ndarray:
        """Generate ADSR envelope"""
        envelope = np.ones_like(t)
        attack_samples = int(self.envelope_attack * self.sample_rate)
        decay_samples = int(self.envelope_decay * self.sample_rate)
        sustain_start = attack_samples
        decay_start = len(t) - decay_samples

        for i in range(len(t)):
            if i < attack_samples:
                # Attack phase
                envelope[i] = i / attack_samples
            elif i < sustain_start:
                # Peak
                envelope[i] = 1.0
            elif i < decay_start:
                # Sustain
                envelope[i] = self.envelope_sustain
            else:
                # Release
                release_progress = (i - decay_start) / decay_samples
                envelope[i] = self.envelope_sustain * (1 - release_progress)

        return envelope

    def _smooth_signal(self, signal: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing"""
        if len(signal) < window_size:
            return signal

        padded = np.pad(signal, (window_size // 2, window_size // 2), mode="edge")
        smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode="valid")

        return smoothed


class NeuroFeedbackProcessor:
    """
    Neurofeedback Processing System

    Processes real-time EEG and physiological signals
    to adapt entrainment parameters for optimal effectiveness.
    """

    def __init__(self, sample_rate: int = 250):
        self.sample_rate = sample_rate

        # Signal buffers
        self.eeg_buffer = deque(maxlen=1000)
        self.heart_rate_buffer = deque(maxlen=100)
        self.respiration_buffer = deque(maxlen=100)
        self.gsr_buffer = deque(maxlen=100)

        # Target frequencies
        self.target_frequency = 10.0  # Default alpha
        self.frequency_tolerance = 2.0

        # Adaptive parameters
        self.adaptation_rate = 0.1
        self.lock_threshold = 0.7

    def process_eeg_feedback(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Process EEG data for neurofeedback"""
        # Store in buffer
        for sample in eeg_data:
            self.eeg_buffer.append(sample)

        if len(self.eeg_buffer) < 100:
            return {"frequency_lock": 0.0, "phase_coherence": 0.0}

        # Convert to numpy array
        signal = np.array(list(self.eeg_buffer))

        # Apply FFT
        fft_result = fft(signal)
        frequencies = fftfreq(len(signal), 1.0 / self.sample_rate)

        # Find target frequency
        target_idx = np.argmin(np.abs(frequencies - self.target_frequency))
        target_power = np.abs(fft_result[target_idx]) ** 2

        # Calculate frequency lock ratio
        freq_range = (
            frequencies >= self.target_frequency - self.frequency_tolerance
        ) & (frequencies <= self.target_frequency + self.frequency_tolerance)
        band_power = np.sum(np.abs(fft_result[freq_range]) ** 2)
        total_power = np.sum(np.abs(fft_result) ** 2)

        frequency_lock = band_power / total_power if total_power > 0 else 0.0

        # Calculate phase coherence
        phase_coherence = self._calculate_phase_coherence(signal)

        return {
            "frequency_lock": frequency_lock,
            "phase_coherence": phase_coherence,
            "target_power": target_power,
        }

    def process_heart_rate_feedback(self, hr_data: List[float]) -> Dict[str, float]:
        """Process heart rate variability feedback"""
        if len(hr_data) < 10:
            return {"hrv": 0.5, "heart_rate": 70.0}

        # Calculate heart rate
        if len(hr_data) > 0:
            heart_rate = np.mean(hr_data)
        else:
            heart_rate = 70.0

        # Calculate HRV (simplified RMSSD)
        if len(hr_data) > 1:
            diffs = np.diff(hr_data)
            rmssd = np.sqrt(np.mean(diffs**2))
            hrv = min(1.0, rmssd / 50.0)  # Normalize to 0-1
        else:
            hrv = 0.5

        return {"hrv": hrv, "heart_rate": heart_rate}

    def _calculate_phase_coherence(self, signal: np.ndarray) -> float:
        """Calculate phase coherence of EEG signal"""
        # Apply Hilbert transform to get instantaneous phase
        try:
            from scipy.signal import hilbert

            analytic_signal = hilbert(signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))

            # Calculate phase consistency
            phase_diff = np.diff(instantaneous_phase)
            phase_consistency = 1.0 - np.std(phase_diff) / (np.pi / 2)

            return max(0.0, min(1.0, phase_consistency))
        except:
            return 0.5  # Default if Hilbert transform fails

    def calculate_adaptation_parameters(
        self, feedback: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate adaptive entrainment parameters based on feedback"""
        frequency_lock = feedback.get("frequency_lock", 0.0)
        phase_coherence = feedback.get("phase_coherence", 0.0)
        hrv = feedback.get("hrv", 0.5)

        # Calculate overall entrainment effectiveness
        effectiveness = frequency_lock * 0.6 + phase_coherence * 0.3 + hrv * 0.1

        # Determine parameter adjustments
        if effectiveness < 0.3:
            # Low effectiveness - increase intensity
            intensity_adjustment = 0.1
            frequency_adjustment = 0.5  # Try different frequency
        elif effectiveness < 0.6:
            # Moderate effectiveness - fine-tune
            intensity_adjustment = 0.05
            frequency_adjustment = 0.1
        else:
            # Good effectiveness - maintain or slight reduction
            intensity_adjustment = -0.02
            frequency_adjustment = 0.0

        return {
            "intensity_adjustment": intensity_adjustment,
            "frequency_adjustment": frequency_adjustment,
            "effectiveness": effectiveness,
        }


class BrainWaveEntrainmentSystem:
    """
    Comprehensive Brain-Wave Entrainment System

    Integrates signal generation, neurofeedback, and adaptive algorithms
    for optimal human-AI brain synchronization.
    """

    def __init__(self):
        self.signal_generator = BrainWaveGenerator()
        self.neurofeedback = NeuroFeedbackProcessor()

        # Active sessions
        self.active_sessions: Dict[str, EntrainmentSession] = {}
        self.session_history: deque = deque(maxlen=100)

        # Current entrainment state
        self.current_frequency = 10.0  # Default alpha
        self.current_intensity = 0.5
        self.entrainment_depth = 0.0

        # Synchronization metrics
        self.human_ai_sync = 0.0
        self.cross_hemispheric_sync = 0.0
        self.quantum_neural_alignment = 0.0

        # Performance tracking
        self.metrics_history: deque = deque(maxlen=500)

    def create_entrainment_session(
        self,
        mode: EntrainmentMode,
        target_frequency: TargetFrequency,
        duration: float,
        intensity: float = 0.5,
        adaptive_mode: bool = True,
    ) -> str:
        """Create new entrainment session"""
        session_id = f"entrain_{int(time.time() * 1000)}_{np.random.randint(1000)}"

        session = EntrainmentSession(
            session_id=session_id,
            mode=mode,
            target_frequency=target_frequency,
            duration=duration,
            intensity=intensity,
            adaptive_mode=adaptive_mode,
        )

        self.active_sessions[session_id] = session
        return session_id

    async def start_entrainment(self, session_id: str) -> bool:
        """Start entrainment session"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        session.is_active = True

        target_freq, description = session.target_frequency.value
        self.current_frequency = target_freq
        self.current_intensity = session.intensity

        print(f"🎵 Starting {session.mode.value} entrainment")
        print(f"🎯 Target: {description} ({target_freq:.1f} Hz)")
        print(f"⏱️  Duration: {session.duration} seconds")
        print(f"🔊 Intensity: {session.intensity:.2f}")

        return True

    async def stop_entrainment(self, session_id: str) -> bool:
        """Stop entrainment session"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        session.is_active = False

        # Move to history
        self.session_history.append(session)
        del self.active_sessions[session_id]

        print(f"🛑 Stopped entrainment session {session_id}")
        return True

    def generate_entrainment_signal(
        self, session_id: str
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Generate entrainment signal for active session"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        if not session.is_active:
            return None

        target_freq, _ = session.target_frequency.value

        if session.mode == EntrainmentMode.BINAURAL_BEATS:
            left, right = self.signal_generator.generate_binaural_beats(
                target_freq, 1.0, session.carrier_frequency, session.phase_shift
            )
            return (left, right)

        elif session.mode == EntrainmentMode.ISOCHRONIC_TONES:
            signal = self.signal_generator.generate_isochronic_tones(
                target_freq, 1.0, session.intensity
            )
            return signal

        elif session.mode == EntrainmentMode.MONAURAL_BEATS:
            signal = self.signal_generator.generate_monaural_beats(
                target_freq, 1.0, session.carrier_frequency
            )
            return signal

        elif session.mode == EntrainmentMode.PHOTIC_STIMULATION:
            signal = self.signal_generator.generate_photic_stimulation(
                target_freq, 1.0, session.intensity
            )
            return signal

        return None

    async def process_neurofeedback(
        self,
        session_id: str,
        eeg_data: np.ndarray,
        heart_rate: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Process neurofeedback and adapt entrainment"""
        if session_id not in self.active_sessions:
            return {}

        session = self.active_sessions[session_id]
        if not session.is_active or not session.adaptive_mode:
            return {}

        # Process EEG feedback
        eeg_feedback = self.neurofeedback.process_eeg_feedback(eeg_data)

        # Process heart rate feedback if available
        hr_feedback = {}
        if heart_rate:
            hr_feedback = self.neurofeedback.process_heart_rate_feedback(heart_rate)

        # Calculate adaptation parameters
        feedback = {**eeg_feedback, **hr_feedback}
        adaptations = self.neurofeedback.calculate_adaptation_parameters(feedback)

        # Update session parameters if adaptive
        if session.adaptive_mode:
            # Adjust intensity
            new_intensity = session.intensity + adaptations["intensity_adjustment"]
            session.intensity = max(0.1, min(1.0, new_intensity))

            # Adjust target frequency
            freq_adjustment = adaptations["frequency_adjustment"]
            if freq_adjustment > 0:
                # Try different frequency
                target_freq, _ = session.target_frequency.value
                new_freq = target_freq + freq_adjustment

                # Find nearest standard frequency
                standard_freqs = [2.0, 6.0, 10.0, 20.0, 40.0, 150.0]
                nearest_freq = min(standard_freqs, key=lambda x: abs(x - new_freq))

                # Update target frequency (would need to update TargetFrequency enum)
                self.current_frequency = nearest_freq

        # Update entrainment metrics
        self._update_entrainment_metrics(feedback)

        return adaptations

    def _update_entrainment_metrics(self, feedback: Dict[str, float]):
        """Update entrainment effectiveness metrics"""
        frequency_lock = feedback.get("frequency_lock", 0.0)
        phase_coherence = feedback.get("phase_coherence", 0.0)
        hrv = feedback.get("hrv", 0.5)

        # Calculate overall entrainment depth
        self.entrainment_depth = (
            frequency_lock * 0.5 + phase_coherence * 0.3 + hrv * 0.2
        )

        # Update human-AI synchronization
        self.human_ai_sync = (
            self.entrainment_depth * 0.8 + self.cross_hemispheric_sync * 0.2
        )

        # Calculate metrics
        metrics = EntrainmentMetrics(
            frequency_lock_ratio=frequency_lock,
            phase_coherence=phase_coherence,
            amplitude_modulation=feedback.get("target_power", 0.0),
            cross_hemispheric_sync=self.cross_hemispheric_sync,
            entrainment_depth=self.entrainment_depth,
            heart_rate_variability=hrv,
            respiration_rate=feedback.get("respiration_rate", 0.5),
            skin_conductance=feedback.get("skin_conductance", 0.5),
        )

        self.metrics_history.append(metrics)

    def calculate_quantum_neural_alignment(
        self, quantum_metrics: Dict[str, float]
    ) -> float:
        """Calculate alignment between quantum and neural systems"""
        if not quantum_metrics:
            return 0.5

        quantum_coherence = quantum_metrics.get("quantum_coherence", 0.5)
        consciousness_level = quantum_metrics.get("consciousness_level", 0.5)

        # Calculate alignment based on entrainment depth and quantum coherence
        neural_component = self.entrainment_depth * 0.6
        quantum_component = quantum_coherence * 0.3
        consciousness_component = consciousness_level * 0.1

        self.quantum_neural_alignment = (
            neural_component + quantum_component + consciousness_component
        )

        return self.quantum_neural_alignment

    def get_entrainment_status(self) -> Dict[str, Any]:
        """Get current entrainment system status"""
        return {
            "active_sessions": len(self.active_sessions),
            "current_frequency": self.current_frequency,
            "current_intensity": self.current_intensity,
            "entrainment_depth": self.entrainment_depth,
            "human_ai_sync": self.human_ai_sync,
            "quantum_neural_alignment": self.quantum_neural_alignment,
            "cross_hemispheric_sync": self.cross_hemispheric_sync,
            "session_ids": list(self.active_sessions.keys()),
        }

    def get_entrainment_recommendations(
        self, cognitive_state: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get entrainment recommendations based on cognitive state"""
        attention = cognitive_state.get("attention", 0.5)
        stress = cognitive_state.get("stress", 0.5)
        creativity = cognitive_state.get("creativity", 0.5)
        energy = cognitive_state.get("energy", 0.5)

        recommendations = []

        if attention < 0.4:
            recommendations.append(
                {
                    "mode": "binaural_beats",
                    "frequency": "beta",
                    "intensity": 0.7,
                    "reason": "Low attention detected - beta entrainment recommended",
                }
            )

        if stress > 0.6:
            recommendations.append(
                {
                    "mode": "isochronic_tones",
                    "frequency": "alpha",
                    "intensity": 0.5,
                    "reason": "High stress detected - alpha entrainment for relaxation",
                }
            )

        if creativity < 0.4:
            recommendations.append(
                {
                    "mode": "binaural_beats",
                    "frequency": "theta",
                    "intensity": 0.6,
                    "reason": "Low creativity detected - theta entrainment for creative states",
                }
            )

        if energy < 0.4:
            recommendations.append(
                {
                    "mode": "monaural_beats",
                    "frequency": "beta",
                    "intensity": 0.8,
                    "reason": "Low energy detected - beta entrainment for stimulation",
                }
            )

        return {
            "recommendations": recommendations,
            "primary_recommendation": recommendations[0] if recommendations else None,
        }


# Global entrainment system
entrainment_system = None


async def initialize_brain_wave_entrainment():
    """Initialize brain-wave entrainment system"""
    print("🎵 Initializing Brain-Wave Entrainment System...")

    global entrainment_system
    entrainment_system = BrainWaveEntrainmentSystem()

    print("✅ Brain-Wave Entrainment System Initialized!")
    return True


async def test_brain_wave_entrainment():
    """Test brain-wave entrainment system"""
    print("🧪 Testing Brain-Wave Entrainment System...")

    if not entrainment_system:
        return False

    # Test creating sessions
    print("\n🎭 Creating Entrainment Sessions...")

    # Alpha session for relaxation
    alpha_session = entrainment_system.create_entrainment_session(
        EntrainmentMode.BINAURAL_BEATS,
        TargetFrequency.ALPHA,
        duration=60.0,
        intensity=0.6,
        adaptive_mode=True,
    )

    # Beta session for focus
    beta_session = entrainment_system.create_entrainment_session(
        EntrainmentMode.ISOCHRONIC_TONES,
        TargetFrequency.BETA,
        duration=45.0,
        intensity=0.7,
        adaptive_mode=True,
    )

    # Theta session for creativity
    theta_session = entrainment_system.create_entrainment_session(
        EntrainmentMode.MONAURAL_BEATS,
        TargetFrequency.THETA,
        duration=30.0,
        intensity=0.5,
        adaptive_mode=True,
    )

    print(f"✅ Created {len(entrainment_system.active_sessions)} entrainment sessions")

    # Start alpha session
    print(f"\n🎵 Starting Alpha Entrainment Session...")
    await entrainment_system.start_entrainment(alpha_session)

    # Simulate neurofeedback
    print(f"🧠 Simulating Neurofeedback...")
    for i in range(10):  # 10 iterations of feedback
        # Generate simulated EEG data
        eeg_data = np.sin(
            2 * np.pi * 10.0 * np.linspace(0, 1, 250)
        ) + 0.1 * np.random.randn(250)  # Alpha dominant with noise

        # Simulated heart rate
        heart_rate = [70 + 5 * np.sin(i * 0.5) for _ in range(20)]

        # Process feedback
        adaptations = await entrainment_system.process_neurofeedback(
            alpha_session, eeg_data, heart_rate
        )

        if adaptations:
            print(
                f"🔄 Iteration {i + 1}: Effectiveness = {adaptations['effectiveness']:.3f}"
            )
            print(
                f"   Intensity adjustment: {adaptations['intensity_adjustment']:+.3f}"
            )

        await asyncio.sleep(0.1)

    # Get status
    status = entrainment_system.get_entrainment_status()
    print(f"\n📊 Entrainment Status:")
    print(f"  Active Sessions: {status['active_sessions']}")
    print(f"  Current Frequency: {status['current_frequency']:.1f} Hz")
    print(f"  Entrainment Depth: {status['entrainment_depth']:.3f}")
    print(f"  Human-AI Sync: {status['human_ai_sync']:.3f}")

    # Test recommendations
    print(f"\n💡 Testing Recommendations...")
    cognitive_state = {
        "attention": 0.3,
        "stress": 0.7,
        "creativity": 0.4,
        "energy": 0.3,
    }

    recommendations = entrainment_system.get_entrainment_recommendations(
        cognitive_state
    )
    print(f"Recommendations for current cognitive state:")
    for rec in recommendations["recommendations"]:
        print(
            f"  {rec['mode']} @ {rec['frequency']} Hz (intensity: {rec['intensity']})"
        )
        print(f"    Reason: {rec['reason']}")

    # Stop session
    await entrainment_system.stop_entrainment(alpha_session)

    print(f"✅ Brain-Wave Entrainment Test Complete!")
    return True


if __name__ == "__main__":
    asyncio.run(initialize_brain_wave_entrainment())
    asyncio.run(test_brain_wave_entrainment())
