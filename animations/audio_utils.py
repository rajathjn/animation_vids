"""
Audio generation utilities for animations.
"""

import numpy as np
from numpy.typing import NDArray


def generate_hit_sound(
    frequency: float = 440, duration: float = 0.1, sample_rate: int = 44100, volume: float = 1.0
) -> NDArray[np.int16]:
    """
    Generate a simple hit sound effect with exponential decay.

    The sound mimics a physical impact by combining multiple frequency harmonics
    that decay rapidly, creating a realistic "thud" or "bounce" sound.

    Args:
        frequency: Base frequency in Hz (default 440 = A4 note)
        duration: Length of the sound in seconds
        sample_rate: Audio sample rate (44100 Hz is CD quality)
        volume: Amplitude multiplier (0.0 to 1.0)

    Returns:
        numpy array of int16 audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Combine two sine waves at different frequencies with exponential decay
    wave = (
        np.sin(2 * np.pi * frequency * t) * np.exp(-t * 15)
        + np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 20) * 0.5
    )

    # Scale to int16 range for WAV format
    return (wave * volume * 32767).astype(np.int16)


def generate_ambient_sound(
    duration: float = 12, sample_rate: int = 44100, volume: float = 0.3
) -> NDArray[np.int16]:
    """
    Generate a simple ambient/ASMR-like background sound.

    Creates a calming background atmosphere using pink noise with gentle
    low-frequency oscillations.

    Args:
        duration: Length of ambient sound in seconds
        sample_rate: Audio sample rate
        volume: Amplitude multiplier (0.0 to 1.0)

    Returns:
        numpy array of int16 audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Pink noise with low-frequency oscillations
    noise = np.random.normal(0, 0.05, len(t))

    # Add low-frequency sine waves for a soothing effect
    ambient = noise + 0.02 * np.sin(2 * np.pi * 0.1 * t) + 0.015 * np.sin(2 * np.pi * 0.23 * t)

    return (ambient * 32767 * volume).astype(np.int16)
