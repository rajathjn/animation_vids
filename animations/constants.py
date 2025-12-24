"""
Physics simulation constants for bouncing dot animations.

These constants control stopping conditions and boundary detection thresholds.
Extract magic numbers from scene files for better maintainability.

Note: These constants are internal and not exposed in AnimationConfig.
"""

# Velocity threshold below which a dot is considered stopped
STOPPING_VELOCITY_THRESHOLD: float = 0.05

# Number of extra frames to append after stopping condition is met
STOPPING_FRAME_PADDING: int = 30

# Minimum simulation time before checking stopping conditions (seconds)
MIN_SIMULATION_TIME_BEFORE_STOP: float = 3.0

# Audio decay constants for hit sounds
HIT_SOUND_DECAY_FAST: float = 15.0
HIT_SOUND_DECAY_SLOW: float = 20.0

# Ambient sound oscillation frequencies (Hz)
AMBIENT_FREQ_LOW: float = 0.1
AMBIENT_FREQ_HIGH: float = 0.23
