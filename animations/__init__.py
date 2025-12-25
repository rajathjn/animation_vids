# This file makes the animations folder a Python package

# Make key classes easily importable
from .audio_mixer import AudioMixer
from .bouncing_dot import BouncingDot
from .bouncing_dots import BouncingDots
from .config import AnimationConfig
from .main import render_animation
from .physics import BounceEvent, DotState, PhysicsSimulation, SpatialHashGrid

__all__ = [
    "AnimationConfig",
    "BouncingDot",
    "BouncingDots",
    "render_animation",
    "PhysicsSimulation",
    "BounceEvent",
    "DotState",
    "SpatialHashGrid",
    "AudioMixer",
]
