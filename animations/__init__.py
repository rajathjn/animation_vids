# This file makes the animations folder a Python package

# Make key classes easily importable
from .config import AnimationConfig
from .bouncing_dot import BouncingDot
from .bouncing_dots import BouncingDots
from .main import render_animation
from .physics import PhysicsSimulation, BounceEvent, DotState, SpatialHashGrid
from .audio_mixer import AudioMixer

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
