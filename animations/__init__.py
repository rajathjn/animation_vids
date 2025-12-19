# This file makes the animations folder a Python package

# Make key classes easily importable
from .config import AnimationConfig
from .bouncing_dot import BouncingDot
from .bouncing_dots import BouncingDots
from .main import generate_animation, generate_multiple

__all__ = [
    'AnimationConfig',
    'BouncingDot',
    'BouncingDots',
    'generate_animation',
    'generate_multiple',
]
