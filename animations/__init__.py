# This file makes the animations folder a Python package

# Make key classes easily importable
from .config import AnimationConfig
from .bouncing_dot import BouncingDot
from .bouncing_dots import BouncingDots
from .main import render_animation, render_batch

__all__ = [
    'AnimationConfig',
    'BouncingDot',
    'BouncingDots',
    'render_animation',
    'render_batch',
]
