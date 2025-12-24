"""
Single bouncing dot animation with physics simulation.
"""
from __future__ import annotations

from pathlib import Path

from manim import (
    BLACK,
    Circle,
    Dot,
    Scene,
    ValueTracker,
    VMobject,
    always_redraw,
    linear,
)
import numpy as np
from numpy.typing import NDArray

from .audio_mixer import AudioMixer
from .physics import PhysicsSimulation, create_dot_states_from_config

from .config import AnimationConfig


class BouncingDot(Scene):
    """
    Bouncing dot animation inside a circular boundary with physics simulation.
    
    Features:
        - Gravity-based physics with collision detection
        - Energy loss on bounce (configurable damping)
        - Optional trail effect
        - Ambient and hit sound effects
    
    All settings are loaded from AnimationConfig.
    
    Usage:
        config = AnimationConfig()
        config.override({
            "DAMPING": 0.95,
            "DOT_COLOR": "RED",
        })
        scene = BouncingDot(config=config)
    """
    
    def __init__(self, config: AnimationConfig, **kwargs) -> None:
        """
        Initialize the bouncing dot scene.
        
        Args:
            config: AnimationConfig instance (required).
            **kwargs: Additional arguments passed to Scene
        """
        super().__init__(**kwargs)
        self.config = config
    
    def construct(self) -> None:
        """Build and render the bouncing dot animation."""
        cfg = self.config
        
        # Validate starting position
        self._validate_start_position()
        
        # Setup scene
        self._setup_scene()
        
        # Create dot configuration from single-dot settings
        dot_config = cfg.get_default_dot_config()
        dot_config["start_pos"] = np.array(dot_config["start_pos"])
        dot_config["initial_velocity"] = np.array(dot_config["initial_velocity"])
        
        # Run physics simulation
        dot_states = create_dot_states_from_config([dot_config])
        simulation = PhysicsSimulation(cfg, dot_states, debug=cfg.DEBUG)
        positions_list, bounce_events = simulation.simulate()
        positions = positions_list[0]  # Single dot
        
        # Generate audio
        total_duration = (len(positions) - 1) * cfg.SIMULATION_DT
        audio_dir = Path(cfg.audio_output_dir) if cfg.audio_output_dir else None
        mixer = AudioMixer(cfg, bounce_events, audio_dir=audio_dir, debug=cfg.DEBUG)
        mixed_path = mixer.generate_mixed_audio(total_duration)
        
        # Animate
        self._animate(positions, mixed_path, total_duration)
    
    def _validate_start_position(self) -> None:
        """Validate that the starting position is within the circle boundary."""
        cfg = self.config
        distance_from_center = float(np.linalg.norm(cfg.DOT_START_POS[:2] - cfg.CIRCLE_CENTER[:2]))
        max_allowed_distance = cfg.CIRCLE_RADIUS - cfg.DOT_RADIUS
        
        if distance_from_center > max_allowed_distance:
            raise ValueError(
                f"DOT_START_POS {cfg.DOT_START_POS[:2]} is outside the circle boundary. "
                f"Distance from center: {distance_from_center:.3f}, "
                f"Max allowed (CIRCLE_RADIUS - DOT_RADIUS): {max_allowed_distance:.3f}"
            )
    
    def _setup_scene(self) -> None:
        """Setup the scene with background and boundary circle."""
        cfg = self.config
        self.camera.background_color = BLACK
        
        # Create boundary circle
        circle = Circle(radius=cfg.CIRCLE_RADIUS, color=cfg.CIRCLE_COLOR, stroke_width=cfg.CIRCLE_STROKE_WIDTH)
        self.add(circle)
    
    def _animate(
        self,
        positions: list[NDArray[np.floating]],
        audio_path: Path,
        total_duration: float,
    ) -> None:
        """
        Animate the dot movement with optional trail.
        
        Args:
            positions: List of position arrays from simulation
            audio_path: Path to the mixed audio file
            total_duration: Total animation duration in seconds
        """
        cfg = self.config
        
        # Create initial dot (will be replaced with animated version)
        dot = Dot(point=positions[0], radius=cfg.DOT_RADIUS, color=cfg.DOT_COLOR)
        self.add(dot)
        
        # Animation time tracker
        time_tracker = ValueTracker(0)
        
        if cfg.DEBUG:
            print(f"Animation: {total_duration:.2f}s, {len(positions)} frames, trail={cfg.ENABLE_TRAIL}")
        
        def get_dot_position() -> Dot:
            """Get dot at current animation time."""
            index = min(int(time_tracker.get_value() / cfg.SIMULATION_DT), len(positions) - 1)
            return Dot(point=positions[index], radius=cfg.DOT_RADIUS, color=cfg.DOT_COLOR)
        
        def get_trail_path() -> VMobject:
            """Get trail path at current animation time."""
            index = min(int(time_tracker.get_value() / cfg.SIMULATION_DT), len(positions) - 1)
            trail = VMobject(
                stroke_color=cfg.TRAIL_COLOR,
                stroke_width=cfg.TRAIL_WIDTH,
                stroke_opacity=cfg.TRAIL_OPACITY,
            )
            
            if index >= 2:
                trail_points = positions[:index:cfg.TRAIL_SAMPLE_INTERVAL]
                if len(trail_points) > 1:
                    trail.set_points_as_corners(trail_points)
            
            return trail
        
        # Replace static dot with animated version
        self.remove(dot)
        animated_dot = always_redraw(get_dot_position)
        
        if cfg.ENABLE_TRAIL:
            self.add(always_redraw(get_trail_path), animated_dot)
        else:
            self.add(animated_dot)
        
        # Play animation with audio
        self.add_sound(str(audio_path))
        self.play(
            time_tracker.animate.set_value(total_duration),
            run_time=total_duration,
            rate_func=linear,
        )
        self.wait(1)
