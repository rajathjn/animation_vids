"""
Multiple bouncing dots animation with physics simulation and dot-to-dot collision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

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
from .physics import PhysicsSimulation, DotState, create_dot_states_from_config

from .config import AnimationConfig


class BouncingDots(Scene):
    """
    Multiple bouncing dots animation inside a circular boundary with physics simulation.
    
    Features:
        - Gravity-based physics with collision detection
        - Dot-to-dot collision handling (optimized with spatial hashing)
        - Energy loss on bounce (configurable damping per dot)
        - Optional trail effect per dot
        - Ambient and hit sound effects
    
    Dots configuration is loaded from AnimationConfig.DOTS_JSON.
    If DOTS_JSON is empty and USE_SINGLE_DOT_DEFAULTS is True, uses single dot
    from [animation] section values. Otherwise raises ValueError.
    
    Usage:
        config = AnimationConfig()
        config.override({
            "DOTS_JSON": [
                {"color": "RED", "radius": 0.25, "start_pos": [0, 0, 0],
                 "initial_velocity": [0.5, -5, 0], "damping": 0.95},
            ]
        })
        scene = BouncingDots(config=config)
    """
    
    def __init__(self, config: AnimationConfig, **kwargs) -> None:
        """
        Initialize the bouncing dots scene.
        
        Args:
            config: AnimationConfig instance (required).
            **kwargs: Additional arguments passed to Scene
        
        Raises:
            ValueError: If DOTS_JSON is empty and USE_SINGLE_DOT_DEFAULTS is False
        """
        super().__init__(**kwargs)
        self.config = config
        
        # Load dots configuration from config
        self.dots_config = self._load_dots_config()
    
    def _load_dots_config(self) -> list[dict[str, Any]]:
        """
        Load dots configuration from AnimationConfig.
        
        Returns list of dot configs, each with: color, radius, start_pos, 
        initial_velocity, damping.
        
        If USE_SINGLE_DOT_DEFAULTS is True, missing keys in individual dots
        will default to values from [animation] section.
        
        Raises:
            ValueError: If no dots configured and USE_SINGLE_DOT_DEFAULTS is False
        """
        dots_json = self.config.DOTS_JSON
        
        # Get default dot config if USE_SINGLE_DOT_DEFAULTS is True
        default_dot = None
        if self.config.USE_SINGLE_DOT_DEFAULTS:
            default_dot = self.config.get_default_dot_config()
        
        if dots_json:
            # Process each dot, filling in missing keys with defaults if enabled
            dots: list[dict[str, Any]] = []
            for dot in dots_json:
                # Start with defaults if available, then override with specified values
                if default_dot:
                    dot_copy = default_dot.copy()
                    dot_copy.update(dot)
                else:
                    dot_copy = dot.copy()
                
                # Convert list values to numpy arrays
                if "start_pos" in dot_copy:
                    dot_copy["start_pos"] = np.array(dot_copy["start_pos"])
                if "initial_velocity" in dot_copy:
                    dot_copy["initial_velocity"] = np.array(dot_copy["initial_velocity"])
                dots.append(dot_copy)
            return dots
        
        # DOTS_JSON is empty - check if we should use single dot defaults
        if default_dot:
            default_dot["start_pos"] = np.array(default_dot["start_pos"])
            default_dot["initial_velocity"] = np.array(default_dot["initial_velocity"])
            return [default_dot]
        
        # No dots and no fallback
        raise ValueError(
            "DOTS_JSON is empty and USE_SINGLE_DOT_DEFAULTS is false. "
            "No dots to animate. Either provide DOTS_JSON or set "
            "USE_SINGLE_DOT_DEFAULTS to true."
        )
    
    def construct(self) -> None:
        """Build and render the bouncing dots animation."""
        cfg = self.config
        
        # Validate starting positions
        self._validate_start_positions()
        
        # Setup scene
        self._setup_scene()
        
        # Run physics simulation
        dot_states = create_dot_states_from_config(self.dots_config)
        simulation = PhysicsSimulation(cfg, dot_states, debug=cfg.DEBUG)
        positions_per_dot, bounce_events = simulation.simulate()
        
        # Generate audio
        total_duration = (len(positions_per_dot[0]) - 1) * cfg.SIMULATION_DT
        audio_dir = Path(cfg.audio_output_dir) if cfg.audio_output_dir else None
        mixer = AudioMixer(cfg, bounce_events, audio_dir=audio_dir, debug=cfg.DEBUG)
        mixed_path = mixer.generate_mixed_audio(total_duration, output_filename="bounce_multi_with_audio.wav")
        
        # Animate
        self._animate(dot_states, positions_per_dot, mixed_path, total_duration)
    
    def _validate_start_positions(self) -> None:
        """Validate that all dot starting positions are within the circle boundary."""
        cfg = self.config
        
        for i, dot_config in enumerate(self.dots_config):
            start_pos = dot_config["start_pos"]
            dot_radius = dot_config["radius"]
            distance_from_center = float(np.linalg.norm(start_pos[:2] - cfg.CIRCLE_CENTER[:2]))
            max_allowed_distance = cfg.CIRCLE_RADIUS - dot_radius
            
            if distance_from_center > max_allowed_distance:
                raise ValueError(
                    f"Dot {i} start_pos {start_pos[:2]} is outside the circle boundary. "
                    f"Distance from center: {distance_from_center:.3f}, "
                    f"Max allowed (CIRCLE_RADIUS - dot_radius): {max_allowed_distance:.3f}"
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
        dot_states: list[DotState],
        positions_per_dot: list[list[NDArray[np.floating]]],
        audio_path: Path,
        total_duration: float,
    ) -> None:
        """
        Animate the dots movement with optional trails.
        
        Args:
            dot_states: List of dot state dictionaries with color/radius info
            positions_per_dot: List of position lists, one per dot
            audio_path: Path to the mixed audio file
            total_duration: Total animation duration in seconds
        """
        cfg = self.config
        
        # Animation time tracker
        time_tracker = ValueTracker(0)
        
        if cfg.DEBUG:
            print(f"Animation: {total_duration:.2f}s, {len(positions_per_dot[0])} frames, trail={cfg.ENABLE_TRAIL}")
        
        # Create animated dot and trail getters for each dot
        def make_dot_getter(state: DotState, positions: list[NDArray[np.floating]]):
            def get_dot_position() -> Dot:
                index = min(int(time_tracker.get_value() / cfg.SIMULATION_DT), len(positions) - 1)
                return Dot(point=positions[index], radius=state["radius"], color=state["color"])
            return get_dot_position
        
        def make_trail_getter(state: DotState, positions: list[NDArray[np.floating]]):
            def get_trail_path() -> VMobject:
                index = min(int(time_tracker.get_value() / cfg.SIMULATION_DT), len(positions) - 1)
                trail = VMobject(
                    stroke_color=state["color"],
                    stroke_width=cfg.TRAIL_WIDTH,
                    stroke_opacity=cfg.TRAIL_OPACITY,
                )
                
                if index >= 2:
                    trail_points = positions[:index:cfg.TRAIL_SAMPLE_INTERVAL]
                    if len(trail_points) > 1:
                        trail.set_points_as_corners(trail_points)
                
                return trail
            return get_trail_path
        
        # Add all trails first (behind dots) to fix z-ordering
        if cfg.ENABLE_TRAIL:
            for state, positions in zip(dot_states, positions_per_dot):
                animated_trail = always_redraw(make_trail_getter(state, positions))
                self.add(animated_trail)
        
        # Then add all dots on top
        for state, positions in zip(dot_states, positions_per_dot):
            animated_dot = always_redraw(make_dot_getter(state, positions))
            self.add(animated_dot)
        
        # Play animation with audio
        self.add_sound(str(audio_path))
        self.play(
            time_tracker.animate.set_value(total_duration),
            run_time=total_duration,
            rate_func=linear,
        )
        self.wait(1)
