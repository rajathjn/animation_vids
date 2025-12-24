"""
Multiple bouncing dots animation with physics simulation and dot-to-dot collision.
"""
from manim import *
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from pathlib import Path
from typing import Any

from .config import AnimationConfig
from .audio_utils import generate_hit_sound, generate_ambient_sound


class BouncingDots(Scene):
    """
    Multiple bouncing dots animation inside a circular boundary with physics simulation.
    
    Features:
        - Gravity-based physics with collision detection
        - Dot-to-dot collision handling
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
    
    def __init__(self, config: AnimationConfig, **kwargs):
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
        
        Raises:
            ValueError: If no dots configured and USE_SINGLE_DOT_DEFAULTS is False
        """
        dots_json = self.config.DOTS_JSON
        
        if dots_json:
            # Convert list values to numpy arrays
            dots = []
            for dot in dots_json:
                dot_copy = dot.copy()
                if "start_pos" in dot_copy:
                    dot_copy["start_pos"] = np.array(dot_copy["start_pos"])
                if "initial_velocity" in dot_copy:
                    dot_copy["initial_velocity"] = np.array(dot_copy["initial_velocity"])
                dots.append(dot_copy)
            return dots
        
        # DOTS_JSON is empty - check if we should use single dot defaults
        if self.config.USE_SINGLE_DOT_DEFAULTS:
            default_dot = self.config.get_default_dot_config()
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
        self.camera.background_color = BLACK
        
        # Create boundary circle
        circle = Circle(radius=cfg.CIRCLE_RADIUS, color=BLUE, stroke_width=2)
        self.add(circle)
        
        num_dots = len(self.dots_config)
        collision_distance = cfg.CIRCLE_RADIUS
        
        # Validate that all dot starting positions are within the circle
        for i, dot_config in enumerate(self.dots_config):
            start_pos = dot_config["start_pos"]
            dot_radius = dot_config["radius"]
            distance_from_center = np.linalg.norm(start_pos[:2] - cfg.CIRCLE_CENTER[:2])
            max_allowed_distance = cfg.CIRCLE_RADIUS - dot_radius
            
            if distance_from_center > max_allowed_distance:
                raise ValueError(
                    f"Dot {i} start_pos {start_pos[:2]} is outside the circle boundary. "
                    f"Distance from center: {distance_from_center:.3f}, "
                    f"Max allowed (CIRCLE_RADIUS - dot_radius): {max_allowed_distance:.3f}"
                )
        
        # Initialize dot states from configuration
        dot_states = []
        for dot_config in self.dots_config:
            dot_states.append({
                "pos": dot_config["start_pos"].copy(),
                "vel": dot_config["initial_velocity"].copy(),
                "damping": dot_config["damping"],
                "radius": dot_config["radius"],
                "color": dot_config["color"],
                "positions": [dot_config["start_pos"].copy()],
            })
        
        # Create visual dots
        dots = []
        for state in dot_states:
            dot = Dot(point=state["pos"], radius=state["radius"], color=eval(state["color"]))
            dots.append(dot)
            self.add(dot)
        
        # Physics simulation - pre-calculate all positions
        bounce_times: list[float] = []
        bounce_speeds: list[float] = []
        
        current_time = 0.0
        bounce_count = 0
        
        if cfg.DEBUG:
            print(f"Starting simulation with {num_dots} dots...")
            print(f"Circle radius: {cfg.CIRCLE_RADIUS}")
        
        # Main physics simulation loop
        while current_time < cfg.MAX_SIMULATION_TIME:
            # Update each dot
            for state in dot_states:
                # Apply gravity and update position
                state["vel"] = state["vel"] + cfg.GRAVITY * cfg.SIMULATION_DT
                new_pos = state["pos"] + state["vel"] * cfg.SIMULATION_DT
                
                # Check for collision with circle boundary
                dot_collision_distance = collision_distance - state["radius"]
                distance_from_center = np.linalg.norm(new_pos[:2])
                
                if distance_from_center > dot_collision_distance:
                    # Record bounce data for audio
                    bounce_count += 1
                    speed = np.linalg.norm(state["vel"])
                    bounce_times.append(current_time)
                    bounce_speeds.append(speed)
                    
                    if cfg.DEBUG:
                        print(f"Wall bounce #{bounce_count} at t={current_time:.2f}s")
                    
                    # Collision response - reflect velocity off boundary
                    direction_2d = new_pos[:2] / distance_from_center
                    new_pos[:2] = direction_2d * dot_collision_distance
                    normal_2d = -direction_2d
                    vel_normal = np.dot(state["vel"][:2], normal_2d)
                    
                    if vel_normal < 0:
                        state["vel"][:2] -= 2 * vel_normal * normal_2d
                        state["vel"] *= state["damping"]
                
                state["pos"] = new_pos
            
            # Check for dot-to-dot collisions
            for i in range(num_dots):
                for j in range(i + 1, num_dots):
                    state_i = dot_states[i]
                    state_j = dot_states[j]
                    
                    # Calculate distance between dot centers
                    diff = state_i["pos"] - state_j["pos"]
                    dist = np.linalg.norm(diff[:2])
                    min_dist = state_i["radius"] + state_j["radius"]
                    
                    if dist < min_dist and dist > 0:
                        # Collision detected between dots
                        bounce_count += 1
                        
                        # Calculate relative speed for audio volume
                        relative_vel = state_i["vel"] - state_j["vel"]
                        speed = np.linalg.norm(relative_vel)
                        bounce_times.append(current_time)
                        bounce_speeds.append(speed)
                        
                        if cfg.DEBUG:
                            print(f"Dot collision #{bounce_count} between dot {i} and {j} at t={current_time:.2f}s")
                        
                        # Normal vector from j to i
                        normal = diff[:2] / dist
                        
                        # Separate the dots
                        overlap = min_dist - dist
                        state_i["pos"][:2] += normal * (overlap / 2)
                        state_j["pos"][:2] -= normal * (overlap / 2)
                        
                        # Calculate relative velocity along normal
                        vel_i_normal = np.dot(state_i["vel"][:2], normal)
                        vel_j_normal = np.dot(state_j["vel"][:2], normal)
                        
                        # Only resolve if dots are approaching each other
                        if vel_i_normal - vel_j_normal < 0:
                            avg_damping = (state_i["damping"] + state_j["damping"]) / 2
                            
                            state_i["vel"][:2] += (vel_j_normal - vel_i_normal) * normal * avg_damping
                            state_j["vel"][:2] += (vel_i_normal - vel_j_normal) * normal * avg_damping
            
            # Store positions for animation
            for state in dot_states:
                state["positions"].append(state["pos"].copy())
            
            current_time += cfg.SIMULATION_DT
            
            # Check stopping condition - all dots nearly stopped
            all_stopped = True
            for state in dot_states:
                speed = np.linalg.norm(state["vel"])
                if speed >= 0.05:
                    all_stopped = False
                    break
            
            if all_stopped and current_time > 3:
                for state in dot_states:
                    state["positions"].extend([state["pos"].copy() for _ in range(30)])
                break
        
        if cfg.DEBUG:
            print(f"\nSimulation complete:")
            print(f"  Total positions per dot: {len(dot_states[0]['positions'])}")
            print(f"  Total bounces: {bounce_count}")
            print(f"  Simulation duration: {current_time:.2f}s")
            print()
        
        # Audio generation
        total_duration = (len(dot_states[0]["positions"]) - 1) * cfg.SIMULATION_DT
        
        # Use audio_output_dir if set, otherwise fall back to media/audio
        if cfg.audio_output_dir:
            audio_dir = Path(cfg.audio_output_dir)
        else:
            audio_dir = Path("media/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        sound_effect_dir = Path("sound_effect")
        sound_effect_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate ambient background sound
        ambient_duration = total_duration + 1
        ambient_sound = generate_ambient_sound(duration=ambient_duration, sample_rate=cfg.SAMPLE_RATE)
        ambient_path = audio_dir / "ambient_multi.wav"
        wavfile.write(ambient_path, cfg.SAMPLE_RATE, ambient_sound)
        
        # Generate or load hit sounds
        total_samples = int(ambient_duration * cfg.SAMPLE_RATE)
        hit_audio = np.zeros(total_samples, dtype=np.float32)
        
        use_generated_sound = cfg.USE_GENERATED_SOUND
        sound_effect_path = sound_effect_dir / cfg.SOUND_EFFECT
        
        if not use_generated_sound and not sound_effect_path.exists():
            if cfg.DEBUG:
                print(f"Sound effect not found: {sound_effect_path}, using generated sound")
            use_generated_sound = True
        
        # Calculate speed range for volume normalization
        min_speed = min(bounce_speeds) if bounce_speeds else 0
        max_speed = max(bounce_speeds) if bounce_speeds else 1
        speed_range = max(max_speed - min_speed, 1)
        
        def calculate_volume(speed: float, base: float = 0.2, scale: float = 0.8) -> float:
            return base + scale * (speed - min_speed) / speed_range
        
        if use_generated_sound:
            for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                if i > 0 and (bounce_time - bounce_times[i-1]) < cfg.MIN_BOUNCE_SOUND_INTERVAL:
                    continue
                
                volume = calculate_volume(speed)
                frequency = 440 + (i % 3) * 100
                hit_sound = generate_hit_sound(frequency=frequency, volume=volume, sample_rate=cfg.SAMPLE_RATE)
                
                start_sample = int(bounce_time * cfg.SAMPLE_RATE)
                end_sample = min(start_sample + len(hit_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += hit_sound[:end_sample - start_sample].astype(np.float32)
        else:
            # Load and process sound effect file
            effect_sample_rate, effect_sound = wavfile.read(sound_effect_path)
            
            if len(effect_sound.shape) > 1:
                effect_sound = effect_sound.mean(axis=1)
            
            if effect_sample_rate != cfg.SAMPLE_RATE:
                effect_duration = len(effect_sound) / effect_sample_rate
                new_length = int(effect_duration * cfg.SAMPLE_RATE)
                effect_sound = np.interp(
                    np.linspace(0, len(effect_sound) - 1, new_length),
                    np.arange(len(effect_sound)),
                    effect_sound.astype(np.float32)
                )
            else:
                effect_sound = effect_sound.astype(np.float32)
            
            effect_max = np.abs(effect_sound).max()
            if effect_max > 0:
                effect_sound = effect_sound / effect_max * 32767
            
            for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                if i > 0 and (bounce_time - bounce_times[i-1]) < cfg.MIN_BOUNCE_SOUND_INTERVAL:
                    continue
                
                volume = calculate_volume(speed, base=0.3, scale=0.7)
                start_sample = int(bounce_time * cfg.SAMPLE_RATE)
                end_sample = min(start_sample + len(effect_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += effect_sound[:end_sample - start_sample] * volume
        
        # Mix and normalize audio
        mixed_audio = ambient_sound.astype(np.float32) + hit_audio
        max_val = np.abs(mixed_audio).max()
        if max_val > 32767:
            mixed_audio *= 32767 / max_val
        
        mixed_path = audio_dir / "bounce_multi_with_audio.wav"
        wavfile.write(mixed_path, cfg.SAMPLE_RATE, mixed_audio.astype(np.int16))
        
        # Animation setup
        time_tracker = ValueTracker(0)
        
        if cfg.DEBUG:
            print(f"Animation: {total_duration:.2f}s, {len(dot_states[0]['positions'])} frames, trail={cfg.ENABLE_TRAIL}")
        
        # Create animated dot and trail getters for each dot
        def make_dot_getter(state):
            def get_dot_position() -> Dot:
                index = min(int(time_tracker.get_value() / cfg.SIMULATION_DT), len(state["positions"]) - 1)
                return Dot(point=state["positions"][index], radius=state["radius"], color=eval(state["color"]))
            return get_dot_position
        
        def make_trail_getter(state):
            def get_trail_path() -> VMobject:
                index = min(int(time_tracker.get_value() / cfg.SIMULATION_DT), len(state["positions"]) - 1)
                trail = VMobject(
                    stroke_color=eval(state["color"]),
                    stroke_width=cfg.TRAIL_WIDTH,
                    stroke_opacity=cfg.TRAIL_OPACITY
                )
                
                if index >= 2:
                    trail_points = state["positions"][:index:cfg.TRAIL_SAMPLE_INTERVAL]
                    if len(trail_points) > 1:
                        trail.set_points_as_corners(trail_points)
                
                return trail
            return get_trail_path
        
        # Replace static dots with animated versions
        for dot in dots:
            self.remove(dot)
        
        # Add all trails first (behind dots) to fix z-ordering
        if cfg.ENABLE_TRAIL:
            for state in dot_states:
                animated_trail = always_redraw(make_trail_getter(state))
                self.add(animated_trail)
        
        # Then add all dots on top
        for state in dot_states:
            animated_dot = always_redraw(make_dot_getter(state))
            self.add(animated_dot)
        
        # Play animation with audio
        self.add_sound(mixed_path)
        self.play(
            time_tracker.animate.set_value(total_duration),
            run_time=total_duration,
            rate_func=linear
        )
        self.wait(1)
