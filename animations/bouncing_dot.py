"""
Single bouncing dot animation with physics simulation.
"""
from manim import *
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from pathlib import Path
from typing import Optional

from .config import AnimationConfig
from .audio_utils import generate_hit_sound, generate_ambient_sound


class BouncingDot(Scene):
    """
    Bouncing dot animation inside a circular boundary with physics simulation.
    
    Features:
        - Gravity-based physics with collision detection
        - Energy loss on bounce (configurable damping)
        - Optional trail effect
        - Ambient and hit sound effects
    
    Usage:
        # Use defaults from app.cfg
        scene = BouncingDot()
        
        # Override specific settings
        config = AnimationConfig()
        config.override(damping=0.95, dot_color='RED')
        scene = BouncingDot(config=config)
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None, **kwargs):
        """
        Initialize the bouncing dot scene.
        
        Args:
            config: AnimationConfig instance. If None, loads from app.cfg files.
            **kwargs: Additional arguments passed to Scene
        """
        super().__init__(**kwargs)
        self.config = config if config is not None else AnimationConfig()
    
    def construct(self) -> None:
        """Build and render the bouncing dot animation."""
        cfg = self.config
        
        # Validate that the starting position is within the circle
        distance_from_center = np.linalg.norm(cfg.dot_start_pos[:2] - cfg.circle_center[:2])
        max_allowed_distance = cfg.circle_radius - cfg.dot_radius
        if distance_from_center > max_allowed_distance:
            raise ValueError(
                f"DOT_START_POS {cfg.dot_start_pos[:2]} is outside the circle boundary. "
                f"Distance from center: {distance_from_center:.3f}, "
                f"Max allowed (CIRCLE_RADIUS - DOT_RADIUS): {max_allowed_distance:.3f}"
            )
        
        self.camera.background_color = BLACK
        
        # Create boundary circle
        circle = Circle(radius=cfg.circle_radius, color=BLUE, stroke_width=2)
        self.add(circle)
        
        # Create bouncing dot
        dot_start_pos = cfg.dot_start_pos.copy()
        dot = Dot(point=dot_start_pos, radius=cfg.dot_radius, color=eval(cfg.dot_color))
        self.add(dot)
        
        # Physics simulation - pre-calculate all positions
        positions: list[NDArray] = [dot_start_pos.copy()]
        bounce_times: list[float] = []
        bounce_speeds: list[float] = []
        
        current_pos = dot_start_pos.copy()
        current_vel = cfg.initial_velocity.copy()
        current_time = 0.0
        bounce_count = 0
        collision_distance = cfg.circle_radius - cfg.dot_radius
        
        if cfg.debug:
            print(f"Starting simulation...")
            print(f"Collision distance: {collision_distance}")
        
        # Main physics simulation loop
        while current_time < cfg.max_simulation_time:
            # Apply gravity and update position
            current_vel = current_vel + cfg.gravity * cfg.simulation_dt
            new_pos = current_pos + current_vel * cfg.simulation_dt
            
            # Check for collision with circle boundary
            distance_from_center = np.linalg.norm(new_pos[:2])
            
            if distance_from_center > collision_distance:
                # Record bounce data for audio
                bounce_count += 1
                speed = np.linalg.norm(current_vel)
                bounce_times.append(current_time)
                bounce_speeds.append(speed)
                
                if cfg.debug:
                    print(f"Bounce #{bounce_count} at t={current_time:.2f}s")
                
                # Collision response - reflect velocity off boundary
                direction_2d = new_pos[:2] / distance_from_center
                new_pos[:2] = direction_2d * collision_distance
                normal_2d = -direction_2d
                vel_normal = np.dot(current_vel[:2], normal_2d)
                
                if vel_normal < 0:
                    current_vel[:2] -= 2 * vel_normal * normal_2d
                    current_vel *= cfg.damping
            
            current_pos = new_pos
            positions.append(current_pos.copy())
            current_time += cfg.simulation_dt
            
            # Check stopping condition
            speed = np.linalg.norm(current_vel)
            is_near_bottom = current_pos[1] < -collision_distance + 0.5
            is_on_boundary = distance_from_center > collision_distance - 0.1
            
            if speed < 0.05 and current_time > 3 and is_near_bottom and is_on_boundary:
                positions.extend([current_pos.copy() for _ in range(30)])
                break
        
        if cfg.debug:
            print(f"\nSimulation complete:")
            print(f"  Total positions: {len(positions)}")
            print(f"  Total bounces: {bounce_count}")
            print(f"  Simulation duration: {current_time:.2f}s")
            print(f"  Final position: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
            print(f"  Distance from center: {np.sqrt(current_pos[0]**2 + current_pos[1]**2):.3f}")
            print()
        
        # Audio generation
        total_duration = (len(positions) - 1) * cfg.simulation_dt
        
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
        ambient_sound = generate_ambient_sound(duration=ambient_duration, sample_rate=cfg.sample_rate)
        ambient_path = audio_dir / "ambient.wav"
        wavfile.write(ambient_path, cfg.sample_rate, ambient_sound)
        
        # Generate or load hit sounds
        total_samples = int(ambient_duration * cfg.sample_rate)
        hit_audio = np.zeros(total_samples, dtype=np.float32)
        
        use_generated_sound = cfg.use_generated_sound
        sound_effect_path = sound_effect_dir / cfg.sound_effect
        
        if not use_generated_sound and not sound_effect_path.exists():
            if cfg.debug:
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
                if i > 0 and (bounce_time - bounce_times[i-1]) < cfg.min_bounce_sound_interval:
                    continue
                
                volume = calculate_volume(speed)
                frequency = 440 + (i % 3) * 100
                hit_sound = generate_hit_sound(frequency=frequency, volume=volume, sample_rate=cfg.sample_rate)
                
                start_sample = int(bounce_time * cfg.sample_rate)
                end_sample = min(start_sample + len(hit_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += hit_sound[:end_sample - start_sample].astype(np.float32)
        else:
            # Load and process sound effect file
            effect_sample_rate, effect_sound = wavfile.read(sound_effect_path)
            
            if len(effect_sound.shape) > 1:
                effect_sound = effect_sound.mean(axis=1)
            
            if effect_sample_rate != cfg.sample_rate:
                effect_duration = len(effect_sound) / effect_sample_rate
                new_length = int(effect_duration * cfg.sample_rate)
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
                if i > 0 and (bounce_time - bounce_times[i-1]) < cfg.min_bounce_sound_interval:
                    continue
                
                volume = calculate_volume(speed, base=0.3, scale=0.7)
                start_sample = int(bounce_time * cfg.sample_rate)
                end_sample = min(start_sample + len(effect_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += effect_sound[:end_sample - start_sample] * volume
        
        # Mix and normalize audio
        mixed_audio = ambient_sound.astype(np.float32) + hit_audio
        max_val = np.abs(mixed_audio).max()
        if max_val > 32767:
            mixed_audio *= 32767 / max_val
        
        mixed_path = audio_dir / "bounce_with_audio.wav"
        wavfile.write(mixed_path, cfg.sample_rate, mixed_audio.astype(np.int16))
        
        # Animation setup
        time_tracker = ValueTracker(0)
        
        if cfg.debug:
            print(f"Animation: {total_duration:.2f}s, {len(positions)} frames, trail={cfg.enable_trail}")
        
        def get_dot_position() -> Dot:
            index = min(int(time_tracker.get_value() / cfg.simulation_dt), len(positions) - 1)
            return Dot(point=positions[index], radius=cfg.dot_radius, color=eval(cfg.dot_color))
        
        def get_trail_path() -> VMobject:
            index = min(int(time_tracker.get_value() / cfg.simulation_dt), len(positions) - 1)
            trail = VMobject(
                stroke_color=eval(cfg.trail_color),
                stroke_width=cfg.trail_width,
                stroke_opacity=cfg.trail_opacity
            )
            
            if index >= 2:
                trail_points = positions[:index:cfg.trail_sample_interval]
                if len(trail_points) > 1:
                    trail.set_points_as_corners(trail_points)
            
            return trail
        
        # Replace static dot with animated version
        self.remove(dot)
        animated_dot = always_redraw(get_dot_position)
        
        if cfg.enable_trail:
            self.add(always_redraw(get_trail_path), animated_dot)
        else:
            self.add(animated_dot)
        
        # Play animation with audio
        self.add_sound(mixed_path)
        self.play(
            time_tracker.animate.set_value(total_duration),
            run_time=total_duration,
            rate_func=linear
        )
        self.wait(1)
