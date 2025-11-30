from manim import *
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from pathlib import Path

# Debug mode - set to True to enable console output
DEBUG = False

# Global configuration parameters
# Modify these parameters to easily test different initial setups

# Circle boundary parameters
CIRCLE_CENTER = np.array([0, 0, 0])
CIRCLE_RADIUS = 3  # Units in Manim coordinate system

# Dot parameters
DOT_START_POS = CIRCLE_CENTER.copy()  # Starting position of the dot
DOT_RADIUS = 0.15  # Visual radius of the dot
DOT_COLOR = RED

# Physics parameters
GRAVITY = np.array([0, -9.8, 0])  # Gravity vector (m/s^2)
INITIAL_VELOCITY = np.array([0.5, -3, 0])  # Initial velocity (x, y, z)
DAMPING = 0.85  # Energy retention on bounce (Ex: 0.85 = 85% energy retained)
SIMULATION_DT = 0.005  # Time step for physics simulation
MAX_SIMULATION_TIME = 30  # Maximum simulation time in seconds

# Trail effect configuration
ENABLE_TRAIL = True  # Set to False to disable the trail effect
TRAIL_COLOR = YELLOW
TRAIL_WIDTH = 1
TRAIL_OPACITY = 0.6

# Sound effect configuration
SOUND_EFFECT = "clack.wav"  # Path to the sound effect if any
USE_GENERATED_SOUND = False  # Set to True to use procedurally generated sound instead
MIN_BOUNCE_SOUND_INTERVAL = 0.05  # Minimum time (seconds) between bounce sounds to prevent overlap

# Audio parameters
SAMPLE_RATE = 44100  # CD quality audio sample rate

# Trail sampling (sample every Nth position for performance)
TRAIL_SAMPLE_INTERVAL = 3


def generate_hit_sound(
    frequency: float = 440,
    duration: float = 0.1,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 1.0
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
    # First component: Base frequency with moderate decay
    # Second component: Octave higher (2x frequency) with faster decay
    # The exponential decay (exp(-t * rate)) makes the sound fade out quickly like a real impact
    wave = (
        np.sin(2 * np.pi * frequency * t) * np.exp(-t * 15) +
        np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 20) * 0.5
    )
    
    # Scale to int16 range for WAV format
    return (wave * volume * 32767).astype(np.int16)


def generate_ambient_sound(
    duration: float = 12,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 0.1
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
    # 0.1 Hz and ~0.23 Hz components
    ambient = (
        noise +
        0.02 * np.sin(2 * np.pi * 0.1 * t) +   # 10s period wave
        0.015 * np.sin(2 * np.pi * 0.23 * t)   # ~4s period wave
    )
    
    return (ambient * 32767 * volume).astype(np.int16)


class BouncingDot(Scene):
    """
    Bouncing dot animation inside a circular boundary with physics simulation.
    
    Features:
        - Gravity-based physics with collision detection
        - Energy loss on bounce (configurable damping)
        - Optional trail effect
        - Ambient and hit sound effects
    """
    
    def construct(self) -> None:
        """Build and render the bouncing dot animation."""
        self.camera.background_color = BLACK
        
        # Create boundary circle
        circle = Circle(radius=CIRCLE_RADIUS, color=BLUE, stroke_width=2)
        self.add(circle)
        
        # Create bouncing dot
        dot_start_pos = DOT_START_POS.copy()
        dot = Dot(point=dot_start_pos, radius=DOT_RADIUS, color=DOT_COLOR)
        self.add(dot)
        
        # Physics simulation - pre-calculate all positions
        positions: list[NDArray] = [dot_start_pos.copy()]
        bounce_times: list[float] = []
        bounce_speeds: list[float] = []
        
        current_pos = dot_start_pos.copy()
        current_vel = INITIAL_VELOCITY.copy()
        current_time = 0.0
        bounce_count = 0
        collision_distance = CIRCLE_RADIUS - DOT_RADIUS
        
        if DEBUG:
            print(f"Starting simulation...")
            print(f"Collision distance: {collision_distance}")
        
        # Main physics simulation loop
        while current_time < MAX_SIMULATION_TIME:
            # Apply gravity and update position
            current_vel = current_vel + GRAVITY * SIMULATION_DT
            new_pos = current_pos + current_vel * SIMULATION_DT
            
            # Check for collision with circle boundary
            distance_from_center = np.linalg.norm(new_pos[:2])
            
            if distance_from_center > collision_distance:
                # Record bounce data for audio
                bounce_count += 1
                speed = np.linalg.norm(current_vel)
                bounce_times.append(current_time)
                bounce_speeds.append(speed)
                
                if DEBUG:
                    print(f"Bounce #{bounce_count} at t={current_time:.2f}s")
                
                # Collision response - reflect velocity off boundary
                direction_2d = new_pos[:2] / distance_from_center
                # Move dot back to the collision boundary
                new_pos[:2] = direction_2d * collision_distance
                # Calculate inward normal and reflect velocity
                normal_2d = -direction_2d
                vel_normal = np.dot(current_vel[:2], normal_2d)
                
                if vel_normal < 0:
                    # Reflect: v' = v - 2*(v·n)*n, then apply damping
                    current_vel[:2] -= 2 * vel_normal * normal_2d
                    current_vel *= DAMPING
            
            current_pos = new_pos
            positions.append(current_pos.copy())
            current_time += SIMULATION_DT
            
            # Check stopping condition
            speed = np.linalg.norm(current_vel)
            is_near_bottom = current_pos[1] < -collision_distance + 0.5
            is_on_boundary = distance_from_center > collision_distance - 0.1
            
            if speed < 0.05 and current_time > 3 and is_near_bottom and is_on_boundary:
                # Add extra frames at final position for smooth ending
                positions.extend([current_pos.copy() for _ in range(30)])
                break
        
        # Simulation complete - output summary
        if DEBUG:
            print(f"\nSimulation complete:")
            print(f"  Total positions: {len(positions)}")
            print(f"  Total bounces: {bounce_count}")
            print(f"  Simulation duration: {current_time:.2f}s")
            print(f"  Final position: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
            print(f"  Distance from center: {np.sqrt(current_pos[0]**2 + current_pos[1]**2):.3f}")
            print()
        
        # Audio generation
        total_duration = (len(positions) - 1) * SIMULATION_DT
        audio_dir = Path("media/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        sound_effect_dir = Path("sound_effect")
        sound_effect_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate ambient background sound
        ambient_duration = total_duration + 1
        ambient_sound = generate_ambient_sound(duration=ambient_duration)
        ambient_path = audio_dir / "ambient.wav"
        wavfile.write(ambient_path, SAMPLE_RATE, ambient_sound)
        
        # Generate or load hit sounds
        total_samples = int(ambient_duration * SAMPLE_RATE)
        hit_audio = np.zeros(total_samples, dtype=np.float32)
        
        use_generated_sound = USE_GENERATED_SOUND
        sound_effect_path = sound_effect_dir / SOUND_EFFECT
        
        if not use_generated_sound and not sound_effect_path.exists():
            if DEBUG:
                print(f"Sound effect not found: {sound_effect_path}, using generated sound")
            use_generated_sound = True
        
        # Calculate speed range for volume normalization
        min_speed = min(bounce_speeds) if bounce_speeds else 0
        max_speed = max(bounce_speeds) if bounce_speeds else 1
        speed_range = max(max_speed - min_speed, 1)
        
        def calculate_volume(speed: float, base: float = 0.2, scale: float = 0.8) -> float:
            """Calculate volume based on bounce speed."""
            return base + scale * (speed - min_speed) / speed_range
        
        if use_generated_sound:
            for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                if i > 0 and (bounce_time - bounce_times[i-1]) < MIN_BOUNCE_SOUND_INTERVAL:
                    continue
                
                volume = calculate_volume(speed)
                frequency = 440 + (i % 3) * 100
                hit_sound = generate_hit_sound(frequency=frequency, volume=volume)
                
                start_sample = int(bounce_time * SAMPLE_RATE)
                end_sample = min(start_sample + len(hit_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += hit_sound[:end_sample - start_sample].astype(np.float32)
        else:
            # Load and process sound effect file
            effect_sample_rate, effect_sound = wavfile.read(sound_effect_path)
            
            # Convert to mono if stereo
            if len(effect_sound.shape) > 1:
                effect_sound = effect_sound.mean(axis=1)
            
            # Resample if necessary
            if effect_sample_rate != SAMPLE_RATE:
                effect_duration = len(effect_sound) / effect_sample_rate
                new_length = int(effect_duration * SAMPLE_RATE)
                effect_sound = np.interp(
                    np.linspace(0, len(effect_sound) - 1, new_length),
                    np.arange(len(effect_sound)),
                    effect_sound.astype(np.float32)
                )
            else:
                effect_sound = effect_sound.astype(np.float32)
            
            # Normalize sound effect
            effect_max = np.abs(effect_sound).max()
            if effect_max > 0:
                effect_sound = effect_sound / effect_max * 32767
            
            for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                if i > 0 and (bounce_time - bounce_times[i-1]) < MIN_BOUNCE_SOUND_INTERVAL:
                    continue
                
                volume = calculate_volume(speed, base=0.3, scale=0.7)
                start_sample = int(bounce_time * SAMPLE_RATE)
                end_sample = min(start_sample + len(effect_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += effect_sound[:end_sample - start_sample] * volume
        
        # Mix and normalize audio
        mixed_audio = ambient_sound.astype(np.float32) + hit_audio
        max_val = np.abs(mixed_audio).max()
        if max_val > 32767:
            mixed_audio *= 32767 / max_val
        
        mixed_path = audio_dir / "bounce_with_audio.wav"
        wavfile.write(mixed_path, SAMPLE_RATE, mixed_audio.astype(np.int16))
        
        # Animation setup
        time_tracker = ValueTracker(0)
        
        if DEBUG:
            print(f"Animation: {total_duration:.2f}s, {len(positions)} frames, trail={ENABLE_TRAIL}")
        
        def get_dot_position() -> Dot:
            """Return dot at current time position."""
            index = min(int(time_tracker.get_value() / SIMULATION_DT), len(positions) - 1)
            return Dot(point=positions[index], radius=DOT_RADIUS, color=DOT_COLOR)
        
        def get_trail_path() -> VMobject:
            """Create trail showing path traveled so far."""
            index = min(int(time_tracker.get_value() / SIMULATION_DT), len(positions) - 1)
            trail = VMobject(stroke_color=TRAIL_COLOR, stroke_width=TRAIL_WIDTH, stroke_opacity=TRAIL_OPACITY)
            
            if index >= 2:
                trail_points = positions[:index:TRAIL_SAMPLE_INTERVAL]
                if len(trail_points) > 1:
                    trail.set_points_as_corners(trail_points)
            
            return trail
        
        # Replace static dot with animated version
        self.remove(dot)
        animated_dot = always_redraw(get_dot_position)
        
        if ENABLE_TRAIL:
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
