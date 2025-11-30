from manim import *
import numpy as np
from scipy.io import wavfile
import os

# ========================================
# GLOBAL CONFIGURATION PARAMETERS
# ========================================
# Modify these parameters to easily test different initial setups

# Circle boundary parameters
CIRCLE_CENTER = np.array([0, 0, 0])
CIRCLE_RADIUS = 3  # Units in Manim coordinate system

# Dot parameters
DOT_START_POS = CIRCLE_CENTER.copy()  # Starting position of the dot
DOT_RADIUS = 0.15  # Visual radius of the dot
DOT_COLOR = RED

# Physics parameters
GRAVITY = np.array([0, -9.8, 0])  # Gravity vector (m/s²)
INITIAL_VELOCITY = np.array([0.5, -3, 0])  # Initial velocity (x, y, z)
DAMPING = 0.85  # Energy retention on bounce (0.85 = 85% energy retained)
SIMULATION_DT = 0.005  # Time step for physics simulation
MAX_SIMULATION_TIME = 12  # Maximum simulation time in seconds

# Trail effect configuration
ENABLE_TRAIL = True  # Set to False to disable the trail effect
TRAIL_COLOR = YELLOW
TRAIL_WIDTH = 1
TRAIL_OPACITY = 0.6

# Sound effect configuration
SOUND_EFFECT_PATH = "../sound_effect/clack.wav"  # Path to the clack sound effect
USE_GENERATED_SOUND = False  # Set to True to use procedurally generated sound instead


def generate_hit_sound(frequency=440, duration=0.1, sample_rate=44100, volume=1.0):
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
    # Create time array for the duration of the sound
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Combine two sine waves at different frequencies with exponential decay
    # First component: Base frequency with moderate decay
    # Second component: Octave higher (2x frequency) with faster decay
    # The exponential decay (exp(-t * rate)) makes the sound fade out quickly like a real impact
    wave = (np.sin(2 * np.pi * frequency * t) * np.exp(-t * 15) +
            np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 20) * 0.5)
    
    # Scale to int16 range (-32768 to 32767) for WAV file format
    wave = wave * volume * 32767
    return wave.astype(np.int16)


def generate_ambient_sound(duration=12, sample_rate=44100):
    """
    Generate a simple ambient/ASMR-like background sound.
    
    Creates a calming background atmosphere using pink noise with gentle
    low-frequency oscillations. This mimics the continuous, soft sounds
    found in ASMR videos (like gentle rain or soft wind).
    
    Args:
        duration: Length of ambient sound in seconds
        sample_rate: Audio sample rate
    
    Returns:
        numpy array of int16 audio samples
    """
    # Create time array for the full duration
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate pink noise (random values with normal distribution)
    # This creates a soft, continuous background texture
    noise = np.random.normal(0, 0.05, len(t))
    
    # Add gentle low-frequency oscillations to make it more interesting
    # 0.1 Hz = very slow wave (10 second period) - like breathing
    # 0.23 Hz = slower wave (~4 second period) - adds variety
    ambient = noise + 0.02 * np.sin(2 * np.pi * 0.1 * t) + 0.015 * np.sin(2 * np.pi * 0.23 * t)
    
    # Scale to int16 range with lower volume (30%) for background
    ambient = ambient * 32767 * 0.3
    return ambient.astype(np.int16)


class BouncingDot(Scene):
    """
    Main animation class for a bouncing dot inside a circle with physics simulation.
    
    This scene simulates a dot dropped from the center of a circle with gravity,
    bouncing off the circular boundary with energy loss on each impact.
    Includes audio with ambient background and hit sounds.
    
    Set ENABLE_TRAIL = True in global config to show the trail path.
    """
    
    def construct(self):
        """
        Main construction method called by Manim to build and render the scene.
        """
        # ========================================
        # SCENE SETUP
        # ========================================
        
        self.camera.background_color = BLACK
        
        # Circle parameters - defines the boundary for bouncing
        circle = Circle(radius=CIRCLE_RADIUS, color=BLUE, stroke_width=2)
        self.add(circle)
        
        # Dot parameters - the bouncing object
        dot_start_pos = DOT_START_POS.copy()
        dot = Dot(point=dot_start_pos, radius=DOT_RADIUS, color=DOT_COLOR)
        self.add(dot)
        
        # ========================================
        # PHYSICS SIMULATION - CALCULATE ALL POSITIONS
        # ========================================
        
        # Lists to store simulation data
        positions = [dot_start_pos.copy()]  # All positions the dot will visit
        bounce_times = []  # Timestamps when bounces occur (for audio sync)
        bounce_speeds = []  # Speed at each bounce (for audio volume)
        
        # Current state variables
        current_pos = dot_start_pos.copy()
        current_vel = INITIAL_VELOCITY.copy()
        current_time = 0
        bounce_count = 0
        collision_distance = CIRCLE_RADIUS - DOT_RADIUS
        
        # Debug output
        print(f"Starting simulation...")
        print(f"Circle radius: {CIRCLE_RADIUS}, Dot radius: {DOT_RADIUS}")
        print(f"Collision distance: {collision_distance}")
        
        # Main physics simulation loop - runs until time limit or dot stops moving
        while current_time < MAX_SIMULATION_TIME:
            # ----------------------------------------
            # STEP 1: Apply gravity (constant acceleration)
            # ----------------------------------------
            current_vel = current_vel + GRAVITY * SIMULATION_DT
            
            # ----------------------------------------
            # STEP 2: Update position based on velocity
            # ----------------------------------------
            new_pos = current_pos + current_vel * SIMULATION_DT
            
            # ----------------------------------------
            # STEP 3: Check for collision with circle boundary
            # ----------------------------------------
            distance_from_center = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
            
            # If dot has moved outside the boundary, a collision occurred
            if distance_from_center > collision_distance:
                # Record bounce data for audio generation
                bounce_count += 1
                speed_before_bounce = np.linalg.norm(current_vel)
                bounce_times.append(current_time)
                bounce_speeds.append(speed_before_bounce)
                
                print(f"Bounce #{bounce_count} at time {current_time:.2f}s, distance: {distance_from_center:.3f}")
                
                # ----------------------------------------
                # COLLISION RESPONSE
                # ----------------------------------------
                
                # Calculate the direction from center to dot (outward normal)
                direction_2d = np.array([new_pos[0], new_pos[1]]) / distance_from_center
                
                # Move dot back to exactly the collision boundary
                new_pos[0] = direction_2d[0] * collision_distance
                new_pos[1] = direction_2d[1] * collision_distance
                
                # Calculate the inward normal (points toward center)
                normal_2d = -direction_2d
                
                # Project velocity onto the normal
                vel_normal = np.dot(current_vel[:2], normal_2d)
                
                # Only reflect if moving toward the wall
                if vel_normal < 0:
                    # Reflect velocity using: v' = v - 2*(v·n)*n
                    current_vel[0] = current_vel[0] - 2 * vel_normal * normal_2d[0]
                    current_vel[1] = current_vel[1] - 2 * vel_normal * normal_2d[1]
                    
                    # Apply energy loss due to inelastic collision
                    current_vel = current_vel * DAMPING
                    
                    print(f"  After bounce - vel: ({current_vel[0]:.2f}, {current_vel[1]:.2f}), speed: {np.linalg.norm(current_vel):.2f}")
            
            # Update position for next iteration
            current_pos = new_pos
            
            # Store position for animation
            positions.append(current_pos.copy())
            current_time += SIMULATION_DT
            
            # ----------------------------------------
            # STEP 4: Check stopping condition
            # ----------------------------------------
            speed = np.linalg.norm(current_vel)
            if speed < 0.05 and current_time > 3:
                print(f"Stopping simulation at time {current_time:.2f}s, speed: {speed:.4f}")
                
                # Add extra frames at the final position
                for _ in range(30):
                    positions.append(current_pos.copy())
                break
        
        # ========================================
        # SIMULATION COMPLETE - OUTPUT SUMMARY
        # ========================================
        
        print(f"\nSimulation complete:")
        print(f"  Total positions: {len(positions)}")
        print(f"  Total bounces: {bounce_count}")
        print(f"  Simulation duration: {current_time:.2f}s")
        print(f"  Final position: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
        print(f"  Distance from center: {np.sqrt(current_pos[0]**2 + current_pos[1]**2):.3f}")
        print()
        
        # ========================================
        # AUDIO GENERATION
        # ========================================
        
        total_duration = (len(positions) - 1) * SIMULATION_DT
        
        print("Generating audio files...")
        audio_dir = "media/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        # ----------------------------------------
        # Generate ambient background sound
        # ----------------------------------------
        ambient_duration = total_duration + 1
        ambient_sound = generate_ambient_sound(duration=ambient_duration)
        ambient_path = os.path.join(audio_dir, "ambient.wav")
        wavfile.write(ambient_path, 44100, ambient_sound)
        print(f"  Ambient sound saved to {ambient_path}")
        
        # ----------------------------------------
        # Generate or load hit sounds
        # ----------------------------------------
        sample_rate = 44100
        total_samples = int(ambient_duration * sample_rate)
        hit_audio = np.zeros(total_samples, dtype=np.float32)
        
        if USE_GENERATED_SOUND:
            # Use procedurally generated sound
            if bounce_speeds:
                max_speed = max(bounce_speeds)
                min_speed = min(bounce_speeds)
                speed_range = max_speed - min_speed if max_speed > min_speed else 1
            
            for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                volume = 0.2 + 0.8 * ((speed - min_speed) / speed_range if speed_range > 0 else 1.0)
                frequency = 440 + (i % 3) * 100
                hit_sound = generate_hit_sound(frequency=frequency, volume=volume)
                
                start_sample = int(bounce_time * sample_rate)
                end_sample = min(start_sample + len(hit_sound), total_samples)
                
                if start_sample < total_samples:
                    hit_audio[start_sample:end_sample] += hit_sound[:end_sample - start_sample].astype(np.float32)
                
                print(f"  Bounce {i+1} at {bounce_time:.2f}s, speed: {speed:.2f}, volume: {volume:.2f}")
        else:
            # Use clack.wav sound effect
            clack_path = os.path.join(os.path.dirname(__file__), SOUND_EFFECT_PATH)
            if os.path.exists(clack_path):
                clack_sample_rate, clack_sound = wavfile.read(clack_path)
                
                # Convert to mono if stereo
                if len(clack_sound.shape) > 1:
                    clack_sound = clack_sound.mean(axis=1)
                
                # Resample if necessary
                if clack_sample_rate != sample_rate:
                    # Simple resampling by interpolation
                    clack_duration = len(clack_sound) / clack_sample_rate
                    new_length = int(clack_duration * sample_rate)
                    clack_sound = np.interp(
                        np.linspace(0, len(clack_sound) - 1, new_length),
                        np.arange(len(clack_sound)),
                        clack_sound.astype(np.float32)
                    )
                else:
                    clack_sound = clack_sound.astype(np.float32)
                
                # Normalize clack sound
                clack_max = np.abs(clack_sound).max()
                if clack_max > 0:
                    clack_sound = clack_sound / clack_max * 32767
                
                # Calculate volume based on bounce speed
                if bounce_speeds:
                    max_speed = max(bounce_speeds)
                    min_speed = min(bounce_speeds)
                    speed_range = max_speed - min_speed if max_speed > min_speed else 1
                
                for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                    volume = 0.3 + 0.7 * ((speed - min_speed) / speed_range if speed_range > 0 else 1.0)
                    
                    start_sample = int(bounce_time * sample_rate)
                    end_sample = min(start_sample + len(clack_sound), total_samples)
                    
                    if start_sample < total_samples:
                        sound_to_add = clack_sound[:end_sample - start_sample] * volume
                        hit_audio[start_sample:end_sample] += sound_to_add
                    
                    print(f"  Bounce {i+1} at {bounce_time:.2f}s, speed: {speed:.2f}, volume: {volume:.2f}")
            else:
                print(f"  WARNING: Sound effect not found at {clack_path}")
                print(f"  Falling back to generated sound...")
                # Fallback to generated sound
                for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
                    hit_sound = generate_hit_sound(frequency=440, volume=0.8)
                    start_sample = int(bounce_time * sample_rate)
                    end_sample = min(start_sample + len(hit_sound), total_samples)
                    if start_sample < total_samples:
                        hit_audio[start_sample:end_sample] += hit_sound[:end_sample - start_sample].astype(np.float32)
        
        # ----------------------------------------
        # Mix ambient and hit sounds together
        # ----------------------------------------
        mixed_audio = ambient_sound.astype(np.float32) + hit_audio
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed_audio).max()
        if max_val > 32767:
            mixed_audio = mixed_audio * (32767 / max_val)
        
        mixed_audio = mixed_audio.astype(np.int16)
        
        mixed_path = os.path.join(audio_dir, "bounce_with_audio.wav")
        wavfile.write(mixed_path, sample_rate, mixed_audio)
        print(f"  Mixed audio saved to {mixed_path}")
        print()
        
        # ========================================
        # MANIM ANIMATION SETUP
        # ========================================
        
        time_tracker = ValueTracker(0)
        
        print(f"Animation settings:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Will animate through {len(positions)} positions")
        print(f"  Animation dt: {SIMULATION_DT}")
        print(f"  Trail enabled: {ENABLE_TRAIL}")
        print()
        
        # ----------------------------------------
        # Create animated dot using always_redraw
        # ----------------------------------------
        def get_dot_position():
            """Returns a Dot object at the position corresponding to the current time."""
            t = time_tracker.get_value()
            index = int(t / SIMULATION_DT)
            index = min(index, len(positions) - 1)
            return Dot(point=positions[index], radius=DOT_RADIUS, color=DOT_COLOR)
        
        # ----------------------------------------
        # Create animated trail (if enabled)
        # ----------------------------------------
        def get_trail_path():
            """Create a trail showing the path traveled so far."""
            t = time_tracker.get_value()
            index = int(t / SIMULATION_DT)
            index = min(index, len(positions) - 1)
            
            if index < 2:
                return VMobject(stroke_color=TRAIL_COLOR, stroke_width=TRAIL_WIDTH, stroke_opacity=TRAIL_OPACITY)
            
            trail_obj = VMobject(stroke_color=TRAIL_COLOR, stroke_width=TRAIL_WIDTH, stroke_opacity=TRAIL_OPACITY)
            trail_points = positions[:index:3]
            
            if len(trail_points) > 1:
                trail_obj.set_points_as_corners(trail_points)
            
            return trail_obj
        
        # Remove the original static dot
        self.remove(dot)
        
        # Add the animated dot that updates every frame
        animated_dot = always_redraw(get_dot_position)
        
        # Add trail if enabled
        if ENABLE_TRAIL:
            animated_trail = always_redraw(get_trail_path)
            self.add(animated_trail, animated_dot)
        else:
            self.add(animated_dot)
        
        # ----------------------------------------
        # Add audio and animate
        # ----------------------------------------
        self.add_sound(mixed_path)
        
        self.play(time_tracker.animate.set_value(total_duration), run_time=total_duration, rate_func=linear)
        
        # Hold the final frame
        self.wait(1)
