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
    Includes procedurally generated audio with ambient background and hit sounds.
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
        circle_center = np.array([0, 0, 0])
        circle_radius = 3  # Units in Manim coordinate system
        circle = Circle(radius=circle_radius, color=BLUE, stroke_width=2)
        self.add(circle)
        
        # Dot parameters - the bouncing object
        dot_start_pos = circle_center.copy()  # Start at center of circle
        dot_radius = 0.15  # Visual radius of the dot
        dot = Dot(point=dot_start_pos, radius=dot_radius, color=RED)
        self.add(dot)
        
        # ========================================
        # PHYSICS PARAMETERS
        # ========================================
        
        # Gravity vector pointing downward (negative y-direction)
        gravity = np.array([0, -9.8, 0])  # Standard Earth gravity in m/s²
        
        # Initial velocity - gives the dot a slight horizontal push and downward motion
        velocity = np.array([0.5, -3, 0])  # (x, y, z) components
        
        # Damping factor - simulates energy loss on bounce
        # 0.85 means the dot retains 85% of its energy after each collision
        damping = 0.85
        
        # Time step for physics simulation
        # Smaller dt = more accurate simulation but more computation
        # 0.005s (200 steps per second) provides good accuracy
        dt = 0.005
        
        # Maximum simulation time before forcing stop
        max_time = 12
        
        # ========================================
        # PHYSICS SIMULATION - CALCULATE ALL POSITIONS
        # ========================================
        
        # Lists to store simulation data
        positions = [dot_start_pos.copy()]  # All positions the dot will visit
        bounce_times = []  # Timestamps when bounces occur (for audio sync)
        bounce_speeds = []  # Speed at each bounce (for audio volume)
        
        # Current state variables
        current_pos = dot_start_pos.copy()
        current_vel = velocity.copy()
        current_time = 0
        bounce_count = 0
        
        # Debug output
        print(f"Starting simulation...")
        print(f"Circle radius: {circle_radius}, Dot radius: {dot_radius}")
        print(f"Collision distance: {circle_radius - dot_radius}")
        
        # Main physics simulation loop - runs until time limit or dot stops moving
        while current_time < max_time:
            # ----------------------------------------
            # STEP 1: Apply gravity (constant acceleration)
            # ----------------------------------------
            # Euler integration: v(t+dt) = v(t) + a*dt
            current_vel = current_vel + gravity * dt
            
            # ----------------------------------------
            # STEP 2: Update position based on velocity
            # ----------------------------------------
            # Euler integration: x(t+dt) = x(t) + v(t)*dt
            new_pos = current_pos + current_vel * dt
            
            # ----------------------------------------
            # STEP 3: Check for collision with circle boundary
            # ----------------------------------------
            # Calculate distance from center (ignore z-coordinate for 2D collision)
            distance_from_center = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
            
            # Collision boundary is circle radius minus dot radius
            # This ensures the dot's edge touches the circle, not its center
            collision_distance = circle_radius - dot_radius
            
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
                # This prevents the dot from "escaping" through the wall
                new_pos[0] = direction_2d[0] * collision_distance
                new_pos[1] = direction_2d[1] * collision_distance
                
                # Calculate the inward normal (points toward center)
                # This is the direction we'll reflect the velocity across
                normal_2d = -direction_2d
                
                # Project velocity onto the normal to get the component moving into the wall
                # This tells us how "hard" the dot is hitting the wall
                vel_normal = np.dot(current_vel[:2], normal_2d)
                
                # Only reflect if moving toward the wall (vel_normal < 0)
                # This prevents double-bouncing if the dot is already moving away
                if vel_normal < 0:
                    # Reflect velocity using: v' = v - 2*(v·n)*n
                    # This is the standard formula for reflecting a vector across a normal
                    current_vel[0] = current_vel[0] - 2 * vel_normal * normal_2d[0]
                    current_vel[1] = current_vel[1] - 2 * vel_normal * normal_2d[1]
                    
                    # Apply energy loss due to inelastic collision
                    # Real-world bounces aren't perfectly elastic
                    current_vel = current_vel * damping
                    
                    print(f"  After bounce - vel: ({current_vel[0]:.2f}, {current_vel[1]:.2f}), speed: {np.linalg.norm(current_vel):.2f}")
            
            # Update position for next iteration
            current_pos = new_pos
            
            # Store position for animation
            positions.append(current_pos.copy())
            current_time += dt
            
            # ----------------------------------------
            # STEP 4: Check stopping condition
            # ----------------------------------------
            # Stop if velocity becomes very small (dot has essentially stopped)
            speed = np.linalg.norm(current_vel)
            if speed < 0.05 and current_time > 3:
                print(f"Stopping simulation at time {current_time:.2f}s, speed: {speed:.4f}")
                
                # Add extra frames at the final position
                # This ensures the animation doesn't end abruptly
                for _ in range(30):  # 0.3 more seconds of stillness
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
        
        # Calculate total animation duration from number of positions
        simulation_dt = dt
        total_duration = (len(positions) - 1) * simulation_dt
        
        print("Generating audio files...")
        audio_dir = "media/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        # ----------------------------------------
        # Generate ambient background sound
        # ----------------------------------------
        # Make it slightly longer than animation to ensure full coverage
        ambient_duration = total_duration + 1
        ambient_sound = generate_ambient_sound(duration=ambient_duration)
        ambient_path = os.path.join(audio_dir, "ambient.wav")
        wavfile.write(ambient_path, 44100, ambient_sound)
        print(f"  Ambient sound saved to {ambient_path}")
        
        # ----------------------------------------
        # Generate hit sounds and overlay them
        # ----------------------------------------
        # Create an empty audio buffer to hold all hit sounds
        sample_rate = 44100
        total_samples = int(ambient_duration * sample_rate)
        hit_audio = np.zeros(total_samples, dtype=np.int16)
        
        # Normalize bounce speeds to volume range (0.2 to 1.0)
        # This ensures louder sounds for faster bounces, quieter for slower ones
        if bounce_speeds:
            max_speed = max(bounce_speeds)
            min_speed = min(bounce_speeds)
            speed_range = max_speed - min_speed if max_speed > min_speed else 1
        
        # Generate and place each hit sound at its bounce time
        for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
            # Map speed to volume range linearly
            # Fastest bounce = volume 1.0, slowest = volume 0.2
            volume = 0.2 + 0.8 * ((speed - min_speed) / speed_range if speed_range > 0 else 1.0)
            
            # Vary frequency slightly for each bounce to add variety
            # Alternates between 440Hz, 540Hz, and 640Hz
            frequency = 440 + (i % 3) * 100
            
            # Generate the hit sound with calculated parameters
            hit_sound = generate_hit_sound(frequency=frequency, volume=volume)
            
            # Calculate where in the audio timeline this bounce occurs
            start_sample = int(bounce_time * sample_rate)
            end_sample = min(start_sample + len(hit_sound), total_samples)
            
            # Overlay the hit sound onto the timeline
            # Note: This directly assigns, not adds, so overlapping sounds will replace
            if start_sample < total_samples:
                hit_audio[start_sample:end_sample] = hit_sound[:end_sample - start_sample]
            
            print(f"  Bounce {i+1} at {bounce_time:.2f}s, speed: {speed:.2f}, volume: {volume:.2f}")
        
        # ----------------------------------------
        # Mix ambient and hit sounds together
        # ----------------------------------------
        # Convert to float32 for mixing to avoid overflow
        mixed_audio = ambient_sound.astype(np.float32) + hit_audio.astype(np.float32)
        # ----------------------------------------
        # Mix ambient and hit sounds together
        # ----------------------------------------
        # Convert to float32 for mixing to avoid overflow
        mixed_audio = ambient_sound.astype(np.float32) + hit_audio.astype(np.float32)
        
        # Normalize to prevent clipping (values exceeding int16 range)
        # Find the maximum absolute value in the mixed audio
        max_val = np.abs(mixed_audio).max()
        if max_val > 32767:
            # Scale down to fit within int16 range
            mixed_audio = mixed_audio * (32767 / max_val)
        
        # Convert back to int16 for WAV file format
        mixed_audio = mixed_audio.astype(np.int16)
        
        # Save the final mixed audio file
        mixed_path = os.path.join(audio_dir, "bounce_with_audio.wav")
        wavfile.write(mixed_path, sample_rate, mixed_audio)
        print(f"  Mixed audio saved to {mixed_path}")
        print()
        
        # ========================================
        # MANIM ANIMATION SETUP
        # ========================================
        
        # ValueTracker allows smooth interpolation of a value over time
        # We'll use it to track which position in our simulation to display
        time_tracker = ValueTracker(0)
        
        print(f"Animation settings:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Will animate through {len(positions)} positions")
        print(f"  Animation dt: {simulation_dt}")
        print()
        
        # ----------------------------------------
        # Create animated dot using always_redraw
        # ----------------------------------------
        # always_redraw recreates the object every frame based on time_tracker value
        def get_dot_position():
            """
            Returns a Dot object at the position corresponding to the current time.
            
            This function is called every frame by Manim's always_redraw mechanism.
            It calculates which position index corresponds to the current animation time
            and creates a dot at that location.
            """
            # Get current animation time from the tracker
            t = time_tracker.get_value()
            
            # Calculate which position index corresponds to this time
            # Index = time / time_step (e.g., at t=0.5s with dt=0.005, index=100)
            index = int(t / simulation_dt)
            
            # Clamp to valid range to prevent index errors
            index = min(index, len(positions) - 1)
            
            # Create and return a new dot at the calculated position
            return Dot(point=positions[index], radius=dot_radius, color=RED)
        
        # Remove the original static dot
        self.remove(dot)
        
        # Add the animated dot that updates every frame
        animated_dot = always_redraw(get_dot_position)
        self.add(animated_dot)
        
        # ----------------------------------------
        # Add audio and animate
        # ----------------------------------------
        # Add the audio track to the scene - it will play during rendering
        self.add_sound(mixed_path)
        
        # Animate the time_tracker from 0 to total_duration
        # This drives the entire animation:
        # - run_time: How long the animation takes in real time
        # - rate_func=linear: Ensures constant speed (no easing)
        # As time_tracker changes, always_redraw automatically updates the dot position
        self.play(time_tracker.animate.set_value(total_duration), run_time=total_duration, rate_func=linear)
        
        # Hold the final frame for 1 second
        self.wait(1)


class BouncingDotWithTrail(Scene):
    """
    Enhanced version of BouncingDot that includes a visual trail.
    
    This version shows the path the dot has traveled by drawing a yellow line
    that follows behind it, making the motion more visible and aesthetically pleasing.
    """
    
    def construct(self):
        """
        Main construction method - similar to BouncingDot but with trail visualization.
        """
        # ========================================
        # SCENE SETUP (same as BouncingDot)
        # ========================================
        
        self.camera.background_color = BLACK
        
        # Circle boundary
        circle_center = np.array([0, 0, 0])
        circle_radius = 3
        circle = Circle(radius=circle_radius, color=BLUE, stroke_width=2)
        self.add(circle)
        
        # Dot configuration
        dot_start_pos = circle_center.copy()
        dot_radius = 0.15
        dot = Dot(point=dot_start_pos, radius=dot_radius, color=RED)
        self.add(dot)
        
        # Physics parameters (identical to BouncingDot)
        gravity = np.array([0, -9.8, 0])
        velocity = np.array([0.5, -3, 0])
        damping = 0.85
        dt = 0.005
        max_time = 12
        
        # ========================================
        # PHYSICS SIMULATION (same as BouncingDot)
        # ========================================
        # Note: This is a duplicate of the physics code above
        # In a production environment, you'd want to refactor this into a shared function
        
        positions = [dot_start_pos.copy()]
        
        current_pos = dot_start_pos.copy()
        current_vel = velocity.copy()
        current_time = 0
        
        while current_time < max_time:
            # Apply physics (same as BouncingDot - see above for detailed comments)
            current_vel = current_vel + gravity * dt
            new_pos = current_pos + current_vel * dt
            
            distance_from_center = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
            collision_distance = circle_radius - dot_radius
            
            if distance_from_center > collision_distance:
                direction_2d = np.array([new_pos[0], new_pos[1]]) / distance_from_center
                new_pos[0] = direction_2d[0] * collision_distance
                new_pos[1] = direction_2d[1] * collision_distance
                
                normal_2d = -direction_2d
                vel_normal = np.dot(current_vel[:2], normal_2d)
                
                if vel_normal < 0:
                    current_vel[0] = current_vel[0] - 2 * vel_normal * normal_2d[0]
                    current_vel[1] = current_vel[1] - 2 * vel_normal * normal_2d[1]
                    current_vel = current_vel * damping
            
            current_pos = new_pos
            positions.append(current_pos.copy())
            current_time += dt
            
            # Stop when velocity is negligible
            speed = np.linalg.norm(current_vel)
            if speed < 0.05 and current_time > 3:
                for _ in range(30):
                    positions.append(current_pos.copy())
                break
        
        # ========================================
        # ANIMATION WITH TRAIL
        # ========================================
        
        # Create an empty trail object (will be updated by always_redraw)
        trail = VMobject(stroke_color=YELLOW, stroke_width=1, stroke_opacity=0.6)
        self.add(trail)
        
        # ValueTracker for animation timing
        time_tracker = ValueTracker(0)
        simulation_dt = dt
        total_duration = (len(positions) - 1) * simulation_dt
        
        # ----------------------------------------
        # Animated dot (same as BouncingDot)
        # ----------------------------------------
        def get_dot_position():
            """Get the dot at the current time position."""
            t = time_tracker.get_value()
            index = int(t / simulation_dt)
            index = min(index, len(positions) - 1)
            return Dot(point=positions[index], radius=dot_radius, color=RED)
        
        # ----------------------------------------
        # Animated trail
        # ----------------------------------------
        def get_trail_path():
            """
            Create a trail showing the path traveled so far.
            
            The trail is built from the start position up to the current position,
            sampling every 3rd point to avoid excessive detail and improve performance.
            """
            t = time_tracker.get_value()
            index = int(t / simulation_dt)
            index = min(index, len(positions) - 1)
            
            # Need at least 2 points to draw a line
            if index < 2:
                return VMobject(stroke_color=YELLOW, stroke_width=1, stroke_opacity=0.6)
            
            # Create new trail object for this frame
            trail_obj = VMobject(stroke_color=YELLOW, stroke_width=1, stroke_opacity=0.6)
            
            # Sample every 3rd position to balance visual quality and performance
            # [::3] means "take every 3rd element"
            trail_points = positions[:index:3]
            
            if len(trail_points) > 1:
                # Connect the points with straight line segments
                trail_obj.set_points_as_corners(trail_points)
            
            return trail_obj
        
        # ----------------------------------------
        # Setup and run animation
        # ----------------------------------------
        # Remove static objects and replace with animated versions
        self.remove(dot, trail)
        animated_dot = always_redraw(get_dot_position)
        animated_trail = always_redraw(get_trail_path)
        
        # Add trail first so it appears behind the dot
        self.add(animated_trail, animated_dot)
        
        # Animate (no audio in this version, but could be added)
        self.play(time_tracker.animate.set_value(total_duration), run_time=total_duration, rate_func=linear)
        
        # Brief pause at the end
        self.wait(0.5)

        