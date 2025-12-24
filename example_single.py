"""
Example: Generate a single bouncing dot animation with custom settings.

This example demonstrates:
1. Creating and initializing AnimationConfig
2. Using override() for animation settings
3. Using override_manim() for rendering settings
4. Rendering an animation directly (no temp files or subprocess)

Configuration methods:
    override() - Animation/physics/dots settings
    override_manim() - Manim rendering settings (can add new keys)

See AnimationConfig docstrings for full list of available keys.
"""

from animations import AnimationConfig, render_animation


def main():
    print("\n" + "="*60)
    print("BOUNCING DOT ANIMATION")
    print("="*60 + "\n")
    
    # Create a config instance (loads defaults from animations/app.cfg)
    config = AnimationConfig()
    
    # Override animation settings
    config.override({
        "DAMPING": 0.98,              # Energy loss on bounce (0-1)
        "DOT_COLOR": "RED",           # Dot color
        "DOT_START_X": 0.0,           # Start X position
        "DOT_START_Y": 0.0,           # Start Y position
        "ENABLE_TRAIL": False,        # Enable trail effect
        "TRAIL_COLOR": "YELLOW",      # Trail color
        "INITIAL_VELOCITY_X": -13.0,  # Horizontal velocity
        "INITIAL_VELOCITY_Y": -6.0,   # Vertical velocity
    })
        
    # Render the animation directly
    output_path = render_animation(
        animation_name="BouncingDot",
        config=config
    )
    
    print(f"\n Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
