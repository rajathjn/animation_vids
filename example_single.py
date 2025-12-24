"""
Example: Generate a single bouncing dot animation with custom settings.

This example demonstrates:
1. Creating a custom configuration
2. Overriding specific parameters
3. Rendering an animation directly (no temp files or subprocess)
"""

from animations import AnimationConfig, render_animation


def main():
    print("\n" + "="*60)
    print("BOUNCING DOT ANIMATION")
    print("="*60 + "\n")
    
    # Create a config instance
    config = AnimationConfig()
    
    # Override specific settings
    config.override(
        damping=0.99,              # More energy loss on bounce
        dot_color='RED',           # Red dot
        doct_start_x=0.0,         # Start at center
        dot_start_y=2.5,          # Start higher
        enable_trail=True,         # Enable trail effect
        trail_color='YELLOW',      # Yellow trail
        initial_velocity_x=-3.0,   # Faster horizontal movement
        initial_velocity_y=-6.0,   # Faster downward movemen
    )
    
    # Render the animation directly
    output_path = render_animation(
        animation_name="BouncingDot",
        config=config,
        output_name="red_dot_with_trail",
        quality="medium_quality",  # Options: low_quality, medium_quality, high_quality
        preview=False,
    )
    
    print(f"\n✓ Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
