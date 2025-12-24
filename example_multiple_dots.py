"""
Example: Generate multiple bouncing dots animation with custom dot configurations.

This example demonstrates:
1. Defining custom dot configurations
2. Creating multiple dots with different properties
3. Rendering the animation directly
"""

from animations import AnimationConfig, render_animation


def main():
    # Create a config instance
    config = AnimationConfig()
    
    # Override global settings
    config.override(
        enable_trail=False,
        trail_width=2.0,
        trail_opacity=0.7,
        debug=False,
    )
    
    # Define custom dots configuration
    # Note: You can use plain lists instead of np.array - they'll be converted automatically
    dots = [
        {
            "initial_velocity": [2, -5, 0],
            "damping": 0.95,
            "radius": 0.30,
            "color": "YELLOW",
            "start_pos": [0, 1, 0]
        },
        {
            "initial_velocity": [-10, -1, 0],
            "damping": 0.98,
            "radius": 0.20,
            "color": "PURPLE",
            "start_pos": [1, -1, 0]
        },
        {
            "initial_velocity": [10, -1, 0],
            "damping": 0.99,
            "radius": 0.20,
            "color": "ORANGE",
            "start_pos": [-1, 0.5, 0]
        },
    ]
    
    # Render the animation
    output_path = render_animation(
        animation_name="BouncingDots",
        config=config,
        dots=dots,
        output_name="three_colorful_dots",
        quality="high_quality",
        preview=False,
    )
    
    print(f"\n✓ Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
