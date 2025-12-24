"""
Example: Generate multiple bouncing dots animation with custom dot configurations.

This example demonstrates:
1. Creating and initializing AnimationConfig
2. Using override() for animation settings and DOTS_JSON
3. Using override_manim() for rendering settings
4. Each dot can have its own color, radius, position, velocity, and damping

DOTS_JSON format (list of dictionaries):
    Each dot dict should have:
    - "color": str - Manim color name (e.g., "RED", "BLUE", "YELLOW")
    - "radius": float - Dot radius
    - "start_pos": [x, y, z] - Starting position
    - "initial_velocity": [x, y, z] - Initial velocity vector
    - "damping": float - Energy loss on bounce (0-1)

If DOTS_JSON is empty and USE_SINGLE_DOT_DEFAULTS is true,
a single dot using [animation] section values will be created.
If both are empty/false, a ValueError is raised.
"""

from animations import AnimationConfig, render_animation


def main():
    print("\n" + "="*60)
    print("MULTI-DOT BOUNCING ANIMATION")
    print("="*60 + "\n")
    
    # Create a config instance (loads defaults from animations/app.cfg)
    config = AnimationConfig()
    
    # Override animation settings and define custom dots
    config.override({
        # Global trail settings
        "ENABLE_TRAIL": False,
        "TRAIL_WIDTH": 2.0,
        "TRAIL_OPACITY": 0.7,
        "DEBUG": False,
        
        # Custom dots configuration (overrides app.cfg DOTS_JSON)
        "DOTS_JSON": [
            {
                "color": "YELLOW",
                "radius": 0.30,
                "start_pos": [0, 1, 0],
                "initial_velocity": [2, -5, 0],
                "damping": 0.95,
            },
            {
                "color": "PURPLE",
                "radius": 0.20,
                "start_pos": [1, -1, 0],
                "initial_velocity": [-10, -1, 0],
                "damping": 0.98,
            },
            {
                "color": "ORANGE",
                "radius": 0.20,
                "start_pos": [-1, 0.5, 0],
                "initial_velocity": [10, -1, 0],
                "damping": 0.99,
            },
        ],
    })
    
    # Override manim rendering settings (optional)
    # config.override_manim({
    #     "RENDERER": "cairo",
    #     "PIXEL_WIDTH": 1080,
    #     "PIXEL_HEIGHT": 1920,
    # })
    
    # Render the animation
    output_path = render_animation(
        animation_name="BouncingDots",
        config=config
    )
    
    print(f"\n Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
