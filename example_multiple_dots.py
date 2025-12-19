"""
Example: Generate multiple bouncing dots animation with custom dot configurations.

This example demonstrates:
1. Defining custom dot configurations
2. Creating multiple dots with different properties
3. Generating the animation with custom settings
"""

import numpy as np
from animations import AnimationConfig, generate_animation


def main():
    # Create a config instance
    config = AnimationConfig()
    
    # Override global settings
    config.override(
        ENABLE_TRAIL=True,
        TRAIL_WIDTH=2.0,
        TRAIL_OPACITY=0.7,
        DEBUG=False,
    )
    
    # Define custom dots configuration
    dots = [
        {
            "initial_velocity": np.array([2, -5, 0]),
            "damping": 0.99,
            "radius": 0.32,
            "color": "YELLOW",
            "start_pos": np.array([0, 1, 0])
        },
        {
            "initial_velocity": np.array([-3, -6, 0]),
            "damping": 0.98,
            "radius": 0.5,
            "color": "PURPLE",
            "start_pos": np.array([1, -1, 0])
        },
        {
            "initial_velocity": np.array([10, -1, 0]),
            "damping": 0.99,
            "radius": 0.20,
            "color": "ORANGE",
            "start_pos": np.array([-1, 0.5, 0])
        },
    ]
    
    # Generate the animation
    output_path = generate_animation(
        animation_name="BouncingDots",
        config=config,
        dots=dots,
        output_name="three_colorful_dots",
        quality="h",
        preview=False,
        organize_output=True
    )
    
    print(f"\n✓ Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
