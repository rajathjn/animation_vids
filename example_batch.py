"""
Example: Generate multiple different animations in a batch.

This example demonstrates:
1. Generating multiple animations with different configurations
2. Using the batch generation feature
3. Creating variations of the same animation
"""

import numpy as np
from animations import AnimationConfig, generate_multiple


def main():
    # Define multiple animation configurations
    animations = []
    
    # Animation 1: Simple bouncing dot
    config1 = AnimationConfig()
    config1.override(
        dot_color='BLUE',
        damping=0.99,
        enable_trail=False,
    )
    animations.append({
        "animation_name": "BouncingDot",
        "config": config1,
        "output_name": "simple_blue_dot",
    })
    
    # Animation 2: Red dot with trail
    config2 = AnimationConfig()
    config2.override(
        dot_color='RED',
        damping=0.95,
        enable_trail=True,
        trail_color='YELLOW',
        initial_velocity_x=-4.0,
        initial_velocity_y=-8.0,
    )
    animations.append({
        "animation_name": "BouncingDot",
        "config": config2,
        "output_name": "red_dot_fast_with_trail",
    })
    
    # Animation 3: Multiple dots collision
    config3 = AnimationConfig()
    config3.override(
        enable_trail=True,
        trail_opacity=0.5,
    )
    dots = [
        {
            "initial_velocity": np.array([3, -4, 0]),
            "damping": 0.95,
            "radius": 0.25,
            "color": "RED",
            "start_pos": np.array([-1, 1, 0])
        },
        {
            "initial_velocity": np.array([-3, -4, 0]),
            "damping": 0.95,
            "radius": 0.25,
            "color": "BLUE",
            "start_pos": np.array([1, 1, 0])
        },
    ]
    animations.append({
        "animation_name": "BouncingDots",
        "config": config3,
        "dots": dots,
        "output_name": "two_dots_collision",
    })
    
    # Animation 4: Chaotic multi-dot system
    config4 = AnimationConfig()
    config4.override(enable_trail=False)
    dots_chaotic = [
        {
            "initial_velocity": np.array([i*0.5, -5-i*0.3, 0]),
            "damping": 0.96 + i*0.01,
            "radius": 0.15 + i*0.03,
            "color": ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE"][i],
            "start_pos": np.array([np.sin(i)*1.5, np.cos(i)*1.5, 0])
        }
        for i in range(5)
    ]
    animations.append({
        "animation_name": "BouncingDots",
        "config": config4,
        "dots": dots_chaotic,
        "output_name": "five_dots_chaos",
    })
    
    # Generate all animations
    print("\n" + "="*60)
    print(f"Starting batch generation of {len(animations)} animations")
    print("="*60 + "\n")
    
    results = generate_multiple(
        animations,
        quality='m',        # Medium quality for all
        preview=False,      # No preview
        organize_output=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH GENERATION SUMMARY")
    print("="*60)
    for i, (anim, result) in enumerate(zip(animations, results), 1):
        status = "✓" if result else "✗"
        print(f"{status} {i}. {anim['output_name']}")
        if result:
            print(f"   Output: {result}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
