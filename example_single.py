"""
Example: Generate a single bouncing dot animation with custom settings.

This example demonstrates:
1. Creating a custom configuration
2. Overriding specific parameters
3. Generating a single animation

NOTE: The first run may take 2-5 minutes depending on your system and quality settings.
You'll see real-time progress in the terminal.
"""

import numpy as np
from animations import AnimationConfig, generate_animation


def main():
    print("\n" + "="*60)
    print("BOUNCING DOT ANIMATION GENERATOR")
    print("="*60)
    print("\n>>> This will take a few minutes to generate...")
    print("    Watch the terminal for real-time progress updates\n")
    
    # Create a config instance
    config = AnimationConfig()
    
    # Override specific settings
    config.override(
        damping=0.95,              # More energy loss on bounce
        dot_color='RED',           # Red dot
        enable_trail=True,         # Enable trail effect
        trail_color='YELLOW',      # Yellow trail
        initial_velocity_x=-3.0,   # Faster horizontal movement
        initial_velocity_y=-6.0,   # Faster downward movement
    )
    
    # Generate the animation
    output_path = generate_animation(
        animation_name="BouncingDot",
        config=config,
        output_name="red_dot_with_trail",
        quality="m",               # Medium quality (720p30)
        preview=False,             # Don't auto-preview
        organize_output=True       # Organize into output folder
    )
    
    print(f"\n{'='*60}")
    print(f"[OK] COMPLETE! Animation saved to:")
    print(f"     {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
