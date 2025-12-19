"""
Example: Using app.cfg file for configuration.

This example demonstrates:
1. Creating an app.cfg file in the same directory
2. Letting the config system automatically load it
3. Generating animations without programmatic config
"""

from animations import generate_animation


def main():
    # The AnimationConfig will automatically look for app.cfg in this directory
    # If found, it will override the defaults from animations/app.cfg
    
    print("Generating animation using app.cfg from this directory...")
    print("(If app.cfg doesn't exist here, defaults will be used)\n")
    
    # Generate animation - config is loaded automatically
    output_path = generate_animation(
        animation_name="BouncingDot",
        output_name="config_file_example",
        quality="m",
        preview=False,
        organize_output=True
    )
    
    print(f"\n✓ Animation saved to: {output_path}")
    print("\nTIP: Create an app.cfg file in this directory to customize settings!")
    print("Example app.cfg content:")
    print("""
[animation]
dot_color = PURPLE
damping = 0.95
enable_trail = true
trail_color = YELLOW
    """)


if __name__ == "__main__":
    main()
