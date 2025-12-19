"""
Main orchestrator for generating animation videos.

This module provides a simple interface to configure and generate multiple
animations without manually editing source files or renaming outputs.

Usage:
    from animations.main import generate_animation
    
    # Generate with defaults
    generate_animation("BouncingDot", output_name="my_bounce_video")
    
    # Generate with custom config
    from animations.config import AnimationConfig
    config = AnimationConfig()
    config.override(damping=0.95, dot_color='RED', enable_trail=True)
    generate_animation("BouncingDot", config=config, output_name="red_dot_with_trail")
    
    # Generate multiple dots
    dots = [
        {"initial_velocity": np.array([2, -5, 0]), "damping": 0.96, "radius": 0.2, 
         "color": "YELLOW", "start_pos": np.array([0, 1, 0])},
        {"initial_velocity": np.array([-3, -6, 0]), "damping": 0.97, "radius": 0.18, 
         "color": "PURPLE", "start_pos": np.array([1, -1, 0])},
    ]
    generate_animation("BouncingDots", dots=dots, output_name="two_dots_custom")
"""

import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import tempfile
import numpy as np

if TYPE_CHECKING:
    from .config import AnimationConfig


def generate_animation(
    animation_name: str,
    config: Optional['AnimationConfig'] = None,
    dots: Optional[List[Dict[str, Any]]] = None,
    output_name: Optional[str] = None,
    quality: str = "m",
    preview: bool = False,
    organize_output: bool = True
) -> Path:
    """
    Generate an animation video using manim.
    
    Args:
        animation_name: Name of the animation class ("BouncingDot" or "BouncingDots")
        config: Optional AnimationConfig instance with custom settings
        dots: Optional list of dot configurations (only for BouncingDots)
        output_name: Optional custom name for the output (defaults to animation_name_timestamp)
        quality: Manim quality flag: 'l' (low), 'm' (medium), 'h' (high) (default: 'm')
        preview: Whether to preview the video after generation (default: False)
        organize_output: Whether to organize output into a dedicated folder (default: True)
    
    Returns:
        Path to the generated video file (or output folder if organize_output=True)
    
    Raises:
        ValueError: If animation_name is not recognized
        RuntimeError: If manim rendering fails
    """
    # Validate animation name
    valid_animations = ["BouncingDot", "BouncingDots"]
    if animation_name not in valid_animations:
        raise ValueError(f"Unknown animation: {animation_name}. Valid options: {valid_animations}")
    
    # Import here to avoid circular imports
    from .config import AnimationConfig
    
    # Create config if not provided
    if config is None:
        config = AnimationConfig()
    
    # Generate output name with timestamp
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{animation_name}_{timestamp}"
    
    # Create output folder first (so audio can be saved directly there)
    outputs_dir = Path.cwd() / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_folder = outputs_dir / output_name
    output_folder.mkdir(exist_ok=True)
    
    # Create audio subfolder
    audio_output_dir = output_folder / "audio"
    audio_output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating animation: {animation_name}")
    print(f"Output name: {output_name}")
    print(f"Quality: {quality}")
    print(f"Renderer: cairo")
    print(f"{'='*60}\n")
    
    # Create a temporary Python file that will be passed to manim
    # This file imports the animation class and instantiates it with the config
    temp_dir = Path(tempfile.gettempdir()) / "animation_vids_temp"
    temp_dir.mkdir(exist_ok=True)
    
    temp_script = temp_dir / f"{animation_name.lower()}_temp.py"
    
    # Get the actual project root (where animations folder is)
    project_root = Path.cwd()
    
    # Build the temporary script content
    script_content = f"""
from manim import *
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, r'{project_root}')

from animations.config import AnimationConfig
from animations.{animation_name.lower().replace('bouncing', 'bouncing_')} import {animation_name}

# Create config
config = AnimationConfig()

# Set audio output directory
config.audio_output_dir = r'{audio_output_dir}'

# Override config values
"""
    
    # Add config overrides to the script
    for key, value in config.config.items():
        if isinstance(value, str):
            script_content += f"config.config['{key}'] = '{value}'\n"
        else:
            script_content += f"config.config['{key}'] = {value}\n"
    
    # Add scene class instantiation
    if animation_name == "BouncingDots" and dots is not None:
        # Convert numpy arrays to lists for JSON serialization
        dots_serializable = []
        for dot in dots:
            dot_copy = dot.copy()
            for key in ['initial_velocity', 'start_pos']:
                if key in dot_copy and isinstance(dot_copy[key], np.ndarray):
                    dot_copy[key] = dot_copy[key].tolist()
            dots_serializable.append(dot_copy)
        
        dots_json = json.dumps(dots_serializable)
        script_content += f"""
# Reconstruct dots config from JSON
import json
dots_data = json.loads('{dots_json}')
for dot in dots_data:
    for key in ['initial_velocity', 'start_pos']:
        if key in dot:
            dot[key] = np.array(dot[key])

class {animation_name}Scene({animation_name}):
    def __init__(self, renderer=None, **kwargs):
        super().__init__(config=config, dots=dots_data, **kwargs)
"""
    else:
        script_content += f"""
class {animation_name}Scene({animation_name}):
    def __init__(self, renderer=None, **kwargs):
        super().__init__(config=config, **kwargs)
"""
    
    # Write temp script
    temp_script.write_text(script_content, encoding='utf-8')
    
    # Define output video path
    final_video = output_folder / f"{output_name}.mp4"
    
    try:
        # Build manim command with direct output
        # Note: -o flag works with Cairo renderer, so we use that instead of OpenGL
        cmd = [
            "manim",
            #f"-q{quality}",
            "-o", str(final_video),
            str(temp_script),
            f"{animation_name}Scene",
        ]
        
        if preview:
            cmd.insert(1, "-p")
        
        print(f"Running manim (this may take a few minutes)...\n")
        print(f"Quality: {quality} | Scene: {animation_name}Scene\n")
        print("-" * 60 + "\n")
        
        # Run manim with direct output to terminal for smooth progress bars
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd())
        )
        
        result_returncode = result.returncode
        print("\n" + "-" * 60 + "\n")
        
        if result_returncode != 0:
            raise RuntimeError(f"Manim rendering failed with return code {result_returncode}")
        
        print("[OK] Rendering complete!\n")
        
        # Verify video was created
        if not final_video.exists():
            raise RuntimeError(f"Video file was not created at {final_video}")
        
        print(f"[OK] Video saved: {final_video.name}")
        
        if organize_output:
            # Save metadata
            print("[*] Saving metadata...")
            metadata = {
                "animation_name": animation_name,
                "output_name": output_name,
                "timestamp": datetime.now().isoformat(),
                "quality": quality,
                "renderer": "cairo",
                "config": config.config,
            }
            
            if dots is not None:
                metadata["dots"] = dots_serializable
            
            metadata_file = output_folder / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f"    [OK] Metadata saved\n")
            
            print(f"[OK] Output organized in: {output_folder}")
            print(f"     Video: {final_video.name}")
            print(f"     Metadata: metadata.json")
            if audio_output_dir.exists() and any(audio_output_dir.glob("*.wav")):
                print(f"     Audio: audio/")
            
            return output_folder
        else:
            return final_video
            
    finally:
        # Cleanup temp script
        if temp_script.exists():
            temp_script.unlink()
        
        print(f"\n{'='*60}")
        print(f"Animation generation complete!")
        print(f"{'='*60}\n")


# Convenience function for batch generation
def generate_multiple(animations: List[Dict[str, Any]], **common_kwargs) -> List[Path]:
    """
    Generate multiple animations in sequence.
    
    Args:
        animations: List of animation specifications, each dict containing:
                   - animation_name: str
                   - config: Optional AnimationConfig
                   - dots: Optional list (for BouncingDots)
                   - output_name: Optional str
        **common_kwargs: Common arguments applied to all animations (quality, preview, etc.)
    
    Returns:
        List of output paths (videos or folders)
    
    Example:
        animations = [
            {"animation_name": "BouncingDot", "output_name": "dot_1"},
            {"animation_name": "BouncingDot", "output_name": "dot_2", 
             "config": config_with_trail},
        ]
        generate_multiple(animations, quality='h', preview=False)
    """
    results = []
    
    for i, anim_spec in enumerate(animations, 1):
        print(f"\n{'#'*60}")
        print(f"# Animation {i}/{len(animations)}")
        print(f"{'#'*60}")
        
        # Merge animation-specific and common kwargs
        kwargs = {**common_kwargs, **anim_spec}
        animation_name = kwargs.pop("animation_name")
        
        try:
            result = generate_animation(animation_name, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"✗ Failed to generate {animation_name}: {e}")
            results.append(None)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Batch generation complete!")
    print(f"  Successful: {sum(1 for r in results if r is not None)}/{len(animations)}")
    print(f"{'='*60}\n")
    
    return results
