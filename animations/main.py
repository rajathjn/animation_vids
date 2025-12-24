"""
Main orchestrator for generating animation videos.

This module provides a simple interface to configure and render manim animations
directly without subprocess calls or temp files.

Usage:
    from animations import render_animation, AnimationConfig
    
    # Simple render with defaults
    render_animation("BouncingDot", output_name="my_bounce")
    
    # Render with custom config
    config = AnimationConfig()
    config.override(damping=0.95, dot_color='RED', enable_trail=True)
    render_animation("BouncingDot", config=config, output_name="red_dot")
    
    # Render multiple dots
    dots = [
        {"initial_velocity": [2, -5, 0], "damping": 0.96, "radius": 0.2, 
         "color": "YELLOW", "start_pos": [0, 1, 0]},
        {"initial_velocity": [-3, -6, 0], "damping": 0.97, "radius": 0.18, 
         "color": "PURPLE", "start_pos": [1, -1, 0]},
    ]
    render_animation("BouncingDots", dots=dots, output_name="multi_dots")
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from configparser import ConfigParser
import numpy as np

from manim import config as manim_config, tempconfig

from .config import AnimationConfig
from .bouncing_dot import BouncingDot
from .bouncing_dots import BouncingDots


# Map animation names to classes
ANIMATION_CLASSES = {
    "BouncingDot": BouncingDot,
    "BouncingDots": BouncingDots,
}


def render_animation(
    animation_name: str,
    config: Optional[AnimationConfig] = None,
    dots: Optional[List[Dict[str, Any]]] = None,
    output_name: Optional[str] = None,
    quality: str = "medium_quality",
    preview: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Render an animation directly using manim's Python API.
    
    Args:
        animation_name: Name of the animation class ("BouncingDot" or "BouncingDots")
        config: Optional AnimationConfig instance with custom settings
        dots: Optional list of dot configurations (only for BouncingDots)
        output_name: Optional custom name for the output file
        quality: Manim quality preset: 'low_quality', 'medium_quality', 
                 'high_quality', 'production_quality', or 'fourk_quality'
        preview: Whether to open the video after rendering
        output_dir: Custom output directory (default: ./outputs)
    
    Returns:
        Path to the generated video file
    
    Raises:
        ValueError: If animation_name is not recognized
    """
    # Validate animation name
    if animation_name not in ANIMATION_CLASSES:
        raise ValueError(
            f"Unknown animation: {animation_name}. "
            f"Valid options: {list(ANIMATION_CLASSES.keys())}"
        )
    
    # Create config if not provided
    if config is None:
        config = AnimationConfig()
    
    # Generate output name with timestamp if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{animation_name}_{timestamp}"
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subfolder for this animation
    animation_output_dir = output_dir / output_name
    animation_output_dir.mkdir(exist_ok=True)
    
    # Setup audio output directory
    audio_dir = animation_output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    config.audio_output_dir = str(audio_dir)
    
    print(f"\n{'='*60}")
    print(f"Rendering: {animation_name}")
    print(f"Output: {output_name}")
    print(f"Quality: {quality}")
    print(f"{'='*60}\n")
    
    # Convert dots list to use numpy arrays if provided
    if dots is not None:
        dots = _prepare_dots_config(dots)
    
    # Load manim.cfg from project root if it exists to respect user settings
    project_root = Path.cwd()
    manim_cfg_path = project_root / "manim.cfg"
    manim_settings = {
        "preview": preview,
        "output_file": output_name,
    }
    
    if manim_cfg_path.exists():
        # Read manim.cfg and apply settings from it
        parser = ConfigParser()
        parser.read(manim_cfg_path)
        
        if parser.has_section('CLI'):
            # Apply CLI settings, respecting user's dimensions
            for key, value in parser.items('CLI'):
                if key == 'frame_rate':
                    manim_settings['frame_rate'] = int(value)
                elif key == 'pixel_height':
                    manim_settings['pixel_height'] = int(value)
                elif key == 'pixel_width':
                    manim_settings['pixel_width'] = int(value)
                elif key == 'background_color':
                    manim_settings['background_color'] = value
                elif key == 'background_opacity':
                    manim_settings['background_opacity'] = float(value)
                elif key == 'frame_width':
                    manim_settings['frame_width'] = float(value)
                elif key == 'frame_height':
                    manim_settings['frame_height'] = float(value)
                elif key == 'disable_caching':
                    manim_settings['disable_caching'] = value.lower() == 'true'
                elif key == 'flush_cache':
                    manim_settings['flush_cache'] = value.lower() == 'true'
    else:
        # Fallback to quality preset if no manim.cfg
        manim_settings["quality"] = quality
    
    # Render the scene using manim's tempconfig context manager
    with tempconfig(manim_settings):
        # Create the scene instance
        scene_class = ANIMATION_CLASSES[animation_name]
        
        if animation_name == "BouncingDots" and dots is not None:
            scene = scene_class(config=config, dots=dots)
        else:
            scene = scene_class(config=config)
        
        # Render the scene
        scene.render()
        
        # Get the output video path from manim
        video_path = Path(scene.renderer.file_writer.movie_file_path)
    
    # Copy video to our organized output folder
    final_video_path = animation_output_dir / f"{output_name}.mp4"
    if video_path.exists():
        if video_path != final_video_path:
            shutil.copy2(str(video_path), str(final_video_path))
    else:
        raise RuntimeError(f"Video was not created at {video_path}")
    
    # Save metadata
    _save_metadata(
        animation_output_dir,
        animation_name=animation_name,
        output_name=output_name,
        quality=quality,
        config=config,
        dots=dots,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Rendering complete!")
    print(f"  Video: {final_video_path}")
    print(f"  Folder: {animation_output_dir}")
    print(f"{'='*60}\n")
    
    return final_video_path


def _prepare_dots_config(dots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert dot config lists to numpy arrays."""
    prepared = []
    for dot in dots:
        dot_copy = dot.copy()
        for key in ['initial_velocity', 'start_pos']:
            if key in dot_copy:
                value = dot_copy[key]
                if isinstance(value, list):
                    dot_copy[key] = np.array(value)
        prepared.append(dot_copy)
    return prepared


def _save_metadata(
    output_dir: Path,
    animation_name: str,
    output_name: str,
    quality: str,
    config: AnimationConfig,
    dots: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save animation metadata to JSON."""
    metadata = {
        "animation_name": animation_name,
        "output_name": output_name,
        "timestamp": datetime.now().isoformat(),
        "quality": quality,
        "config": config.to_dict(),
    }
    
    if dots is not None:
        # Convert numpy arrays for JSON serialization
        dots_serializable = []
        for dot in dots:
            dot_copy = {}
            for key, value in dot.items():
                if isinstance(value, np.ndarray):
                    dot_copy[key] = value.tolist()
                else:
                    dot_copy[key] = value
            dots_serializable.append(dot_copy)
        metadata["dots"] = dots_serializable
    
    metadata_file = output_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
