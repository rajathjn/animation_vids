"""
Main orchestrator for generating animation videos.

This module provides a simple interface to configure and render manim animations
directly without subprocess calls or temp files.

Usage:
    from animations import render_animation, AnimationConfig

    # Simple render with defaults from app.cfg
    config = AnimationConfig()
    render_animation("BouncingDot", config=config, output_name="my_bounce")

    # Render with custom config overrides
    config = AnimationConfig()
    config.override({
        "DAMPING": 0.95,
        "DOT_COLOR": "RED",
        "ENABLE_TRAIL": True,
    })
    render_animation("BouncingDot", config=config, output_name="red_dot")

    # Render multiple dots (uses DOTS_JSON from config or override)
    config = AnimationConfig()
    config.override({
        "DOTS_JSON": [
            {"initial_velocity": [2, -5, 0], "damping": 0.96, "radius": 0.2,
             "color": "YELLOW", "start_pos": [0, 1, 0]},
            {"initial_velocity": [-3, -6, 0], "damping": 0.97, "radius": 0.18,
             "color": "PURPLE", "start_pos": [1, -1, 0]},
        ]
    })
    render_animation("BouncingDots", config=config, output_name="multi_dots")
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from manim import Scene, tempconfig

from .bouncing_dot import BouncingDot
from .bouncing_dots import BouncingDots
from .config import AnimationConfig

# Map animation names to classes
ANIMATION_CLASSES: dict[str, Scene] = {
    "BouncingDot": BouncingDot,
    "BouncingDots": BouncingDots,
}


def render_animation(
    animation_name: str,
    config: AnimationConfig,
    output_name: str | None = None,
    preview: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """
    Render an animation directly using manim's Python API.

    Args:
        animation_name: Name of the animation class ("BouncingDot" or "BouncingDots")
        config: AnimationConfig instance (required). Initialize and use override() to customize.
        output_name: Optional custom name for the output file
        preview: Whether to open the video after rendering
        output_dir: Custom output directory (default: from config.MEDIA_DIR)

    Returns:
        Path to the generated video file

    Raises:
        ValueError: If animation_name is not recognized
    """
    # Validate animation name
    if animation_name not in ANIMATION_CLASSES:
        raise ValueError(
            f"Unknown animation: {animation_name}. Valid options: {list(ANIMATION_CLASSES.keys())}"
        )

    # Generate output name with timestamp if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{animation_name}_{timestamp}"

    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd() / config.MEDIA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subfolder for this animation
    animation_output_dir = output_dir / output_name
    animation_output_dir.mkdir(exist_ok=True)

    # Setup audio output directory
    audio_dir = animation_output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    config.audio_output_dir = str(audio_dir)

    print(f"\n{'=' * 60}")
    print(f"Rendering: {animation_name}")
    print(f"Output: {output_name}")
    print(f"Resolution: {config.PIXEL_WIDTH}x{config.PIXEL_HEIGHT}")
    print(f"Renderer: {config.RENDERER}")
    print(f"{'=' * 60}\n")

    # Track rendering time using perf_counter for precise measurement
    render_start_time = time.perf_counter()

    # Get manim settings from config and add output file
    manim_settings = config.get_manim_config()
    manim_settings["preview"] = preview
    manim_settings["output_file"] = output_name

    # Render the scene using manim's tempconfig context manager
    with tempconfig(manim_settings):
        # Create the scene instance
        scene_class: Scene = ANIMATION_CLASSES[animation_name]
        scene: Scene = scene_class(config=config)

        # Render the scene
        scene.render()

        # Get the output video path from manim
        video_path = Path(scene.renderer.file_writer.movie_file_path)

    # Calculate elapsed time
    render_elapsed = time.perf_counter() - render_start_time

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
        config=config,
        render_time=render_elapsed,
    )

    # Format elapsed time
    minutes = int(render_elapsed // 60)
    seconds = render_elapsed % 60
    time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

    print(f"\n{'=' * 60}")
    print(" Rendering complete!")
    print(f"  Time elapsed: {time_str}")
    print(f"  Video: {final_video_path}")
    print(f"  Folder: {animation_output_dir}")
    print(f"{'=' * 60}\n")

    return final_video_path


def _save_metadata(
    output_dir: Path,
    animation_name: str,
    output_name: str,
    config: AnimationConfig,
    render_time: float = 0.0,
) -> None:
    """Save animation metadata to JSON."""
    metadata = {
        "animation_name": animation_name,
        "output_name": output_name,
        "timestamp": datetime.now().isoformat(),
        "render_time_seconds": render_time,
        "config": _serialize_config(config.to_dict()),
    }

    metadata_file = output_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _serialize_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Serialize config dict, converting numpy arrays and ManimColor objects to JSON-serializable types."""
    def convert_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(item) for item in value]
        else:
            # Convert any non-serializable type (ManimColor, etc.) to string
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)
    
    result = {}
    for section, values in config_dict.items():
        result[section] = convert_value(values)
    return result
