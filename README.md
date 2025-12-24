# Animation Video Generator

A clean, simple framework for generating bouncing ball physics animations using Manim's Python API directly - no subprocess calls or temp files needed.

## 🎯 Features

- **Direct rendering**: Uses manim's Python API - no subprocess or temp files
- **Config-based setup**: Define animation parameters in `app.cfg` files
- **Modular design**: Separate classes for single and multiple bouncing dots
- **Simple API**: Import, configure, and render videos programmatically
- **Organized output**: Each video stored in its own folder with metadata

## 📁 Project Structure

```
animation_vids/
├── animations/
│   ├── __init__.py           # Package exports
│   ├── config.py             # AnimationConfig class
│   ├── app.cfg               # Default configuration
│   ├── audio_utils.py        # Audio generation utilities
│   ├── bouncing_dot.py       # Single dot animation
│   ├── bouncing_dots.py      # Multiple dots animation
│   └── main.py               # Main orchestrator
├── outputs/                   # Generated videos (organized by name)
├── example_single.py          # Example: Single dot
├── example_multiple_dots.py   # Example: Multiple dots
├── example_batch.py           # Example: Batch generation
├── manim.cfg                  # Manim configuration
└── README.md
```

## 🚀 Quick Start

### 1. Render with Defaults

```python
from animations import render_animation

# Render a simple bouncing dot
render_animation("BouncingDot", output_name="my_first_animation")
```

### 2. Render with Custom Config

```python
from animations import AnimationConfig, render_animation

# Create config and override settings
config = AnimationConfig()
config.override(
    damping=0.95,
    dot_color='RED',
    enable_trail=True,
    trail_color='YELLOW',
)

# Render animation
render_animation(
    animation_name="BouncingDot",
    config=config,
    output_name="red_dot_with_trail",
    quality="high_quality"
)
```

### 3. Render Multiple Dots

```python
from animations import AnimationConfig, render_animation

# Define dots - use plain lists (automatically converted to numpy arrays)
dots = [
    {
        "initial_velocity": [2, -5, 0],
        "damping": 0.96,
        "radius": 0.22,
        "color": "YELLOW",
        "start_pos": [0, 1, 0]
    },
    {
        "initial_velocity": [-3, -6, 0],
        "damping": 0.97,
        "radius": 0.18,
        "color": "PURPLE",
        "start_pos": [1, -1, 0]
    },
]

config = AnimationConfig()
config.override(enable_trail=True)

render_animation(
    animation_name="BouncingDots",
    config=config,
    dots=dots,
    output_name="two_dots"
)
```

### 4. Batch Rendering

```python
from animations import render_batch, AnimationConfig

config1 = AnimationConfig()
config1.override(dot_color='BLUE')

config2 = AnimationConfig()
config2.override(dot_color='RED', enable_trail=True)

animations = [
    {"animation_name": "BouncingDot", "output_name": "blue_dot", "config": config1},
    {"animation_name": "BouncingDot", "output_name": "red_dot", "config": config2},
]

render_batch(animations, quality='medium_quality')
```

## ⚙️ Configuration

### Using app.cfg Files

Create an `app.cfg` file in your script's directory to override defaults:

```ini
[animation]
# Circle boundary
circle_radius = 3.5

# Dot parameters
dot_color = PURPLE
dot_radius = 0.25

# Physics
damping = 0.95
gravity_y = -12.0

# Trail
enable_trail = true
trail_color = YELLOW
trail_width = 2.0

# Debug
debug = true
```

### Config Lookup Order

1. Script's directory (`app.cfg` in the same folder as your Python script)
2. `animations/app.cfg` (default configuration)
3. Hardcoded defaults in `config.py`

### Available Parameters

| Category | Parameters |
|----------|-----------|
| **Circle** | `circle_center_x/y/z`, `circle_radius` |
| **Dot** | `dot_start_x/y/z`, `dot_radius`, `dot_color` |
| **Physics** | `gravity_x/y/z`, `initial_velocity_x/y/z`, `damping`, `simulation_dt`, `max_simulation_time` |
| **Trail** | `enable_trail`, `trail_color`, `trail_width`, `trail_opacity`, `trail_sample_interval` |
| **Audio** | `sound_effect`, `use_generated_sound`, `min_bounce_sound_interval`, `sample_rate` |
| **Debug** | `debug` |

## 🎬 Quality Settings

Quality presets (manim standard):
- `'low_quality'` - 480p15 (fast preview)
- `'medium_quality'` - 720p30 (default)
- `'high_quality'` - 1080p60
- `'production_quality'` - 1440p60
- `'fourk_quality'` - 4K 60fps

## 📦 Output Structure

Each generated animation creates an organized output folder:

```
outputs/
└── my_animation/
    ├── my_animation.mp4          # Final video
    ├── metadata.json             # Configuration used
    ├── media/                    # Manim working files
    └── audio/                    # Audio files
        ├── ambient.wav
        └── bounce_with_audio.wav
```

## 📝 Examples

Run the example scripts:

```bash
python example_single.py
python example_multiple_dots.py
python example_batch.py
```

## 🛠️ API Reference

### `render_animation()`

```python
render_animation(
    animation_name: str,           # "BouncingDot" or "BouncingDots"
    config: AnimationConfig = None,
    dots: List[Dict] = None,       # For BouncingDots only
    output_name: str = None,       # Auto-generated if None
    quality: str = "medium_quality",
    preview: bool = False,         # Open video after render
    output_dir: Path = None,       # Default: ./outputs
) -> Path
```

### `render_batch()`

```python
render_batch(
    animations: List[Dict],        # List of animation specs
    **common_kwargs                # Applied to all animations
) -> List[Path]
```

### `AnimationConfig`

```python
config = AnimationConfig()
config.override(damping=0.95, dot_color='RED')
config.to_dict()  # Get as plain dictionary
```

## 🐛 Troubleshooting

**Issue**: Import errors
- Ensure you're running from the project root
- Check that `animations/__init__.py` exists

**Issue**: Config not loading
- Verify `app.cfg` is in the correct directory
- Check INI syntax (use `=` not `:`)
- Enable debug mode: `config.override(debug=True)`

## 📄 License

MIT License

## 🙏 Acknowledgments

- Built with [Manim Community Edition](https://www.manim.community/)
- Physics simulation using NumPy and SciPy
