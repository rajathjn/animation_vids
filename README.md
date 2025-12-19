# Animation Video Generator

A configurable framework for generating bouncing ball physics animations using Manim.

## 🎯 Features

- **Config-based setup**: Define animation parameters in `app.cfg` files (similar to manim.cfg)
- **Modular design**: Separate classes for single and multiple bouncing dots
- **Easy generation**: Import, configure, and generate videos programmatically
- **Organized output**: Each video stored in its own timestamped folder with metadata
- **No file editing**: Generate multiple variations without editing source files
- **Platform independent**: Uses subprocess to call manim, works on all platforms

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
├── extras/                    # Backup files and examples
├── example_single.py          # Example: Single dot
├── example_multiple_dots.py   # Example: Multiple dots
├── example_batch.py           # Example: Batch generation
├── manim.cfg                  # Manim configuration
└── README.md
```

## 🚀 Quick Start

### 1. Generate with Defaults

```python
from animations import generate_animation

# Generate a simple bouncing dot
generate_animation("BouncingDot", output_name="my_first_animation")
```

### 2. Generate with Custom Config

```python
from animations import AnimationConfig, generate_animation

# Create config and override settings
config = AnimationConfig()
config.override(
    damping=0.95,
    dot_color='RED',
    enable_trail=True,
    trail_color='YELLOW',
)

# Generate animation
generate_animation(
    animation_name="BouncingDot",
    config=config,
    output_name="red_dot_with_trail",
    quality="h"  # High quality
)
```

### 3. Generate Multiple Dots

```python
import numpy as np
from animations import AnimationConfig, generate_animation

# Define dots
dots = [
    {
        "initial_velocity": np.array([2, -5, 0]),
        "damping": 0.96,
        "radius": 0.22,
        "color": "YELLOW",
        "start_pos": np.array([0, 1, 0])
    },
    {
        "initial_velocity": np.array([-3, -6, 0]),
        "damping": 0.97,
        "radius": 0.18,
        "color": "PURPLE",
        "start_pos": np.array([1, -1, 0])
    },
]

config = AnimationConfig()
config.override(enable_trail=True)

generate_animation(
    animation_name="BouncingDots",
    config=config,
    dots=dots,
    output_name="two_dots"
)
```

### 4. Batch Generation

```python
from animations import generate_multiple

animations = [
    {
        "animation_name": "BouncingDot",
        "output_name": "simple_blue_dot",
        "config": config1
    },
    {
        "animation_name": "BouncingDot",
        "output_name": "red_dot_fast",
        "config": config2
    },
]

generate_multiple(animations, quality='m', preview=False)
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

## 📦 Output Structure

Each generated animation creates an organized output folder:

```
outputs/
└── my_animation_20231218_143022/
    ├── my_animation_20231218_143022.mp4  # Final video
    ├── metadata.json                      # Configuration used
    └── audio/                             # Audio files
        ├── ambient.wav
        └── bounce_with_audio.wav
```

## 🔧 Advanced Usage

### Custom Caller Directory Config

The config system automatically detects the directory of the Python script that imports the animation:

```python
# In /my_project/generate.py
from animations import generate_animation

# This will look for /my_project/app.cfg first
generate_animation("BouncingDot")
```

### Programmatic Config Override

```python
config = AnimationConfig()
config.override(
    damping=0.95,
    dot_color='RED',
    initial_velocity_y=-10.0
)
```

### Multiple Dots with Different Properties

```python
dots = [
    {
        "initial_velocity": np.array([i*0.5, -5-i*0.3, 0]),
        "damping": 0.96 + i*0.01,
        "radius": 0.15 + i*0.03,
        "color": ["RED", "GREEN", "BLUE", "YELLOW"][i],
        "start_pos": np.array([np.sin(i)*1.5, np.cos(i)*1.5, 0])
    }
    for i in range(4)
]
```

## 📝 Examples

Run the example scripts to see the framework in action:

```bash
python example_single.py
python example_multiple_dots.py
python example_batch.py
```

## 🎬 Quality Settings

- `'l'` - Low quality (480p15)
- `'m'` - Medium quality (720p30) - default
- `'h'` - High quality (1080p60)

Note: The actual resolution is defined in `manim.cfg` (currently 1920x1080 @ 60fps)

## 🛠️ Development

### Adding New Animations

1. Create a new animation class in `animations/`
2. Make it accept `AnimationConfig` in `__init__`
3. Add it to `animations/__init__.py` exports
4. Update `main.py` to support the new animation name

### Modifying the Configuration

- Edit `animations/app.cfg` for new defaults
- Add new parameters to `AnimationConfig` class in `config.py`
- Update documentation

## 📚 Migration from Old System

Old workflow:
```bash
# 1. Edit bounce.py parameters
# 2. Run manim
manim bounce.py BouncingDot
# 3. Manually rename video
```

New workflow:
```python
# All in one script
config = AnimationConfig()
config.override(damping=0.95)
generate_animation("BouncingDot", config=config, output_name="custom_dot")
```

## 🐛 Troubleshooting

**Issue**: Video not found after generation
- Check `media/videos/` directory structure
- Ensure manim completed successfully (check terminal output)

**Issue**: Config not loading
- Verify `app.cfg` is in the correct directory
- Check INI syntax (use `=` not `:`)
- Enable debug mode: `config.override(debug=True)`

**Issue**: Import errors
- Ensure you're running from the project root
- Check that `animations/__init__.py` exists

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with [Manim Community Edition](https://www.manim.community/)
- Physics simulation using NumPy and SciPy
