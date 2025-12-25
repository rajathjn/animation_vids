# Animation Vids

A clean, simple framework for generating bouncing ball physics animations using Manim's Python API directly - no subprocess calls or temp files needed.

## Features

- **Direct rendering**: Uses manim's Python API - no subprocess or temp files
- **Config-based setup**: Define animation parameters in `app.cfg`
- **Modular design**: Separate classes for single and multiple bouncing dots
- **Simple API**: Import, configure, and render videos programmatically
- **Physics simulation**: Accurate gravity, damping, and collision detection
- **Audio support**: Generate bounce sounds and mix with ambient audio

## Installation

### Prerequisites
- Python 3.12+
- Manim Community Edition

### Setup

#### Using uv (fastest)
```bash
cd animation_vids
uv sync
```

#### Using pip
```bash
cd animation_vids
pip install -e .
```

#### Manual installation
```bash
pip install manim>=0.19.1
```

## Quick Start

### 1. Basic Usage - Single Bouncing Dot

```python
from animations import AnimationConfig, render_animation

# Create config (loads defaults from animations/app.cfg)
config = AnimationConfig()

# Override specific settings
config.override({
    "DAMPING": 0.98,
    "DOT_COLOR": "RED",
    "ENABLE_TRAIL": True,
})

# Render animation
render_animation("BouncingDot", config=config, output_name="red_bouncing_dot")
```

### 2. Multiple Bouncing Dots

```python
from animations import AnimationConfig, render_animation

config = AnimationConfig()

# Define dots configuration
dots = [
    {
        "color": "YELLOW",
        "radius": 0.22,
        "start_pos": [0, 1, 0],
        "initial_velocity": [2, -5, 0],
        "damping": 0.96,
    },
    {
        "color": "PURPLE",
        "radius": 0.18,
        "start_pos": [1, -1, 0],
        "initial_velocity": [-3, -6, 0],
        "damping": 0.97,
    },
]

config.override({"DOTS_JSON": dots, "ENABLE_TRAIL": True})

render_animation("BouncingDots", config=config, output_name="multi_bounce")
```

### 3. Custom Rendering Settings

```python
from animations import AnimationConfig, render_animation

config = AnimationConfig()
config.override({"DAMPING": 0.95})

# Override manim rendering settings
config.override_manim({
    "pixel_height": 1440,
    "pixel_width": 2560,
    "frame_rate": 60,
})

render_animation("BouncingDot", config=config, output_name="hd_animation")
```

## Configuration

See [Configuration Guide](docs/CONFIGURATION.md) for detailed configuration options and usage.

## API Reference

See [API Reference](docs/API_REFERENCE.md) for complete API documentation.

## Troubleshooting
See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions to common issues.

## License

AGPL-3.0-or-later

## Acknowledgments

- Built with [Manim Community Edition](https://www.manim.community/)
- Physics simulation using NumPy
- AI-assisted development with GitHub Copilot/Perplexity
