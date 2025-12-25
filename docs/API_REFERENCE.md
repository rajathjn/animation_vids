# API Reference

## `render_animation()`

Renders a single animation with the given configuration.

```python
render_animation(
    animation_name: str,           # "BouncingDot" or "BouncingDots"
    config: AnimationConfig = None,
    output_name: str = None,       # Auto-generated if None
    output_dir: Path = None,       # Default: ./outputs
) -> Path
```

**Parameters:**
- `animation_name`: Name of animation class to render
- `config`: AnimationConfig instance with settings
- `output_name`: Name for output folder (auto-generated if None)
- `output_dir`: Where to save output videos

**Returns:** Path to the generated video file

## `AnimationConfig`

Main configuration class for animations.

```python
config = AnimationConfig()
# Load defaults from animations/app.cfg

config.override({key: value, ...})
# Override animation/physics/dots settings (uppercase keys)

config.override_manim({key: value, ...})
# Override Manim rendering settings

config.to_dict()
# Get entire config as dictionary
```

**Methods:**
- `override(dict)` - Override animation settings
- `override_manim(dict)` - Override Manim rendering settings
- `to_dict()` - Get config as plain dictionary

## Animation Classes

Available animations to pass to `render_animation()`:

- `"BouncingDot"` - Single bouncing dot
- `"BouncingDots"` - Multiple bouncing dots (configured via `DOTS_JSON`)
