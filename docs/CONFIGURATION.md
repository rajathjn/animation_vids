# Configuration Guide

## Overview

Configuration uses INI-style `app.cfg` files. The default configuration is loaded from `animations/app.cfg`.

## Configuration Sections

### [animation]
Main animation parameters:
- `CIRCLE_RADIUS` - Boundary circle radius (default: 3.5)
- `DOT_COLOR` - Dot color using Manim color names (default: BLUE)
- `DOT_RADIUS` - Size of the bouncing dot (default: 0.25)
- `DAMPING` - Energy loss on bounce, 0-1 (default: 0.95)
- `GRAVITY_Y` - Downward acceleration (default: -12.0)
- `ENABLE_TRAIL` - Enable dot trail (default: false)
- `TRAIL_COLOR` - Color of the trail (default: YELLOW)
- `USE_SINGLE_DOT_DEFAULTS` - Use [animation] defaults if DOTS_JSON empty (default: true)

### [manim_render]
Rendering settings passed to Manim:
- `pixel_height` - Video height (default: 1080)
- `pixel_width` - Video width (default: 1920)
- `frame_rate` - Frames per second (default: 30)

### [audio]
Audio generation settings:
- `USE_GENERATED_SOUND` - Generate synthetic bounce sounds (default: true)
- `MIN_BOUNCE_SOUND_INTERVAL` - Minimum time between bounce sounds (default: 0.05)
- `SAMPLE_RATE` - Audio sample rate (default: 44100)

## Using override()

The `override()` method accepts a dictionary of uppercase keys:

```python
config.override({
    "DAMPING": 0.95,
    "DOT_COLOR": "RED",
    "ENABLE_TRAIL": True,
    "GRAVITY_Y": -10.0,
})
```

## Using override_manim()

For Manim-specific rendering settings:

```python
config.override_manim({
    "pixel_height": 1440,
    "pixel_width": 2560,
    "frame_rate": 60,
})
```
