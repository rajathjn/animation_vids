"""
Configuration management for animations.

Loads settings from app.cfg in the animations package directory.
Users must initialize AnimationConfig and use override() with a dictionary
to modify values. This config is then propagated to all modules.
"""
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import json
from typing import Any


class AnimationConfig:
    """
    Manages configuration for bouncing animations.
    
    Loads defaults from animations/app.cfg (same directory as this module).
    Use override() with a dictionary to modify values programmatically.
    
    Example:
        config = AnimationConfig()
        config.override({
            "DAMPING": 0.95,
            "DOT_COLOR": "RED",
            "ENABLE_TRAIL": True,
        })
    """
    
    def __init__(self):
        """
        Initialize configuration by loading from app.cfg.
        
        The app.cfg file must exist in the same directory as this module.
        """
        self._config: dict[str, Any] = {}
        self._manim_config: dict[str, Any] = {}
        self._dots_config: dict[str, Any] = {}
        
        # Optional output directory for audio files (set by main.py)
        self.audio_output_dir: str | None = None
        
        # Load from animations/app.cfg (required)
        config_path = Path(__file__).parent / "app.cfg"
        if not config_path.exists():
            raise FileNotFoundError(
                f"app.cfg not found at {config_path}. "
                "This file is required for the animation system to work."
            )
        self._load_from_file(config_path)
    
    def _load_from_file(self, config_path: Path) -> None:
        """Load configuration from an INI-format file."""
        parser = ConfigParser()
        parser.read(config_path)
        
        # Load [animation] section
        if parser.has_section("animation"):
            for key, value in parser.items("animation"):
                self._config[key.upper()] = self._parse_value(value)
        
        # Load [manim] section
        if parser.has_section("manim"):
            for key, value in parser.items("manim"):
                self._manim_config[key.upper()] = self._parse_value(value)
        
        # Load [dots] section
        if parser.has_section("dots"):
            for key, value in parser.items("dots"):
                if key.upper() == "DOTS_JSON":
                    # Parse JSON array for dots configuration
                    try:
                        self._dots_config["DOTS_JSON"] = json.loads(value) if value.strip() else []
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in DOTS_JSON: {e}")
                else:
                    self._dots_config[key.upper()] = self._parse_value(value)
    
    def _parse_value(self, value: Any) -> Any:
        """
        Parse string value to appropriate Python type.
        
        - Booleans: 'true'/'false' (case-insensitive)
        - Numbers: All numeric values are treated as float
        - Strings: Everything else (colors, filenames, etc.)
        - Non-strings: returned as-is
        """
        # If not a string, return as-is
        if not isinstance(value, str):
            return value
        
        value = value.strip()
        
        # Boolean values
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        
        # Try to parse as float (all numerics are float)
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def override(self, overrides: dict[str, Any]) -> None:
        """
        Override animation configuration values programmatically.
        
        Args:
            overrides: Dictionary of configuration keys and values to override.
                       Keys must be uppercase to match config format.
        
        Available keys for [animation] section:
            Circle boundary:
                CIRCLE_CENTER_X (float): X coordinate of circle center
                CIRCLE_CENTER_Y (float): Y coordinate of circle center  
                CIRCLE_CENTER_Z (float): Z coordinate of circle center
                CIRCLE_RADIUS (float): Radius of the boundary circle
            
            Dot parameters:
                DOT_START_X (float): Starting X position of dot
                DOT_START_Y (float): Starting Y position of dot
                DOT_START_Z (float): Starting Z position of dot
                DOT_RADIUS (float): Radius of the dot
                DOT_COLOR (str): Color name (e.g., 'RED', 'BLUE', 'YELLOW')
            
            Physics parameters:
                GRAVITY_X (float): X component of gravity
                GRAVITY_Y (float): Y component of gravity (typically negative)
                GRAVITY_Z (float): Z component of gravity
                INITIAL_VELOCITY_X (float): Initial X velocity
                INITIAL_VELOCITY_Y (float): Initial Y velocity
                INITIAL_VELOCITY_Z (float): Initial Z velocity
                DAMPING (float): Energy loss on bounce (0-1, higher = less loss)
                SIMULATION_DT (float): Simulation time step
                MAX_SIMULATION_TIME (float): Maximum simulation duration
            
            Trail effect:
                ENABLE_TRAIL (bool): Enable/disable trail effect
                TRAIL_COLOR (str): Trail color name
                TRAIL_WIDTH (float): Trail stroke width
                TRAIL_OPACITY (float): Trail opacity (0-1)
                TRAIL_FADING_TIME (float): Trail fading time in seconds
            
            Sound effect:
                SOUND_EFFECT (str): Sound effect filename
                USE_GENERATED_SOUND (bool): Use generated vs file sound
                MIN_BOUNCE_SOUND_INTERVAL (float): Min time between sounds
                SAMPLE_RATE (int): Audio sample rate
            
            Debug:
                DEBUG (bool): Enable debug output
        
        Available keys for [dots] section:
            USE_SINGLE_DOT_DEFAULTS (bool): Use [animation] values when DOTS_JSON empty
            DOTS_JSON (list): List of dot config dicts with keys:
                - color (str): Dot color
                - radius (float): Dot radius
                - start_pos (list): [x, y, z] starting position
                - initial_velocity (list): [x, y, z] initial velocity
                - damping (float): Energy loss on bounce
        
        Example:
            config.override({
                "DAMPING": 0.95,
                "DOT_COLOR": "RED",
                "ENABLE_TRAIL": True,
                "DOTS_JSON": [
                    {"color": "RED", "radius": 0.2, "start_pos": [0,0,0], 
                     "initial_velocity": [1,-5,0], "damping": 0.95}
                ]
            })
        """
        for key, value in overrides.items():
            key_upper = key.upper()
            
            # Check if it's a dots config key
            if key_upper in self._dots_config:
                self._dots_config[key_upper] = value if key_upper == "DOTS_JSON" else self._parse_value(value)
            else:
                # Animation config (existing or new keys)
                self._config[key_upper] = self._parse_value(value)
    
    def override_manim(self, overrides: dict[str, Any]) -> None:
        """
        Override manim rendering configuration values programmatically.
        
        This method allows overriding existing manim settings and adding new
        key-value pairs that will be passed directly to manim's tempconfig().
        
        Args:
            overrides: Dictionary of manim configuration keys and values.
                       Keys are converted to uppercase for storage but passed
                       as lowercase to manim.
        
        Default keys from [manim] section in app.cfg:
            PIXEL_WIDTH (int): Video width in pixels (default: 2160)
            PIXEL_HEIGHT (int): Video height in pixels (default: 3840)
            FRAME_RATE (float): Video frame rate (default: 60)
            RENDERER (str): 'cairo' or 'opengl' (default: cairo)
            BACKGROUND_COLOR (str): Background color name (default: BLACK)
            MEDIA_DIR (str): Output directory for media files (default: outputs)
            FRAME_WIDTH (float): Scene frame width (default: 8.0)
            FRAME_HEIGHT (float): Scene frame height (default: 14.22)
        
        Additional manim config keys you can add:
            DISABLE_CACHING (bool): Disable manim's caching
            FLUSH_CACHE (bool): Clear cache before rendering
            PREVIEW (bool): Open video after rendering
            WRITE_TO_MOVIE (bool): Write output to movie file
            SAVE_LAST_FRAME (bool): Save the last frame as image
            VERBOSITY (str): Logging verbosity level
            ... any other manim config option
        
        Example:
            config.override_manim({
                "RENDERER": "opengl",
                "PIXEL_WIDTH": 1920,
                "PIXEL_HEIGHT": 1080,
                "DISABLE_CACHING": True,
                "PREVIEW": True,
            })
        
        Raises:
            ValueError: If RENDERER is not 'cairo' or 'opengl'
        """
        for key, value in overrides.items():
            key_upper = key.upper()
            
            # Validate RENDERER
            if key_upper == "RENDERER" and value not in ("cairo", "opengl"):
                raise ValueError(
                    f"RENDERER must be 'cairo' or 'opengl', got '{value}'"
                )
            
            # Add or update manim config (allows new keys)
            self._manim_config[key_upper] = self._parse_value(value)
    
    # =========================================================================
    # [animation] section properties
    # =========================================================================
    
    @property
    def CIRCLE_CENTER(self) -> np.ndarray:
        """Get circle center as numpy array."""
        return np.array([
            self._config.get("CIRCLE_CENTER_X", 0.0),
            self._config.get("CIRCLE_CENTER_Y", 0.0),
            self._config.get("CIRCLE_CENTER_Z", 0.0),
        ])
    
    @property
    def CIRCLE_RADIUS(self) -> float:
        """Get circle radius."""
        return float(self._config.get("CIRCLE_RADIUS", 3.0))
    
    @property
    def CIRCLE_COLOR(self) -> str:
        """Get circle boundary color."""
        return str(self._config.get("CIRCLE_COLOR", "BLUE"))
    
    @property
    def CIRCLE_STROKE_WIDTH(self) -> float:
        """Get circle boundary stroke width."""
        return float(self._config.get("CIRCLE_STROKE_WIDTH", 2.0))
    
    @property
    def DOT_START_POS(self) -> np.ndarray:
        """Get dot starting position as numpy array."""
        return np.array([
            self._config.get("DOT_START_X", 0.0),
            self._config.get("DOT_START_Y", 2.5),
            self._config.get("DOT_START_Z", 0.0),
        ])
    
    @property
    def DOT_RADIUS(self) -> float:
        """Get dot radius."""
        return float(self._config.get("DOT_RADIUS", 0.2))
    
    @property
    def DOT_COLOR(self) -> str:
        """Get dot color."""
        return str(self._config.get("DOT_COLOR", "YELLOW"))
    
    @property
    def GRAVITY(self) -> np.ndarray:
        """Get gravity vector as numpy array."""
        return np.array([
            self._config.get("GRAVITY_X", 0.0),
            self._config.get("GRAVITY_Y", -9.8),
            self._config.get("GRAVITY_Z", 0.0),
        ])
    
    @property
    def INITIAL_VELOCITY(self) -> np.ndarray:
        """Get initial velocity as numpy array."""
        return np.array([
            self._config.get("INITIAL_VELOCITY_X", -2.0),
            self._config.get("INITIAL_VELOCITY_Y", -4.8),
            self._config.get("INITIAL_VELOCITY_Z", 0.0),
        ])
    
    @property
    def DAMPING(self) -> float:
        """Get damping coefficient."""
        return float(self._config.get("DAMPING", 0.99))
    
    @property
    def SIMULATION_DT(self) -> float:
        """Get simulation time step."""
        return float(self._config.get("SIMULATION_DT", 0.001))
    
    @property
    def MAX_SIMULATION_TIME(self) -> float:
        """Get maximum simulation time."""
        return float(self._config.get("MAX_SIMULATION_TIME", 600.0))
    
    @property
    def ENABLE_TRAIL(self) -> bool:
        """Check if trail effect is enabled."""
        return bool(self._config.get("ENABLE_TRAIL", False))
    
    @property
    def TRAIL_COLOR(self) -> str:
        """Get trail color."""
        return str(self._config.get("TRAIL_COLOR", "RED"))
    
    @property
    def TRAIL_WIDTH(self) -> float:
        """Get trail width."""
        return float(self._config.get("TRAIL_WIDTH", 1.0))
    
    @property
    def TRAIL_OPACITY(self) -> float | list[float]:
        """Get trail opacity (can be float or list of floats for gradient)."""
        value = self._config.get("TRAIL_OPACITY", 0.6)
        if isinstance(value, (list, tuple)):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return float(value)
    
    @property
    def TRAIL_FADING_TIME(self) -> float:
        """Get trail fading time in seconds."""
        if self._config.get("TRAIL_FADING_TIME") == "None":
            return None
        return float(self._config.get("TRAIL_FADING_TIME", 2.0))
    
    @property
    def SOUND_EFFECT(self) -> str:
        """Get sound effect filename."""
        return str(self._config.get("SOUND_EFFECT", "clack.wav"))
    
    @property
    def USE_GENERATED_SOUND(self) -> bool:
        """Check if generated sound should be used."""
        return bool(self._config.get("USE_GENERATED_SOUND", False))
    
    @property
    def MIN_BOUNCE_SOUND_INTERVAL(self) -> float:
        """Get minimum bounce sound interval."""
        return float(self._config.get("MIN_BOUNCE_SOUND_INTERVAL", 0.05))
    
    @property
    def SAMPLE_RATE(self) -> int:
        """Get audio sample rate."""
        return int(self._config.get("SAMPLE_RATE", 44100))
    
    @property
    def DEBUG(self) -> bool:
        """Check if debug mode is enabled."""
        return bool(self._config.get("DEBUG", False))
    
    # =========================================================================
    # [dots] section properties - CAPITALIZED
    # =========================================================================
    
    @property
    def USE_SINGLE_DOT_DEFAULTS(self) -> bool:
        """Check if single dot defaults should be used when DOTS_JSON is empty."""
        return bool(self._dots_config.get("USE_SINGLE_DOT_DEFAULTS", True))
    
    @property
    def DOTS_JSON(self) -> list[dict[str, Any]]:
        """
        Get dots configuration as a list of dictionaries.
        
        Each dict should contain:
            - color (str): Dot color name
            - radius (float): Dot radius
            - start_pos (list): [x, y, z] starting position
            - initial_velocity (list): [x, y, z] initial velocity
            - damping (float): Energy loss coefficient
        """
        return self._dots_config.get("DOTS_JSON", [])
    
    # =========================================================================
    # [manim] section properties
    # =========================================================================
    
    @property
    def MEDIA_DIR(self) -> str:
        """Get media output directory from manim config."""
        return str(self._manim_config.get("MEDIA_DIR", "outputs"))
    
    @property
    def PIXEL_WIDTH(self) -> int:
        """Get video width in pixels."""
        return int(self._manim_config.get("PIXEL_WIDTH", 2160))
    
    @property
    def PIXEL_HEIGHT(self) -> int:
        """Get video height in pixels."""
        return int(self._manim_config.get("PIXEL_HEIGHT", 3840))
    
    @property
    def FRAME_RATE(self) -> float:
        """Get video frame rate."""
        return float(self._manim_config.get("FRAME_RATE", 60))
    
    @property
    def RENDERER(self) -> str:
        """Get renderer type (cairo or opengl)."""
        return str(self._manim_config.get("RENDERER", "cairo"))
    
    @property
    def BACKGROUND_COLOR(self) -> str:
        """Get background color."""
        return str(self._manim_config.get("BACKGROUND_COLOR", "BLACK"))
    
    @property
    def FRAME_WIDTH(self) -> float:
        """Get scene frame width."""
        return float(self._manim_config.get("FRAME_WIDTH", 8.0))
    
    @property
    def FRAME_HEIGHT(self) -> float:
        """Get scene frame height."""
        return float(self._manim_config.get("FRAME_HEIGHT", 14.22))
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def get_manim_config(self) -> dict[str, Any]:
        """
        Get manim configuration as a dictionary for tempconfig().
        
        Returns all [manim] section values (including any added via override_manim)
        in the format expected by manim (lowercase keys).
        """
        # Convert all manim config keys to lowercase for manim's tempconfig()
        config_dict: dict[str, Any] = {}
        for key, value in self._manim_config.items():
            key_lower = key.lower()
            
            # Convert pixel dimensions to integers
            if key_lower in ("pixel_width", "pixel_height"):
                config_dict[key_lower] = int(value)
            else:
                config_dict[key_lower] = value
        
        return config_dict
    
    def get_default_dot_config(self) -> dict[str, Any]:
        """
        Get a single dot configuration from [animation] section values.
        
        Used when USE_SINGLE_DOT_DEFAULTS is True and DOTS_JSON is empty.
        
        Returns:
            Dict with keys: color, radius, start_pos, initial_velocity, damping
        """
        return {
            "color": self.DOT_COLOR,
            "radius": self.DOT_RADIUS,
            "start_pos": self.DOT_START_POS.tolist(),
            "initial_velocity": self.INITIAL_VELOCITY.tolist(),
            "damping": self.DAMPING,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Return all config sections as a plain dictionary for serialization."""
        return {
            "animation": self._config.copy(),
            "manim": self._manim_config.copy(),
            "dots": self._dots_config.copy(),
        }
    
    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"AnimationConfig("
            f"animation={len(self._config)} settings, "
            f"manim={len(self._manim_config)} settings, "
            f"dots={len(self.DOTS_JSON)} configured)"
        )
