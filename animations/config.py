"""
Configuration management for animations.
Loads settings from app.cfg files using INI format (mimicking manim.cfg structure).
"""
from configparser import ConfigParser
from pathlib import Path
import numpy as np
from typing import Optional, Any
import inspect


class AnimationConfig:
    """
    Manages configuration for bouncing animations.
    
    Loads defaults from animations/app.cfg, then overrides with app.cfg 
    found in the directory of the calling script (if exists).
    """
    
    def __init__(self, override_config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            override_config_path: Optional path to a specific config file to use for overrides
        """
        # Default values (fallback if no config files exist)
        self._defaults = {
            # Circle boundary
            'CIRCLE_CENTER_X': 0.0,
            'CIRCLE_CENTER_Y': 0.0,
            'CIRCLE_CENTER_Z': 0.0,
            'CIRCLE_RADIUS': 3.0,
            
            # Dot parameters
            'DOT_START_X': 0.0,
            'DOT_START_Y': 2.5,
            'DOT_START_Z': 0.0,
            'DOT_RADIUS': 0.2,
            'DOT_COLOR': 'YELLOW',
            
            # Physics parameters
            'GRAVITY_X': 0.0,
            'GRAVITY_Y': -9.8,
            'GRAVITY_Z': 0.0,
            'INITIAL_VELOCITY_X': -2.0,
            'INITIAL_VELOCITY_Y': -4.8,
            'INITIAL_VELOCITY_Z': 0.0,
            'DAMPING': 0.99,
            'SIMULATION_DT': 0.001,
            'MAX_SIMULATION_TIME': 240.0,
            
            # Trail effect
            'ENABLE_TRAIL': False,
            'TRAIL_COLOR': 'RED',
            'TRAIL_WIDTH': 1.0,
            'TRAIL_OPACITY': 0.6,
            'TRAIL_SAMPLE_INTERVAL': 3,
            
            # Sound effect
            'SOUND_EFFECT': 'clack.wav',
            'USE_GENERATED_SOUND': False,
            'MIN_BOUNCE_SOUND_INTERVAL': 0.05,
            'SAMPLE_RATE': 44100,
            
            # Debug
            'DEBUG': False,
        }
        
        # Store config values
        self.config = self._defaults.copy()
        
        # Optional output directory for audio files (set by main.py)
        self.audio_output_dir: Optional[str] = None
        
        # Load from animations/app.cfg (default config)
        default_config_path = Path(__file__).parent / "app.cfg"
        if default_config_path.exists():
            self._load_from_file(default_config_path)
        
        # Find and load config from caller's directory
        if override_config_path is None:
            caller_dir = self._get_caller_directory()
            if caller_dir:
                caller_config_path = caller_dir / "app.cfg"
                if caller_config_path.exists():
                    self._load_from_file(caller_config_path)
        else:
            if override_config_path.exists():
                self._load_from_file(override_config_path)
    
    def _get_caller_directory(self) -> Optional[Path]:
        """
        Get the directory of the script that called the animation class.
        Uses stack inspection to find the first frame outside this module.
        """
        try:
            # Get the call stack
            frame = inspect.currentframe()
            if frame is None:
                return None
            
            # Walk up the stack to find the first frame outside this module and manim
            while frame is not None:
                frame_info = inspect.getframeinfo(frame)
                frame_file = frame_info.filename
                
                # Skip frames from this config module and manim internals
                if frame_file and 'config.py' not in frame_file and 'manim' not in frame_file.lower():
                    caller_path = Path(frame_file).resolve()
                    if caller_path.exists():
                        return caller_path.parent
                
                frame = frame.f_back
            
            return None
        except:
            return None
    
    def _load_from_file(self, config_path: Path) -> None:
        """Load configuration from an INI-format file."""
        parser = ConfigParser()
        parser.read(config_path)
        
        # Load animation settings if section exists
        if parser.has_section('animation'):
            for key, value in parser.items('animation'):
                # Convert key to uppercase to match our internal format
                key_upper = key.upper()
                if key_upper in self.config:
                    self.config[key_upper] = self._parse_value(value, key_upper)
    
    def _parse_value(self, value: str, key: str) -> Any:
        """Parse string value to appropriate Python type."""
        value = value.strip()
        
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string (for colors, filenames, etc.)
        return value
    
    def override(self, **kwargs) -> None:
        """
        Override configuration values programmatically.
        
        Example:
            config.override(DAMPING=0.95, DOT_COLOR='RED')
            # or lowercase (will be converted):
            config.override(damping=0.95, dot_color='RED')
        """
        for key, value in kwargs.items():
            # Convert to uppercase to match internal format
            key_upper = key.upper()
            if key_upper in self.config:
                self.config[key_upper] = value
    
    # Convenience properties for accessing common config groups
    
    @property
    def circle_center(self) -> np.ndarray:
        """Get circle center as numpy array."""
        return np.array([
            self.config['CIRCLE_CENTER_X'],
            self.config['CIRCLE_CENTER_Y'],
            self.config['CIRCLE_CENTER_Z']
        ])
    
    @property
    def circle_radius(self) -> float:
        """Get circle radius."""
        return self.config['CIRCLE_RADIUS']
    
    @property
    def dot_start_pos(self) -> np.ndarray:
        """Get dot starting position as numpy array."""
        return np.array([
            self.config['DOT_START_X'],
            self.config['DOT_START_Y'],
            self.config['DOT_START_Z']
        ])
    
    @property
    def dot_radius(self) -> float:
        """Get dot radius."""
        return self.config['DOT_RADIUS']
    
    @property
    def dot_color(self) -> str:
        """Get dot color."""
        return self.config['DOT_COLOR']
    
    @property
    def gravity(self) -> np.ndarray:
        """Get gravity vector as numpy array."""
        return np.array([
            self.config['GRAVITY_X'],
            self.config['GRAVITY_Y'],
            self.config['GRAVITY_Z']
        ])
    
    @property
    def initial_velocity(self) -> np.ndarray:
        """Get initial velocity as numpy array."""
        return np.array([
            self.config['INITIAL_VELOCITY_X'],
            self.config['INITIAL_VELOCITY_Y'],
            self.config['INITIAL_VELOCITY_Z']
        ])
    
    @property
    def damping(self) -> float:
        """Get damping coefficient."""
        return self.config['DAMPING']
    
    @property
    def simulation_dt(self) -> float:
        """Get simulation time step."""
        return self.config['SIMULATION_DT']
    
    @property
    def max_simulation_time(self) -> float:
        """Get maximum simulation time."""
        return self.config['MAX_SIMULATION_TIME']
    
    @property
    def enable_trail(self) -> bool:
        """Check if trail effect is enabled."""
        return self.config['ENABLE_TRAIL']
    
    @property
    def trail_color(self) -> str:
        """Get trail color."""
        return self.config['TRAIL_COLOR']
    
    @property
    def trail_width(self) -> float:
        """Get trail width."""
        return self.config['TRAIL_WIDTH']
    
    @property
    def trail_opacity(self) -> float:
        """Get trail opacity."""
        return self.config['TRAIL_OPACITY']
    
    @property
    def trail_sample_interval(self) -> int:
        """Get trail sampling interval."""
        return int(self.config['TRAIL_SAMPLE_INTERVAL'])
    
    @property
    def sound_effect(self) -> str:
        """Get sound effect filename."""
        return self.config['SOUND_EFFECT']
    
    @property
    def use_generated_sound(self) -> bool:
        """Check if generated sound should be used."""
        return self.config['USE_GENERATED_SOUND']
    
    @property
    def min_bounce_sound_interval(self) -> float:
        """Get minimum bounce sound interval."""
        return self.config['MIN_BOUNCE_SOUND_INTERVAL']
    
    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        return int(self.config['SAMPLE_RATE'])
    
    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config['DEBUG']
    
    def to_dict(self) -> dict:
        """Return config as a plain dictionary for serialization."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"AnimationConfig({len(self.config)} settings)"
