"""
Audio mixing utilities for bouncing dot animations.

Handles generation and mixing of ambient sounds and hit sounds
based on physics bounce events.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from .audio_utils import generate_ambient_sound, generate_hit_sound
from .config import AnimationConfig
from .physics import BounceEvent


class AudioMixer:
    """
    Generates and mixes audio for bouncing dot animations.

    Combines ambient background sound with hit sounds triggered by
    bounce events from physics simulation.

    Example:
        config = AnimationConfig()
        bounce_events = [BounceEvent(time=0.5, speed=5.0), ...]
        mixer = AudioMixer(config, bounce_events, audio_dir=Path("output/audio"))
        audio_path = mixer.generate_mixed_audio(duration=10.0)
    """

    def __init__(
        self,
        config: AnimationConfig,
        bounce_events: list[BounceEvent],
        audio_dir: Path | None = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the audio mixer.

        Args:
            config: AnimationConfig instance with audio settings
            bounce_events: List of bounce events from physics simulation
            audio_dir: Directory to save audio files (default: media/audio)
            debug: Enable debug output
        """
        self.config = config
        self.bounce_events = bounce_events
        self.debug = debug

        # Set up directories
        if audio_dir:
            self.audio_dir = audio_dir
        elif config.audio_output_dir:
            self.audio_dir = Path(config.audio_output_dir)
        else:
            self.audio_dir = Path("media/audio")

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.sound_effect_dir = Path("sound_effect")
        self.sound_effect_dir.mkdir(parents=True, exist_ok=True)

    def generate_mixed_audio(
        self,
        duration: float,
        output_filename: str = "bounce_with_audio.wav",
    ) -> Path:
        """
        Generate mixed audio with ambient sound and hit sounds.

        Args:
            duration: Total duration of the audio in seconds
            output_filename: Name of the output WAV file

        Returns:
            Path to the generated audio file
        """
        cfg = self.config
        sample_rate = cfg.SAMPLE_RATE

        # Add 1 second buffer to duration
        ambient_duration = duration + 1.0
        total_samples = int(ambient_duration * sample_rate)

        # Generate ambient background
        ambient_sound = generate_ambient_sound(
            duration=ambient_duration,
            sample_rate=sample_rate,
        )
        ambient_path = self.audio_dir / "ambient.wav"
        wavfile.write(ambient_path, sample_rate, ambient_sound)

        # Generate hit sounds based on bounce events
        hit_audio = self._generate_hit_audio(total_samples, sample_rate)

        # Mix ambient and hits
        mixed_audio = self._mix_audio(ambient_sound, hit_audio)

        # Save mixed audio
        mixed_path = self.audio_dir / output_filename
        wavfile.write(mixed_path, sample_rate, mixed_audio)

        if self.debug:
            print(f"Audio generated: {mixed_path}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Bounce sounds: {len(self.bounce_events)}")

        return mixed_path

    def _generate_hit_audio(
        self,
        total_samples: int,
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """
        Generate hit sound audio track based on bounce events.

        Args:
            total_samples: Total number of audio samples
            sample_rate: Audio sample rate

        Returns:
            Float32 array of hit sounds
        """
        cfg = self.config
        hit_audio = np.zeros(total_samples, dtype=np.float32)

        if not self.bounce_events:
            return hit_audio

        # Extract bounce times and speeds
        bounce_times = [event.time for event in self.bounce_events]
        bounce_speeds = [event.speed for event in self.bounce_events]

        # Calculate volume normalization range
        min_speed = min(bounce_speeds)
        max_speed = max(bounce_speeds)
        speed_range = max(max_speed - min_speed, 1.0)

        # Determine if using generated or file-based sounds
        use_generated_sound = cfg.USE_GENERATED_SOUND
        sound_effect_path = self.sound_effect_dir / cfg.SOUND_EFFECT

        if not use_generated_sound and not sound_effect_path.exists():
            if self.debug:
                print(f"Sound effect not found: {sound_effect_path}, using generated sound")
            use_generated_sound = True

        if use_generated_sound:
            self._add_generated_hits(
                hit_audio,
                bounce_times,
                bounce_speeds,
                min_speed,
                speed_range,
                sample_rate,
                total_samples,
            )
        else:
            self._add_file_hits(
                hit_audio,
                bounce_times,
                bounce_speeds,
                min_speed,
                speed_range,
                sample_rate,
                total_samples,
                sound_effect_path,
            )

        return hit_audio

    def _add_generated_hits(
        self,
        hit_audio: NDArray[np.float32],
        bounce_times: list[float],
        bounce_speeds: list[float],
        min_speed: float,
        speed_range: float,
        sample_rate: int,
        total_samples: int,
    ) -> None:
        """Add procedurally generated hit sounds to audio track."""
        cfg = self.config
        min_interval = cfg.MIN_BOUNCE_SOUND_INTERVAL

        for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
            # Skip if too close to previous bounce
            if i > 0 and (bounce_time - bounce_times[i - 1]) < min_interval:
                continue

            # Calculate volume based on speed
            volume = self._calculate_volume(speed, min_speed, speed_range, base=0.2, scale=0.8)

            # Vary frequency slightly for each bounce
            frequency = 440 + (i % 3) * 100
            hit_sound = generate_hit_sound(
                frequency=frequency,
                volume=volume,
                sample_rate=sample_rate,
            )

            # Add to hit audio track
            start_sample = int(bounce_time * sample_rate)
            end_sample = min(start_sample + len(hit_sound), total_samples)

            if start_sample < total_samples:
                hit_audio[start_sample:end_sample] += hit_sound[: end_sample - start_sample].astype(
                    np.float32
                )

    def _add_file_hits(
        self,
        hit_audio: NDArray[np.float32],
        bounce_times: list[float],
        bounce_speeds: list[float],
        min_speed: float,
        speed_range: float,
        sample_rate: int,
        total_samples: int,
        sound_effect_path: Path,
    ) -> None:
        """Add file-based hit sounds to audio track."""
        cfg = self.config
        min_interval = cfg.MIN_BOUNCE_SOUND_INTERVAL

        # Load and process sound effect file
        effect_sample_rate, effect_sound = wavfile.read(sound_effect_path)

        # Convert stereo to mono if needed
        if len(effect_sound.shape) > 1:
            effect_sound = effect_sound.mean(axis=1)

        # Resample if needed
        if effect_sample_rate != sample_rate:
            effect_duration = len(effect_sound) / effect_sample_rate
            new_length = int(effect_duration * sample_rate)
            effect_sound = np.interp(
                np.linspace(0, len(effect_sound) - 1, new_length),
                np.arange(len(effect_sound)),
                effect_sound.astype(np.float32),
            )
        else:
            effect_sound = effect_sound.astype(np.float32)

        # Normalize
        effect_max = np.abs(effect_sound).max()
        if effect_max > 0:
            effect_sound = effect_sound / effect_max * 32767

        # Add hit sounds at each bounce
        for i, (bounce_time, speed) in enumerate(zip(bounce_times, bounce_speeds)):
            # Skip if too close to previous bounce
            if i > 0 and (bounce_time - bounce_times[i - 1]) < min_interval:
                continue

            volume = self._calculate_volume(speed, min_speed, speed_range, base=0.3, scale=0.7)
            start_sample = int(bounce_time * sample_rate)
            end_sample = min(start_sample + len(effect_sound), total_samples)

            if start_sample < total_samples:
                hit_audio[start_sample:end_sample] += (
                    effect_sound[: end_sample - start_sample] * volume
                )

    def _calculate_volume(
        self,
        speed: float,
        min_speed: float,
        speed_range: float,
        base: float = 0.2,
        scale: float = 0.8,
    ) -> float:
        """
        Calculate volume based on bounce speed.

        Args:
            speed: Speed at bounce
            min_speed: Minimum speed in all bounces
            speed_range: Range of speeds (max - min)
            base: Base volume level
            scale: Volume scaling factor

        Returns:
            Volume multiplier (0.0 to 1.0)
        """
        return base + scale * (speed - min_speed) / speed_range

    def _mix_audio(
        self,
        ambient: NDArray[np.int16],
        hits: NDArray[np.float32],
    ) -> NDArray[np.int16]:
        """
        Mix ambient and hit audio tracks with normalization.

        Args:
            ambient: Ambient background audio (int16)
            hits: Hit sounds audio (float32)

        Returns:
            Mixed and normalized int16 audio
        """
        mixed = ambient.astype(np.float32) + hits

        # Normalize if clipping would occur
        max_val = np.abs(mixed).max()
        if max_val > 32767:
            mixed *= 32767 / max_val

        return mixed.astype(np.int16)
