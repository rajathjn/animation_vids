"""
Visualization of all waveforms in the generate_ambient_sound function.
Shows each component wave separately, then combined, and finally the output.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def visualize_ambient_sound_waves(
    duration: float = 2.0, sample_rate: int = 44100, volume: float = 0.1
):
    """
    Visualize all waveform components from the generate_ambient_sound function.

    Args:
        duration: Length of ambient sound in seconds (shorter for visualization)
        sample_rate: Audio sample rate
        volume: Amplitude multiplier (0.0 to 1.0)
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # === Individual Components ===

    # 1. Pink noise with low-frequency oscillations
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.05, len(t))

    # 2. First low-frequency sine wave (0.1 Hz)
    sine_wave_1 = 0.02 * np.sin(2 * np.pi * 0.1 * t)

    # 3. Second low-frequency sine wave (0.23 Hz)
    sine_wave_2 = 0.015 * np.sin(2 * np.pi * 0.23 * t)

    # 4. Combined ambient (all waves together, before scaling)
    ambient_combined = noise + sine_wave_1 + sine_wave_2

    # 5. Final output (scaled to int16)
    output = (ambient_combined * 32767 * volume).astype(np.int16)

    # === Plotting ===
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(
        "Waveform Visualization: generate_ambient_sound()", fontsize=16, fontweight="bold", y=0.98
    )

    gs = GridSpec(5, 1, figure=fig, hspace=0.4)

    # For better visualization, use a subset of samples
    # Show first 0.5 seconds for detail, or full duration if shorter
    display_duration = min(0.5, duration)
    display_samples = int(sample_rate * display_duration)
    t_display = t[:display_samples]

    # Color scheme
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    # Box 1: Noise
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_display, noise[:display_samples], color=colors[0], linewidth=0.5, alpha=0.8)
    ax1.set_title("1. Noise: np.random.normal(0, 0.05, len(t))", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, display_duration)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333")
        spine.set_linewidth(1.5)

    # Box 2: Sine Wave 1 (0.1 Hz)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_display, sine_wave_1[:display_samples], color=colors[1], linewidth=1.5)
    ax2.set_title("2. Sine Wave 1: 0.02 × sin(2π × 0.1 × t)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, display_duration)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")
        spine.set_linewidth(1.5)

    # Box 3: Sine Wave 2 (0.23 Hz)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t_display, sine_wave_2[:display_samples], color=colors[2], linewidth=1.5)
    ax3.set_title("3. Sine Wave 2: 0.015 × sin(2π × 0.23 × t)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlim(0, display_duration)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#f8f9fa")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333")
        spine.set_linewidth(1.5)

    # Box 4: All waves combined (overlay)
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(
        t_display, noise[:display_samples], color=colors[0], linewidth=0.5, alpha=0.5, label="Noise"
    )
    ax4.plot(
        t_display,
        sine_wave_1[:display_samples],
        color=colors[1],
        linewidth=1.5,
        alpha=0.7,
        label="Sine 0.1Hz",
    )
    ax4.plot(
        t_display,
        sine_wave_2[:display_samples],
        color=colors[2],
        linewidth=1.5,
        alpha=0.7,
        label="Sine 0.23Hz",
    )
    ax4.plot(
        t_display,
        ambient_combined[:display_samples],
        color=colors[3],
        linewidth=1,
        alpha=0.9,
        label="Combined",
    )
    ax4.set_title("4. All Waves Combined (Overlay)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Amplitude")
    ax4.set_xlim(0, display_duration)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#f8f9fa")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#333")
        spine.set_linewidth(1.5)

    # Box 5: Final Output (int16)
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t_display, output[:display_samples], color=colors[4], linewidth=0.5)
    ax5.set_title(
        f"5. Final Output: (ambient × 32767 × {volume}).astype(np.int16)",
        fontsize=11,
        fontweight="bold",
    )
    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("Amplitude (int16)")
    ax5.set_xlim(0, display_duration)
    ax5.grid(True, alpha=0.3)
    ax5.set_facecolor("#f8f9fa")
    for spine in ax5.spines.values():
        spine.set_edgecolor("#333")
        spine.set_linewidth(1.5)

    # Add text annotation with formula summary
    formula_text = (
        "Formula: output = (noise + 0.02×sin(2π×0.1×t) + 0.015×sin(2π×0.23×t)) × 32767 × volume"
    )
    fig.text(0.5, 0.01, formula_text, ha="center", fontsize=10, style="italic", color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def visualize_full_duration(duration: float = 12.0, sample_rate: int = 44100, volume: float = 0.1):
    """
    Visualize the full duration of the ambient sound with downsampling for performance.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(t))
    sine_wave_1 = 0.02 * np.sin(2 * np.pi * 0.1 * t)
    sine_wave_2 = 0.015 * np.sin(2 * np.pi * 0.23 * t)
    ambient_combined = noise + sine_wave_1 + sine_wave_2
    output = (ambient_combined * 32767 * volume).astype(np.int16)

    # Downsample for visualization (every 100th sample)
    step = 100
    t_ds = t[::step]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle(f"Full Duration ({duration}s) - Downsampled View", fontsize=16, fontweight="bold")

    waves = [
        (noise[::step], "Noise", "#e74c3c"),
        (sine_wave_1[::step], "Sine Wave 1 (0.1 Hz)", "#3498db"),
        (sine_wave_2[::step], "Sine Wave 2 (0.23 Hz)", "#2ecc71"),
        (ambient_combined[::step], "All Waves Combined", "#9b59b6"),
        (output[::step], "Final Output (int16)", "#f39c12"),
    ]

    for ax, (wave, title, color) in zip(axes, waves):
        ax.plot(t_ds, wave, color=color, linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f8f9fa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
            spine.set_linewidth(1.5)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Ambient Sound Waveform Visualization")
    print("=" * 60)
    print("\nShowing detailed view (first 0.5 seconds)...")
    visualize_ambient_sound_waves(duration=2.0)

    print("\nShowing full duration view (12 seconds, downsampled)...")
    visualize_full_duration(duration=12.0)
