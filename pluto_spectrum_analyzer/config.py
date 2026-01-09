"""Application configuration defaults and validation.

Defines the SpectrumConfig dataclass and default values. This module should not
import UI or SDR classes, and it should stay focused on configuration data only.
"""

from dataclasses import dataclass


@dataclass
class SpectrumConfig:
    """
    Configuration for the spectrum viewer.

    Notes
    Span is the sampled bandwidth.
    For a real time FFT spectrum, visible span equals sample rate.
    """

    # Default Pluto URI for gadget mode.
    uri: str = "ip:192.168.2.1"

    # Center frequency for the LO.
    center_hz: int = 2_437_000_000

    # Sample rate controls instantaneous span. RF BW should not exceed span.
    sample_rate_hz: int = 20_000_000
    rf_bw_hz: int = 20_000_000

    # Gain settings.
    gain_db: int = 55
    gain_mode: str = "manual"

    # FFT and update timing.
    fft_size: int = 8192
    update_ms: int = 100
    buffer_factor: int = 4
    overlap: float = 0.5

    # UI responsiveness (hover).
    hover_rate_hz: int = 25

    # RBW controls.
    rbw_mode: str = "Manual"
    rbw_hz: float = 4894.0
    window: str = "Blackman Harris"

    # VBW controls.
    vbw_mode: str = "Auto"
    vbw_hz: float = 3_000.0

    # Detector/trace controls.
    detector: str = "RMS"
    trace_type: str = "Clear Write"
    trace2_enabled: bool = False
    avg_count: int = 10
    avg_mode: str = "RMS"

    # Display scaling.
    ref_level_db: float = 0.0
    display_range_db: float = 100.0

    # Spectrogram defaults.
    spectrogram_mode: str = "PSD (Welch)"

    # Measurement helpers.
    measurement_mode: bool = False
    dc_remove: bool = True
    dc_blank_bins: int = 1
