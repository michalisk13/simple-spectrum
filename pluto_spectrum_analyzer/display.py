"""Display-focused decimation and quantization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from pluto_spectrum_analyzer.protocol import EngineSpectrogramFrame, EngineSpectrumFrame


@dataclass(frozen=True)
class DisplayConfig:
    max_spectrum_bins: int
    max_spectrogram_cols: int
    spectrogram_quantize: bool


def _decimate_indices(count: int, max_bins: int) -> np.ndarray:
    if max_bins <= 0:
        raise ValueError("max_bins must be positive")
    if count <= max_bins:
        return np.arange(count, dtype=int)
    indices = np.linspace(0, count - 1, num=max_bins, dtype=int)
    indices = np.unique(indices)
    if indices[0] != 0:
        indices = np.insert(indices, 0, 0)
    if indices[-1] != count - 1:
        indices = np.append(indices, count - 1)
    return indices


def decimate_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    indices = _decimate_indices(x.size, max_bins)
    dec_x = x[indices]
    dec_y = y[indices]
    if dec_x.size > 1 and dec_x[0] > dec_x[-1]:
        dec_x = dec_x[::-1]
        dec_y = dec_y[::-1]
    return dec_x, dec_y


def apply_spectrum_display(
    frame: EngineSpectrumFrame,
    cfg: DisplayConfig,
) -> EngineSpectrumFrame:
    if frame.n_bins <= cfg.max_spectrum_bins:
        return frame
    x = np.linspace(frame.freq_start_hz, frame.freq_stop_hz, frame.n_bins, dtype=np.float64)
    _, dec_y = decimate_xy(x, frame.y, cfg.max_spectrum_bins)
    return replace(frame, n_bins=int(dec_y.size), y=dec_y)


def _decimate_spectrogram_row(row_db: np.ndarray, max_cols: int) -> np.ndarray:
    indices = _decimate_indices(row_db.size, max_cols)
    return row_db[indices]


def _quantize_spectrogram_row(row_db: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    if db_max <= db_min:
        return np.zeros_like(row_db, dtype=np.uint8)
    scale = 255.0 / (db_max - db_min)
    normalized = (row_db - db_min) * scale
    clipped = np.clip(normalized, 0.0, 255.0)
    return clipped.astype(np.uint8)


def apply_spectrogram_display(
    frame: EngineSpectrogramFrame,
    cfg: DisplayConfig,
) -> tuple[EngineSpectrogramFrame, bool, str]:
    working = frame
    if frame.col_count > cfg.max_spectrogram_cols:
        row_db = _decimate_spectrogram_row(frame.row_db, cfg.max_spectrogram_cols)
        working = replace(frame, row_db=row_db, col_count=int(row_db.size))
    if cfg.spectrogram_quantize:
        row_db = _quantize_spectrogram_row(working.row_db, working.db_min, working.db_max)
        working = replace(working, row_db=row_db, col_count=int(row_db.size))
        return working, True, "u8"
    return working, False, "f32"
