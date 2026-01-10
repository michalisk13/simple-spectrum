// Typed representations of WebSocket frames consumed by the frontend.

import type { EngineStatus } from "../api/types";

// Shared base fields sent with every WebSocket frame.
export type WebSocketFrameBase = {
  // Protocol version (v1.0 as of this client).
  proto_version: string;
  // Frame discriminator used by the backend protocol contract.
  type: string;
  // Monotonic timestamp in nanoseconds for ordering on the client.
  ts_monotonic_ns: number;
  // Per-session sequence number for this frame.
  seq: number;
  // Unique session identifier for this WebSocket connection.
  session_id: string;
};

// Status frame sent immediately on connect and periodically afterward.
export type StatusFrame = WebSocketFrameBase &
  EngineStatus & {
    // Explicitly narrowed discriminator for status frames.
    type: "status";
  };

// Metadata sent ahead of spectrum payloads.
export type SpectrumMetaFrame = WebSocketFrameBase & {
  type: "spectrum_meta";
  payload_id: string;
  freq_start_hz: number;
  freq_stop_hz: number;
  n_bins: number;
  y_units: "dBFS";
  detector: string;
  trace_mode: string;
  rbw_hz: number;
  vbw_hz: number;
  fft_size: number;
  window: string;
  averaging_alpha: number | null;
  dtype: "f32";
  endianness: "LE";
};

// Metadata sent ahead of spectrogram payloads.
export type SpectrogramMetaFrame = WebSocketFrameBase & {
  type: "spectrogram_meta";
  payload_id: string;
  freq_start_hz: number;
  freq_stop_hz: number;
  n_cols: number;
  row_ts_monotonic_ns: number;
  db_min: number;
  db_max: number;
  colormap: string;
  quantized: boolean;
  dtype: "u8" | "f32";
  endianness: "LE" | null;
};

// Parsed spectrum payload paired with its metadata.
export type SpectrumFrame = {
  meta: SpectrumMetaFrame;
  payload: Float32Array;
};

// Parsed spectrogram payload paired with its metadata.
export type SpectrogramFrame = {
  meta: SpectrogramMetaFrame;
  payload: Uint8Array | Float32Array;
};

// Union of known WebSocket frames for type-safe handling.
export type WebSocketFrame = StatusFrame | SpectrumMetaFrame | SpectrogramMetaFrame;
