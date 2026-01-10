// Typed representations of the REST API payloads used by the frontend.

// Shared error shape from the backend engine error frames.
export type EngineError = {
  // Monotonic timestamp in nanoseconds when the error was generated.
  ts_monotonic_ns: number;
  // Stable error code for programmatic handling.
  error_code: string;
  // Human-friendly error message for UI display.
  message: string;
  // Optional structured error details.
  details?: Record<string, unknown> | null;
  // True when the error is recoverable without restarting the app.
  recoverable: boolean;
  // Optional wall-clock timestamp for logging.
  ts_unix_ms?: number | null;
};

// Status snapshot returned by the backend engine.
export type EngineStatus = {
  // Monotonic timestamp in nanoseconds for the snapshot.
  ts_monotonic_ns: number;
  // True when the SDR is connected and streaming.
  connected: boolean;
  // Optional SDR URI in use.
  uri?: string | null;
  // Human-readable device name for display.
  device_name: string;
  // Current center frequency (Hz).
  center_hz: number;
  // Current span (Hz).
  span_hz: number;
  // SDR sample rate (Hz).
  sample_rate_hz: number;
  // RF bandwidth (Hz).
  rf_bw_hz: number;
  // Gain mode string (manual/auto/etc.).
  gain_mode: string;
  // Gain in dB.
  gain_db: number;
  // FFT size used for processing.
  fft_size: number;
  // Window function name.
  window: string;
  // Resolution bandwidth (Hz).
  rbw_hz: number;
  // Video bandwidth (Hz).
  vbw_hz: number;
  // Target update rate (Hz).
  update_hz_target: number;
  // Actual update rate (Hz).
  update_hz_actual: number;
  // Spectrum frames per second.
  spectrum_fps: number;
  // Spectrogram rows per second.
  spectrogram_fps: number;
  // Legacy drop count from engine worker.
  frame_drop_count: number;
  // Last processing time (ms).
  processing_ms: number;
  // Rolling average processing time (ms).
  avg_processing_ms: number;
  // Rolling average processing time per frame (ms).
  frame_processing_ms_avg: number;
  // Total frames dropped in the WebSocket fanout.
  frames_dropped: number;
  // Whether spectrogram streaming is enabled.
  spectrogram_enabled: boolean;
  // Spectrogram row rate (Hz).
  spectrogram_rate: number;
  // Spectrogram history depth (seconds).
  spectrogram_time_span_s: number;
  // Optional status message for UI.
  message?: string | null;
  // Optional wall-clock timestamp for display.
  ts_unix_ms?: number | null;
};

// REST response from GET /api/status.
export type ApiStatusResponse = {
  // Latest status snapshot from the engine.
  status: EngineStatus;
  // Optional error metadata if the engine reported one.
  error: EngineError | null;
};

// App configuration settings mirrored from the backend SpectrumConfig dataclass.
export type SpectrumConfig = {
  uri: string;
  center_hz: number;
  sample_rate_hz: number;
  rf_bw_hz: number;
  gain_db: number;
  gain_mode: string;
  fft_size: number;
  update_ms: number;
  buffer_factor: number;
  overlap: number;
  hover_rate_hz: number;
  rbw_mode: string;
  rbw_hz: number;
  window: string;
  vbw_mode: string;
  vbw_hz: number;
  detector: string;
  trace_type: string;
  trace2_enabled: boolean;
  avg_count: number;
  avg_mode: string;
  ref_level_db: number;
  display_range_db: number;
  spectrogram_mode: string;
  max_spectrum_bins: number;
  max_spectrogram_cols: number;
  spectrogram_quantize: boolean;
  measurement_mode: boolean;
  dc_remove: boolean;
  dc_blank_bins: number;
};

// Stream metadata for the spectrogram pipeline.
export type StreamMetadata = {
  spectrogram_enabled: boolean;
  spectrogram_rate: number;
  spectrogram_time_span_s: number;
};

// REST response from GET/POST /api/config.
export type ApiConfigResponse = {
  config: SpectrumConfig;
  stream: StreamMetadata;
};

// Payload for POST /api/config (partial updates).
export type ConfigUpdatePayload = Partial<SpectrumConfig> & Partial<StreamMetadata>;

// Standard ok/status response from SDR commands.
export type ApiCommandResponse = {
  ok: boolean;
  status: EngineStatus;
  error?: EngineError | null;
  message?: string;
};

// List of preset names from GET /api/presets.
export type ApiPresetsResponse = {
  presets: string[];
};

// Payload for POST /api/presets/apply.
export type ApplyPresetRequest = {
  name: string;
  measure_detector?: string;
};

// Response from POST /api/presets/apply.
export type ApplyPresetResponse = {
  ok: boolean;
  config: ApiConfigResponse;
};

// Client-side API error shape for fetch failures.
export type ApiError = {
  // Human-readable message to show in notifications.
  message: string;
  // HTTP status code, when applicable.
  status?: number;
  // Optional detail payload from the response body.
  details?: string;
};
