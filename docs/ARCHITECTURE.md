## Architecture overview

This project is transitioning from a PyQt-only UI to a web UI with a shared,
headless backend. The goal is to keep all SDR/DSP logic in Python while
streaming only display artifacts (spectrum traces, spectrogram rows, metadata)
to clients.

### Key components

#### Engine (headless)
* Owns the SDR connection lifecycle (connect, disconnect, reconnect).
* Owns the worker lifecycle for DSP processing.
* Owns the current config and applies updates transactionally.
* Publishes internal engine frames for clients (PyQt and FastAPI).

#### Worker (DSP thread)
* Performs SDR reads and DSP computations (FFT, detectors, spectrogram slices).
* Emits internal frames (spectrum + spectrogram) back to the engine.
* No UI or web dependencies.

#### Protocol module
* Defines the Protocol Contract v1.0 JSON schema for wire frames.
* Defines binary payload header helpers (SPAY header).
* Defines internal engine frame dataclasses and helpers to serialize to wire frames.

#### PyQt UI (reference client)
* Subscribes to engine frames and renders spectrum/spectrogram plots.
* Never instantiates the SDR or worker directly.
* Acts as the reference client until the web UI reaches feature parity.

#### Web server (future)
* FastAPI REST for control (config, status, connect/disconnect).
* WebSocket for real-time streaming of meta + binary payload frames.
* Uses the same Engine as PyQt for consistent behavior.

### Data flow (current + target)
1. PyQt or FastAPI calls Engine methods (connect/apply_config).
2. Engine starts Worker and receives engine frames.
3. Engine publishes engine frames to subscribers (PyQt now, FastAPI later).
4. FastAPI WebSocket will convert engine frames to wire frames and stream with
   metadata + binary payloads.

### No-raw-IQ rule
* Only display artifacts are ever streamed: spectrum traces, spectrogram rows,
  and metadata.
* Raw IQ never leaves the Python backend.

### Performance guardrails (future)
* Decimation and quantization in the Engine for stable bandwidth at high FFT sizes.
* Wire frames carry metadata + binary payloads in the Protocol Contract v1.0 format.
