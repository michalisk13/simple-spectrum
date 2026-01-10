## Architecture block diagram

```mermaid
flowchart LR
    subgraph Clients
        Qt[PyQt UI - reference client]
        Web[Web UI - future]
    end

    subgraph Backend
        Engine[Engine - headless]
        Worker[SpectrumWorker - DSP thread]
        SDR[Pluto SDR - pyadi-iio]
        Protocol[Protocol v1.0 helpers]
        REST[FastAPI REST - future]
        WS[FastAPI WebSocket - future]
    end

    Qt -->|subscribe engine frames| Engine
    Web -->|REST control| REST
    Web -->|WS stream meta+binary| WS
    REST --> Engine
    WS --> Engine
    Engine -->|config apply| Worker
    Worker -->|engine frames| Engine
    Worker --> SDR
    Engine --> Protocol
    WS --> Protocol
```

Notes:
* Engine publishes internal engine frames to clients.
* Protocol helpers serialize engine frames into wire frames for WS.
* Only spectrum/spectrogram display artifacts and metadata are streamed.
