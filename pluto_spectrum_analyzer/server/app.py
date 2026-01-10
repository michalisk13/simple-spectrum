"""FastAPI application factory for the spectrum analyzer server."""

from __future__ import annotations

from fastapi import FastAPI

from pluto_spectrum_analyzer.config import SpectrumConfig
from pluto_spectrum_analyzer.engine import Engine
from pluto_spectrum_analyzer.server.routes import router
from pluto_spectrum_analyzer.server.ws import router as ws_router


def create_app(engine: Engine | None = None) -> FastAPI:
    cfg = SpectrumConfig()
    app = FastAPI(title="Pluto Spectrum Analyzer")
    app.state.engine = engine or Engine(cfg)
    app.include_router(router)
    app.include_router(ws_router)
    return app


# Provide a default app instance for non-factory uvicorn usage.
app = create_app()
