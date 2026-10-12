"""Local node-graph app server (FastAPI, 127.0.0.1 only)."""

from vlm_pipeline.app.main import create_app

__all__ = ["create_app"]
