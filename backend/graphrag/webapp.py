"""Backward compatibility wrapper for webapp.serve_app - now uses FastAPI."""

from __future__ import annotations

from pathlib import Path

# For backward compatibility, re-export the FastAPI-based serve_app
from .server import serve_app

__all__ = ["serve_app"]

