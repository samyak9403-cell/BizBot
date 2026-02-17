"""WSGI entry point for production deployment (Render, Gunicorn, etc.)."""

from src.api import create_app

app = create_app()
