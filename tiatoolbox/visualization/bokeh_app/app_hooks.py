"""Hooks to be executed upon specific events in bokeh app."""

import sys
from contextlib import suppress

import requests
from bokeh.application.application import SessionContext


def on_session_destroyed(session_context: SessionContext) -> None:
    """Hook to be executed when a session is destroyed."""
    user = session_context.request.arguments["user"]
    with suppress(requests.exceptions.ReadTimeout):
        requests.get(f"http://127.0.0.1:5000/tileserver/reset/{user}", timeout=5)
    sys.exit()
