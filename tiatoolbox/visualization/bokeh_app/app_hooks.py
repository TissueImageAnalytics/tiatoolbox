"""Hooks to be executed upon specific events in bokeh app."""

import os
import sys
from contextlib import suppress

import requests
from bokeh.application.application import SessionContext

PORT = os.environ.get("TIATOOLBOX_TILESERVER_PORT", "5000")


def on_session_destroyed(session_context: SessionContext) -> None:
    """Hook to be executed when a session is destroyed."""
    user = session_context.request.arguments["user"]
    with suppress(requests.exceptions.ReadTimeout):
        requests.get(
            f"http://127.0.0.1:{PORT}/tileserver/reset/{user}",
            timeout=5,
        )
    sys.exit()
