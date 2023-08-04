"""Hooks to be executed upon specific events in bokeh app."""
import os
import sys

import requests


def on_session_destroyed(session_context):
    """Hook to be executed when a session is destroyed."""
    user = session_context.request.arguments["user"]
    host = os.environ.get("HOST2")
    if host is None:
        host = "127.0.0.1"
        sys.exit()
    requests.get(f"http://{host}:5000/tileserver/reset/{user}")
