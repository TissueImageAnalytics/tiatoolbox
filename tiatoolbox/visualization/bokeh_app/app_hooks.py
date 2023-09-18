"""Hooks to be executed upon specific events in bokeh app."""
import sys

import requests


def on_session_destroyed(session_context):
    """Hook to be executed when a session is destroyed."""
    user = session_context.request.arguments["user"]
    requests.get(f"http://127.0.0.1:5000/tileserver/reset/{user}", timeout=1000)
    sys.exit()
