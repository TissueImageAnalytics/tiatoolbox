"""Hooks to be executed upon specific events in bokeh app."""
import sys


def on_session_destroyed(session_context):  # noqa: ARG001
    """Hook to be executed when a session is destroyed."""
    sys.exit()
