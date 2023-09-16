"""Hooks to be executed upon specific events in bokeh app."""
import sys

from tiatoolbox import logger


def on_session_destroyed(session_context):
    """Hook to be executed when a session is destroyed."""
    logger.info(
        "Session destroyed for user %s.",
        session_context.request.arguments["user"],
    )
    sys.exit()
