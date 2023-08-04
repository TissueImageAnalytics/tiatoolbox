import os
import sys

import requests


def on_session_destroyed(session_context):
    # If present, this function executes when the server closes session.
    user = session_context.request.arguments["user"]
    host = os.environ.get("HOST2")
    if host is None:
        host = "127.0.0.1"
        sys.exit()
    requests.get(f"http://{host}:5000/tileserver/reset/{user}")
