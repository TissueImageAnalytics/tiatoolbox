import os
import sys
import urllib
from pathlib import PureWindowsPath

import requests


def make_safe_name(name):
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")


def on_session_destroyed(session_context):
    # If present, this function executes when the server closes session.
    user = session_context.request.arguments["user"]
    host = os.environ.get("HOST2")
    if host is None:
        host = "127.0.0.1"
        sys.exit()
    fname = r"/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs"
    fname = make_safe_name(fname)
    print("cleaning up...")
    requests.get(f"http://{host}:5000/tileserver/reset/{user}")
