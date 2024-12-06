"""Module to run TileServer for testing purpose."""

from flask_cors import CORS

from tiatoolbox.visualization import TileServer
from tiatoolbox.wsicore import WSIReader

svs = "/path/to/fsspec.json"

reader = WSIReader.open(svs)

# Initialize and run the TileServer
tile_server = TileServer(
    title="Tiatoolbox TileServer",
    layers={"layer": reader},
)
CORS(tile_server, send_wildcard=True)


tile_server.run(host="127.0.0.1", port=5000)
