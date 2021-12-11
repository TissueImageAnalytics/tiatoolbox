from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader

wsi = WSIReader.open("/home/john/Downloads/CMU-1.svs")
# wsi = WSIReader.open("/home/john/Downloads/test1.jp2")
app = TileServer(
    title="Testing TileServer",
    layers={
        "My SVS": wsi,
    },
)
app.run(port=8080)
