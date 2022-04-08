"""Simple Flask WSGI apps to display tiles as slippery maps."""
import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
from flask import Flask, Response, send_file
from flask.templating import render_template

from tiatoolbox import data
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.wsicore.wsireader import WSIReader


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader]):
            A dictionary mapping layer names to :obj:`WSIReader` objects
            to display.

    Examples:
        >>> from tiatoolbox.wsiscore.wsireader import WSIReader
        >>> from tiatoolbox.visualization.tileserver import TileServer
        >>> wsi = WSIReader.open("CMU-1.svs")
        >>> app = TileServer(
        ...     title="Testing TileServer",
        ...     layers={
        ...         "My SVS": wsi,
        ...     },
        ... )
        >>> app.run()
    """

    def __init__(self, title: str, layers: Dict[str, WSIReader]) -> None:
        super().__init__(
            __name__,
            template_folder=data._local_sample_path(
                Path("visualization") / "templates"
            ),
            static_url_path="",
            static_folder=data._local_sample_path(Path("visualization") / "static"),
        )
        self.tia_title = title
        self.tia_layers = layers
        self.tia_pyramids = {
            key: ZoomifyGenerator(reader) for key, reader in self.tia_layers.items()
        }
        self.route(
            "/layer/<layer>/zoomify/TileGroup<int:tile_group>/"
            "<int:z>-<int:x>-<int:y>.jpg"
        )(
            self.zoomify,
        )
        self.route("/")(self.index)

    def zoomify(
        self, layer: str, tile_group: int, z: int, x: int, y: int  # skipcq: PYL-w0613
    ) -> Response:
        """Serve a Zoomify tile for a particular layer.

        Note that this should not be called directly, but will be called
        automatically by the Flask framework when a client requests a
        tile at the registered URL.

        Args:
            layer (str):
                The layer name.
            tile_group (int):
                The tile group. Currently unused.
            z (int):
                The zoom level.
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.

        Returns:
            Response:
                The tile image response.

        """
        try:
            pyramid = self.tia_pyramids[layer]
        except KeyError:
            return Response("Layer not found", status=404)
        try:
            tile_image = pyramid.get_tile(level=z, x=x, y=y)
        except IndexError:
            return Response("Tile not found", status=404)
        image_io = io.BytesIO()
        tile_image.save(image_io, format="JPEG")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/jpeg")

    def index(self) -> Response:
        """Serve the index page.

        Returns:
            Response: The index page.

        """
        layers = [
            {
                "name": name,
                "url": f"/layer/{name}/zoomify/{{TileGroup}}/{{z}}-{{x}}-{{y}}.jpg",
                "size": [int(x) for x in reader.info.slide_dimensions],
                "mpp": float(np.mean(reader.info.mpp)),
            }
            for name, reader in self.tia_layers.items()
        ]
        return render_template(
            "index.html", title=self.tia_title, layers=json.dumps(layers)
        )
