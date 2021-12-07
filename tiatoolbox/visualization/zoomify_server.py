import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
from flask import Flask, send_file
from flask.templating import render_template

from tiatoolbox import data
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.wsicore.wsireader import WSIReader


class ZoomifyViewer(Flask):
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
            self.tile,
        )
        self.route("/")(self.index)

    def tile(self, layer: str, tile_group: int, z: int, x: int, y: int):
        """Serve a tile.

        Args:
            layer (str): The layer name.
            tile_group (int): The tile group.
            z (int): The zoom level.
            x (int): The x coordinate.
            y (int): The y coordinate.

        Returns:
            bytes: The tile JPEG data.

        """
        pyramid = self.tia_pyramids[layer]
        try:
            tile_image = pyramid.get_tile(level=z, x=x, y=y)
        except IndexError:
            return "", 404
        image_io = io.BytesIO()
        tile_image.save(image_io, format="JPEG")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/jpeg")

    def index(self):
        """Serve the index page.

        Returns:
            str: The index page.

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
