"""Simple Flask WSGI apps to display tiles as slippery maps."""
import io
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from flask import Flask, Response, send_file
from flask.templating import render_template
from PIL import Image

from tiatoolbox import data
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.utils.visualization import colourise_image
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader | str] | List[WSIReader | str]):
            A dictionary mapping layer names to image paths or
            :obj:`WSIReader` objects to display. May also be a list,
            in which case generic names 'layer-1', 'layer-2' etc.
            will be used.
            If layer is a single-channel low-res overlay, it will be
            colourized using the 'viridis' colourmap

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

    def __init__(
        self,
        title: str,
        layers: Union[Dict[str, Union[WSIReader, str]], List[Union[WSIReader, str]]],
    ) -> None:
        super().__init__(
            __name__,
            template_folder=data._local_sample_path(
                Path("visualization") / "templates"
            ),
            static_url_path="",
            static_folder=data._local_sample_path(Path("visualization") / "static"),
        )
        self.tia_title = title

        # Generic layer names if none provided.
        if isinstance(layers, list):
            layers = {f"layer-{i}": p for i, p in enumerate(layers)}
        # Set up the layer dict.
        layer_def = {}
        meta = None
        for i, key in enumerate(layers):
            layer = layers[key]

            if isinstance(layer, (str, Path)):
                layer_path = Path(layer)
                if layer_path.suffix in [".jpg", ".png"]:
                    # Assume its a low-res heatmap.
                    layer = Image.open(layer_path)
                    layer = np.array(layer)
                else:
                    layer = WSIReader.open(layer_path)

            if isinstance(layer, np.ndarray):
                # Make into rgb if single channel.
                layer = colourise_image(layer)
                layer = VirtualWSIReader(layer, info=meta)

            layer_def[key] = layer
            if i == 0:
                meta = layer.info

        self.tia_layers = layer_def
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
