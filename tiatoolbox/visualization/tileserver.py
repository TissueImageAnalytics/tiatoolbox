"""Simple Flask WSGI apps to display tiles as slippery maps."""

from __future__ import annotations

import copy
import io
import json
import os
import secrets
import sys
import tempfile
import urllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from flask import Flask, Response, jsonify, make_response, request, send_file
from flask.templating import render_template
from matplotlib import colormaps
from PIL import Image
from shapely.geometry import Point

from tiatoolbox import data, logger
from tiatoolbox.annotation import AnnotationStore, SQLiteStore
from tiatoolbox.tools.pyramid import AnnotationTileGenerator, ZoomifyGenerator
from tiatoolbox.utils.misc import add_from_dat, store_from_dat
from tiatoolbox.utils.postproc_defs import MultichannelToRGB
from tiatoolbox.utils.visualization import AnnotationRenderer, colourise_image
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.colors import Colormap

    from tiatoolbox.wsicore import WSIMeta


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader | str] | List[WSIReader | str]):
            A dictionary mapping layer names to image paths, annotation paths,
            or :obj:`WSIReader` objects to display. The dictionary should have
            a 'slide' key which is the base slide for the visualization.
            May also be a list, in which case generic names 'slide', 'layer-1',
            'layer-2' etc. will be used. First entry in list will be assumed to
            be the base slide. If a layer is a single-channel low-res overlay,
            it will be colourized using the 'viridis' colourmap.

    Examples:
        >>> from tiatoolbox.wsicore.wsireader import WSIReader
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

    def __init__(  # noqa: PLR0915
        self: TileServer,
        title: str,
        layers: dict[str, WSIReader | str] | list[WSIReader | str],
        renderer: AnnotationRenderer | None = None,
    ) -> None:
        """Initialize :class:`TileServer`."""
        super().__init__(
            __name__,
            template_folder=data._local_sample_path(  # noqa: SLF001
                Path("visualization") / "templates",
            ),
            static_url_path="",
            static_folder=data._local_sample_path(  # noqa: SLF001
                Path("visualization") / "static",
            ),
        )
        self.title = title
        self.layers = {}
        self.pyramids = {}
        self.renderer = renderer
        self.overlap = 0
        if renderer is None:  # pragma: no branch
            self.renderer = AnnotationRenderer(
                score_prop="type",
                thickness=-1,
                edge_thickness=1,
                zoomed_out_strat="scale",
                max_scale=8,
                blur_radius=0,
            )
        self.slide_mpps = {}
        self.renderers = {}
        self.overlaps = {}

        # Generic layer names if none provided.
        if isinstance(layers, list):
            layers_dict = {"slide": layers[0]}
            for i, p in enumerate(layers[1:]):
                layers_dict[f"layer-{i+1}"] = p
            layers = layers_dict
        # Set up the layer dict.
        meta = None
        # if layers provided directly, not using with app,
        # so just set default session_id
        self.default_session_id = len(layers) > 0
        if self.default_session_id:
            self.layers["default"] = {}
            self.pyramids["default"] = {}
            self.renderers["default"] = copy.deepcopy(self.renderer)
        for i, (key, layer) in enumerate(layers.items()):
            layer = self._get_layer_as_wsireader(layer, meta)  # noqa: PLW2901

            self.layers["default"][key] = layer

            if isinstance(layer, WSIReader):
                self.pyramids["default"][key] = ZoomifyGenerator(layer)
            else:
                self.pyramids["default"][key] = layer  # it's an AnnotationTileGenerator

            if i == 0:
                meta = layer.info  # base slide info
                self.slide_mpps["default"] = meta.mpp

        self.route(
            "/tileserver/layer/<layer>/<session_id>/zoomify/TileGroup<int:tile_group>/"
            "<int:z>-<int:x>-<int:y>@<int:res>x.jpg",
        )(
            self.zoomify,
        )
        self.route("/")(self.index)
        self.route("/tileserver/session_id")(self.session_id)
        self.route("/tileserver/color_prop", methods=["PUT"])(self.change_prop)
        self.route("/tileserver/slide", methods=["PUT"])(self.change_slide)
        self.route("/tileserver/cmap", methods=["PUT"])(self.change_mapper)
        self.route(
            "/tileserver/annotations",
            methods=["PUT"],
        )(self.load_annotations)
        self.route("/tileserver/overlay", methods=["PUT"])(self.change_overlay)
        self.route("/tileserver/commit", methods=["POST"])(self.commit_db)
        self.route("/tileserver/renderer/<prop>", methods=["PUT"])(self.update_renderer)
        self.route("/tileserver/reset/<session_id>", methods=["PUT"])(self.reset)
        self.route("/tileserver/secondary_cmap", methods=["PUT"])(
            self.change_secondary_cmap,
        )
        self.route("/tileserver/prop_names/<ann_type>")(self.get_properties)
        self.route("/tileserver/prop_values/<prop>/<ann_type>")(
            self.get_property_values,
        )
        self.route("/tileserver/color_prop", methods=["GET"])(self.get_color_prop)
        self.route("/tileserver/slide", methods=["GET"])(self.get_slide)
        self.route("/tileserver/cmap", methods=["GET"])(self.get_mapper)
        self.route("/tileserver/annotations", methods=["GET"])(self.get_annotations)
        self.route("/tileserver/overlay", methods=["GET"])(self.get_overlay)
        self.route("/tileserver/renderer/<prop>", methods=["GET"])(self.get_renderer)
        self.route("/tileserver/secondary_cmap", methods=["GET"])(
            self.get_secondary_cmap,
        )
        self.route("/tileserver/tap_query/<x>/<y>")(self.tap_query)
        self.route("/tileserver/prop_range", methods=["PUT"])(self.prop_range)
        self.route("/tileserver/channels", methods=["GET"])(self.get_channels)
        self.route("/tileserver/channels", methods=["PUT"])(self.set_channels)
        self.route("/tileserver/enhance", methods=["PUT"])(self.set_enhance)
        self.route("/tileserver/shutdown", methods=["POST"])(self.shutdown)

    def _get_session_id(self: TileServer) -> str:
        """Get the session_id from the request.

        Returns:
            str: The session_id name.

        """
        if self.default_session_id:
            return "default"
        return request.cookies.get("session_id")

    @staticmethod
    def _get_cmap(cmap: str | dict) -> Colormap:
        """Get the colourmap from the string sent."""
        if cmap == "None":
            return colormaps["jet"]
        if isinstance(cmap, str):
            return colormaps[cmap]

        def cmapp(x: Colormap) -> Colormap:
            """Dictionary colormap callable wrapper."""
            return cmap[x]

        return cmapp

    def _get_layer_as_wsireader(
        self: TileServer,
        layer: str | np.ndarray | WSIReader,
        meta: WSIMeta,
    ) -> WSIReader:
        """Gets appropriate image provider for layer.

        Args:
            layer (str | ndarray | WSIReader):
                A reference to an image or annotations to be displayed.
            meta (WSIMeta):
                The metadata of the base slide.

        Returns:
            WSIReader or AnnotationTileGenerator:
                The appropriate image source for the layer.

        """
        if isinstance(layer, (str, Path)):
            layer_path = Path(layer)
            if layer_path.suffix in [".jpg", ".png"]:
                # Assume it's a low-res heatmap.
                layer = np.array(Image.open(layer_path))
            elif layer_path.suffix == ".db":
                # Assume it's an annotation store.
                layer = AnnotationTileGenerator(
                    meta,
                    SQLiteStore(layer_path),
                    self.renderers["default"],
                    overlap=self.overlap,
                )
            elif layer_path.suffix == ".geojson":
                # Assume annotations in geojson format
                layer = AnnotationTileGenerator(
                    meta,
                    SQLiteStore.from_geojson(layer_path),
                    self.renderers["default"],
                    overlap=self.overlap,
                )
            else:
                # Assume it's a WSI.
                return WSIReader.open(layer_path)

        if isinstance(layer, np.ndarray):
            # Make into rgb if single channel.
            layer = colourise_image(layer)
            return VirtualWSIReader(layer, info=meta)

        if isinstance(layer, AnnotationStore):
            layer = AnnotationTileGenerator(
                meta,
                layer,
                self.renderers["default"],
                overlap=self.overlap,
            )

        return layer

    def zoomify(
        self: TileServer,
        layer: str,
        session_id: str,
        tile_group: int,  # skipcq: PYL-W0613  #noqa: ARG002
        z: int,
        x: int,
        y: int,
        res: int,
    ) -> Response:
        """Serve a Zoomify tile for a particular layer.

        Note that this should not be called directly, but will be called
        automatically by the Flask framework when a client requests a
        tile at the registered URL.

        Args:
            layer (str):
                The layer name.
            session_id (str):
                Session ID. Unique ID to disambiguate requests from different sessions.
            tile_group (int):
                The tile group. Currently unused.
            z (int):
                The zoom level.
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.
            res (int):
                Resolution to save the tiles at.
                Helps to specify high resolution tiles. Valid options are 1 and 2.

        Returns:
            flask.Response:
                The tile image response.

        """
        try:
            pyramid = self.pyramids[session_id][layer]
            if isinstance(self.layers[session_id][layer], VirtualWSIReader):
                interpolation = "nearest"
                transparent_value = 0
            else:
                interpolation = "optimise"
                transparent_value = None
            if isinstance(pyramid, AnnotationTileGenerator):
                interpolation = None
        except KeyError:
            return Response("Layer not found", status=404)
        try:
            tile_image = pyramid.get_tile(
                level=z,
                x=x,
                y=y,
                res=res,
                interpolation=interpolation,
                transparent_value=transparent_value,
            )
        except IndexError:
            return Response("Tile not found", status=404)
        image_io = io.BytesIO()
        tile_image.save(image_io, format="webp")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/webp")

    @staticmethod
    def update_types(sq: SQLiteStore) -> tuple:
        """Get the available types from the store."""
        types = sq.pquery("props['type']")
        types = [t for t in types if t is not None]
        return tuple(types)

    @staticmethod
    def decode_safe_name(name: str) -> Path:
        """Decode a URL-safe name."""
        return Path(urllib.parse.unquote(name).replace("\\", os.sep))

    def get_ann_layer(
        self: TileServer,
        session_id: str,
    ) -> AnnotationTileGenerator | ValueError:
        """Get the annotation layer for a session_id."""
        for layer in self.pyramids[session_id].values():
            if isinstance(layer, AnnotationTileGenerator):
                return layer
        msg = "No annotation layer found."
        raise ValueError(msg)

    def index(self: TileServer) -> str:
        """Serve the index page.

        Returns:
            flask.Response:
                The index page.

        """
        session_id = self._get_session_id()
        layers = [
            {
                "name": name,
                "url": f"/tileserver/layer/{name}/default/zoomify/"
                "{TileGroup}/{z}-{x}-{y}@1x.jpg",
                "size": [int(x) for x in layer.info.slide_dimensions],
                "mpp": float(np.mean(layer.info.mpp)),
            }
            for name, layer in self.layers[session_id].items()
        ]

        return render_template(
            "index.html",
            title=self.title,
            layers=json.dumps(layers),
        )

    def change_prop(self: TileServer) -> str:
        """Change the property to colour annotations by."""
        prop = request.form["prop"]
        session_id = self._get_session_id()
        self.renderers[session_id].score_prop = json.loads(prop)

        return "done"

    def session_id(self: TileServer) -> Response:
        """Set up a new session."""
        # respond with a random cookie to disambiguate sessions
        resp = make_response("done")
        session_id = "default" if self.default_session_id else secrets.token_urlsafe(16)
        resp.set_cookie("session_id", session_id, httponly=True)  # skipcq: PTC-W6003
        self.renderers[session_id] = copy.deepcopy(self.renderer)
        self.overlaps[session_id] = 0
        self.layers[session_id] = {}
        self.pyramids[session_id] = {}
        return resp

    def reset(self: TileServer, session_id: str) -> str:
        """Reset the tileserver."""
        del self.layers[session_id]
        del self.pyramids[session_id]
        del self.slide_mpps[session_id]
        del self.renderers[session_id]
        del self.overlaps[session_id]
        return "done"

    def change_slide(self: TileServer) -> str:
        """Change the slide."""
        session_id = self._get_session_id()
        slide_path = request.form["slide_path"]
        slide_path = self.decode_safe_name(slide_path)

        self.layers[session_id] = {"slide": WSIReader.open(Path(slide_path))}
        self.pyramids[session_id] = {
            "slide": ZoomifyGenerator(self.layers[session_id]["slide"], tile_size=256),
        }
        if self.layers[session_id]["slide"].info.mpp is None:
            self.layers[session_id]["slide"].info.mpp = [1, 1]
        self.slide_mpps[session_id] = self.layers[session_id]["slide"].info.mpp

        return "done"

    def change_mapper(self: TileServer) -> str:
        """Change the colour mapper for the overlay."""
        session_id = self._get_session_id()
        cmap = json.loads(request.form["cmap"])
        if isinstance(cmap, dict):
            cmap = dict(zip(cmap["keys"], cmap["values"]))
            self.renderers[session_id].score_fn = lambda x: x
        self.renderers[session_id].mapper = cmap
        self.renderers[session_id].function_mapper = None

        return "done"

    def change_secondary_cmap(self: TileServer) -> str:
        """Change the type-specific colour mapper for the overlay."""
        session_id = self._get_session_id()
        cmap = json.loads(request.form["cmap"])
        type_id = request.form["type_id"]
        prop = request.form["prop"]
        cmapp = self._get_cmap(cmap)

        cmap_dict = {"type": json.loads(type_id), "score_prop": prop, "mapper": cmapp}
        self.renderers[session_id].secondary_cmap = cmap_dict

        return "done"

    def update_renderer(self: TileServer, prop: str) -> str:
        """Update a property in the renderer.

        Args:
            prop (str): The property to update.

        """
        session_id = self._get_session_id()
        val = request.form["val"]
        val = json.loads(val)
        if val in ["None", "null"]:
            val = None
        self.renderers[session_id].__setattr__(prop, val)
        self.renderers[session_id].__setattr__(prop, val)
        if prop == "blur_radius":
            self.overlaps[session_id] = int(1.5 * val)
            self.get_ann_layer(session_id).overlap = self.overlaps[session_id]
        return "done"

    def load_annotations(self: TileServer) -> str:
        """Load annotations from a dat file.

        Adds to an existing store if one is already present,
        otherwise creates a new store.

        Returns:
            str: A jsonified list of types.

        """
        session_id = self._get_session_id()
        file_path = request.form["file_path"]
        model_mpp = json.loads(request.form["model_mpp"])
        file_path = self.decode_safe_name(file_path)

        for layer in self.pyramids[session_id].values():
            if isinstance(layer, AnnotationTileGenerator):
                add_from_dat(
                    layer.store,
                    file_path,
                    np.array(model_mpp) / np.array(self.slide_mpps[session_id]),
                )
                types = self.update_types(layer.store)
                return json.dumps(types)

        sq = store_from_dat(
            file_path,
            np.array(model_mpp) / np.array(self.slide_mpps[session_id]),
        )
        tmp_path = Path(tempfile.gettempdir()) / "temp.db"
        sq.dump(tmp_path)
        sq = SQLiteStore(tmp_path)
        self.pyramids[session_id]["overlay"] = AnnotationTileGenerator(
            self.layers[session_id]["slide"].info,
            sq,
            self.renderers[session_id],
            overlap=self.overlaps[session_id],
        )
        self.layers[session_id]["overlay"] = self.pyramids[session_id]["overlay"]
        types = self.update_types(sq)
        return json.dumps(types)

    def change_overlay(self: TileServer) -> str:
        """Change the overlay.

        If the path points to some annotations, the current overlay
        is replaced with the new one. If the path points to an image,
        it is added as a new layer.

        Returns:
            str: A jsonified list of types.

        """
        session_id = self._get_session_id()
        overlay_path = request.form["overlay_path"]
        overlay_path = self.decode_safe_name(overlay_path)

        if overlay_path.suffix in [".jpg", ".png", ".tiff", ".svs", ".ndpi", ".mrxs"]:
            layer = f"layer{len(self.pyramids[session_id])}"
            if overlay_path.suffix == ".tiff":
                self.layers[session_id][layer] = OpenSlideWSIReader(
                    overlay_path,
                    mpp=self.layers[session_id]["slide"].info.mpp[0],
                )
            elif overlay_path.suffix in [".jpg", ".png"]:
                self.layers[session_id][layer] = VirtualWSIReader(
                    Path(overlay_path),
                    info=self.layers[session_id]["slide"].info,
                )
            else:
                self.layers[session_id][layer] = WSIReader.open(overlay_path)
            self.pyramids[session_id][layer] = ZoomifyGenerator(
                self.layers[session_id][layer],
            )
            return json.dumps(layer)
        if overlay_path.suffix == ".geojson":
            sq = SQLiteStore.from_geojson(overlay_path)
        elif overlay_path.suffix == ".dat":
            sq = store_from_dat(overlay_path)
        if overlay_path.suffix == ".db":
            sq = SQLiteStore(overlay_path, auto_commit=False)
        else:
            # make a temporary db for the new annotations
            tmp_path = Path(tempfile.gettempdir()) / "temp.db"
            sq.dump(tmp_path)
            sq = SQLiteStore(tmp_path)

        for layer in self.pyramids[session_id].values():
            if isinstance(layer, AnnotationTileGenerator):
                layer.store = sq
                logger.info("Loaded %d annotations.", len(sq))
                types = self.update_types(sq)
                return json.dumps(types)
        self.pyramids[session_id]["overlay"] = AnnotationTileGenerator(
            self.layers[session_id]["slide"].info,
            sq,
            self.renderers[session_id],
            overlap=self.overlaps[session_id],
        )
        logger.info(
            "Loaded %d annotations.",
            len(self.pyramids[session_id]["overlay"].store),
        )
        self.layers[session_id]["overlay"] = self.pyramids[session_id]["overlay"]
        types = self.update_types(sq)
        return json.dumps(types)

    def get_properties(self: TileServer, ann_type: str) -> str:
        """Get all the properties of the annotations in the store.

        Args:
            ann_type (str): The type of annotations to get the properties for.

        Returns:
            str: A jsonified list of the properties.

        """
        session_id = self._get_session_id()
        where = None
        if ann_type != "all":
            where = f'props["type"]=={ann_type}'
        ann_props = self.get_ann_layer(session_id).store.pquery(
            select="*",
            where=where,
            unique=False,
        )
        props = []
        for prop_dict in ann_props.values():
            props.extend(list(prop_dict.keys()))
        return json.dumps(list(set(props)))

    def get_property_values(self: TileServer, prop: str, ann_type: str) -> str:
        """Get all the values of a property in the store.

        Args:
            prop (str): The property to get the values of.
            ann_type (str): The type of annotations to get the values for.

        Returns:
            str: A jsonified list of the values of the property.
        """
        session_id = self._get_session_id()
        where = None
        if ann_type != "all":
            where = f'props["type"]=={ann_type}'
        if "overlay" not in self.pyramids[session_id]:
            return json.dumps([])
        ann_props = self.get_ann_layer(session_id).store.pquery(
            select=f"props['{prop}']",
            where=where,
            unique=True,
        )
        return json.dumps(list(ann_props))

    def commit_db(self: TileServer) -> str:
        """Commit changes to the current store.

        If the store is not already associated with a .db file,
        the save_path is used to create a new .db file.

        """
        session_id = self._get_session_id()
        save_path = request.form["save_path"]
        save_path = self.decode_safe_name(save_path)
        for layer in self.pyramids[session_id].values():
            if isinstance(layer, AnnotationTileGenerator):
                if (
                    layer.store.path.suffix == ".db"
                    and layer.store.path.name != "temp.db"
                ):
                    logger.info("%s*.db committed.", layer.store.path.stem)
                    layer.store.commit()
                else:
                    layer.store.commit()
                    layer.store.dump(str(save_path))
                    logger.info("db saved to %s.", save_path)
                return "done"
        return "nothing to save"

    def get_color_prop(self: TileServer) -> Response:
        """Get the property used to color annotations from renderer."""
        session_id = self._get_session_id()
        return jsonify(self.renderers[session_id].score_prop)

    def get_slide(self: TileServer) -> Response:
        """Get the slide metadata."""
        session_id = self._get_session_id()
        info = self.layers[session_id]["slide"].info.as_dict()
        info["file_path"] = str(info["file_path"])
        return jsonify(info)

    def get_mapper(self: TileServer) -> Response:
        """Get the mapper used to color annotations from renderer."""
        session_id = self._get_session_id()
        mapper = self.renderers[session_id].raw_mapper
        return jsonify(mapper)

    def get_annotations(self: TileServer) -> Response:
        """Get the annotations in the specified bounds."""
        session_id = self._get_session_id()
        bounds = json.loads(request.form["bounds"])
        where = json.loads(request.form["where"])
        annotations = self.get_ann_layer(session_id).store.query(
            geometry=bounds,
            where=where,
        )
        annotations = [
            {"geom": ann.geometry.wkt, "properties": ann.properties}
            for ann in annotations.values()
        ]
        return jsonify(annotations)

    def get_overlay(self: TileServer) -> Response:
        """Get the overlay info."""
        session_id = self._get_session_id()
        return jsonify(str(self.get_ann_layer(session_id).store.path))

    def get_renderer(self: TileServer, prop: str) -> Response:
        """Get the requested property from the renderer."""
        session_id = self._get_session_id()
        return jsonify(getattr(self.renderers[session_id], prop))

    def get_secondary_cmap(self: TileServer) -> Response:
        """Get the secondary cmap from the renderer."""
        session_id = self._get_session_id()
        mapper = self.renderers[session_id].secondary_cmap
        mapper["mapper"] = mapper["mapper"].__class__.__name__
        return jsonify(mapper)

    def tap_query(self: TileServer, x: float, y: float) -> Response:
        """Query for annotations at a point.

        Args:
            x (float): The x coordinate.
            y (float): The y coordinate.

        Returns:
            Response: The jsonified dict of the properties of the
            smallest annotation returned from the query at the point.

        """
        session_id = self._get_session_id()
        anns = self.get_ann_layer(session_id).store.query(
            Point(x, y),
        )
        if len(anns) == 0:
            return json.dumps({})
        return jsonify(list(anns.values())[-1].properties)

    def prop_range(self: TileServer) -> str:
        """Set the range which the color mapper will map to.

        It will create an appropriate function to map the range to the
        range [0, 1], and set the renderers score_fn to this function.

        """
        session_id = self._get_session_id()
        prop_range = json.loads(request.form["range"])
        if prop_range is None:
            self.renderers[session_id].score_fn = lambda x: x
            return "done"
        minv, maxv = prop_range
        self.renderers[session_id].score_fn = lambda x: (x - minv) / (maxv - minv)
        return "done"

    def get_channels(self: TileServer) -> Response:
        """Get the channels of the slide."""
        session_id = self._get_session_id()
        if isinstance(self.layers[session_id]["slide"].post_proc, MultichannelToRGB):
            return jsonify(
                {
                    "channels": self.layers[session_id]["slide"].post_proc.color_dict,
                    "active": self.layers[session_id]["slide"].post_proc.channels,
                },
            )
        return jsonify({"channels": {}, "active": []})

    def set_channels(self: TileServer) -> str:
        """Set the channels of the slide."""
        session_id = self._get_session_id()
        if isinstance(self.layers[session_id]["slide"].post_proc, MultichannelToRGB):
            channels = json.loads(request.form["channels"])
            active = json.loads(request.form["active"])
            self.layers[session_id]["slide"].post_proc.color_dict = channels
            self.layers[session_id]["slide"].post_proc.channels = active
            self.layers[session_id]["slide"].post_proc.is_validated = False
        return "done"

    def set_enhance(self: TileServer) -> str:
        """Set the enhance factor of the slide."""
        session_id = self._get_session_id()
        enhance = json.loads(request.form["val"])
        if isinstance(self.layers[session_id]["slide"].post_proc, MultichannelToRGB):
            self.layers[session_id]["slide"].post_proc.enhance = enhance
        return "done"

    @staticmethod
    def shutdown() -> None:
        """Shutdown the tileserver."""
        sys.exit()
