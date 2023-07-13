import json
import operator
import os
import pickle
import sys
import tempfile
import urllib
from cmath import pi
from pathlib import Path, PureWindowsPath
from shutil import rmtree
from threading import Thread

import matplotlib.cm as cm
import numpy as np
import requests
import torch

# Bokeh basics
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import (
    BasicTickFormatter,
    BoxEditTool,
    Button,
    CheckboxButtonGroup,
    CheckboxGroup,
    Circle,
    ColorBar,
    ColorPicker,
    ColumnDataSource,
    Div,
    Dropdown,
    FuncTickFormatter,
    GraphRenderer,
    HoverTool,
    Line,
    LinearColorMapper,
    MultiChoice,
    PointDrawTool,
    RadioButtonGroup,
    Segment,
    Select,
    Slider,
    Spinner,
    StaticLayoutProvider,
    TabPanel,
    Tabs,
    TapTool,
    TextInput,
    Toggle,
)
from bokeh.models.tiles import WMTSTileSource
from bokeh.plotting import figure
from bokeh.util import token
from flask_cors import CORS
from requests.adapters import HTTPAdapter, Retry

from tiatoolbox.annotation.dsl import SQL_GLOBALS, SQLTriplet
from tiatoolbox.models.architecture import fetch_pretrained_weights

# from tiatoolbox.models.architecture.nuclick import NuClick
# from tiatoolbox.models.engine.interactive_segmentor import (
#     InteractiveSegmentor,
#     IOInteractiveSegmentorConfig,
# )
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer, random_colors
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.visualization.ui_utils import get_level_by_extent
from tiatoolbox.wsicore.wsireader import WSIReader


# Define helper functions/classes
# region
class DummyAttr:
    def __init__(self, val):
        self.item = val


class UIWrapper:
    def __getitem__(self, key):
        # gets active ui element - implement later
        return win_dicts[active][key]


def get_view_bounds(dims, plot_size):
    """helper to get the current view bounds"""
    pad = int(np.mean(dims) / 10)
    aspect_ratio = plot_size[0] / plot_size[1]
    large_dim = np.argmax(np.array(dims) / plot_size)

    if large_dim == 1:
        x_range_start = -0.5 * (dims[1] * aspect_ratio - dims[0]) - aspect_ratio * pad
        x_range_end = (
            dims[1] * aspect_ratio
            - 0.5 * (dims[1] * aspect_ratio - dims[0])
            + aspect_ratio * pad
        )
        y_range_start = -dims[1] - pad
        y_range_end = pad
        # UI["p"].x_range.min_interval = ?
    else:
        x_range_start = -aspect_ratio * pad
        x_range_end = dims[0] + pad * aspect_ratio
        y_range_start = (
            -dims[0] / aspect_ratio + 0.5 * (dims[0] / aspect_ratio - dims[1]) - pad
        )
        y_range_end = 0.5 * (dims[0] / aspect_ratio - dims[1]) + pad
    return x_range_start, x_range_end, y_range_start, y_range_end


def to_num(x):
    """helper to convert a str representation of a number to an appropriate
    numerical value."""
    if not isinstance(x, str):
        return x
    if x == "None":
        return None
    try:
        return int(x)
    except:
        return float(x)


def get_from_config(keys, default=None):
    """helper to get a value from a config dict"""
    c_dict = config.config
    for k in keys:
        if k in c_dict:
            c_dict = c_dict[k]
        else:
            return default
    return c_dict


def make_ts(route, z_levels, init_z=4):
    """helper to make a tile source"""
    if init_z is None:
        init_z = UI["vstate"].init_z
        # if UI["vstate"].init:
        # init_z=4
    print(f"init z is: {init_z}")
    sf = 2 ** (z_levels - 9 - init_z + 4)  # 2 ** (z_levels - 9)
    ts = WMTSTileSource(
        name="WSI provider",
        url=route,
        attribution="",
        snap_to_zoom=False,
        min_zoom=0,
        max_zoom=z_levels - 1,
    )
    ts.tile_size = 256
    ts.initial_resolution = (
        40211.5 * sf * (2 / (100 * pi))  # * (256/512)
    )  # 156543.03392804097    40030 great circ
    ts.x_origin_offset = 0  # 5000000
    # ts.y_origin_offset=-2500000
    ts.y_origin_offset = sf * 10294144.78 * (2 / (100 * pi))
    ts.wrap_around = False
    # ts.max_zoom=10
    # ts.min_zoom=10
    return ts


def to_float_rgb(rgb):
    """Helper to convert from int to float rgb(a) tuple"""
    return tuple(v / 255 for v in rgb)


def to_int_rgb(rgb):
    """Helper to convert from float to int rgb(a) tuple"""
    return tuple(int(v * 255) for v in rgb)


def name2type(name):
    name = UI["vstate"].orig_types[name]
    if isinstance(name, str):
        return f'"{name}"'
    return name


def name2type_key(name):
    # do we need this?
    try:
        return int(name)
    except:
        return f"{name}"


def hex2rgb(hex_val):
    return tuple(int(hex_val[i : i + 2], 16) / 255 for i in (1, 3, 5))


def rgb2hex(rgb):
    """covert float rgb(a) tuple to hex string"""
    return "#{:02x}{:02x}{:02x}".format(*to_int_rgb(rgb))


def make_color_seq_from_cmap(cmap):
    if cmap is None:
        return [
            rgb2hex((1.0, 1.0, 1.0)),
            rgb2hex((1.0, 1.0, 1.0)),
        ]  # no colors if using dict
    colors = []
    for v in np.linspace(0, 1, 50):
        colors.append(rgb2hex(cmap(v)))
    return colors


def make_safe_name(name):
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")


def make_color_dict(types):
    colors = random_colors(len(types))
    # grab colors out of config["colour_dict"] if possible, otherwise use random
    type_colours = {}
    for i, t in enumerate(types):
        if str(t) in config["colour_dict"]:
            type_colours[t] = to_float_rgb(config["colour_dict"][str(t)])
        else:
            type_colours[t] = (*colors[i], 1)
    return type_colours


def set_alpha_glyph(glyph, alpha):
    """Sets both fill and line alpha for a glyph"""
    glyph.fill_alpha = alpha
    glyph.line_alpha = alpha


def get_mapper_for_prop(prop, enforce_dict=False):
    # find out the unique values of the chosen property
    print(prop)
    resp = UI["s"].get(f"http://{host2}:5000/tileserver/prop_values/{prop}/all")
    prop_vals = json.loads(resp.text)
    # guess what cmap should be
    if (len(prop_vals) > 10 or len(prop_vals) == 0) and not enforce_dict:
        cmap = (
            "viridis" if UI["cmap_select"].value == "Dict" else UI["cmap_select"].value
        )
    else:
        cmap = make_color_dict(prop_vals)
    return cmap


def update_mapper():
    if UI["vstate"].types is not None:
        # UI["vstate"].mapper = make_color_dict(UI["vstate"].types)
        update_renderer("mapper", UI["vstate"].mapper)


def update_renderer(prop, value):
    if prop == "mapper":
        if value == "dict":
            if UI["cprop_input"].value == "type":
                value = UI["vstate"].mapper  # put the type mapper dict back
            else:
                value = get_mapper_for_prop(
                    UI["cprop_input"].value[0], enforce_dict=True
                )
            UI["color_bar"].color_mapper.palette = make_color_seq_from_cmap(None)
        if not isinstance(value, dict):
            UI["color_bar"].color_mapper.palette = make_color_seq_from_cmap(
                cm.get_cmap(value)
            )
            UI["color_bar"].visible = True
        return UI["s"].put(
            f"http://{host2}:5000/tileserver/cmap", data={"cmap": json.dumps(value)}
        )
    return UI["s"].put(
        f"http://{host2}:5000/tileserver/renderer/{prop}",
        data={"val": json.dumps(value)},
    )


def build_predicate():
    """Builds a predicate function from the currently selected types,
    and the filter input.
    """
    preds = [
        f'props["type"]=={name2type(l.label)}'
        for l in UI["type_column"].children
        if l.active and l.label in UI["vstate"].types
    ]
    if len(preds) == len(UI["type_column"].children):
        preds = []
    combo = "None"
    if len(preds) > 0:
        combo = "(" + ") | (".join(preds) + ")"
    if UI["filter_input"].value not in ["None", ""]:
        if combo == "None":
            combo = UI["filter_input"].value
        else:
            combo = "(" + combo + ") & (" + UI["filter_input"].value + ")"

    print(combo)
    update_renderer("where", combo)
    return combo


def initialise_slide():
    UI["vstate"].mpp = UI["vstate"].wsi.info.mpp
    if UI["vstate"].mpp is None:
        UI["vstate"].mpp = [1, 1]
    UI["vstate"].dims = UI["vstate"].wsi.info.slide_dimensions
    slide_name = UI["vstate"].wsi.info.file_path.stem
    UI["vstate"].types = []
    UI["vstate"].props = []
    pad = int(np.mean(UI["vstate"].dims) / 10)
    plot_size = np.array([UI["p"].width, UI["p"].height])
    aspect_ratio = plot_size[0] / plot_size[1]
    large_dim = np.argmax(np.array(UI["vstate"].dims) / plot_size)

    UI["vstate"].micron_formatter.args["mpp"] = UI["vstate"].mpp[0]
    if slide_name in config["initial_views"]:
        lims = config["initial_views"][slide_name]
        UI["p"].x_range.start = lims[0]
        UI["p"].x_range.end = lims[2]
        UI["p"].y_range.start = -lims[3]
        UI["p"].y_range.end = -lims[1]
    # if two windows open and new slide is 'related' to the other, use the same view
    elif len(win_dicts) == 2 and (
        win_dicts[0]["vstate"].slide_path.stem in win_dicts[1]["vstate"].slide_path.stem
        or win_dicts[1]["vstate"].slide_path.stem
        in win_dicts[0]["vstate"].slide_path.stem
        or win_dicts[1]["vstate"].dims == win_dicts[0]["vstate"].dims
    ):
        # view should already be correct, pass
        pass
    else:
        if large_dim == 1:
            UI["p"].x_range.start = (
                -0.5 * (UI["vstate"].dims[1] * aspect_ratio - UI["vstate"].dims[0])
                - aspect_ratio * pad
            )
            UI["p"].x_range.end = (
                UI["vstate"].dims[1] * aspect_ratio
                - 0.5 * (UI["vstate"].dims[1] * aspect_ratio - UI["vstate"].dims[0])
                + aspect_ratio * pad
            )
            UI["p"].y_range.start = -UI["vstate"].dims[1] - pad
            UI["p"].y_range.end = pad
            # UI["p"].x_range.min_interval = ?
        else:
            UI["p"].x_range.start = -aspect_ratio * pad
            UI["p"].x_range.end = UI["vstate"].dims[0] + pad * aspect_ratio
            UI["p"].y_range.start = (
                -UI["vstate"].dims[0] / aspect_ratio
                + 0.5 * (UI["vstate"].dims[0] / aspect_ratio - UI["vstate"].dims[1])
                - pad
            )
            UI["p"].y_range.end = (
                0.5 * (UI["vstate"].dims[0] / aspect_ratio - UI["vstate"].dims[1]) + pad
            )
        print(
            f"ranges are x: {UI['p'].x_range.start} - {UI['p'].x_range.end}, y: {UI['p'].y_range.start} - {UI['p'].y_range.end}"
        )
    # UI["p"].x_range.bounds = (UI["p"].x_range.start - 2 * pad, UI["p"].x_range.end + 2 * pad)
    # UI["p"].y_range.bounds = (UI["p"].y_range.start - 2 * pad, UI["p"].y_range.end + 2 * pad)
    # UI["p"]._trigger_event()

    init_z = get_level_by_extent((0, UI["p"].y_range.start, UI["p"].x_range.end, 0))
    UI["vstate"].init_z = init_z
    print(f"initial_zoom: {init_z}")
    print(UI["vstate"].wsi.info.as_dict())


def initialise_overlay():
    UI["vstate"].colors = list(UI["vstate"].mapper.values())
    # UI["vstate"].types = [
    #     t for t in UI["vstate"].types
    # ]  # UI["vstate"].mapper.keys()]
    now_active = {b.label: b.active for b in UI["type_column"].children}
    print(UI["vstate"].types)
    print(now_active)
    for t in sorted(UI["vstate"].types):
        if str(t) not in now_active.keys():
            UI["type_column"].children.append(
                Toggle(
                    label=str(t),
                    active=True,
                    width=130,
                    height=30,
                    max_width=130,
                    sizing_mode="stretch_width",
                )
            )
            UI["type_column"].children[-1].on_click(layer_select_cb)
            try:
                UI["color_column"].children.append(
                    ColorPicker(
                        color=to_int_rgb(UI["vstate"].mapper[t][0:3]),
                        name=str(t),
                        width=60,
                        min_width=60,
                        max_width=70,
                        height=30,
                        sizing_mode="stretch_width",
                    )
                )
            except KeyError:
                UI["color_column"].children.append(
                    ColorPicker(
                        color=to_int_rgb(UI["vstate"].mapper[to_num(t)][0:3]),
                        name=str(t),
                        width=60,
                        height=30,
                        min_width=60,
                        max_width=70,
                        sizing_mode="stretch_width",
                    )
                )
            UI["color_column"].children[-1].on_change(
                "color", bind_cb_obj(UI["color_column"].children[-1], color_input_cb)
            )

    for b in UI["type_column"].children.copy():
        if (
            b.label not in UI["vstate"].types
            and b.label not in UI["vstate"].layer_dict.keys()
        ):
            print(f"removing {b.label}")
            UI["type_column"].children.remove(b)
    for c in UI["color_column"].children.copy():
        if c.name not in UI["vstate"].types and "slider" not in c.name:
            UI["color_column"].children.remove(c)

    build_predicate()


def add_layer(lname):
    UI["type_column"].children.append(
        Toggle(
            label=lname,
            active=True,
            width=130,
            height=40,
            max_width=130,
            sizing_mode="stretch_width",
        )
    )
    if lname == "nodes":
        UI["type_column"].children[-1].active = (
            UI["p"].renderers[UI["vstate"].layer_dict[lname]].glyph.line_alpha > 0
        )
    if lname == "edges":
        UI["type_column"].children[-1].active = (
            UI["p"].renderers[UI["vstate"].layer_dict[lname]].visible
        )
    UI["type_column"].children[-1].on_click(
        bind_cb_obj_tog(UI["type_column"].children[-1], fixed_layer_select_cb)
    )
    UI["color_column"].children.append(
        Slider(
            start=0,
            end=1,
            value=0.5,
            step=0.01,
            title=lname,
            height=40,
            width=100,
            max_width=90,
            sizing_mode="stretch_width",
            name=f"{lname}_slider",
        )
    )
    UI["color_column"].children[-1].on_change(
        "value", bind_cb_obj(UI["color_column"].children[-1], layer_slider_cb)
    )

    # layer_boxes=[Toggle(label=t, active=a, width=100) for t,a in now_active.items()]
    # lcolors=[ColorPicker(color=col[0:3], name=t, width=60) for col, t in zip(UI["vstate"].colors, UI["vstate"].types)]


class TileGroup:
    def __init__(self):
        self.group = 1

    def get_grp(self):
        self.group = self.group + 1
        return self.group


class ColorCycler:
    def __init__(self, colors=None):
        if colors is None:
            colors = ["red", "blue", "lime", "yellow", "cyan", "magenta", "orange"]
        self.colors = colors
        self.index = -1

    def get_next(self):
        self.index = (self.index + 1) % len(self.colors)
        return self.colors[self.index]

    def get_random(self):
        return str(np.random.choice(self.colors))

    def generate_random(self):
        return rgb2hex(np.random.choice(256, 3) / 255)


def change_tiles(layer_name="overlay"):
    grp = tg.get_grp()

    if layer_name == "graph" and layer_name not in UI["vstate"].layer_dict.keys():
        # for layer_key in UI["vstate"].layer_dict.keys():
        #     if layer_key in ["rect", "pts", "nodes", "edges"]:
        #         continue
        #     grp = tg.get_grp()
        #     ts = make_ts(
        #         f"http://{host}:{port}/tileserver/layer/{layer_key}/{UI['user']}/zoomify/TileGroup{grp}"
        #         + r"/{z}-{x}-{y}@{UI["vstate"].res}x.jpg",
        #     )
        #     UI["p"].renderers[UI["vstate"].layer_dict[layer_key]].tile_source = ts
        return

    ts = make_ts(
        f"http://{host}:{port}/tileserver/layer/{layer_name}/{UI['user']}/zoomify/TileGroup{grp}"
        + r"/{z}-{x}-{y}"
        + f"@{UI['vstate'].res}x.jpg",
        UI["vstate"].num_zoom_levels,
    )
    if layer_name in UI["vstate"].layer_dict:
        UI["p"].renderers[UI["vstate"].layer_dict[layer_name]].tile_source = ts
    else:
        UI["p"].add_tile(
            ts,
            smoothing=True,
            alpha=UI["overlay_alpha"].value,
            level="image",
            render_parents=False,
        )
        for layer_key in UI["vstate"].layer_dict.keys():
            if layer_key in ["rect", "pts", "nodes", "edges"]:
                continue
            grp = tg.get_grp()
            ts = make_ts(
                f"http://{host}:{port}/tileserver/layer/{layer_key}/{UI['user']}/zoomify/TileGroup{grp}"
                + r"/{z}-{x}-{y}"
                + f"@{UI['vstate'].res}x.jpg",
                UI["vstate"].num_zoom_levels,
            )
            UI["p"].renderers[UI["vstate"].layer_dict[layer_key]].tile_source = ts
        UI["vstate"].layer_dict[layer_name] = len(UI["p"].renderers) - 1

    print(UI["vstate"].layer_dict)
    print(UI["p"].renderers)


def run_app():
    app = TileServer(
        title="Testing TileServer",
        layers={
            # "slide": UI["vstate"].wsi,
        },
    )
    CORS(app, send_wildcard=True)
    app.run(host="127.0.0.1", threaded=False)


class ViewerState:
    def __init__(self, slide_path=None):
        if slide_path is not None:
            self.wsi = WSIReader.open(slide_path)
            self.slide_path = slide_path
            self.mpp = self.wsi.info.mpp
            if self.mpp is None:
                self.mpp = [1, 1]
            self.dims = self.wsi.info.slide_dimensions
            print(f"nzoom_levs: {self.num_zoom_levels}")
        else:
            self.dims = [30000, 20000]
            self.mpp = None
            self.slide_path = None
            self.num_zoom_levels = 0
        self.mapper = {}
        self.colors = list(self.mapper.values())
        self.cprop = None
        self.init_z = None
        self.types = list(self.mapper.keys())
        self.layer_dict = {"slide": 0, "rect": 1, "pts": 2}
        self.update_state = 0
        self.thickness = -1
        self.model_mpp = 0
        self.init = True
        self.micron_formatter = FuncTickFormatter(
            args={"mpp": 0.1},
            code="""
                return Math.round(tick*mpp)
                """,
        )
        self.current_model = "hovernet"
        self.props = []
        self.props_old = []
        self.to_update = set()
        self.graph = []
        self.res = 2

    def __setattr__(self, __name: str, __value) -> None:
        if __name == "types":
            self.__dict__["mapper"] = make_color_dict(__value)
            self.__dict__["colors"] = list(self.mapper.values())
            if self.cprop == "type":
                update_mapper()
            # we will standardise the types to strings, keep dict of originals
            self.__dict__["orig_types"] = {str(x): x for x in __value}
            __value = [str(x) for x in __value]

        if __name == "wsi":
            z = ZoomifyGenerator(__value, tile_size=256)
            self.__dict__["num_zoom_levels"] = z.level_count

        self.__dict__[__name] = __value


# endregion


# Define UI callbacks
# region
def res_switch_cb(attr, old, new):
    if new == 0:
        UI["vstate"].res = 1
    elif new == 1:
        UI["vstate"].res = 2
    else:
        raise ValueError("Invalid resolution")
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay", "slide"])


def slide_toggle_cb(attr):
    if UI["p"].renderers[0].alpha == 0:
        UI["p"].renderers[0].alpha = UI["slide_alpha"].value
    else:
        UI["p"].renderers[0].alpha = 0.0


def node_select_cb(attr, old, new):
    # do something on node select if desired
    pass


def overlay_toggle_cb(attr):
    for i in range(5, len(UI["p"].renderers)):
        if isinstance(UI["p"].renderers[i], GraphRenderer):
            # set_graph_alpha(UI["p"].renderers[i], new)
            continue
        if UI["p"].renderers[i].alpha == 0:
            UI["p"].renderers[i].alpha = UI["overlay_alpha"].value
        else:
            UI["p"].renderers[i].alpha = 0.0


def folder_input_cb(attr, old, new):
    populate_slide_list(config["slide_folder"], new)


def populate_layer_list(slide_name, overlay_path: Path):
    if overlay_path is None:
        return
    file_list = []
    for ext in [
        "*.db",
        "*.dat",
        "*.geojson",
        "*.png",
        "*.jpg",
        "*.pkl",
        "*.tiff",
    ]:  # and '*.tiff'?
        file_list.extend(list(overlay_path.glob(str(Path("*") / ext))))
        file_list.extend(list(overlay_path.glob(ext)))
    file_list = [(str(p), str(p)) for p in sorted(file_list) if slide_name in str(p)]
    UI["layer_drop"].menu = file_list


def populate_slide_list(slide_folder, search_txt=None):
    file_list = []
    len_slidepath = len(slide_folder.parts)
    for ext in ["*.svs", "*ndpi", "*.tiff", "*.mrxs", "*.jpg", "*.png"]:
        file_list.extend(list(Path(slide_folder).glob(str(Path("*") / ext))))
        file_list.extend(list(Path(slide_folder).glob(ext)))
    if search_txt is None:
        file_list = [
            (str(Path(*p.parts[len_slidepath:])), str(Path(*p.parts[len_slidepath:])))
            for p in sorted(file_list)
        ]
    else:
        file_list = [
            (str(Path(*p.parts[len_slidepath:])), str(Path(*p.parts[len_slidepath:])))
            for p in sorted(file_list)
            if search_txt in str(p)
        ]

    UI["slide_select"].options = file_list


def layer_folder_input_cb(attr, old, new):
    # unused at the moment
    file_list = []
    for ext in ["*.db", "*.dat", "*.geojson", "*.png", "*.jpg", ".tiff"]:
        file_list.extend(list(Path(new).glob("*\\" + ext)))
    file_list = [(str(p), str(p)) for p in sorted(file_list)]
    UI["layer_drop"].menu = file_list
    return file_list


def filter_input_cb(attr, old, new):
    """Change predicate to be used to filter annotations"""
    build_predicate()
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def cprop_input_cb(attr, old, new):
    """Change property to colour by"""
    if len(new) == 0:
        return
    if new[0] == "type":
        cmap = UI["vstate"].mapper
    else:
        cmap = get_mapper_for_prop(new[0])
    UI["vstate"].cprop = new[0]
    update_renderer("mapper", cmap)
    UI["s"].put(
        f"http://{host2}:5000/tileserver/color_prop", data={"prop": json.dumps(new[0])}
    )
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def cmap_builder_cb(attr, old, new):
    """add a property to the colormap and make a ColorPicker for it,
    then add it to the UI["cmap_picker_column"]. if new < old, remove the
    ColorPicker wit the deselected property from the cmap_picker_column"""
    if len(new) > len(old):
        new_prop = set(new).difference(set(old))
        new_prop = new_prop.pop()

        color_picker = ColorPicker(
            title=new_prop,
            color=color_cycler.get_next(),
            width=100,
            height=50,
        )
        color_picker.on_change("color", cmap_picker_cb)
        UI["cmap_picker_column"].children.append(color_picker)
    else:
        old_prop = set(old).difference(set(new))
        old_prop = old_prop.pop()
        for i, cp in enumerate(UI["cmap_picker_column"].children):
            if cp.title == old_prop:
                UI["cmap_picker_column"].children.pop(i)
                break


def cmap_picker_cb(attr, old, new):
    """update the colormap with the new color"""
    pass  # anything needed here?


def set_graph_alpha(g_renderer, value):
    # set all components of graph to given alpha value
    g_renderer.node_renderer.glyph.fill_alpha = value
    g_renderer.node_renderer.glyph.line_alpha = value
    g_renderer.edge_renderer.glyph.line_alpha = value


def slide_alpha_cb(attr, old, new):
    UI["p"].renderers[0].alpha = new
    # UI["p"].renderers[0].tile_source.max_zoom=7
    # UI["p"].renderers[1].tile_source.max_zoom=7


def overlay_alpha_cb(attr, old, new):
    for i in range(5, len(UI["p"].renderers)):
        if isinstance(UI["p"].renderers[i], GraphRenderer):
            # set_graph_alpha(UI["p"].renderers[i], new)
            pass
        else:
            UI["p"].renderers[i].alpha = new


def pt_size_cb(attr, old, new):
    UI["vstate"].graph_node.size = 2 * new


def edge_size_cb(attr, old, new):
    update_renderer("edge_thickness", new)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def opt_buttons_cb(attr, old, new):
    old_thickness = UI["vstate"].thickness
    if 0 in new:
        UI["vstate"].thickness = -1
        update_renderer("thickness", -1)
    else:
        UI["vstate"].thickness = 1
        update_renderer("thickness", 1)
    if old_thickness != UI["vstate"].thickness:
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])
    if 1 in new:
        UI["p"].xaxis[0].formatter = UI["vstate"].micron_formatter
        UI["p"].yaxis[0].formatter = UI["vstate"].micron_formatter
    else:
        UI["p"].xaxis[0].formatter = BasicTickFormatter()
        UI["p"].yaxis[0].formatter = BasicTickFormatter()
    if 2 in new:
        UI["p"].ygrid.grid_line_color = "gray"
        UI["p"].xgrid.grid_line_color = "gray"
        UI["p"].ygrid.grid_line_alpha = 0.6
        UI["p"].xgrid.grid_line_alpha = 0.6
    else:
        UI["p"].ygrid.grid_line_alpha = 0
        UI["p"].xgrid.grid_line_alpha = 0
    print(UI["p"].ygrid)
    print(UI["p"].grid)


def cmap_select_cb(attr, old, new):
    update_renderer("mapper", new)
    # change_tiles('overlay')
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def blur_spinner_cb(attr, old, new):
    update_renderer("blur_radius", new)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def scale_spinner_cb(attr, old, new):
    update_renderer("max_scale", new)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def slide_select_cb(attr, old, new):
    """setup the newly chosen slide"""
    if len(new) == 0:
        return
    slide_path = Path(config["slide_folder"]) / Path(new[0])
    # reset the data sources for glyph overlays
    UI["pt_source"].data = {"x": [], "y": []}
    UI["box_source"].data = {"x": [], "y": [], "width": [], "height": []}
    UI["node_source"].data = {"x_": [], "y_": [], "node_color_": []}
    UI["edge_source"].data = {"x0_": [], "y0_": [], "x1_": [], "y1_": []}
    UI["hover"].tooltips = None
    if len(UI["p"].renderers) > 5:
        for r in UI["p"].renderers[5:].copy():
            UI["p"].renderers.remove(r)
    UI["vstate"].layer_dict = {"slide": 0, "rect": 1, "pts": 2, "nodes": 3, "edges": 4}
    UI["vstate"].slide_path = slide_path
    """
    for c in UI["color_column"].children.copy():
        if "_slider" in c.name:
            UI["color_column"].children.remove(c)
    for b in UI["type_column"].children.copy():
        if "layer" in b.label or "graph" in b.label:
            UI["type_column"].children.remove(b)
    """
    UI["color_column"].children = []
    UI["type_column"].children = []
    print(UI["p"].renderers)
    print(slide_path)
    populate_layer_list(slide_path.stem, config["overlay_folder"])
    UI["vstate"].wsi = WSIReader.open(slide_path)
    initialise_slide()
    # fname='-*-'.join(attr.item.split('\\'))
    fname = make_safe_name(str(slide_path))
    print(fname)
    print(UI["vstate"].mpp)
    UI["s"].put(f"http://{host2}:5000/tileserver/slide", data={"slide_path": fname})
    change_tiles("slide")
    # if len(UI["p"].renderers)==1:
    # r=UI["p"].rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
    # UI["p"].add_tools(BoxEditTool(renderers=[r], num_objects=1))
    # UI["p"].x_range.bounds=MinMaxBounds(0,UI["vstate"].dims[0])
    # UI["p"].y_range.bounds=(0,-UI["vstate"].dims[1])

    # load the overlay and graph automatically for demo purposes
    if config["auto_load"]:
        for f in UI["layer_drop"].menu:
            dummy_attr = DummyAttr(f[0])
            layer_drop_cb(dummy_attr)


def layer_drop_cb(attr):
    """setup the newly chosen overlay"""
    if Path(attr.item).suffix == ".pkl":
        # its a graph
        do_feats = False
        with open(attr.item, "rb") as f:
            graph_dict = pickle.load(f)
        node_cm = cm.get_cmap("viridis")
        num_nodes = graph_dict["coordinates"].shape[0]
        if "score" in graph_dict:
            UI["node_source"].data = {
                "x_": graph_dict["coordinates"][:, 0],
                "y_": -graph_dict["coordinates"][:, 1],
                "node_color_": [
                    rgb2hex(node_cm(to_num(v))) for v in graph_dict["score"]
                ],
            }
        else:
            # default to green
            UI["node_source"].data = {
                "x_": graph_dict["coordinates"][:, 0],
                "y_": -graph_dict["coordinates"][:, 1],
                "node_color_": [rgb2hex((0, 1, 0))] * num_nodes,
            }
        UI["edge_source"].data = {
            "x0_": [
                graph_dict["coordinates"][i, 0] for i in graph_dict["edge_index"][0, :]
            ],
            "y0_": [
                -graph_dict["coordinates"][i, 1] for i in graph_dict["edge_index"][0, :]
            ],
            "x1_": [
                graph_dict["coordinates"][i, 0] for i in graph_dict["edge_index"][1, :]
            ],
            "y1_": [
                -graph_dict["coordinates"][i, 1] for i in graph_dict["edge_index"][1, :]
            ],
        }
        # UI["edge_source"].data = {
        #     "xs": [[graph_dict["coordinates"][inds[0], 0], graph_dict["coordinates"][inds[1], 0]] for inds in graph_dict["edge_index"].T],
        #     "ys": [[-graph_dict["coordinates"][inds[0], 1], -graph_dict["coordinates"][inds[1], 1]] for inds in graph_dict["edge_index"].T],
        # }

        # graph_layout = dict(
        #     zip(
        #         UI["node_source"].data["index"],
        #         [
        #             # (x / (4 * UI["vstate"].mpp[0]), -y / (4 * UI["vstate"].mpp[1]))
        #             (x, -y)
        #             for x, y in graph_dict["coordinates"]
        #         ],
        #     )
        # )
        add_layer("edges")
        add_layer("nodes")
        change_tiles("graph")
        if "graph_overlay" not in UI["type_cmap_select"].options:
            UI["type_cmap_select"].options = UI["type_cmap_select"].options + [
                "graph_overlay"
            ]

        # add additional data to graph datasource
        for key in graph_dict:
            if key == "feat_names":
                graph_feat_names = graph_dict[key]
                do_feats = True
            try:
                if (
                    key in ["edge_index", "coordinates"]
                    or len(graph_dict[key]) != num_nodes
                ):
                    continue
            except TypeError:
                continue  # not arraylike, cant add to node data
            UI["node_source"].data[key] = graph_dict[key]

        if do_feats:
            for i in range(graph_dict["feats"].shape[1]):
                if i > 9:
                    break  # more than 10 wont really fit, ignore rest
                UI["node_source"].data[graph_feat_names[i]] = graph_dict["feats"][:, i]

            TOOLTIPS = [
                ("Index", "$index"),
                ("(x,y)", "($x, $y)"),
            ]
            TOOLTIPS.extend(
                [
                    (graph_feat_names[i], f"@{graph_feat_names[i]}")
                    for i in range(np.minimum(graph_dict["feats"].shape[1], 9))
                ]
            )
            UI["hover"].tooltips = TOOLTIPS

        return

    # fname='-*-'.join(attr.item.split('\\'))
    fname = make_safe_name(attr.item)
    resp = UI["s"].put(
        f"http://{host2}:5000/tileserver/overlay", data={"overlay_path": fname}
    )
    resp = json.loads(resp.text)

    if Path(attr.item).suffix in [".db", ".dat", ".geojson"]:
        UI["vstate"].types = resp
        props = UI["s"].get(f"http://{host2}:5000/tileserver/prop_names/all")
        UI["vstate"].props = json.loads(props.text)
        # update the color type by prop menu
        UI["type_cmap_select"].options = [t for t in UI["vstate"].types]
        if len(UI["node_source"].data["x_"]) > 0:
            UI["type_cmap_select"].options.append("graph_overlay")
        # UI["type_cmap_select"].options = UI["vstate"].props
        # update the color type by prop menu
        UI["type_cmap_select"].options = [t for t in UI["vstate"].types]
        if len(UI["node_source"].data["x_"]) > 0:
            UI["type_cmap_select"].options.append("graph_overlay")
        UI["cprop_input"].options = UI["vstate"].props
        # subcat_select.options = ["All"] + list(
        #     set([UI["p"].split("_")[0] for p in UI["vstate"].props])
        # )
        UI["cprop_input"].options.append("None")
        UI["cmap_builder_input"].options = UI["vstate"].props
        if not UI["vstate"].props == UI["vstate"].props_old:
            # if color by prop no longer exists, reset to type
            if (
                len(UI["cprop_input"].value) == 0
                or UI["cprop_input"].value[0] not in UI["vstate"].props
            ):
                UI["cprop_input"].value = ["type"]
                update_mapper()
            UI["vstate"].props_old = UI["vstate"].props
        initialise_overlay()
        change_tiles("overlay")
    else:
        add_layer(resp)
        change_tiles(resp)


def layer_select_cb(attr):
    build_predicate()
    # change_tiles('overlay')
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def fixed_layer_select_cb(obj, attr):
    print(UI["vstate"].layer_dict)
    key = UI["vstate"].layer_dict[obj.label]
    if obj.label == "edges":
        if not UI["p"].renderers[key].visible:  # line_alpha == 0:
            UI["p"].renderers[
                key
            ].visible = True  # line_alpha = UI["overlay_alpha"].value
            # UI["p"].renderers[key].node_renderer.glyph.line_alpha = UI["overlay_alpha"].value
            # UI["p"].renderers[key].edge_renderer.glyph.line_alpha = UI["overlay_alpha"].value
        else:
            UI["p"].renderers[key].visible = False  # line_alpha = 0.0
            # UI["p"].renderers[key].node_renderer.glyph.line_alpha = 0.0
            # UI["p"].renderers[key].edge_renderer.glyph.line_alpha = 0.0
    elif obj.label == "nodes":
        if UI["p"].renderers[key].glyph.fill_alpha == 0:
            UI["p"].renderers[key].glyph.fill_alpha = UI["overlay_alpha"].value
            UI["p"].renderers[key].glyph.line_alpha = UI["overlay_alpha"].value
            # UI["p"].renderers[key].edge_renderer.glyph.line_alpha = UI["overlay_alpha"].value
        else:
            UI["p"].renderers[key].glyph.fill_alpha = 0.0
            UI["p"].renderers[key].glyph.line_alpha = 0.0
    else:
        if UI["p"].renderers[key].alpha == 0:
            UI["p"].renderers[key].alpha = float(obj.name)
        else:
            obj.name = str(UI["p"].renderers[key].alpha)  # save old alpha
            UI["p"].renderers[key].alpha = 0.0


def layer_slider_cb(obj, attr, old, new):
    if obj.name.split("_")[0] == "nodes":
        set_alpha_glyph(
            UI["p"].renderers[UI["vstate"].layer_dict[obj.name.split("_")[0]]].glyph,
            new,
        )
    elif obj.name.split("_")[0] == "edges":
        UI["p"].renderers[
            UI["vstate"].layer_dict[obj.name.split("_")[0]]
        ].glyph.line_alpha = new
    else:
        UI["p"].renderers[UI["vstate"].layer_dict[obj.name.split("_")[0]]].alpha = new


def color_input_cb(obj, attr, old, new):
    print(new)
    UI["vstate"].mapper[UI["vstate"].orig_types[obj.name]] = (*hex2rgb(new), 1)
    if UI["vstate"].cprop == "type":
        update_renderer("mapper", UI["vstate"].mapper)
    # change_tiles('overlay')
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def bind_cb_obj(cb_obj, cb):
    def wrapped(attr, old, new):
        cb(cb_obj, attr, old, new)

    return wrapped


def bind_cb_obj_tog(cb_obj, cb):
    def wrapped(attr):
        cb(cb_obj, attr)

    return wrapped


def swap_cb(attr):
    val = UI["type_cmap_select"].value
    if len(val) == 0:
        return
    if "_exp" in val[0]:
        UI["type_cmap_select"].value = [val[0][:-4]]
    else:
        UI["type_cmap_select"].value = [val[0] + "_exp"]
    type_cmap_cb(None, None, UI["type_cmap_select"].value)


def model_drop_cb(attr):
    UI["vstate"].current_model = attr.item


def to_model_cb(attr):
    if UI["vstate"].current_model == "hovernet":
        segment_on_box(attr)
    elif UI["vstate"].current_model == "nuclick":
        nuclick_on_pts(attr)
    else:
        print("unknown model")


def type_cmap_cb(attr, old, new):
    if len(new) == 0:
        UI["type_cmap_select"].options = UI["vstate"].types + ["graph_overlay"]
        UI["s"].put(
            f"http://{host2}:5000/tileserver/secondary_cmap",
            data={
                "type_id": json.dumps("None"),
                "prop": "None",
                "cmap": json.dumps("viridis"),
            },
        )
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])
        return
    if len(new) == 1:
        # find out what still has to be selected
        if new[0] in UI["vstate"].types + ["graph_overlay"]:
            if new[0] == "graph_overlay":
                UI["type_cmap_select"].options = [
                    key
                    for key in UI["node_source"].data.keys()
                    if key not in ["x_", "y_", "node_color_"]
                ] + [new[0]]
            else:
                UI["type_cmap_select"].options = UI["vstate"].props + [new[0]]
        elif new[0] in UI["vstate"].props:
            UI["type_cmap_select"].options = (
                UI["vstate"].types + [new[0]] + ["graph_overlay"]
            )
    else:
        # both selected, update the renderer
        if new[1] in UI["vstate"].types:
            # make sure the type is the first one
            UI["type_cmap_select"].value = [new[1], new[0]]
            return
        if new[0] == "graph_overlay":
            # adjust the node color in source if prop exists
            if new[1] in UI["node_source"].data:
                node_cm = cm.get_cmap("viridis")
                UI["node_source"].data["node_color_"] = [
                    rgb2hex(node_cm(to_num(v))) for v in UI["node_source"].data[new[1]]
                ]
            return
        cmap = get_mapper_for_prop(new[1])  # separate cmap select ?
        UI["s"].put(
            f"http://{host2}:5000/tileserver/secondary_cmap",
            data={
                "type_id": json.dumps(UI["vstate"].orig_types[new[0]]),
                "prop": new[1],
                "cmap": json.dumps(cmap),
            },
        )

        UI["color_bar"].color_mapper.palette = make_color_seq_from_cmap(
            cm.get_cmap("viridis")
        )
        UI["color_bar"].visible = True
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])


def save_cb(attr):
    if config["overlay_folder"] is None:
        # save in slide folder instead
        save_path = make_safe_name(
            str(
                config["slide_folder"]
                / (UI["vstate"].slide_path.stem + "_saved_anns.db")
            )
        )
    else:
        save_path = make_safe_name(
            str(
                config["overlay_folder"]
                / (UI["vstate"].slide_path.stem + "_saved_anns.db")
            )
        )
    UI["s"].post(
        f"http://{host2}:5000/tileserver/commit", data={"save_path": save_path}
    )


def subcat_select_cb(attr, old, new):
    if new == "All":
        UI["cmap_builder_input"].options = UI["vstate"].props
        return
    new_opts = [p for p in UI["vstate"].props if new in p]
    UI["cmap_builder_input"].options = new_opts + [
        a for a in UI["cmap_builder_input"].value if a not in new_opts
    ]


# run NucleusInstanceSegmentor on a region of wsi defined by the box in box_source
def segment_on_box(attr):
    print(UI["vstate"].types)
    # thumb=UI["vstate"].wsi.slide_thumbnail(resolution=8, units='mpp')
    thumb = UI["vstate"].wsi.slide_thumbnail()
    # conv_mpp=wsi.convert_resolution_units(1.25, 'power', 'mpp')[0]
    conv_mpp = UI["vstate"].dims[0] / thumb.shape[1]
    print(f'box tl: {UI["box_source"].data["x"][0]}, {UI["box_source"].data["y"][0]}')
    x = round(
        (UI["box_source"].data["x"][0] - 0.5 * UI["box_source"].data["width"][0])
        / conv_mpp
    )
    y = -round(
        (UI["box_source"].data["y"][0] + 0.5 * UI["box_source"].data["height"][0])
        / conv_mpp
    )
    width = round(UI["box_source"].data["width"][0] / conv_mpp)
    height = round(UI["box_source"].data["height"][0] / conv_mpp)
    print(x, y, width, height)

    # img_tile=wsi.read_rect((x,y),(width,height))
    mask = np.zeros((thumb.shape[0], thumb.shape[1]))
    mask[y : y + height, x : x + width] = 1

    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=4,
        num_postproc_workers=8,
        batch_size=24,
    )

    UI["vstate"].model_mpp = inst_segmentor.ioconfig.save_resolution["resolution"]
    tile_output = inst_segmentor.predict(
        [UI["vstate"].slide_path],
        [mask],
        save_dir="sample_tile_results/",
        mode="wsi",
        # resolution=UI["vstate"].mpp,
        # units='mpp',
        on_gpu=torch.cuda.is_available(),
        crash_on_exception=True,
    )

    # fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
    fname = make_safe_name(".\\sample_tile_results\\0.dat")
    print(fname)
    resp = UI["s"].put(
        f"http://{host2}:5000/tileserver/annotations",
        data={"file_path": fname, "model_mpp": json.dumps(UI["vstate"].model_mpp)},
    )
    UI["vstate"].types = json.loads(resp.text)

    # update the props options if needed
    props = UI["s"].get(f"http://{host2}:5000/tileserver/prop_names/all")
    UI["vstate"].props = json.loads(props.text)
    # UI["type_cmap_select"].options = UI["vstate"].props
    UI["cprop_input"].options = UI["vstate"].props
    # subcat_select.options = ["All"] + list(set([UI["p"].split("_")[0] for p in UI["vstate"].props]))
    UI["cmap_builder_input"].options = UI["vstate"].props
    if not UI["vstate"].props == UI["vstate"].props_old:
        update_mapper()
        UI["vstate"].props_old = UI["vstate"].props

    # update_mapper()
    # type_drop.menu=[(str(t),str(t)) for t in UI["vstate"].types]
    rmtree(r"./sample_tile_results")
    initialise_overlay()
    change_tiles("overlay")

    return tile_output


# run nuclick on user selected points in pt_source
# def nuclick_on_pts(attr):
#     x = np.round(np.array(UI["pt_source"].data["x"]))
#     y = -np.round(np.array(UI["pt_source"].data["y"]))

#     model = NuClick(5, 1)
#     # pretrained_weights = r"C:\Users\meast\app_data\NuClick_Nuclick_40xAll.pth"
#     fetch_pretrained_weights(
#         "nuclick_original-pannuke", r"./nuclick_weights.pth", overwrite=False
#     )
#     saved_state_dict = torch.load(r"./nuclick_weights.pth", map_location="cpu")
#     model.load_state_dict(saved_state_dict, strict=True)
#     UI["vstate"].model_mpp = 0.25
#     ioconf = IOInteractiveSegmentorConfig(
#         input_resolutions=[{"resolution": 0.25, "units": "mpp"}], patch_size=(128, 128)
#     )
#     inst_segmentor = InteractiveSegmentor(
#         num_loader_workers=0,
#         batch_size=16,
#         model=model,
#     )

#     points = np.vstack([x, y]).T
#     points = points / (ioconf.input_resolutions[0]["resolution"] / UI["vstate"].mpp[0])
#     print(points.shape)
#     nuclick_output = inst_segmentor.predict(
#         [UI["vstate"].slide_path],
#         [points],
#         ioconfig=ioconf,
#         save_dir="/app_data/sample_tile_results/",
#         patch_size=(128, 128),
#         resolution=0.25,
#         units="mpp",
#         on_gpu=False,
#         save_output=True,
#     )

#     # fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
#     fname = make_safe_name("\\app_data\\sample_tile_results\\0.dat")
#     print(fname)
#     resp = UI["s"].put(
#         f"http://{host2}:5000/tileserver/load_annotations/{fname}/{UI['vstate'].model_mpp}"
#     )
#     print(resp.text)
#     UI["vstate"].types = json.loads(resp.text)
#     update_mapper()
#     rmtree(Path(r"/app_data/sample_tile_results"))
#     initialise_overlay()
#     change_tiles("overlay")


# endregion

# set up main window
slide_wins = row(
    children=[],
    name="slide_windows",
    sizing_mode="stretch_both",
)
# and the controls
control_tabs = Tabs(tabs=[], name="ui_layout")


def make_window(vstate):
    win_num = str(len(windows))
    if len(windows) == 1:
        slide_wins.children[0].width = 800
        p = figure(
            x_range=slide_wins.children[0].x_range,
            y_range=slide_wins.children[0].y_range,
            x_axis_type="linear",
            y_axis_type="linear",
            width=800,
            height=1000,
            # max_width=1700,
            # max_height=1000,
            # width_policy="max",
            # height_policy="max",
            # tooltips=TOOLTIPS,
            tools=tool_str,
            active_scroll="wheel_zoom",
            output_backend="webgl",
            hidpi=True,
            match_aspect=False,
            lod_factor=200000,
            # lod_interval=500,
            # lod_threshold=10,
            # lod_timeout=200,
            sizing_mode="stretch_both",
            name=f"slide_window{win_num}",
        )
        init_z = first_z[0]
    else:
        p = figure(
            x_range=(0, vstate.dims[0]),
            y_range=(0, vstate.dims[1]),
            x_axis_type="linear",
            y_axis_type="linear",
            width=1700,
            height=1000,
            # max_width=1700,
            # max_height=1000,
            # width_policy="max",
            # height_policy="max",
            # tooltips=TOOLTIPS,
            tools=tool_str,
            active_scroll="wheel_zoom",
            output_backend="webgl",
            hidpi=True,
            match_aspect=False,
            lod_factor=200000,
            # lod_interval=500,
            # lod_threshold=10,
            # lod_timeout=200,
            sizing_mode="stretch_both",
            name=f"slide_window{win_num}",
        )
        init_z = get_level_by_extent((0, p.y_range.start, p.x_range.end, 0))
        first_z[0] = init_z
    p.axis.visible = False
    # initialise_slide()
    p.toolbar.tools[1].zoom_on_axis = False

    s = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=0.1,
    )
    # status_forcelist=[ 500, 502, 503, 504 ])
    s.mount("http://", HTTPAdapter(max_retries=retries))

    resp = s.get(f"http://{host2}:5000/tileserver/session_id")
    print(f"cookies are: {resp.cookies}")
    cookies = resp.cookies
    user = resp.cookies.get("session_id")
    if curdoc().session_context:
        curdoc().session_context.request.arguments["user"] = user
    vstate.init_z = init_z
    ts1 = make_ts(
        f"http://{host}:{port}/tileserver/layer/slide/{user}/zoomify/TileGroup1"
        + r"/{z}-{x}-{y}"
        + f"@{vstate.res}x.jpg",
        vstate.num_zoom_levels,
        # 4,
    )
    print(p.renderers)
    print(p.y_range)
    p.add_tile(ts1, smoothing=True, level="image", render_parents=True)
    print(p.y_range)
    print(f"max zoom is: {p.renderers[0].tile_source.max_zoom}")

    p.grid.grid_line_color = None
    box_source = ColumnDataSource({"x": [], "y": [], "width": [], "height": []})
    pt_source = ColumnDataSource({"x": [], "y": []})
    r = p.rect("x", "y", "width", "height", source=box_source, fill_alpha=0)
    c = p.circle("x", "y", source=pt_source, color="red", size=5)
    p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
    p.add_tools(PointDrawTool(renderers=[c]))
    p.add_tools(TapTool())
    if get_from_config(["opts", "hover_on"], 0) == 0:
        p.toolbar.active_inspect = None

    p.renderers[0].tile_source.max_zoom = 10

    node_source = ColumnDataSource({"x_": [], "y_": [], "node_color_": []})
    edge_source = ColumnDataSource({"x0_": [], "y0_": [], "x1_": [], "y1_": []})
    vstate.graph_node = Circle(x="x_", y="y_", fill_color="node_color_", size=5)
    vstate.graph_edge = Segment(x0="x0_", y0="y0_", x1="x1_", y1="y1_")
    p.add_glyph(node_source, vstate.graph_node)
    if not get_from_config(["opts", "nodes_on"], True):
        p.renderers[-1].glyph.fill_alpha = 0
        p.renderers[-1].glyph.line_alpha = 0
    p.add_glyph(edge_source, vstate.graph_edge)
    if not get_from_config(["opts", "edges_on"], False):
        p.renderers[-1].visible = False
    vstate.layer_dict["nodes"] = len(p.renderers) - 2
    vstate.layer_dict["edges"] = len(p.renderers) - 1
    hover = HoverTool(renderers=[p.renderers[-2]])
    p.add_tools(hover)

    color_bar = ColorBar(
        color_mapper=LinearColorMapper(
            make_color_seq_from_cmap(cm.get_cmap("viridis"))
        ),
        label_standoff=12,
    )
    if get_from_config(["opts", "colorbar_on"], 1) == 1:
        p.add_layout(color_bar, "below")

    # Define UI elements
    res_switch = RadioButtonGroup(labels=["1x", "2x"], active=1)

    slide_alpha = Slider(
        title="Slide Alpha",
        start=0,
        end=1,
        step=0.05,
        value=1.0,
        width=200,
        # max_width=200,
        sizing_mode="stretch_width",
        name=f"slide_alpha{win_num}",
    )

    overlay_alpha = Slider(
        title="Overlay Alpha",
        start=0,
        end=1,
        step=0.05,
        value=0.75,
        width=200,
        # max_width=200,
        sizing_mode="stretch_width",
        name=f"overlay_alpha{win_num}",
    )

    edge_size_spinner = Spinner(
        title="Edge thickness:",
        low=0,
        high=10,
        step=1,
        value=1,
        width=60,
        # max_width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"edge_size{win_num}",
    )

    pt_size_spinner = Spinner(
        title="Pt. Size:",
        low=0,
        high=10,
        step=1,
        value=4,
        width=60,
        # max_width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"pt_size{win_num}",
    )

    slide_toggle = Toggle(
        label="Slide",
        active=True,
        button_type="success",
        width=90,
        # max_width=90,
        sizing_mode="stretch_width",
        name=f"slide_toggle{win_num}",
    )
    overlay_toggle = Toggle(
        label="Overlay",
        active=True,
        button_type="success",
        width=90,
        # max_width=90,
        sizing_mode="stretch_width",
        name=f"overlay_toggle{win_num}",
    )
    filter_input = TextInput(
        value="None",
        title="Filter:",
        sizing_mode="stretch_width",
        name=f"filter{win_num}",
    )
    # cprop_input = TextInput(
    #    value="type", title="CProp:", max_width=300, sizing_mode="stretch_width"
    # )
    cprop_input = MultiChoice(
        title="Colour by:",
        max_items=1,
        options=[get_from_config(["default_cprop"], "type")],
        value=[get_from_config(["default_cprop"], "type")],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        # max_width=300,
        name=f"cprop{win_num}",
    )
    slide_select = MultiChoice(
        title="Select Slide:",
        max_items=1,
        options=[get_from_config(["first_slide"], "*")],
        value=[get_from_config(["first_slide"], "*")],
        search_option_limit=5000,
        # max_width=300,
        sizing_mode="stretch_width",
        name=f"slide_select{win_num}",
    )
    cmmenu = [
        ("jet", "jet"),
        ("coolwarm", "coolwarm"),
        ("viridis", "viridis"),
        ("dict", "dict"),
    ]
    cmap_select = Select(
        title="Cmap",
        options=cmmenu,
        width=60,
        value=get_from_config(["UI_settings", "mapper"], "coolwarm"),
        # max_width=60,
        height=45,
        sizing_mode="stretch_width",
        name=f"cmap{win_num}",
    )
    blur_spinner = Spinner(
        title="Blur:",
        low=0,
        high=20,
        step=1,
        value=get_from_config(["UI_settings", "blur_radius"], 0),
        width=60,
        height=50,
        # max_width=60,
        sizing_mode="stretch_width",
        name=f"blur{win_num}",
    )
    scale_spinner = Spinner(
        title="max scale:",
        low=0,
        high=540,
        step=8,
        value=get_from_config(["UI_settings", "max_scale"], 16),
        width=60,
        # max_width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"scale{win_num}",
    )
    to_model_button = Button(
        label="Go",
        button_type="success",
        width=60,
        max_width=60,
        sizing_mode="stretch_width",
        name=f"to_model{win_num}",
    )
    model_drop = Dropdown(
        label="Choose Model",
        button_type="warning",
        menu=["hovernet", "nuclick"],
        width=120,
        max_width=120,
        sizing_mode="stretch_width",
        name=f"model_drop{win_num}",
    )
    type_cmap_select = MultiChoice(
        title="Colour type by property:",
        max_items=2,
        options=["*"],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        # max_width=300,
        name=f"type_cmap{win_num}",
    )
    layer_boxes = [
        Toggle(
            label=t, active=True, width=100, max_width=100, sizing_mode="stretch_width"
        )
        for t in vstate.types
    ]
    lcolors = [
        ColorPicker(color=col[0:3], width=60, max_width=60, sizing_mode="stretch_width")
        for col in vstate.colors
    ]
    layer_folder_input = TextInput(
        value=str(config["overlay_folder"]),
        title="Overlay Folder:",
        # max_width=300,
        sizing_mode="stretch_width",
        name=f"layer_folder{win_num}",
    )
    layer_drop = Dropdown(
        label="Add Overlay",
        button_type="warning",
        menu=[None],
        # max_width=300,
        sizing_mode="stretch_width",
        name=f"layer_drop{win_num}",
    )
    opt_buttons = CheckboxButtonGroup(
        labels=["Filled", "Microns", "Grid"],
        active=[0],
        # max_width=300,
        sizing_mode="stretch_width",
        name=f"opt_buttons{win_num}",
    )
    save_button = Button(
        label="Save",
        button_type="success",
        max_width=90,
        sizing_mode="stretch_width",
        name=f"save_button{win_num}",
    )
    cmap_builder_input = MultiChoice(
        title="Choose props and colors:",
        max_items=10,
        options=["*"],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        # max_width=300,
        name=f"cmap_builder_input{win_num}",
    )
    cmap_builder_button = Button(
        label="Build Cmap",
        button_type="success",
        max_width=90,
        sizing_mode="stretch_width",
        name=f"cmap_builder_button{win_num}",
    )
    cmap_picker_column = column(children=[], name=f"cmap_picker_column{win_num}")
    subcat_select = Select(
        title="subcat", options=["All"], value="All", name=f"subcat_select{win_num}"
    )
    mixing_type_select = RadioButtonGroup(
        labels=["lin", "max", "avg", "prod", "pow", "softm"],
        active=4,
        name=f"mixing_type_select{win_num}",
    )
    add_postproc_button = Button(
        label="Add Postproc",
        button_type="success",
        max_width=90,
        sizing_mode="stretch_width",
        name=f"postproc_button{win_num}",
    )

    # associate callback functions to the widgets
    slide_alpha.on_change("value", slide_alpha_cb)
    overlay_alpha.on_change("value", overlay_alpha_cb)
    res_switch.on_change("active", res_switch_cb)
    pt_size_spinner.on_change("value", pt_size_cb)
    edge_size_spinner.on_change("value", edge_size_cb)
    slide_select.on_change("value", slide_select_cb)
    save_button.on_click(save_cb)
    cmap_select.on_change("value", cmap_select_cb)
    blur_spinner.on_change("value", blur_spinner_cb)
    scale_spinner.on_change("value", scale_spinner_cb)
    to_model_button.on_click(to_model_cb)
    model_drop.on_click(model_drop_cb)
    layer_drop.on_click(layer_drop_cb)
    opt_buttons.on_change("active", opt_buttons_cb)
    slide_toggle.on_click(slide_toggle_cb)
    overlay_toggle.on_click(overlay_toggle_cb)
    filter_input.on_change("value", filter_input_cb)
    cprop_input.on_change("value", cprop_input_cb)
    node_source.selected.on_change("indices", node_select_cb)
    type_cmap_select.on_change("value", type_cmap_cb)
    cmap_builder_input.on_change("value", cmap_builder_cb)
    subcat_select.on_change("value", subcat_select_cb)

    vstate.cprop = config["default_cprop"]

    type_column = column(children=layer_boxes, name=f"type_column{win_num}")
    color_column = column(
        children=lcolors, sizing_mode="stretch_width", name=f"color_column{win_num}"
    )

    # create the layout
    slide_row = row([slide_toggle, slide_alpha], sizing_mode="stretch_width")
    overlay_row = row([overlay_toggle, overlay_alpha], sizing_mode="stretch_width")
    cmap_row = row(
        [cmap_select, scale_spinner, blur_spinner],
        sizing_mode="stretch_width",
    )
    model_row = row(
        [to_model_button, model_drop, save_button], sizing_mode="stretch_width"
    )
    type_select_row = row(
        children=[type_column, color_column], sizing_mode="stretch_width"
    )

    # make element dictionaries
    UI_elements_1 = dict(
        zip(
            [
                "slide_select",
                "layer_drop",
                "slide_row",
                "overlay_row",
                "filter_input",
                "cprop_input",
                "cmap_row",
                "type_cmap_select",
                "model_row",
                "type_select_row",
            ],
            [
                slide_select,
                layer_drop,
                slide_row,
                overlay_row,
                filter_input,
                cprop_input,
                cmap_row,
                type_cmap_select,
                model_row,
                type_select_row,
            ],
        )
    )
    if "UI_elements_1" in config:
        ui_layout = column(
            [
                UI_elements_1[el]
                for el in config["UI_elements_1"]
                if config["UI_elements_1"][el] == 1
            ],
            sizing_mode="stretch_width",
        )
    else:
        ui_layout = column(
            list(UI_elements_1.values()),
            sizing_mode="stretch_width",
        )

    UI_elements_2 = dict(
        zip(
            [
                "opt_buttons",
                "pt_size_spinner",
                "edge_size_spinner",
                "res_switch",
                "mixing_type_select",
                "cmap_builder_input",
                "cmap_picker_column",
                "cmap_builder_button",
                "add_postproc_button",
            ],
            [
                opt_buttons,
                pt_size_spinner,
                edge_size_spinner,
                res_switch,
                mixing_type_select,
                cmap_builder_input,
                cmap_picker_column,
                cmap_builder_button,
                add_postproc_button,
            ],
        )
    )
    if "UI_elements_2" in config:
        extra_options = column(
            [
                UI_elements_2[el]
                for el in config["UI_elements_2"]
                if config["UI_elements_2"][el] == 1
            ],
        )
    else:
        extra_options = column(
            list(UI_elements_2.values()),
        )

    if len(windows) == 0:
        controls.append(
            TabPanel(
                child=Tabs(
                    tabs=[
                        TabPanel(child=ui_layout, title="Main"),
                        TabPanel(child=extra_options, title="More Opts"),
                    ],
                ),
                title="window 1",
            )
        )
        controls.append(TabPanel(child=Div(), title="window 2"))
        windows.append(p)
    else:
        control_tabs.tabs[1] = TabPanel(
            child=Tabs(
                tabs=[
                    TabPanel(child=ui_layout, title="Main"),
                    TabPanel(child=extra_options, title="More Opts"),
                ],
            ),
            title="window 2",
            closable=True,
        )
        slide_wins.children.append(p)

    return {
        **UI_elements_1,
        **UI_elements_2,
        "p": p,
        "vstate": vstate,
        "s": s,
        "box_source": box_source,
        "pt_source": pt_source,
        "node_source": node_source,
        "edge_source": edge_source,
        "hover": hover,
        "user": user,
        "color_column": color_column,
        "type_column": type_column,
        "overlay_alpha": overlay_alpha,
        "cmap_select": cmap_select,
        "color_bar": color_bar,
        "slide_alpha": slide_alpha,
    }


# main ui containers
UI = UIWrapper()
windows = []
controls = []
win_dicts = []

color_cycler = ColorCycler()
tg = TileGroup()
do_feats = False
tool_str = "pan,wheel_zoom,reset,save"

active = 0

# some setup
# config["base_folder"] = r"E:\TTB_vis_folder"

req_args = []
do_doc = False
if curdoc().session_context is not None:
    req_args = curdoc().session_context.request.arguments
    do_doc = True
print(f"req args are: {req_args}")

is_deployed = False
rand_id = token.generate_session_id()
print(f"rand id is: {rand_id}")
first_z = [1]

if is_deployed:
    host = os.environ.get("HOST")
    host2 = os.environ.get("HOST2")
    port = os.environ.get("PORT")
else:
    host = "127.0.0.1"
    host2 = "127.0.0.1"
    port = "5000"


def cleanup_session(session_context):
    # If present, this function executes when the server closes a session.
    sys.exit()


def update():
    if UI["vstate"].update_state == 2:
        for layer in UI["vstate"].to_update:
            if layer in UI["vstate"].layer_dict:
                change_tiles(layer)
        UI["vstate"].update_state = 0
        UI["vstate"].to_update = set()
    if UI["vstate"].update_state == 1:
        UI["vstate"].update_state = 2


def control_tabs_cb(attr, old, new):
    global active
    if new == 1 and len(slide_wins.children) == 1:
        # make new window
        win_dicts.append(make_window(ViewerState(win_dicts[0]["vstate"].slide_path)))
        win_dicts[1]["vstate"].thickness = win_dicts[0]["vstate"].thickness
        bounds = get_view_bounds(
            UI["vstate"].dims, np.array([UI["p"].width, UI["p"].height])
        )
        active = new
        if "UI_settings" in config:
            for k in config["UI_settings"]:
                update_renderer(k, config["UI_settings"][k])
            if config["default_cprop"] is not None:
                UI["s"].put(
                    f"http://{host2}:5000/tileserver/color_prop/",
                    data={"prop": json.dumps(config["default_cprop"])},
                )
        slide_select_cb(None, None, new=[UI["vstate"].slide_path])
        if "default_type_cprop" in config:
            UI["type_cmap_select"].value = list(config["default_type_cprop"].values())
        # initialise_slide()
        populate_slide_list(config["slide_folder"])
        populate_layer_list(
            Path(UI["vstate"].slide_path).stem, config["overlay_folder"]
        )
        win_dicts[0]["vstate"].init_z = get_level_by_extent(
            (0, bounds[2], bounds[1], 0)
        )
        UI["vstate"].init = False
    else:
        active = new


def control_tabs_remove_cb(attr, old, new):
    global active
    if len(new) == 1:
        # remove the second window
        slide_wins.children.pop()
        slide_wins.children[0].width = 1700  # set back to original size
        control_tabs.tabs.append(TabPanel(child=Div(), title="window 2"))
        win_dicts.pop()
        active = 0


class DocConfig:
    def __init__(self) -> None:
        config = {}
        config["colour_dict"] = {}
        config["initial_views"] = {}
        config["default_cprop"] = "type"
        config["demo_name"] = "TIAvis"
        config["base_folder"] = Path("/app_data")
        config["slide_folder"] = config["base_folder"].joinpath("slides")
        config["overlay_folder"] = config["base_folder"].joinpath("overlays")
        self.config = config
        self.sys_args = None

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config

    def set_sys_args(self, argv):
        self.sys_args = argv

    def setup_doc(self, doc):
        sys_args = self.sys_args
        if len(sys_args) > 1 and sys_args[1] != "None":
            base_folder = Path(sys_args[1])
            if len(req_args) > 0:
                print(req_args)
                config["demo_name"] = str(req_args["demo"][0], "utf-8")
                base_folder = base_folder.joinpath(str(req_args["demo"][0], "utf-8"))
            slide_folder = base_folder.joinpath("slides")
            overlay_folder = base_folder.joinpath("overlays")
            if not overlay_folder.exists():
                overlay_folder = None
        if len(sys_args) == 3:
            slide_folder = Path(sys_args[1])
            overlay_folder = Path(sys_args[2])
        # load a color_dict and/or slide initial view windows from a json file
        config_file = list(overlay_folder.glob("*config.json"))
        config = self.config
        if len(config_file) > 0:
            config_file = config_file[0]
            if (config_file).exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                    print(config)
                # if "config["colour_dict"]" in config:
                #     self.config["colour_dict"] = config["config["colour_dict"]"]
                #     print(self.config["colour_dict"])
                # if "config["initial_views"]" in config:
                #     self.config["initial_views"] = config["config["initial_views"]"]
                #     print(self.config["initial_views"])
                # if "config["default_cprop"]" in config:
                #     self.config["default_cprop"] = config["config["default_cprop"]"]
                #     print(self.config["default_cprop"])

        config["base_folder"] = base_folder
        config["slide_folder"] = slide_folder
        config["overlay_folder"] = overlay_folder
        config["demo_name"] = "demo_name"

        # get any extra info from query url
        if "slide" in req_args:
            config["first_slide"] = str(req_args["slide"][0], "utf-8")
            if "window" in req_args:
                if "initial_views" not in config:
                    config["initial_views"] = {}
                config["initial_views"][Path(config["first_slide"]).stem] = [
                    int(s) for s in str(req_args["window"][0], "utf-8")[1:-1].split(",")
                ]
        self.config = config
        self.config["auto_load"] = get_from_config(["auto_load"], 0) == 1
        # UI["vstate"].slide_path = r"E:\\TTB_vis_folder\\slides\\TCGA-SC-A6LN-01Z-00-DX1.svs"
        # UI["vstate"].slide_path=Path(r'/tiatoolbox/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs')

        # start tile server
        if not is_deployed:
            proc = Thread(target=run_app, daemon=True)
            proc.start()

        # set initial slide to first one in base folder
        slide_list = []
        for ext in ["*.svs", "*ndpi", "*.tiff", "*.mrxs", "*.png", "*.jpg"]:
            slide_list.extend(list(config["slide_folder"].glob(ext)))
            slide_list.extend(list(config["slide_folder"].glob(str(Path("*") / ext))))
        first_slide_path = slide_list[0]
        if "first_slide" in config:
            first_slide_path = config["slide_folder"] / config["first_slide"]

        # make initial window
        win_dicts.append(make_window(ViewerState(first_slide_path)))
        # set up any initial ui settings from config file
        if "UI_settings" in config:
            for k in config["UI_settings"]:
                update_renderer(k, config["UI_settings"][k])
        if self.config["default_cprop"] is not None:
            UI["s"].put(
                f"http://{host2}:5000/tileserver/color_prop",
                data={"prop": json.dumps(config["default_cprop"])},
            )
        # open up initial slide
        slide_select_cb(None, None, new=[UI["vstate"].slide_path])
        if "default_type_cprop" in config:
            UI["type_cmap_select"].value = list(config["default_type_cprop"].values())
        # initialise_slide()
        populate_slide_list(config["slide_folder"])
        populate_layer_list(
            Path(UI["vstate"].slide_path).stem, config["overlay_folder"]
        )
        UI["vstate"].init = False

        # # set up main window
        # slide_wins = row(
        #     children=windows,
        #     name="slide_windows",
        #     sizing_mode="stretch_both",
        # )

        # control_tabs = Tabs(tabs=controls, name="ui_layout")

        slide_wins.children = windows
        control_tabs.tabs = controls

        control_tabs.on_change("active", control_tabs_cb)
        control_tabs.on_change("tabs", control_tabs_remove_cb)

        print("windows and controls are:")
        print(windows)
        print(controls)
        print(doc.template)
        doc.template_variables["demo_name"] = config["demo_name"]
        doc.add_periodic_callback(update, 220)
        doc.add_root(slide_wins)
        # curdoc().add_root(ui_layout)
        doc.add_root(control_tabs)
        # curdoc().add_root(opts_column)
        doc.title = "Tiatoolbox Visualization Tool"
        # if "template" in config:
        #    curdoc().template = str(Path(r"C:\Users\meast\app_data\templates") / config["template"])
        return slide_wins, control_tabs


config = DocConfig()
if do_doc:
    config.set_sys_args(sys.argv)
    doc = curdoc()
    slide_wins, control_tabs = config.setup_doc(doc)