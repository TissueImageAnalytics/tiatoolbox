from shutil import rmtree
from bokeh.models import Selection, TileRenderer
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader
from PIL import Image
from bokeh.models import Plot, BasicTickFormatter, FuncTickFormatter, CheckboxButtonGroup, CustomJS, ColumnDataSource, Panel, Slider, Toggle, FileInput, DataRange1d, TextInput, Button, Dropdown, BoxEditTool, CheckboxGroup, ColorPicker, Range1d
from bokeh.models import GraphRenderer, Circle, StaticLayoutProvider, TapTool, PointDrawTool
from bokeh.layouts import layout, row, column
from bokeh.io.showing import show
from bokeh.core.properties import MinMaxBounds
from pathlib import Path
from bokeh.plotting import figure
from bokeh.models.tiles import WMTSTileSource, TMSTileSource
from cmath import pi
from tiatoolbox.tools.pyramid import AnnotationTileGenerator, ZoomifyGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer, random_colors
from tiatoolbox.annotation.storage import AnnotationStore, SQLiteStore, DictionaryStore, Annotation
from tiatoolbox.annotation.dsl import SQLTriplet, SQL_GLOBALS
from tiatoolbox.models.architecture.nuclick import NuClick
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.engine.interactive_segmentor import InteractiveSegmentor, IOInteractiveSegmentorConfig
from tiatoolbox.utils.misc import mpp2objective_power, objective_power2mpp
from pathlib import Path
#import sqlite3
from shapely.geometry import Polygon, Point
import numpy as np
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader
import requests
from flask_cors import CORS
from threading import Thread
import operator
#from bokeh.embed import server_document
#from pyproj import Geod, CRS, Transformer
import sys
from tiatoolbox.visualization.ui_utils import get_level_by_extent
import pickle
import torch

# Pandas for data management
import pandas as pd
from pathlib import Path

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs

#Define helper functions

def make_ts(route,mpp=0.2525):
    #crs = CRS.from_epsg(3857)
    #proj = Transformer.from_crs(crs,crs.geodetic_crs)
    sf=1.00301
    #sf=sf/mpp
    sf=(sf/0.5015)*2**(vstate.num_zoom_levels-10)    #*(vstate.maxds/32.0063)
    ts=WMTSTileSource(name="WSI provider", url=route, attribution="", snap_to_zoom=False, min_zoom=0, max_zoom=vstate.num_zoom_levels-1)
    ts.tile_size=256
    ts.initial_resolution=40211.5*sf*(2/(100*pi))   #156543.03392804097    40030 great circ
    ts.x_origin_offset=0#5000000
    #ts.y_origin_offset=-2500000
    ts.y_origin_offset=sf*(10247680*(2/(100*pi))  + 438.715 +38.997+13-195.728+0.82)  #10160000,   509.3
    ts.wrap_around=False
    #ts.max_zoom=10
    #ts.min_zoom=10
    return ts

def to_int_rgb(rgb):
    """Helper to convert from float to int rgb(a) tuple"""
    return tuple(int(v*255) for v in rgb)

def name2type(name):
    try:
        return int(name)
    except:
        return f'"{name}"'

def name2type_key(name):
    try:
        return int(name)
    except:
        return f'{name}'

def hex2rgb(hex):
    print(hex)
    return tuple(int(hex[i:i+2], 16)/255 for i in (1, 3, 5))

def update_mapper():
    colors = random_colors(len(vstate.types))
    vstate.mapper = {key: (*color, 1) for key, color in zip(vstate.types, colors)}
    print(vstate.mapper)
    renderer.mapper = lambda x: vstate.mapper[x]


def build_predicate():
    #pred1=eval("props['type']=='class1'", SQL_GLOBALS, {})
    #pred2=eval("props['type']=='class2'", SQL_GLOBALS, {})
    preds=[eval(f'props["type"]=={name2type(l.label)}', SQL_GLOBALS, {}) for l in box_column.children if l.active]
    if len(preds)==len(box_column.children):
        preds=[]
    combo=None
    if len(preds)>0:       
        combo=preds[0]
        for pred in preds[1:]:
            combo=SQLTriplet(combo, operator.or_, pred)
    if filter_input.value!='None':
        combo = SQLTriplet(eval(filter_input.value, SQL_GLOBALS, {}), operator.and_, combo)

    vstate.renderer.where=combo
    return combo

def build_predicate_callable():
    get_types=[name2type_key(l.label) for l in box_column.children if l.active]
    if len(get_types)==len(box_column.children):
        if filter_input.value == 'None':
            vstate.renderer.where=None
            return None
        
    if filter_input.value == 'None':
        def pred(props):
            return props['type'] in get_types
    else:
        def pred(props):
            print(eval(filter_input.value))
            return eval(filter_input.value) and props['type'] in get_types
    vstate.renderer.where=pred
    return pred

def initialise_slide():
    vstate.mpp=wsi[0].info.mpp
    vstate.dims = wsi[0].info.slide_dimensions

    pad=int(np.mean(vstate.dims)/50)
    plot_size=np.array([1700,1000])
    large_dim=np.argmax(np.array(vstate.dims)/plot_size)

    vstate.micron_formatter.args['mpp']=vstate.mpp[0]
    if large_dim==1:
        p.x_range.start = -0.5*(vstate.dims[1]*1.7-vstate.dims[0])-1.7*pad
        p.x_range.end = vstate.dims[1]*1.7-0.5*(vstate.dims[1]*1.7-vstate.dims[0])+1.7*pad
        p.y_range.start = -vstate.dims[1] - pad
        p.y_range.end = pad
        #p.x_range.min_interval = ?
    else:
        p.x_range.start = -1.7*pad
        p.x_range.end = vstate.dims[0] + pad*1.7
        p.y_range.start = -vstate.dims[0]/1.7 + 0.5*(vstate.dims[0]/1.7-vstate.dims[1])-pad
        p.y_range.end = 0.5*(vstate.dims[0]/1.7-vstate.dims[1])+pad

    p.x_range.bounds = (p.x_range.start-2*pad, p.x_range.end+2*pad)
    p.y_range.bounds = (p.y_range.start-2*pad, p.y_range.end+2*pad)

    opt_level,_ = wsi[0]._find_optimal_level_and_downsample(1.0,'power')
    z =ZoomifyGenerator(wsi[0])
    vstate.num_zoom_levels = z.level_count
    print(f'nzoom_levs: {vstate.num_zoom_levels}')
    vstate.maxds = wsi[0].info.level_downsamples[np.minimum(opt_level+1, len(wsi[0].info.level_downsamples)-1)]
    #vstate.maxds = wsi[0].info.level_downsamples[opt_level]
    zlev = get_level_by_extent((0, p.y_range.start, p.x_range.end, 0))
    print(f'initial_zoom: {zlev}')
    #vstate.maxds = 2 ** (9 - zlev - 1)
    print(vstate.maxds)
    print(wsi[0].info.as_dict())
    #p.x_range.bounds = (0, vstate.dims[0])
    #p.y_range.bounds = (-vstate.dims[1], 0)
    


def initialise_overlay():
    vstate.colors=list(vstate.mapper.values())
    vstate.types=[str(t) for t in vstate.types]#vstate.mapper.keys()]
    now_active={b.label: b.active for b in box_column.children}
    print(vstate.types)
    print(now_active)
    for t in vstate.types:
        if str(t) not in now_active.keys():
            box_column.children.append(Toggle(label=str(t), active=True, width=100))
            box_column.children[-1].on_click(layer_select_cb)
            try:
                color_column.children.append(ColorPicker(color=to_int_rgb(vstate.mapper[t][0:3]), name=str(t), width=60))
            except KeyError:
                color_column.children.append(ColorPicker(color=to_int_rgb(vstate.mapper[int(t)][0:3]), name=str(t), width=60))
            color_column.children[-1].on_change('color', bind_cb_obj(color_column.children[-1], color_input_cb))

    for b in box_column.children.copy():
        if b.label not in vstate.types and b.label not in vstate.layer_dict.keys():
            print(f'removing {b.label}')
            box_column.children.remove(b)
    for c in color_column.children.copy():
        if c.name not in vstate.types and 'slider' not in c.name:
            color_column.children.remove(c)

    build_predicate_callable()

def add_layer(lname):
    box_column.children.append(Toggle(label=lname, active=True, width=100))
    box_column.children[-1].on_click(bind_cb_obj_tog(box_column.children[-1], fixed_layer_select_cb))
    color_column.children.append(Slider(start=0, end=1, value=0.5, step=0.01, title=lname, width=100, name=f'{lname}_slider'))
    color_column.children[-1].on_change('value', bind_cb_obj(color_column.children[-1], layer_slider_cb))

    #layer_boxes=[Toggle(label=t, active=a, width=100) for t,a in now_active.items()]
    #lcolors=[ColorPicker(color=col[0:3], name=t, width=60) for col, t in zip(vstate.colors, vstate.types)]

class TileGroup():
    def __init__(self):
        self.group=1
    
    def get_grp(self):
        self.group=self.group+1
        return self.group
tg=TileGroup()

def change_tiles(layer_name='overlay'):
    
    grp=tg.get_grp()
   
    if layer_name=='graph' and layer_name not in vstate.layer_dict.keys():
        p.renderers.append(graph)
        vstate.layer_dict[layer_name]=len(p.renderers)-1
        for layer_key in vstate.layer_dict.keys():
            if layer_key in ['rect', 'pts', 'graph']:
                continue
            grp=tg.get_grp()
            ts=make_ts(f'http://127.0.0.1:5000/layer/{layer_key}/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg', mpp=vstate.mpp)
            p.renderers[vstate.layer_dict[layer_key]].tile_source=ts
        return

    ts=make_ts(f'http://127.0.0.1:5000/layer/{layer_name}/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg', mpp=vstate.mpp)
    if layer_name in vstate.layer_dict.keys():
        p.renderers[vstate.layer_dict[layer_name]].tile_source=ts
    else:
        p.add_tile(ts, smoothing = True, alpha=overlay_alpha.value, level='overlay', render_parents=False)
        for layer_key in vstate.layer_dict.keys():
            if layer_key in ['rect', 'pts', 'graph']:
                continue
            grp=tg.get_grp()
            ts=make_ts(f'http://127.0.0.1:5000/layer/{layer_key}/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg', mpp=vstate.mpp)
            p.renderers[vstate.layer_dict[layer_key]].tile_source=ts
        vstate.layer_dict[layer_name]=len(p.renderers)-1

    print(vstate.layer_dict)
    print(p.renderers)

class ViewerState():
    def __init__(self):
        self.dims=[30000,20000]
        self.mpp=None
        self.maxds=None
        #self.mapper={'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)}
        self.mapper={}
        self.colors=list(self.mapper.values())
        self.types=list(self.mapper.keys())
        self.layer_dict={'slide': 0,'rect': 1, 'pts': 2}
        self.renderer=[]
        self.num_zoom_levels=0
        self.slide_path=None
        self.update_state=0
        self.model_mpp=0
        self.micron_formatter=None
        self.current_model='hovernet'

vstate=ViewerState()

base_folder='E:\TTB_vis_folder'
#base_folder='/tiatoolbox/app_data'
if len(sys.argv)>1 and sys.argv[1]!='None':
    base_folder=sys.argv[1]
#geo_path=Path(r'E:\TTB_vis_test\ff7d5488-60e1-4c6e-9eb4-d495bb7565b1\TCGA-SC-AA5Z-01Z-00-DX1-2.geojson')
#slide_path=Path(r'E:\TTB_vis_test\ff7d5488-60e1-4c6e-9eb4-d495bb7565b1\TCGA-SC-AA5Z-01Z-00-DX1.svs')
vstate.slide_path=r'E:\TTB_vis_folder\slides\TCGA-SC-A6LN-01Z-00-DX1.svs'
#vstate.slide_path=Path(r'/tiatoolbox/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs')

wsi = [WSIReader.open(vstate.slide_path)]
renderer=AnnotationRenderer('type', {'class1': (1,0,0,1), 'class2': (0,0,1,1), 'class3': (0,1,0,1)}, thickness=-1, edge_thickness=1)
vstate.renderer=renderer

vstate.dims=wsi[0].info.slide_dimensions

vstate.mpp=wsi[0].info.mpp
vstate.maxds=wsi[0].info.level_downsamples[-1]

def run_app():

    app = TileServer(
        title="Testing TileServer",
        layers={
        "slide": wsi[0],
        #"overlay": tile_gen #(wsi, SQ)
        },
        state=vstate,
        )
    CORS(app, send_wildcard=True)   
    app.run(threaded=False)

#start tile server
proc=Thread(target=run_app, daemon=True)
proc.start()

TOOLTIPS=[
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("Scores", "[@c1, @c2]"),
    ]

#set up main window
vstate.micron_formatter = FuncTickFormatter(args={'mpp': 0.1}, code="""
    return Math.round(tick*mpp)
    """)
p = figure(x_range=(0, vstate.dims[0]), y_range=(0,-vstate.dims[1]),x_axis_type="linear", y_axis_type="linear",
width=1700,height=1000, tooltips=TOOLTIPS, output_backend="canvas", hidpi=False, match_aspect=False, lod_factor=100, lod_interval=500, lod_threshold=10, lod_timeout=200)
initialise_slide()
ts1=make_ts(r'http://127.0.0.1:5000/layer/slide/zoomify/TileGroup1/{z}-{x}-{y}.jpg', vstate.mpp)
print(p.renderers)
print(p.y_range)
p.add_tile(ts1, smoothing = True, level='image', render_parents=False)
print(p.y_range)
print(f'max zoom is: {p.renderers[0].tile_source.max_zoom}')

p.grid.grid_line_color=None
p.match_aspect=True
box_source=ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})
pt_source=ColumnDataSource({'x': [], 'y': []})
r=p.rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
c = p.circle('x', 'y', source=pt_source, color='red', size=5)
p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
p.add_tools(PointDrawTool(renderers=[c]))
p.add_tools(TapTool())
tslist=[]

print(p.extra_y_ranges)
print(p.y_scale)
print(p.x_scale)
p.renderers[0].tile_source.max_zoom=10

node_source = ColumnDataSource({'index': []})
edge_source = ColumnDataSource({'start': [], 'end': []})
graph = GraphRenderer()
graph.node_renderer.data_source = node_source
graph.edge_renderer.data_source = edge_source
graph.node_renderer.glyph = Circle(radius=25, fill_color='green')


#Define UI elements
slide_alpha = Slider(
    title="Adjust alpha WSI",
    start=0,
    end=1,
    step=0.05,
    value=1.0,
    width=200,
)

overlay_alpha = Slider(
        title="Adjust alpha Overlay",
        start=0,
        end=1,
        step=0.05,
        value=0.75,
        width=200
    )

slide_toggle = Toggle(label="Slide", button_type="success", width=90)
overlay_toggle = Toggle(label="Overlay", button_type="success", width=90)
filter_input = TextInput(value='None', title="Filter:")
cprop_input = TextInput(value="type", title="CProp:")
folder_input = TextInput(value=base_folder, title="Img Folder:")
cmmenu=[('jet','jet'),('coolwarm', 'coolwarm'),('dict',"{'class1': (1,0,0,1), 'class2': (0,0,1,1), 'class3': (0,1,0,1)}")]
cmap_drop = Dropdown(label="Colourmap", button_type="warning", menu=cmmenu)
file_drop = Dropdown(label="Choose Slide", button_type="warning", menu=[None])
to_model_button=Button(label="Go", button_type="success", width=60)
model_drop=Dropdown(label="Choose Model", button_type="warning", menu=['hovernet', 'nuclick'], width=100)
layer_boxes=[Toggle(label=t, active=True, width=100) for t in vstate.types]
lcolors=[ColorPicker(color=col[0:3], width=60) for col in vstate.colors]
layer_folder_input = TextInput(value=base_folder, title="Overlay Folder:")
layer_drop = Dropdown(label="Add Overlay", button_type="warning", menu=[None])
opt_buttons = CheckboxButtonGroup(labels=['Filled', 'Microns', 'Grid'], active=[0])
save_button=Button(label='Save', button_type="success")


#Define UI callbacks
def slide_toggle_cb(attr):
    print('meep')
    if p.renderers[0].alpha==0:
        p.renderers[0].alpha=slide_alpha.value
    else:
        p.renderers[0].alpha=0.0

def node_select_cb(attr, old, new):
    print(f'selected is: {new}')
    vstate.mapper={new[0]: (1,0,0,1)}
    vstate.renderer.mapper = lambda x: vstate.mapper[x]
    vstate.update_state=1
    

def overlay_toggle_cb(attr):
    print('meep')
    for i in range(3,len(p.renderers)):
        if p.renderers[i].alpha==0:
            p.renderers[i].alpha=overlay_alpha.value
        else:
            p.renderers[i].alpha=0.0

def folder_input_cb(attr, old, new):
    file_list=[]
    for ext in ['*.svs','*ndpi','*.tiff','*.mrxs']:#,'*.png','*.jpg']:
        file_list.extend(list(Path(new).glob('*\\'+ext)))
    file_list=[(str(p),str(p)) for p in file_list]
    file_drop.menu=file_list
    
    file_list=[]
    for ext in ['*.db','*.dat','*.geojson','*.png','*.jpg','*.tiff','*.pkl']:
        file_list.extend(list(Path(new).glob('*\\'+ext)))
    file_list=[(str(p),str(p)) for p in file_list]
    layer_drop.menu=file_list

def populate_layer_list(slide_name, folder_path):
    file_list=[]
    for ext in ['*.db','*.dat','*.geojson','*.png','*.jpg','*.pkl']:   #and '*.tiff'?
        file_list.extend(list(folder_path.glob('*\\'+ext)))
    file_list=[(str(p),str(p)) for p in file_list if slide_name in str(p)]
    layer_drop.menu=file_list

def layer_folder_input_cb(attr, old, new):
    #unused at the moment
    file_list=[]
    for ext in ['*.db','*.dat','*.geojson','*.png','*.jpg','.tiff']:
        file_list.extend(list(Path(new).glob('*\\'+ext)))
    file_list=[(str(p),str(p)) for p in file_list]
    layer_drop.menu=file_list
    return file_list

def filter_input_cb(attr, old, new):
    resp = requests.get(f'http://127.0.0.1:5000/changepredicate/{new}')
    #change_tiles('overlay')
    vstate.update_state=1
    #proc=Thread(target=change_tiles, args=('overlay',))
    #proc.start()
    #proc.join()

def cprop_input_cb(attr, old, new):
    resp = requests.get(f'http://127.0.0.1:5000/changeprop/{new}')
    #change_tiles('overlay')
    vstate.update_state=1

def set_graph_alpha(g_renderer, value):
    #set all components of graph to given alpha value
    g_renderer.node_renderer.glyph.fill_alpha=value
    g_renderer.node_renderer.glyph.line_alpha=value
    g_renderer.edge_renderer.glyph.line_alpha=value

def slide_alpha_cb(attr,old,new):
    print('meep')
    p.renderers[0].alpha=new
    #p.renderers[0].tile_source.max_zoom=7
    #p.renderers[1].tile_source.max_zoom=7

def overlay_alpha_cb(attr,old,new):
    print('meep')
    for i in range(3,len(p.renderers)):
        if isinstance(p.renderers[i], GraphRenderer):
            #set_graph_alpha(p.renderers[i], new)
            pass
        else:
            p.renderers[i].alpha=new

def opt_buttons_cb(attr, old, new):    
    old_thickness = vstate.renderer.thickness
    if 0 in new:
        vstate.renderer.thickness=-1
    else:
        vstate.renderer.thickness=1
    if old_thickness != vstate.renderer.thickness:
        vstate.update_state=1
    if 1 in new:
        p.xaxis[0].formatter=vstate.micron_formatter
        p.yaxis[0].formatter=vstate.micron_formatter
    else:
        p.xaxis[0].formatter=BasicTickFormatter()
        p.yaxis[0].formatter=BasicTickFormatter()
    if 2 in new:
        p.ygrid.grid_line_color='gray'
        p.xgrid.grid_line_color='gray'
        p.ygrid.grid_line_alpha = 0.6
        p.xgrid.grid_line_alpha = 0.6
    else:
        p.ygrid.grid_line_alpha = 0
        p.xgrid.grid_line_alpha = 0
    print(p.ygrid)
    print(p.grid)
    
    

def cmap_drop_cb(attr):
    resp = requests.get(f'http://127.0.0.1:5000/changecmap/{attr.item}')
    #change_tiles('overlay')
    vstate.update_state=1

def file_drop_cb(attr):
    """setup the newly chosen slide"""
    pt_source.data={'x': [], 'y': []}
    box_source.data={'x': [], 'y': [], 'width': [], 'height': []}
    if len(p.renderers)>3:
        for r in p.renderers[3:].copy():
            p.renderers.remove(r)
    #p.extra_x_ranges={'oldx': p.x_range}
    vstate.layer_dict={'slide':0,'rect':1, 'pts': 2}
    vstate.slide_path=attr.item
    for c in color_column.children.copy():
        if '_slider' in c.name:
            color_column.children.remove(c)
    for b in box_column.children.copy():
        if 'layer' in b.label:
            box_column.children.remove(b)
    print(p.renderers)
    print(attr.item)
    populate_layer_list(Path(attr.item).stem, Path(vstate.slide_path).parents[1])
    wsi[0] = WSIReader.open(attr.item)
    initialise_slide()
    fname='-*-'.join(attr.item.split('\\'))
    print(fname)
    print(vstate.mpp)
    resp = requests.get(f'http://127.0.0.1:5000/changeslide/slide/{fname}')
    change_tiles('slide')
    #if len(p.renderers)==1:
        #r=p.rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
        #p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
    #p.x_range.bounds=MinMaxBounds(0,vstate.dims[0])
    #p.y_range.bounds=(0,-vstate.dims[1])

def layer_drop_cb(attr):
    """setup the newly chosen overlay"""
    if Path(attr.item).suffix=='.pkl':
        with open(attr.item,'rb') as f:
            graph_dict=pickle.load(f)
        node_source.data={'index': list(range(graph_dict['x'].shape[0]))}
        edge_source.data={'start': graph_dict['edge_index'][0,:], 'end': graph_dict['edge_index'][1,:]}
        graph_layout = dict(zip(node_source.data['index'], [(x/(4*vstate.mpp[0]), -y/(4*vstate.mpp[1])) for x, y in graph_dict['coordinates']]))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        add_layer('graph')
        change_tiles('graph')
        return

    print(attr.item)
    fname='-*-'.join(attr.item.split('\\'))
    print(fname)
    resp = requests.get(f'http://127.0.0.1:5000/changeoverlay/{fname}')
    print(vstate.types)
    if resp.text=='overlay':
        update_mapper()
        initialise_overlay()
    else:
        add_layer(resp.text)
    change_tiles(resp.text)
    #change_tiles('slide')

def layer_select_cb(attr):
    build_predicate_callable()
    #change_tiles('overlay')
    vstate.update_state=1

def fixed_layer_select_cb(obj, attr):
    print(vstate.layer_dict)
    key=vstate.layer_dict[obj.label]
    if obj.label=='graph':
        if p.renderers[key].node_renderer.glyph.fill_alpha==0:
            p.renderers[key].node_renderer.glyph.fill_alpha=overlay_alpha.value
            p.renderers[key].node_renderer.glyph.line_alpha=overlay_alpha.value
            p.renderers[key].edge_renderer.glyph.line_alpha=overlay_alpha.value
        else:
            p.renderers[key].node_renderer.glyph.fill_alpha=0.0
            p.renderers[key].node_renderer.glyph.line_alpha=0.0
            p.renderers[key].edge_renderer.glyph.line_alpha=0.0
    else:
        if p.renderers[key].alpha==0:
            p.renderers[key].alpha=overlay_alpha.value
        else:
            p.renderers[key].alpha=0.0

def layer_slider_cb(obj, attr, old, new):
    if isinstance(p.renderers[vstate.layer_dict[obj.name.split('_')[0]]], GraphRenderer):
        set_graph_alpha(p.renderers[vstate.layer_dict[obj.name.split('_')[0]]], new)
    else:
        p.renderers[vstate.layer_dict[obj.name.split('_')[0]]].alpha=new

def color_input_cb(obj, attr, old, new):
    print(new)
    vstate.mapper[name2type_key(obj.name)]=(*hex2rgb(new), 1)
    if vstate.renderer.score_prop=='type':
        vstate.renderer.mapper=lambda x: vstate.mapper[x]
    #change_tiles('overlay')
    vstate.update_state=1
    
def bind_cb_obj(cb_obj, cb):
    def wrapped(attr, old, new):
        cb(cb_obj, attr, old, new)

    return wrapped

def bind_cb_obj_tog(cb_obj, cb):
    def wrapped(attr):
        cb(cb_obj, attr)

    return wrapped

def model_drop_cb(attr):
    vstate.current_model=attr.item

def to_model_cb(attr):
    if vstate.current_model == 'hovernet':
        segment_on_box(attr)
    elif vstate.current_model == 'nuclick':
        nuclick_on_pts(attr)
    else:
        print('unknown model')

def save_cb(attr):
    resp = requests.get(f'http://127.0.0.1:5000/commit')


#run NucleusInstanceSegmentor on a region of wsi defined by the box in box_source
def segment_on_box(attr):
    print(vstate.types)
    #thumb=wsi[0].slide_thumbnail(resolution=8, units='mpp')
    thumb=wsi[0].slide_thumbnail()
    #conv_mpp=wsi.convert_resolution_units(1.25, 'power', 'mpp')[0]
    conv_mpp=vstate.dims[0]/thumb.shape[1]
    print(f'box tl: {box_source.data["x"][0]}, {box_source.data["y"][0]}')
    x=round((box_source.data['x'][0]-0.5*box_source.data['width'][0])/conv_mpp)
    y=-round((box_source.data['y'][0]+0.5*box_source.data['height'][0])/conv_mpp)
    width=round(box_source.data['width'][0]/conv_mpp)
    height=round(box_source.data['height'][0]/conv_mpp)
    print(x,y,width,height)


    #img_tile=wsi.read_rect((x,y),(width,height))
    #print(img_tile.shape)
    mask=np.zeros((thumb.shape[0],thumb.shape[1]))
    mask[y:y+height,x:x+width]=1
    #mask_r = VirtualWSIReader(mask, mpp=(0.5, 0.5))

    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=6,
        num_postproc_workers=12,
        batch_size=24,
    )
    print(inst_segmentor.ioconfig.save_resolution)
    #inst_segmentor.ioconfig.save_resolution['resolution'] = vstate.mpp[0]
    for res in inst_segmentor.ioconfig.input_resolutions:
        print(res)
        #res['resolution'] = vstate.mpp[0]
    for res in inst_segmentor.ioconfig.output_resolutions:
        print(res)
        #res['resolution'] = vstate.mpp[0]

    vstate.model_mpp=inst_segmentor.ioconfig.save_resolution['resolution']
    tile_output = inst_segmentor.predict(
        [vstate.slide_path],
        [mask],
        save_dir="sample_tile_results/",
        mode="wsi",
        #resolution=vstate.mpp,
        #units='mpp',
        on_gpu=True,
        crash_on_exception=True,
    )

    fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
    print(fname)
    resp = requests.get(f'http://127.0.0.1:5000/loadannotations/{fname}')

    #types = SQ.query_property("props['type']", [0,0,100000,100000], distinct=True)
    print(vstate.types)
    #print(len(SQ))
    update_mapper()
    #type_drop.menu=[(str(t),str(t)) for t in vstate.types]
    rmtree(r'./sample_tile_results')
    initialise_overlay()
    change_tiles('overlay')

    return tile_output

#run nuclick on user selected points in pt_source
def nuclick_on_pts(attr):
    x=np.round(np.array(pt_source.data['x']))
    y=-np.round(np.array(pt_source.data['y']))

    model = NuClick(5, 1)
    pretrained_weights=r'E:\TTB_vis_folder\NuClick_Nuclick_40xAll.pth'
    saved_state_dict = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)
    vstate.model_mpp = 0.25
    ioconf = IOInteractiveSegmentorConfig(input_resolutions=[{'resolution': 0.25, 'units': 'mpp'}], patch_size=(128, 128))
    inst_segmentor = InteractiveSegmentor(
        num_loader_workers=0,
        batch_size=16,
        model=model,
    )
    #print(inst_segmentor.ioconfig.save_resolution)
    points = np.vstack([x,y]).T
    points = points/(ioconf.input_resolutions[0]['resolution']/vstate.mpp[0])
    print(points.shape)
    nuclick_output = inst_segmentor.predict(
        [vstate.slide_path],
        [points],
        ioconfig=ioconf,
        save_dir="sample_tile_results/",
        patch_size=(128,128),
        resolution=0.25,
        units='mpp',
        on_gpu=True,
        save_output=True,
    )
    print(nuclick_output)

    fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
    print(fname)
    resp = requests.get(f'http://127.0.0.1:5000/loadannotations/{fname}')
    update_mapper()
    rmtree(r'./sample_tile_results')
    initialise_overlay()
    change_tiles('overlay')

#associate callback functions to the widgets
slide_alpha.on_change('value', slide_alpha_cb)
overlay_alpha.on_change('value', overlay_alpha_cb)
folder_input.on_change('value', folder_input_cb)
save_button.on_click(save_cb)
cmap_drop.on_click(cmap_drop_cb)
file_drop.on_click(file_drop_cb)
to_model_button.on_click(to_model_cb)
model_drop.on_click(model_drop_cb)
layer_drop.on_click(layer_drop_cb)
opt_buttons.on_change('active', opt_buttons_cb)
slide_toggle.on_click(slide_toggle_cb)
overlay_toggle.on_click(overlay_toggle_cb)
filter_input.on_change('value', filter_input_cb)
cprop_input.on_change('value', cprop_input_cb)
node_source.selected.on_change('indices', node_select_cb)

folder_input_cb(None, None, base_folder)
populate_layer_list(Path(vstate.slide_path).stem, Path(vstate.slide_path).parents[1])
#populate_layer_list(Path(attr.item).stem, Path(vstate.slide_path).parents[1])

box_column=column(children=layer_boxes)
color_column=column(children=lcolors)
l=layout(
    [
        [p, [
            folder_input,
            save_button,
            file_drop,
            layer_drop,
            row([slide_toggle, slide_alpha]),
            row([overlay_toggle, overlay_alpha]),
            filter_input,
            cprop_input, 
            cmap_drop,
            opt_buttons,
            row([to_model_button, model_drop]),
            #type_drop,
            row(children=[box_column, color_column]),
            #box_column,
            #layer_folder_input,
            ]
        ],
    ]
)

def cleanup_session(session_context):
    # If present, this function executes when the server closes a session.
    print('cleaning up..')
    
#script = server_document("http://127.0.0.1:5006/render_demo")
#print(script)
def update():
    if vstate.update_state==2:
        change_tiles('overlay')
        vstate.update_state=0
    if vstate.update_state==1:
        vstate.update_state=2
    

curdoc().add_periodic_callback(update, 220)
curdoc().add_root(l)
curdoc().on_session_destroyed(cleanup_session)
