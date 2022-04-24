from msilib.schema import CustomAction
from shutil import rmtree
from turtle import color
from bokeh.models import Selection, TileRenderer
from numpy import source
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader
from PIL import Image
from bokeh.models import Plot, CustomJSExpr, CustomJS, ColumnDataSource, Panel, Slider, Toggle, FileInput, DataRange1d, TextInput, Button, Dropdown, BoxEditTool, CheckboxGroup, ColorPicker, Range1d
from bokeh.layouts import layout, row, column
from bokeh.core.properties import MinMaxBounds
from pathlib import Path
from bokeh.plotting import figure
from bokeh.models.tiles import WMTSTileSource, TMSTileSource
from cmath import pi
from tiatoolbox.tools.pyramid import AnnotationTileGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer, random_colors
from tiatoolbox.annotation.storage import AnnotationStore, SQLiteStore, DictionaryStore, Annotation
from tiatoolbox.annotation.dsl import SQLTriplet, SQL_GLOBALS
#from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
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
from bokeh.embed import server_document
from pyproj import Geod, CRS, Transformer
import sys

# Pandas for data management
import pandas as pd
from pathlib import Path

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs

def make_ts(route,mpp=0.2525):
    crs = CRS.from_epsg(3857)
    proj = Transformer.from_crs(crs,crs.geodetic_crs)
    sf=1.00301
    #sf=sf/mpp
    sf=(sf/0.5015)*(vstate.maxds/32.0063)
    ts=WMTSTileSource(name="WSI provider", url=route, attribution="")
    ts.tile_size=256
    ts.initial_resolution=40211.5*sf*(2/(100*pi))   #156543.03392804097    40030 great circ
    ts.x_origin_offset=0#5000000
    #ts.y_origin_offset=-2500000
    ts.y_origin_offset=sf*(10247680*(2/(100*pi))  + 438.715 +38.997+13-195.728+0.82)  #10160000,   509.3
    ts.wrap_around=False
    ts.max_zoom=10
    #ts.min_zoom=10
    return ts

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
    return tuple(int(hex[i:i+2], 16)/255 for i in (1, 3, 5))

def update_mapper():
    colors = random_colors(len(vstate.types))
    vstate.mapper = {key: tuple((np.array([*color, 1])*255).astype(np.uint8)) for key, color in zip(vstate.types, colors)}
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
    

def initialise_slide():
    vstate.mpp=np.minimum(wsi[0].info.mpp[0],1.0)
    vstate.dims = wsi[0].info.slide_dimensions
    opt_level,_ = wsi[0]._find_optimal_level_and_downsample(1.25,'power')
    print(opt_level)
    vstate.maxds = wsi[0].info.level_downsamples[np.minimum(opt_level+1, len(wsi[0].info.level_downsamples)-1)]
    print(wsi[0].info.as_dict())
    #p.x_range.bounds = (0, vstate.dims[0])
    #p.y_range.bounds = (-vstate.dims[1], 0)
    plot_size=np.array([1700,1000])
    large_dim=np.argmax(np.array(vstate.dims)/plot_size)
    
    if large_dim==1:
        p.x_range.start = -0.5*(vstate.dims[1]*1.7-vstate.dims[0])
        p.x_range.end = vstate.dims[1]*1.7-0.5*(vstate.dims[1]*1.7-vstate.dims[0])
        p.y_range.start = -vstate.dims[1]
        p.y_range.end = 0
    else:
        p.x_range.start = 0
        p.x_range.end = vstate.dims[0]
        p.y_range.start = -vstate.dims[0]/1.7 + 0.5*(vstate.dims[0]/1.7-vstate.dims[1])
        p.y_range.end = 0.5*(vstate.dims[0]/1.7-vstate.dims[1])


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
                color_column.children.append(ColorPicker(color=vstate.mapper[t][0:3], name=str(t), width=60))
            except KeyError:
                color_column.children.append(ColorPicker(color=vstate.mapper[int(t)][0:3], name=str(t), width=60))
            color_column.children[-1].on_change('color', bind_cb_obj(color_column.children[-1], color_input_cb))

    for b in box_column.children.copy():
        if b.label not in vstate.types and b.label not in vstate.layer_dict.keys():
            print(f'removing {b.label}')
            box_column.children.remove(b)
    for c in color_column.children.copy():
        if c.name not in vstate.types and 'slider' not in c.name:
            color_column.children.remove(c)

    build_predicate()

def add_layer(lname):
    box_column.children.append(Toggle(label=lname, active=True, width=100))
    box_column.children[-1].on_click(bind_cb_obj_tog(box_column.children[-1], fixed_layer_select_cb))
    color_column.children.append(Slider(start=0, end=1, value=0.5, step=0.01, title=lname, width=100, name=f'{lname}_slider'))
    color_column.children[-1].on_change('value', bind_cb_obj(color_column.children[-1], layer_slider_cb))

    #layer_boxes=[Toggle(label=t, active=a, width=100) for t,a in now_active.items()]
    #lcolors=[ColorPicker(color=col[0:3], name=t, width=60) for col, t in zip(vstate.colors, vstate.types)]

class ViewerState():
    def __init__(self):
        self.dims=[30000,20000]
        self.mpp=None
        self.maxds=None
        #self.mapper={'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)}
        self.mapper={}
        self.colors=list(self.mapper.values())
        self.types=list(self.mapper.keys())
        self.layer_dict={'slide': 0,'rect': 1}
        self.renderer=[]
        self.slide_path=None

vstate=ViewerState()

base_folder='E:\TTB_vis_folder'
if len(sys.argv)>1 and sys.argv[1]!='None':
    base_folder=sys.argv[1]
#geo_path=Path(r'E:\TTB_vis_test\ff7d5488-60e1-4c6e-9eb4-d495bb7565b1\TCGA-SC-AA5Z-01Z-00-DX1-2.geojson')
#slide_path=Path(r'E:\TTB_vis_test\ff7d5488-60e1-4c6e-9eb4-d495bb7565b1\TCGA-SC-AA5Z-01Z-00-DX1.svs')
vstate.slide_path=Path(r'E:\TTB_vis_test\a654c733-cd9a-424c-b60f-56f6afc6a620\TCGA-SC-A6LN-01Z-00-DX1.svs')

def rand_props(pval):
    props={
        'type': str(np.random.choice(['class1', 'class2', 'class3'],1)[0]),
        'prob': pval,#np.random.rand(1)[0],
    }
    return props

def prop_check(prop):
    if prop['type']=='class1':
        return True
    return False

prop_check = 'props["type"] == "class1"'
wsi = [WSIReader.open(vstate.slide_path)]
renderer=AnnotationRenderer('type', {'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)}, thickness=-1)#, prop_check)
vstate.renderer=renderer
#renderer=AnnotationRenderer('score', ['red', 'blue', 'green'], prop_check)
#renderer=AnnotationRenderer('score', ['red', 'blue', 'bob'])
vstate.dims=wsi[0].info.slide_dimensions
#mpp=wsi.info.mpp[0]
#vstate.dims=dims
vstate.mpp=wsi[0].info.mpp[0]
vstate.maxds=wsi[0].info.level_downsamples[-1]
#wsi=[None]

def run_app():
    #tile_gen=AnnotationTileGenerator(wsi.info, SQ, renderer)
    #vstate.types=SQ.query_property("props['type']",[0,0,*dims],distinct=True)

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

proc=Thread(target=run_app, daemon=True)
proc.start()

TOOLTIPS=[
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("Scores", "[@c1, @c2]"),
    ]

#SQ = SQLiteStore()
ts1=make_ts(r'http://127.0.0.1:5000/layer/slide/zoomify/TileGroup1/{z}-{x}-{y}.jpg', vstate.mpp)
#ts2=make_ts(r'http://127.0.0.1:5000/layer/overlay/zoomify/TileGroup1/{z}-{x}-{y}.jpg')

p = figure(x_range=(0, vstate.dims[0]), y_range=(0,-vstate.dims[1]),x_axis_type="linear", y_axis_type="linear",
width=1700,height=1000, tooltips=TOOLTIPS,output_backend="canvas", hidpi=True, match_aspect=False)
print(p.renderers)
print(p.y_range)
p.add_tile(ts1, smoothing = True, level='image')
print(p.y_range)
print(f'max zoom is: {p.renderers[0].tile_source.max_zoom}')
#p.add_tile(ts2)
p.grid.grid_line_color=None
p.match_aspect=True
box_source=ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})
r=p.rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
tslist=[]
initialise_slide()
print(p.extra_y_ranges)
print(p.extra_y_scales)
p.renderers[0].tile_source.max_zoom=10

setmax=CustomJS(args=dict(p=p), code="""
        p.renderers[0].tile_source.max_zoom=10;
    """)

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
#toggle3 = Button(label="Submit Filter", button_type="success")
#toggle4 = Button(label="Submit score_prop", button_type="success")

filter_input = TextInput(value='None', title="Filter:")
cprop_input = TextInput(value="type", title="CProp:")
folder_input = TextInput(value=base_folder, title="Img Folder:")
cmmenu=[('jet','jet'),('coolwarm', 'coolwarm'),('dict',"{'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)}")]
cmap_drop = Dropdown(label="Colourmap", button_type="warning", menu=cmmenu)
file_drop = Dropdown(label="Choose Slide", button_type="warning", menu=[None])
to_model_button=Button(label="To Model", button_type="success")
#type_drop=Dropdown(label="Types", button_type="warning", menu=[None])
layer_boxes=[Toggle(label=t, active=True, width=100) for t in vstate.types]
lcolors=[ColorPicker(color=col[0:3], width=60) for col in vstate.colors]
layer_folder_input = TextInput(value=base_folder, title="Overlay Folder:")
layer_drop = Dropdown(label="Add Overlay", button_type="warning", menu=[None])

class TileGroup():
    def __init__(self):
        self.group=1
    
    def get_grp(self):
        self.group=self.group+1
        return self.group
tg=TileGroup()

def change_tiles(layer_name='overlay'):
    
    grp=tg.get_grp()
    #ts1=make_ts(f'http://127.0.0.1:5000/layer/slide/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg')
    #p.add_tile(ts1, smoothing=False)
    
    #p.add_tile(ts2, smoothing=False)
    ts=make_ts(f'http://127.0.0.1:5000/layer/{layer_name}/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg', mpp=vstate.mpp)
    if layer_name in vstate.layer_dict.keys():
        p.renderers[vstate.layer_dict[layer_name]].tile_source=ts
    else:
        #p.renderers.append(TileRenderer(tile_source=ts))
        p.add_tile(ts, smoothing = True, alpha=overlay_alpha.value, level='overlay')
        for layer_key in vstate.layer_dict.keys():
            if layer_key=='rect':
                continue
            grp=tg.get_grp()
            ts=make_ts(f'http://127.0.0.1:5000/layer/{layer_key}/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg', mpp=vstate.mpp)
            p.renderers[vstate.layer_dict[layer_key]].tile_source=ts
        vstate.layer_dict[layer_name]=len(p.renderers)-1

    print(vstate.layer_dict)
    print(p.renderers)

def slide_toggle_cb(attr):
    print('meep')
    if p.renderers[0].alpha==0:
        p.renderers[0].alpha=slide_alpha.value
    else:
        p.renderers[0].alpha=0.0
    
#print(p.renderers[0].alpha)
def overlay_toggle_cb(attr):
    print('meep')
    for i in range(2,len(p.renderers)):
        if p.renderers[i].alpha==0:
            p.renderers[i].alpha=overlay_alpha.value
        else:
            p.renderers[i].alpha=0.0

def folder_input_cb(attr, old, new):
    file_list=[]
    for ext in ['*.svs','*ndpi','*.tiff']:#,'*.png','*.jpg']:
        file_list.extend(list(Path(new).glob('*\\'+ext)))
    file_list=[(str(p),str(p)) for p in file_list]
    file_drop.menu=file_list
    
    file_list=[]
    for ext in ['*.db','*.dat','*.geojson','*.png','*.jpg','*.tiff']:
        file_list.extend(list(Path(new).glob('*\\'+ext)))
    file_list=[(str(p),str(p)) for p in file_list]
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
    change_tiles('overlay')
    #proc=Thread(target=change_tiles, args=('overlay',))
    #proc.start()
    #proc.join()

def cprop_input_cb(attr, old, new):
    resp = requests.get(f'http://127.0.0.1:5000/changeprop/{new}')
    change_tiles('overlay')

slide_toggle.on_click(slide_toggle_cb)
#slide_toggle.js_on_click(setmax)
overlay_toggle.on_click(overlay_toggle_cb)
filter_input.on_change('value', filter_input_cb)
cprop_input.on_change('value', cprop_input_cb)

def slide_alpha_cb(attr,old,new):
    print('meep')
    p.renderers[0].alpha=new
    #p.renderers[0].tile_source.max_zoom=7
    #p.renderers[1].tile_source.max_zoom=7

def overlay_alpha_cb(attr,old,new):
    print('meep')
    for i in range(2,len(p.renderers)):
        p.renderers[i].alpha=new

def cmap_drop_cb(attr):
    resp = requests.get(f'http://127.0.0.1:5000/changecmap/{attr.item}')
    change_tiles('overlay')

def file_drop_cb(attr):
    """setup the newly chosen slide"""
    if len(p.renderers)>2:
        for r in p.renderers[2:]:
            p.renderers.remove(r)
    #p.extra_x_ranges={'oldx': p.x_range}
    vstate.layer_dict={'slide':0,'rect':1}
    vstate.slide_path=attr.item
    for c in color_column.children.copy():
        if '_slider' in c.name:
            color_column.children.remove(c)
    for b in box_column.children.copy():
        if 'layer' in b.label:
            box_column.children.remove(b)
    print(p.renderers)
    print(attr.item)
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
    print(attr.item)
    fname='-*-'.join(attr.item.split('\\'))
    print(fname)
    resp = requests.get(f'http://127.0.0.1:5000/changeoverlay/{fname}')
    print(resp)
    if resp.text=='overlay':
        update_mapper()
        initialise_overlay()
    else:
        add_layer(resp.text)
    change_tiles(resp.text)
    #change_tiles('slide')

def layer_select_cb(attr):
    build_predicate()
    change_tiles('overlay')

def fixed_layer_select_cb(obj, attr):
    print(vstate.layer_dict)
    key=vstate.layer_dict[obj.label]
    if p.renderers[key].alpha==0:
        p.renderers[key].alpha=overlay_alpha.value
    else:
        p.renderers[key].alpha=0.0

def layer_slider_cb(obj, attr, old, new):
    p.renderers[vstate.layer_dict[obj.name.split('_')[0]]].alpha=new

def color_input_cb(obj, attr, old, new):
    print(new)
    vstate.mapper[name2type_key(obj.name)]=(*hex2rgb(new), 255)
    if vstate.renderer.score_prop=='type':
        vstate.renderer.mapper=lambda x: vstate.mapper[x]
    change_tiles('overlay')
    
def bind_cb_obj(cb_obj, cb):
    def wrapped(attr, old, new):
        cb(cb_obj, attr, old, new)

    return wrapped

def bind_cb_obj_tog(cb_obj, cb):
    def wrapped(attr):
        cb(cb_obj, attr)

    return wrapped

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
    inst_segmentor.ioconfig.save_resolution['resolution'] = vstate.mpp
    for res in inst_segmentor.ioconfig.input_resolutions:
        res['resolution'] = vstate.mpp
    for res in inst_segmentor.ioconfig.output_resolutions:
        res['resolution'] = vstate.mpp

    tile_output = inst_segmentor.predict(
        [vstate.slide_path],
        [mask],
        save_dir="sample_tile_results/",
        mode="wsi",
        resolution=vstate.mpp,
        units='mpp',
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
    


slide_alpha.on_change('value', slide_alpha_cb)
overlay_alpha.on_change('value', overlay_alpha_cb)
folder_input.on_change('value', folder_input_cb)
cmap_drop.on_click(cmap_drop_cb)
file_drop.on_click(file_drop_cb)
to_model_button.on_click(segment_on_box)
#layer_folder_input.on_change('value', layer_folder_input_cb)
layer_drop.on_click(layer_drop_cb)

#layer_folder_input_cb(None, None, base_folder)
folder_input_cb(None, None, base_folder)

box_column=column(children=layer_boxes)
color_column=column(children=lcolors)
l=layout(
    [
        [p, [
            folder_input,
            file_drop,
            layer_drop,
            row([slide_toggle, slide_alpha]),
            row([overlay_toggle, overlay_alpha]),
            filter_input,
            cprop_input, 
            cmap_drop, 
            to_model_button,
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

curdoc().add_root(l)
curdoc().on_session_destroyed(cleanup_session)
