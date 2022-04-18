from shutil import rmtree
from bokeh.models import Selection, TileRenderer
from numpy import source
from tiatoolbox.wsicore.wsireader import get_wsireader
from PIL import Image
from bokeh.models import Plot, ColumnDataSource, Panel, Slider, Toggle, FileInput, TextInput, Button, Dropdown, BoxEditTool, CheckboxGroup, ColorPicker
from bokeh.layouts import layout, row, column
from pathlib import Path
from bokeh.plotting import figure
from bokeh.models.tiles import WMTSTileSource
from cmath import pi
from tiatoolbox.tools.pyramid import AnnotationTileGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer, random_colors
from tiatoolbox.annotation.storage import AnnotationStore, SQLiteStore, DictionaryStore, Annotation
from tiatoolbox.annotation.dsl import SQLTriplet, SQL_GLOBALS
#from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
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

# Pandas for data management
import pandas as pd
from pathlib import Path

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs

def make_ts(route):
  sf=1
  ts=WMTSTileSource(name="WSI provider", url=route, attribution="")
  ts.tile_size=256
  ts.initial_resolution=40211.5*sf*(2/(100*pi))   #156543.03392804097    40030 great circ
  ts.x_origin_offset=0#5000000
  #ts.y_origin_offset=-2500000
  ts.y_origin_offset=10247680*sf*(2/(100*pi))  + 438.715 +38.997+13-195.728  #10160000,   509.3
  ts.wrap_around=False
  ts.max_zoom=9
  #ts.min_zoom=10
  return ts

def build_predicate():
    #pred1=eval("props['type']=='class1'", SQL_GLOBALS, {})
    #pred2=eval("props['type']=='class2'", SQL_GLOBALS, {})
    preds=[eval(f"props['type']=='{l.label}'", SQL_GLOBALS, {}) for l in layer_boxes if l.active]
    combo=None
    if len(preds)>0:       
        combo=preds[0]
        for pred in preds[1:]:
            combo=SQLTriplet(combo, operator.or_, pred)
    if filter_input.value=='None':
        return combo

    return SQLTriplet(eval(filter_input.value, SQL_GLOBALS, {}), operator.and_, combo)
    

def initialise_slide():
    vstate.mpp=wsi.info.mpp

def initialise_overlay():
    vstate.colors=[]

class ViewerState():
    def __init__(self):
        self.dims=[]
        self.mpp=None
        self.mapper={'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)}
        self.colors=list(self.mapper.values())
        self.types=list(self.mapper.keys())

vstate=ViewerState()

geo_path=Path(r'E:\TTB_vis_test\ff7d5488-60e1-4c6e-9eb4-d495bb7565b1\TCGA-SC-AA5Z-01Z-00-DX1-2.geojson')
slide_path=Path(r'E:\TTB_vis_test\ff7d5488-60e1-4c6e-9eb4-d495bb7565b1\TCGA-SC-AA5Z-01Z-00-DX1.svs')

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
wsi = WSIReader.open(slide_path)
renderer=AnnotationRenderer('type', {'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)})#, prop_check)
#renderer=AnnotationRenderer('score', ['red', 'blue', 'green'], prop_check)
#renderer=AnnotationRenderer('score', ['red', 'blue', 'bob'])
dims=wsi.info.slide_dimensions
mpp=wsi.info.mpp[0]
vstate.dims=dims
vstate.mpp=mpp

def run_app(geo_path, wsi):
    SQ=SQLiteStore.from_geojson(geo_path)
    n=len(SQ)
    for i,key in enumerate(SQ.keys()):
        #SQ.patch(key, properties={'score': np.random.rand(1)[0]})
        SQ.patch(key, properties=rand_props(i/n))

    SQ.append(Annotation(Point(np.array([12000,10000])),properties=rand_props(0.8)))
    tile_gen=AnnotationTileGenerator(wsi.info, SQ, renderer)
    vstate.types=SQ.query_property("props['type']",[0,0,*dims],distinct=True)

    app = TileServer(
        title="Testing TileServer",
        layers={
        "slide": wsi,
        "overlay": tile_gen #(wsi, SQ)
        },
        state=vstate,
        )
    CORS(app, send_wildcard=True)   
    app.run(threaded=False)

proc=Thread(target=run_app, args=(geo_path,wsi,), daemon=True)
proc.start()

TOOLTIPS=[
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("Scores", "[@c1, @c2]"),
    ]

#SQ = SQLiteStore()
ts1=make_ts(r'http://127.0.0.1:5000/layer/slide/zoomify/TileGroup1/{z}-{x}-{y}.jpg')
ts2=make_ts(r'http://127.0.0.1:5000/layer/overlay/zoomify/TileGroup1/{z}-{x}-{y}.jpg')

p = figure(x_range=(0, dims[0]/3.5), y_range=(0,-dims[1]/3.5),x_axis_type="linear", y_axis_type="linear",
width=1500,height=900, tooltips=TOOLTIPS, lod_factor=20,output_backend="webgl")
print(p.renderers)
p.add_tile(ts1)
p.add_tile(ts2)
p.grid.grid_line_color=None
p.match_aspect=True
box_source=ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})
r=p.rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
p.add_tools(BoxEditTool(renderers=[r], num_objects=1))

slide_alpha = Slider(
    title="Adjust alpha WSI",
    start=0,
    end=1,
    step=0.05,
    value=1.0
)

overlay_alpha = Slider(
        title="Adjust alpha Overlay",
        start=0,
        end=1,
        step=0.05,
        value=0.8
    )

slide_toggle = Toggle(label="Show Slide", button_type="success")
overlay_toggle = Toggle(label="Show Overlay", button_type="success")
#toggle3 = Button(label="Submit Filter", button_type="success")
#toggle4 = Button(label="Submit score_prop", button_type="success")

filter_input = TextInput(value='props["type"] == "class1"', title="Filter:")
cprop_input = TextInput(value="type", title="CProp:")
folder_input = TextInput(value="folder", title="Img Folder:")
cmmenu=[('jet','jet'),('coolwarm', 'coolwarm'),('dict',"{'class1': (255,0,0,255), 'class2': (0,0,255,255), 'class3': (0,255,0,255)}")]
cmap_drop = Dropdown(label="Colourmap", button_type="warning", menu=cmmenu)
file_drop = Dropdown(label="File", button_type="warning", menu=[None])
to_model_button=Button(label="To Model", button_type="success")
type_drop=Dropdown(label="Types", button_type="warning", menu=[None])
layer_boxes=[Toggle(label=t, active=True, width=100) for t in vstate.types]
lcolors=[ColorPicker(color=col[0:3], width=60) for col in vstate.colors]

class TileGroup():
    def __init__(self):
        self.group=1
    
    def get_grp(self):
        self.group=self.group+1
        return self.group
tg=TileGroup()
layer_list=['slide','overlay']

def change_tiles(layer_num=1):

    #removing the renderer corresponding to the tile layer
    #p.renderers = [x for x in p.renderers if not str(x).startswith('TileRenderer')]
    ##inserting the new tile renderer
    grp=tg.get_grp()
    #ts1=make_ts(f'http://127.0.0.1:5000/layer/slide/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg')
    #p.add_tile(ts1, smoothing=False)
    ts2=make_ts(f'http://127.0.0.1:5000/layer/{layer_list[layer_num]}/zoomify/TileGroup{grp}' +r'/{z}-{x}-{y}.jpg')
    #p.add_tile(ts2, smoothing=False)
    p.renderers[layer_num].tile_source=ts2

def slide_toggle_cb(attr):
    print('meep')
    if p.renderers[0].alpha==0:
        p.renderers[0].alpha=slide_alpha.value
    else:
        p.renderers[0].alpha=0.0
    
#print(p.renderers[0].alpha)
def overlay_toggle_cb(attr):
    print('meep')
    if p.renderers[1].alpha==0:
        p.renderers[1].alpha=overlay_alpha.value
    else:
        p.renderers[1].alpha=0.0

def filter_input_cb(attr, old, new):
    resp = requests.get(f'http://127.0.0.1:5000/changepredicate/{new}')
    #change_tiles()
    proc=Thread(target=change_tiles)
    proc.start()
    proc.join()

def cprop_input_cb(attr, old, new):
    resp = requests.get(f'http://127.0.0.1:5000/changeprop/{new}')
    change_tiles()

slide_toggle.on_click(slide_toggle_cb)
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
    p.renderers[1].alpha=new 
    print(p.renderers[1].alpha)

def folder_input_cb(attr, old, new):
    file_list=[]
    for ext in ['*.svs','*ndpi','*.tiff','*.png','*.jpg']:
        file_list.extend(list(Path(new).glob(ext)))
    file_list=[(str(p),str(p)) for p in file_list]
    file_drop.menu=file_list

def cmap_drop_cb(attr):
    resp = requests.get(f'http://127.0.0.1:5000/changecmap/{attr.item}')
    change_tiles()

def file_drop_cb(attr):
    print(attr.item)
    fname='-*-'.join(attr.item.split('\\'))
    print(fname)
    resp = requests.get(f'http://127.0.0.1:5000/changeslide/slide/{fname}')
    change_tiles(0)
    wsi = WSIReader.open(slide_path)

#run NucleusInstanceSegmentor on a region of wsi defined by the box in box_source
def segment_on_box(attr):
    print(vstate.types)
    thumb=wsi.slide_thumbnail()
    #conv_mpp=wsi.convert_resolution_units(1.25, 'power', 'mpp')[0]
    conv_mpp=0.25*dims[0]/thumb.shape[1]
    x=int((box_source.data['x'][0]-0.5*box_source.data['width'][0])/conv_mpp)
    y=-int((box_source.data['y'][0]+0.5*box_source.data['height'][0])/conv_mpp)
    width=int(box_source.data['width'][0]/conv_mpp)
    height=int(box_source.data['height'][0]/conv_mpp)
    print(x,y,width,height)


    #img_tile=wsi.read_rect((x,y),(width,height))
    #print(img_tile.shape)
    mask=np.zeros((thumb.shape[0],thumb.shape[1]))
    mask[y:y+height,x:x+width]=1

    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=2,
        num_postproc_workers=2,
        batch_size=4,
    )
    inst_segmentor.ioconfig.save_resolution['resolution'] = mpp
    for res in inst_segmentor.ioconfig.input_resolutions:
        res['resolution'] = mpp
    for res in inst_segmentor.ioconfig.output_resolutions:
        res['resolution'] = mpp

    tile_output = inst_segmentor.predict(
        [slide_path],
        [mask],
        save_dir="sample_tile_results/",
        mode="wsi",
        resolution=mpp,
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
    colors = random_colors(len(vstate.types))
    vstate.mapper = {key: (*color, 1) for key, color in zip(vstate.types, colors)}
    print(vstate.mapper)
    renderer.mapper = lambda x: vstate.mapper[x]
    type_drop.menu=[(str(t),str(t)) for t in vstate.types]
    change_tiles(1)
    rmtree(r'./sample_tile_results')

    return tile_output
    


slide_alpha.on_change('value', slide_alpha_cb)
overlay_alpha.on_change('value', overlay_alpha_cb)
folder_input.on_change('value', folder_input_cb)
cmap_drop.on_click(cmap_drop_cb)
file_drop.on_click(file_drop_cb)
to_model_button.on_click(segment_on_box)

l=layout(
    [
        [p, [slide_alpha,
            slide_toggle,
            overlay_alpha,
            overlay_toggle,
            folder_input,
            filter_input,
            cprop_input,
            file_drop, 
            cmap_drop, 
            to_model_button,
            type_drop,
            row(children=[column(layer_boxes),column(lcolors)])]
        ],
    ]
)

curdoc().add_root(l)
