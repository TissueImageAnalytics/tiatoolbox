.. _visualization:

Visualization interface
=======================

Launching the interface
-----------------------

Start the interface using the command::

    tiatoolbox visualize --img-input path\to\slides --img-input path\to\overlays

alternatively just one path can be provided; in this case it is assumed that slides and overlays are in subdirectories of that provided directory called 'slides' and 'overlays' respectively::

    tiatoolbox visualize --img-input path\to\parent_of_slides_and _overlays

In the folder(s) that your command pointed to, should be the things that you want to visualize, following the conventions in the next section.

.. _data_format:

Data Format Conventions and File Structure
------------------------------------------

in the slides folder should be all the slides you want to use, and the overlays folder should contain whatever graphs, segmentations, heatmaps etc you are interesting in overlaying over the slides.

When a slide is selected in the interface, any valid overlay file that can be found that *contains the same name* (not including extension) will be available to overlay upon it.

Segmentation
^^^^^^^^^^^^

The best way of getting segmentations (in the form of contours) into the visualization is by putting them in an AnnotationStore (more information about hte tiatoolbox annotation store can be found at :obj:`storage <tiatoolbox.annotation.storage>`.  The other options are .geojson, or a hovernet -style .dat, both of which can usually be loaded within the interface but will incur a small delay while the data in converted internally into an AnnotationStore.

If your annotations are in a geojson format following the sort of thing QuPath would output, that should be ok. Contours stored following hovernet-style output in a .dat file should also work. An overview of the data structure in these formats is below.

Hovernet style::

    sample_dict = {nuc_id: {
                    box: List[],
                    centroid: List[],
                    contour: List[List[]],
                    prob: float,
                    type: int
        ... #can add as many additional properties as we want...
                    }
    ... # other instances
    }


geojson::

    {"type":"Feature",
    "geometry":{
        "type":"Polygon",
        "coordinates":[[[21741, 49174.09],[21737.84, 49175.12],[21734.76, 49175.93],[21729.85, 49179.85],[21726.12, 49184.84],[21725.69, 49187.95],[21725.08, 49191],[21725.7, 49194.04],[21726.15, 49197.15],[21727.65, 49199.92],[21729.47, 49202.53],[21731.82, 49204.74],[21747.53, 49175.23],[21741, 49174.09]]]},
        "properties":{"object_type":"detection","isLocked":false}
    }


If your data is not in one of these formats, it is usually fairly straightforward to build an annotation store out of your model outputs.\
A small script of 6-10 lines is usually all that is required. There are example code snippets illustrating how to create an annotation store in a variety of common scenarios in the examples section.
Most use-cases should be covered in there, or something close enough that a few tweaks to a snippet will do what is needed.

Heatmaps
^^^^^^^^

will display a low-res heatmap in .jpg or .png format. Should be the same aspect ratio as the WSI it will be overlaid on. When creating the image, keep in mind that white regions (255,255,255) will be made transparent.

Single channel images can also be used but are not recommended; they should take values between 0 and 255 and will simply be put through a viridis colormap. 0 values will become white background.

Whole Slide Overlays
^^^^^^^^^^^^^^^^^^^^

Can overlay multiple WSI's on top of eachother as separate layers

Graphs
^^^^^^

Graphs can also be overlaid. These should be provided in a dictionary format, saved as a .json file.
eg::

    graph_dict = {  'edge_index': 2 x n_edges array of indices of pairs of connected nodes
		'coordinates': n x 2 array of x,y coordinates for each graph node
		}


Additional features can be added to nodes by adding extra keys to the dictionary, eg:

::

    graph_dict = {  'edge_index': 2 x n_edges array of indices of pairs of connected nodes
    'coordinates': n x 2 array of x,y coordinates for each graph node
    'feats': n x n_features array of features for each node
    'feat_names': list n_features names for each feature
    }


It will be possible to colour the nodes by these features in the interface, and the top 10 will appear in a tooltip when hovering over a node (you will have to turn on the hovertool in the small toolbar to the right of the main window to enable this, it is disabled by default.)
The display of nodes and edges can be toggled on/off independently in the right hand panel of the interface (note, edges will be turned off by default).


.. _examples:

Annotation Store examples
-------------------------

Patch Predictions
^^^^^^^^^^^^^^^^^

lets say you have patch level predictions for a model. The top left corner
of each patch, and two predicted scores are in a .csv file. Patch size is 512.

::

    results_path = Path("path/to/results.csv")
    SQ = SQLiteStore()
    patch_df = pd.read_csv(results_path)
    annotations = []
    for i, row in patch_df.iterrows():
        x = row["x"]
        y = row["y"]
        properties = {"score1": row["score1"], "score2": row["score2"]}
        annotations.append(
            Annotation(Polygon.from_bounds(x, y, x + 512, y + 512), properties=properties)
        )
    SQ.append_many(annotations)
    SQ.dump("path/to/filename.db")   # filename should contain its associated slides name

When loading the above in the interface, you will be able to select any of the properties to colour the overlay by.

geojson outputs
^^^^^^^^^^^^^^^

While .geojson files can be loaded in the interface directly, it is often more convenient to convert them to a .db file first, as this will avoid the delay while the geojson is converted to an annotation store.
The tiatoolbox AnnotationStore class provides a method to do this.

::

    geojson_path = Path("path/to/annotations.geojson")
    SQ1 = SQLiteStore.from_geojson(geojson_path)
    SQ1.dump("path/to/annotations.db")

Raw contours and properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a collection of raw centroids or detection contours with corresponding properties/scores, you can easily convert these to an annotation store.

::

    centroid_list = [[1, 4], [3, 2]] # etc...
    # if its contours each element is a list of points instead
    properties_list = [
        {"score": "some_score", "class": "some_class"},
        {"score": "other _score", "class": "other_class"},
        # etc...
    ]

    annotations = []

    for annotation, properties in zip(centroid_list, properties_list):
        props = {"score": properties["score"], "type": properties["class"]}
        annotations.append(
            Annotation(Point(annotation), props)
        )  # use Polygon() instead if its a contour
    SQ.append_many(annotations)
    SQ.create_index("area", '"area"')  # create index on area for faster querying
    SQ.dump("path/to/annotations.db")

Note that in the above we saved the 'class' property as 'type' - this is because the UI treats the 'type' property as a special property, and will allow you to toggle annotations of a specific type on/off, in addition to other functionality.

Graphs example
^^^^^^^^^^^^^^

Lets say you have a graph defined by nodes and edges,
and associated node properties. The following example demonstrates how to package this into a .json file

::

    graph_dict = {  'edge_index': 2 x n_edges array of indices of pairs of connected nodes
        'coordinates': n x 2 array of x,y coordinates for each graph node
        'feats': n x n_features array of features for each node
        'feat_names': list n_features names for each feature
        }

    with open("path/to/graph.json", "w") as f:
        json.dump(graph_dict, f)

Modifying an existing annotation store
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have an existing annotation store and want to add/change
properties of annotations (or can also do similarly for geometry)

::

    # lets assume you have calculated a score in some way, that you want to add to
    # the annotations in a store
    scores = [0.9, 0.5]

    SQ = SQLiteStore("path/to/annotations.db")
    # use the SQLiteStore.patch_many method to replace the properties dict
    # for each annotation.
    new_props = {}
    for i, (key, annotation) in enumerate(SQ.items()):
        new_props[key] = annotation.properties  # get existing props
        new_props[key]["score"] = scores[i]  # add the new score

    SQ.patch_many(
        SQ.keys(), properties_iter=new_props
    )  # replace the properties dict for each annotation

Merging two annotation stores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The interface will only open one annotation store at a time. If you have annotations
belonging to the same slide in different stores that you want to display
at the same time, just put them all in the same store as follows

::

    SQ1 = SQLiteStore("path/to/annotations1.db")
    SQ2 = SQLiteStore("path/to/annotations2.db")
    anns = list(SQ1.items())
    SQ2.append_many(anns)  # SQ2 .db file now contains all annotations from SQ1 too

Shifting coordinates
^^^^^^^^^^^^^^^^^^^^

Lets say you have some annotations that were created on a slide, and you want to grab the annotations in a particular region and display them on a tile from that slide. You will need their coordinates to be relative to the tile. You can do this as follows

::

    top_left = [2048, 1024]  # top left of tile
    tile_size = 1024  # tile size
    SQ1 = SQLiteStore("path/to/annotations.db")
    query_geom = Polygon.from_bounds(
        top_left[0], top_left[1], top_left[0] + tile_size, top_left[1] + tile_size
    )
    SQ2 = SQLiteStore()
    tile_anns = SQ1.query(query_geom) # get all annotations in the tile
    SQ2.append_many(tile_anns.values(), tile_anns.keys()) # add them to a new store


    def translate_geom(geom):
        return geom.translate(-top_left[0], -top_left[1])


    SQ2.transform(translate_geom)  # translate so coordinates relative to top left of tile
    SQ2.dump("path/to/tile_annotations.db")

.. _interface:

General UI Controls and Options
-------------------------------

.. image:: images/visualize_interface.png
    :width: 100%
    :align: center
    :alt: visualize interface

Colormaps/colouring by score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have selected a slide with the slide dropdown, you can overlays by repeatedly choosing files containing overlays from the overlay drop menu. They will be put on there as separate layers. In the case of segmentations, if your segmentations have the 'type' property as one of their properties, this can additionally be used to show/hide annotations of that specific type. Colors can be individually selected for each type also if the randomly-generated colour scheme is not suitable.

You can select the property that will be used to colour annotations in the colour_by box. The corresponding property should be either categorical (strings or ints), in which case a dict-based colour mapping should be used, or a float between 0-1 in which case a matplotlib colourmap should be applied.
There is also the option for the special case 'color' to be used - if your annotations have a property called color, this will be assumed to be an rgb value in the form of a tuple (r, g, b) of floats between 0-1 for each annotation which will be used directly without any mapping.

The 'colour type by property' box allows annotations of the specified type to be coloured by a different property to the 'global' one. For example, this could be used to have all detections coloured according to their type, but for Glands, colour by some feature describing them instead (their area, for example)

Running models
^^^^^^^^^^^^^^

Regions of the image can be selected, using either a box select or points, which can be sent to a model via selecting the model in the drop-down menu and then clicking go. Available so far are hovernet and nuclick.

To save the annotations resulting from a model, or loaded from a .geojson or .dat (will be saved as a SQLiteStore .db file which will be far quicker to load) use the save button (for the moment it is just saved in a file '{slide_name}\_saved_anns.db' in the overlays folder).

Dual window mode
^^^^^^^^^^^^^^^^

A second window can be opened by selecting the 'window 2' tab in the top right. This will open the currently selected slide in a second window. The overlay shown in each window can be controlled independently to allow comparison of different overlays, or viewing of a model output side-by-side with the unoverlaid slide, or ground truth annotations. Slide navigation will be linked between both windows.
Two different slides can also be opened in the two windows, although this will only be useful in cases where the two slides are registered so that a shared coordinate space/slide navigation makes sense.

Inspecting annotations
^^^^^^^^^^^^^^^^^^^^^^

Annotations can be inspected by double clicking on them. This will open a popup showing the annotation in more detail, and allowing the properties to be viewed in a sortable table.

Zoomed out plotting
^^^^^^^^^^^^^^^^^^^

By default, the interface is set up to show only larger annotations while zoomed out. Smaller annotations which would be too small to see clearly while zoomed out will not be displayed. The 'max-scale' value can be changed to control the zoom level at which this happens. A larger value will mean smaller annotations remain visible at more zoomed out scale. If you want all annotations to be displayed always regardless of zoom, just type in a large value (1000+) to set it to its max. In the case of very many annotations, this may result in some loading lag when zoomed out.

Other options
^^^^^^^^^^^^^

There are a few options for how annotations are displayed. You can change the colourmap used in the colormap field if you are colouring objects according to a continuous property (should be between 0-1) - by entering the text of a matplotlib cmap.
The buttons 'filled', 'mpp', 'grid', respectively toggle between filled and outline only rendering of annotations, using mpp or baseline pixels as the scale for the plot, and showing a grid overlay.

A filter can be applied to annotations using the filter box. For example, entering props\['score'\]>0.5 would show only annotations for which the 'score' property  is greater than 0.5.
See the annotation store documentation on valid 'where' statements for more details.

.. _config:

Config files
------------

A json config file can be placed in the overlays folder, to customize various aspects of the UI and annotation display when visualizing overlays in that location. This is especially useful for customising online demos. An example .json explaining all the fields is shown below.

There are settings to control how slides are loaded:

::

    {
    "initial_views": {
        "slideA": [0,19000,35000,44000],    # if a slide with specified name is opened, initial view window will be set to this
        "slideB": [44200,59100,69700,76600]
            },
    "auto_load": 1,     # if 1, upon opening a slide will also load all annotations associated with it
    "first_slide": "slideA.svs",            # initial slide to open upon launching viewer

Settings to control how annotations are displayed, including default colours for specific types, and default properties to colour by:

::

    "colour_dict": {
        "typeA": [252, 161, 3, 255],   # annotations whose 'type' property matches these, will display in the specified color
        "typeB": [3, 252, 40, 255]
    },
    "default_cprop": "some_property",     # default property to color annotations by
    "default_type_cprop": {               # a property to colour a specific type by
    "type": "Gland",
    "cprop": "Explanation"
    },

There are settings to control the initial values of some UI settings:

::

    "UI_settings": {
        "blur_radius": 0,           # applies a blur to rendererd annotations
        "edge_thickness": 0,        # thickness of boundaries drawn around annotation geometries (0=off)
        "mapper": "jet",            # default colormapper to use when coloring by a continuous property
        "max_scale": 32             # controls zoom level at which small annotations are no longer rendered (larger val->smaller
    },                              # annotations visible when zoomed out)
    "opts": {
        "edges_on": 0,              # graph edges are shown or hidden by default
        "nodes_on": 1,              # graph nodes are shown or hidden by default
        "colorbar_on": 1,           # whether colorbar is shown below main window
        "hover_on": 1
    },

and the ability to toggle on or off specific UI elements:

::

    "UI_elements_1": {              # controls which UI elements are visible
        "slide_select": 1,          # slide select box
        "layer_drop": 1,            # overlay select drop down
        "slide_row": 1,             # slide alpha toggle and slider
        "overlay_row": 1,           # overlay alpha toggle and slider
        "filter_input": 1,          # filter text input box
        "cprop_input": 1,           # box to select which property to color annotations by ('color by' box)
        "cmap_row": 1,              # row of UI elements with colormap select, blur, max_scale
        "type_cmap_select": 1,      # UI element to select a secondary colormap for a specific type (i.e 'color type by' box)
        "model_row": 0,             # UI elements to chose and run a model
        "type_select_row": 1        # buttom group for toggling specific types of annotations on/off
    },

::

    "UI_elements_2": {              # controls visible UI elements on second tab in UI
        "opt_buttons": 1,           # UI elements providing a few options including if annotations should be filled/outline only
        "pt_size_spinner": 1,       # control for point size and graph node size
        "edge_size_spinner": 1,     # control for edge thickness
        "res_switch": 1,            # allows to switch to lower res tiles for faster loading
        "mixing_type_select": 1,    # select mixing type for multi-property cmap builder
        "cmap_builder_input": 1,    # property select box for multi-prop cmap builder
        "cmap_picker_column": 1,    # controls color chosen for each property in multi-prop cmap
        "cmap_builder_button": 1    # button to build the multi-prop cmap
    }
    }
