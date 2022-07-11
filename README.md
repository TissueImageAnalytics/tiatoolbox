This visualization tool is in the process of being added to tiatoolbox, but for the moment is not usable there. This tiatoolbox fork makes it available, but it is a work in progress and will probably have a few issues. If you find one let me know! also, if there is any sort of visualization task you'd like to do which seems like it might fit in this tool but which can't be easily done with it at the moment, please suggest it.

## Setup

Install tiatoolbox into a conda environment as normal from this fork.

Install a couple of additional dependencies with:

pip install bokeh
pip install flask-cors

enter command:
python setup.py install    while in the cloned tiatoolbox top directory.

start the interface using:

tiatoolbox visualize --img-input path\to\slides --img-input path\to\overlays
alternatively just one path can be provided; in this case it is assumed that slides and overlays are in subdirectories of that provided directory called 'slides' and 'overlays' respectively.

In the folder(s) that your command pointed to, should be the things that you want to visualize, following the conventions in the next section.

## Data format conventions/filestructure

in the slides folder should be all the slides you want to use, and the overlays folder should contain whatever graphs, segmentations, heatmaps etc you are interesting in overlaying over the slides.

When a slide is selected in the interface, any valid overlay file that can be found that contains the same name (not including extension) will be available to overlay upon it. 

### Segmentation:

The best way of getting segmentations (in the form of contours) into the visualization is by putting them in an AnnotationStore. The other options are .geojson, and .dat.

If your annotatins are in a geojson format following the sort of thing QuPath would output, that should also be ok. Contours stored following hovernet-style output in a .dat file should also work

Hovernet style:
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




geojson:
{"type":"Feature",
"geometry":{
	"type":"Polygon",	
	"coordinates":[[[21741, 49174.09],[21737.84, 49175.12],[21734.76, 49175.93],[21729.85, 49179.85],[21726.12, 49184.84],[21725.69, 49187.95],[21725.08, 49191],[21725.7, 49194.04],[21726.15, 49197.15],[21727.65, 49199.92],[21729.47, 49202.53],[21731.82, 49204.74],[21747.53, 49175.23],[21741, 49174.09]]]},
	"properties":{"object_type":"detection","isLocked":false}
}

### Heatmaps:

will display a low-res heatmap in .jpg or .png format. Should be the same aspect ratio as the WSI it will be overlaid on.

### Whole Slide Overlays:

Can overlay multiple WSI's on top of eachother as separate layers

### Graphs:

Graphs can also be overlaid. Should be in a dictionary format, saved as a pickled .pkl file.
eg:

graph_dict = {  'edge_index': 2 x n_edges array of indices of pairs of connected nodes
		'coordinates': n x 2 array of x,y coordinates for each graph node
		}


## Other stuff:

### colormaps/colouring by score:

You can select the property that will be used to colour annotations in the colour_prop box. The corresponding property should be either categorical (strings or ints), in which case a dict-based colour mapping will be used, or a float between 0-1 in which case a matplotlib colourmap will be applied.
There is also the option for the special case 'color' to be used - if your annotations have a property called color, this will be assumed to be an rgb value for each annotation which will be used directly without any mapping.

Once you have selected a slide with the slide dropdown, you can add any number of overlays by repeatedly choosing files containing overlays from the overlay drop menu. They will be put on there as separate layers. In the case of segmentations, if your segmentations have the 'type' property as one of their properties, this can additionally be used to show/hide annotations of that specific type. Colors can be individually selected for each type also if the randomly-generated colour scheme is not suitable.

### Running models:

Regions of the image can be selected, using either a box select or points, which can be sent to a model via selecting the model in the drop-down menu and then clicking go. Available so far are hovernet and nuclick. 

To save the annotations resulting from a model, or loaded from a .geojson or .dat (will be saved as a SQLiteStore .db file which will be far quicker to load) use the save button (for the moment it is just saved in a file '{slide_name}_saved_anns.db' in the overlays folder). 

### Other options:

There are a few options for how annotations are displayed. You can change the colourmap used in the colormap field if you are colouring objects according to a continuous property (should be between 0-1) - by entering the text of a matplotlib cmap.
The buttons 'filled', 'mpp', 'grid', respectively toggle between filled and outline only rendering of annotations, using mpp or baseline pixels as the scale for the plot, and showing a grid overlay.



<p align="center">
  <img src="https://raw.githubusercontent.com/TissueImageAnalytics/tiatoolbox/develop/docs/tiatoolbox-logo.png">
</p>
<h1 align="center">TIA Toolbox</h1>
<p align="center">
  <a href="https://tia-toolbox.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/tia-toolbox/badge/?version=latest" alt="Documentation Status" />
  </a>
  <a href="https://travis-ci.com/github/TissueImageAnalytics/tiatoolbox">
    <img src="https://app.travis-ci.com/TissueImageAnalytics/tiatoolbox.svg?branch=master" alt="Travis CI Status" />
  </a>
    <a href="https://codecov.io/gh/TissueImageAnalytics/tiatoolbox">
      <img src="https://codecov.io/gh/TissueImageAnalytics/tiatoolbox/branch/master/graph/badge.svg?token=7UZEMacQHm"/>
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg"/>
    </a>
    <a href="https://github.com/TissueImageAnalytics/tiatoolbox/tree/master#license">
      <img src="https://img.shields.io/badge/license-BSD--3--clause-orange" />
    </a>
  <br>
    <br>
  <a href="https://badge.fury.io/py/tiatoolbox">
    <img src="https://badge.fury.io/py/tiatoolbox.svg" alt="PyPI Status" />
  </a>
    <a href="https://pepy.tech/project/tiatoolbox">
      <img src="https://static.pepy.tech/personalized-badge/tiatoolbox?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads"/>
    </a>
    <br>
    <a href="https://anaconda.org/conda-forge/tiatoolbox">
      <img src="https://img.shields.io/conda/vn/conda-forge/tiatoolbox" />
    </a>
    <a href="https://anaconda.org/conda-forge/tiatoolbox">
        <img src="https://anaconda.org/conda-forge/tiatoolbox/badges/downloads.svg" />
    </a>

  <br>
    <br>
  <a href="https://doi.org/10.1101/2021.12.23.474029"><img src="https://img.shields.io/badge/bioRxiv-10.1101%2F2021.12.23.474029-blue" alt="DOI"></a>
  <a href="https://zenodo.org/badge/latestdoi/267705904"><img src="https://zenodo.org/badge/267705904.svg" alt="DOI"></a>
</p>

Computational Pathology Toolbox developed at the TIA Centre

## Getting Started

### All Users

This package is for those interested in digital pathology: including graduate students, medical staff, members of the TIA Centre and of PathLAKE, and anyone, anywhere, who may find it useful. We will continue to improve this package, taking account of developments in pathology, microscopy, computing and related disciplines. Please send comments and criticisms to **[tia@dcs.warwick.ac.uk](mailto:tialab@dcs.warwick.ac.uk)**.

**`tiatoolbox`** is a multipurpose name that we use for 1) a certain computer program, 2) a Python package of related programs, created by us at the TIA Centre to help people get started in Digital Pathology, 3) this repository, 4) a certain virtual environment.


### Developers

Anyone wanting to contribute to this repository, please first look at our [Wiki](https://github.com/TissueImageAnalytics/tiatoolbox/wiki) and at our web page for [contributors](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/CONTRIBUTING.rst). See also the *Prepare for development* section of this document.

### Links, if needed
The [bash](https://www.gnu.org/software/bash) shell is available on all commonly encountered platforms. Commands in this README are in bash. Windows users can use the command prompt to install conda and python packages.

[conda](https://github.com/conda/conda) is a management system for software packages and [virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html). To get `conda`, download [Anaconda](https://www.anaconda.com/), which includes hundreds of the most useful Python packages, using 2GB disk space. Alternatively, [miniconda](https://docs.conda.io/en/latest/miniconda.html) uses 400MB, and packages can be added as needed.

[Github](https://github.com/about) is powered by the version control system [git](https://git-scm.com/), which has many users and uses. In Github, it is used to track versions of code and other documents.


### Examples Taster

1. [Click here](https://github.com/TissueImageAnalytics/tiatoolbox/tree/develop/examples) for [jupyter notebooks](https://jupyter.org/), hosted on the web, with demos of `tiatoolbox`. All necessary resources to run the notebooks are remotely provided, so you don't need to have Python installed on your computer.
2. Click on a filename with suffix `.ipynb` and the notebook will open in your browser.
3. Click on one of the two blue checkboxes in your browser window labelled either **Open in Colab** or **Open in Kaggle**: [colab](https://colab.research.google.com/notebooks/intro.ipynb#) and [kaggle](https://www.kaggle.com/) are websites providing free-of-charge platforms for running jupyter notebooks.
4. Operate the notebook in your browser, editing, inserting or deleting cells as desired.
5. Changes you make to the notebook will last no longer than your colab or kaggle session.

### Install Python package

If you wish to use our programs, perhaps without developing them further, run the command `pip install tiatoolbox` or `pip install --ignore-installed --upgrade tiatoolbox` to upgrade from an existing installation.
Detailed installation instructions can be found in the [documentation](https://tia-toolbox.readthedocs.io/en/latest/installation.html).

To understand better how the programs work, study the jupyter notebooks referred to under the heading **Examples Taster**.

### Command Line
tiatoolbox supports various features through command line. For more information, please try `tiatoolbox --help`

### Prepare for development

Prepare a computer as a convenient platform for further development of the Python package `tiatoolbox` and related programs as follows.
1. Install [pre-requisite software](https://tia-toolbox.readthedocs.io/en/latest/installation.html)
2. Open a terminal window<br/>

```sh
    $ cd <future-home-of-tiatoolbox-directory>
```

3. Download a complete copy of the `tiatoolbox`.

```sh
    $ git clone https://github.com/TissueImageAnalytics/tiatoolbox.git
```

4. Change directory to `tiatoolbox`

```sh
    $ cd tiatoolbox
```

5. Create virtual environment for TIAToolbox using

```sh
    $ conda env create -f requirements.dev.conda.yml # for linux/mac only.
    $ conda activate tiatoolbox-dev
```
or

```sh
    $ conda create -n tiatoolbox-dev python=3.8 # select version of your choice
    $ conda activate tiatoolbox-dev
    $ pip install -r requirements_dev.txt
```
6. To use the packages installed in the environment, run the command:

```sh
    $ conda activate tiatoolbox-dev
```


### License

The source code TIA Toolbox (tiatoolbox) as hosted on GitHub is released under the [The 3-Clause BSD License].

The full text of the licence is included in [LICENSE](https://raw.githubusercontent.com/TissueImageAnalytics/tiatoolbox/develop/LICENSE).

[The 3-Clause BSD License]: https://opensource.org/licenses/BSD-3-Clause

### Auxiliary Files

Auxiliary files, such as pre-trained model weights downloaded from the TIA Centre webpage (https://warwick.ac.uk/tia/), are provided under the [Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Dual License

If you would like to use any of the source code or auxiliary files (e.g. pre-trained model weights) under a different license agreement please contact the Tissue Image Analytics (TIA) Centre at the University of Warwick (tia@dcs.warwick.ac.uk).
