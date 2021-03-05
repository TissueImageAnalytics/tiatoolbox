<p align="center">
  <img src="docs/tia_logo.png">
</p>
<h1 align="center">TIA Toolbox</h1>
<p align="center">
  <a href="https://tia-toolbox.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/tia-toolbox/badge/?version=latest" alt="Documentation Status" />
  </a>
  <a href="https://travis-ci.org/TIA-Lab/tiatoolbox">
    <img src="https://travis-ci.org/TIA-Lab/tiatoolbox.svg?branch=master" alt="Travis CI Status" />
  </a>
  <a href="https://badge.fury.io/py/tiatoolbox">
    <img src="https://badge.fury.io/py/tiatoolbox.svg" alt="PyPI Status" />
  </a>
</p>

Computational Pathology Toolbox developed by TIA Lab

# Getting Started

First, install

-   OpenSlide [here](https://openslide.org/download/).
-   OpenJPEG [here](https://www.openjpeg.org/), using
    `conda install -c conda-forge openjpeg>=2.3.0`.

## Git

1.Create the virtual environment

    conda env create -f requirements.conda.yml

or (on Windows)

    conda env create -f requirements.win64.conda.yml

or

    pip install -r requirements.txt

2.Clone the tiatoolbox

    git clone https://github.com/TIA-Lab/tiatoolbox.git

3.Please try

    python -m tiatoolbox -h

## pip

    pip install tiatoolbox

## conda

    conda env create -n tiatoolbox python=3.{6,7,8}
    conda activate tiatoolbox
    pip install tiatoolbox

# tiatoolbox --help

    usage: tiatoolbox [-h] [--version] [--verbose VERBOSE]
                     {read-bounds,
                      save-tiles,
                      slide_info,
                      slide-thumbnail,
                      stainnorm,
                      }
                     ...

    positional arguments:
      {slide_info}

    read-bounds         usage: tiatoolbox read-bounds -h
    save-tiles          usage: tiatoolbox save-tiles -h
    slide-info          usage: tiatoolbox slide-info -h
    slide-thumbnail     usage: tiatoolbox slide-thumbnail -h
    stainnorm           usage: tiatoolbox stainnorm -h

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program`s version number and exit
      --verbose             VERBOSE

# License

The source code TIA Toolbox (tiatoolbox) as hosted on GitHub is released under the [GNU General Public License (Version 3)].

The full text of the licence is included in [LICENSE.md](https://raw.githubusercontent.com/TIA-Lab/tiatoolbox/develop/LICENSE.md).

[gnu general public license (version 3)]: https://www.gnu.org/licenses/gpl-3.0.html

## Auxiliary Files

Auxiliary files, such as pre-trained model weights downloaded from the TIA Lab webpage (https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox), are provided under the [Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Dual License

If you would like to use any of the source code or auxiliary files (e.g. pre-trained model weights) under a different license agreement please contact the Tissue Image Analytics (TIA) Lab at the University of Warwick (tialab@dcs.warwick.ac.uk).
