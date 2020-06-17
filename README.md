<p align="center">
  <img width="450" height="200" src="https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox/files/tialab_logo.png">
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

Please try

    python -m tiatoolbox -h

Getting Started
===============

First, install OpenSlide [here](https://openslide.org/download/). Then,
create and activate the conda environment:

pip
---

    pip install -r requirements.txt

conda
-----

    conda env create --name tiatoolbox --file requirements.conda.yml
    conda activate tiatoolbox

tiatoolbox --help
=================

    usage: tiatoolbox [-h] [--version] [--verbose VERBOSE]
                     {slide_info}
                     ...

    positional arguments:
      {slide_info}
        slide_info          usage: python -m tiatoolbox slide_info -h

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program`s version number and exit
      --verbose VERBOSE
