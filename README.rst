.. raw:: html

   <p align="center">
     <img width="450" height="200" src=docs/tialab_logo.png>
   </p>

===========
TIA Toolbox
===========

Computational Pathology Toolbox developed by TIA Lab

Please try

::

    python -m tiatoolbox -h

Getting Started
===============

First, install OpenSlide `here <https://openslide.org/download/>`__. Then, create and
activate the conda environment:

pip
----

::

    pip install -r requirements_dev.txt

conda
-----
::

    conda env create --name tiatoolbox --file requirements.conda.yml
    conda activate tiatoolbox

tiatoolbox --help
=======================

::

    usage: tiatoolbox [-h] [--version] [--verbose VERBOSE]
                     {slide_info}
                     ...

    positional arguments:
      {slide_info}
        slide_info          usage: python -m tiatoolbox slide_info -h

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      --verbose VERBOSE

