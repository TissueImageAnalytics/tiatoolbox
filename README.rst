.. raw:: html

   <p align="center">
     <img width="450" height="200" src=docs/tialab_logo.png>
   </p>

tiatoolbox-private
==================

Computational Pathology Toolbox developed by TIA Lab

Please try

::

    python -m tiatoolbox -h

Getting Started
===============

First, install OpenSlide `here <https://openslide.org/download/>`__. For
Windows, extract the OpenSlide binaries at
*C:\\tools\\openslide\\openslide-win64-20171122*. Then, create and
activate the conda environment:

::

    conda env create --name tiatoolbox --file requirements.conda.yml
    conda activate tiatoolbox

python tiatoolbox.py -h
=======================

::

    usage: tiatoolbox.py [-h] [--version] [--verbose VERBOSE]
                     {read_region,generate_tiles,extract_patches,merge_patches,slide_info}
                     ...

    positional arguments:
      {read_region,generate_tiles,extract_patches,merge_patches,slide_info}
        read_region         usage: python tiatoolbox.py read_region -h
        generate_tiles      usage: python tiatoolbox.py generate_tiles -h
        extract_patches     usage: python tiatoolbox.py extract_patches -h
        merge_patches       usage: python tiatoolbox.py merge_patches -h
        slide_info          usage: python tiatoolbox.py slide_info -h

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      --verbose VERBOSE

