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
###### Computational Pathology Toolbox developed by TIA Lab

# Prerequisites

This package is aimed at begining graduate students and, hopefully, at medical staff interested in digital pathology. Please send comments and criticisms to **tialab@dcs.warwick.ac.uk**

We assume some knowledge of the following topics, though, initially, it may be sufficient to glance through a brief introduction using Google.

1. **`Anaconda`** is a distribution of the Python language containing many useful Python packages. To offset its many advantages, it's quite large (about 2Gb). But that is not too much space for a modern home computer.
2. **`Conda`**is an app, found in Anaconda, giving convenient management of large software packages.
3. When working on a computer, one works in a certain *environment*, in which certain commands and procedures make sense. When working on large **Python** projects, such as ours, it is often convenient to work in a *virtual environment*. These virtual environments are controlled by `conda`. 
4. Github: the site where this repository is found.
5. `git, a language giving control of different versions of the same program. A modern novelist might find it useful for keeping different versions of a novel, but usually it's used for software development.

Getting Started
===============

1. Make a directory (also known as a *folder*) into which you will download this repositor. You can call it anything you like. We will call it tb.
2. Navigate to the directory `tb`
3. Clone (download a copy) of the official repository for `tiatoolbox`using the typed command
   `conda clone https://github.com/TIA-Lab/tiatoolbox.git` 
4. Change into the new subdirectory `tb/tiatoolbox`.
5. Create the virtual environment `tiatoolbox` with
   `conda env create -f requirements.conda.yml`
   (The name `tiatoolbox` for the environment is hidden inside the requirements file.)
6. `conda activate tiatoolbox`
7. Check success with `condo list`, which lists all the Python packages in the `tiatoolbox` virtual environment and also the version of Python that will be used in that environment.

The above steps have installed the package `tiatoolbox`and created a function, also called `tiatoolbox`, whose usage is given with the command
`tiatoolbox --help`.




License
=======

The source code TIA Toolbox (tiatoolbox) as hosted on GitHub is released under the [GNU General Public License (Version 3)].

The full text of the licence is included in [LICENSE.md](https://raw.githubusercontent.com/TIA-Lab/tiatoolbox/develop/LICENSE.md).

[GNU General Public License (Version 3)]: https://www.gnu.org/licenses/gpl-3.0.html


Auxiliary Files
---------------

Auxiliary files, such as pre-trained model weights downloaded from the TIA Lab webpage (https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox), are provided under the [Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).


Dual License
------------

If you would like to use any of the source code or auxiliary files (e.g. pre-trained model weights) under a different license agreement please contact the Tissue Image Analytics (TIA) Lab at the University of Warwick (tialab@dcs.warwick.ac.uk).
