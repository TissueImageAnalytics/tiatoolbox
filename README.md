<p align="center">
  <img src="https://raw.githubusercontent.com/TIA-Lab/tiatoolbox/develop/docs/tia_logo.png">
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
    <a href="https://codecov.io/gh/TIA-Lab/tiatoolbox">
      <img src="https://codecov.io/gh/TIA-Lab/tiatoolbox/branch/master/graph/badge.svg?token=7UZEMacQHm"/>
    </a>
</p>

Computational Pathology Toolbox developed at the TIA Centre

## Getting Started

### All Users

This package is for those interested in digital pathology: including graduate students, medical staff, members of the TIA Centre and of PathLAKE, and anyone, anywhere, who may find it useful. We will continue to improve this package, taking account of developments in pathology, microscopy, computing and related disciplines. Please send comments and criticisms to **[tia@dcs.warwick.ac.uk](mailto:tialab@dcs.warwick.ac.uk)**.

**`tiatoolbox`** is a multipurpose name that we use for 1) a certain computer program, 2) a Python package of related programs, created by us at the TIA Centre to help people get started in Digital Pathology, 3) this repository, 4) a certain virtual environment.


### Developers

Anyone wanting to contribute to this repository, please first look at our [Wiki](https://github.com/TIA-Lab/tiatoolbox/wiki) and at our web page for [contributors](https://github.com/TIA-Lab/tiatoolbox/blob/master/CONTRIBUTING.rst). See also the *Prepare for development* section of this document.

### Links, if needed
The [bash](https://www.gnu.org/software/bash) shell is available on all commonly encountered platforms. Commands in this README are in bash. Windows users can use the command prompt to install conda and python packages.


[`conda`](https://github.com/conda/conda) is a management system for software packages and [virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html). To get `conda`, download [Anaconda](https://www.anaconda.com/), which includes hundreds of the most useful Python packages, using 2GB disk space. Alternatively, [miniconda](https://docs.conda.io/en/latest/miniconda.html) uses 400MB, and packages can be added as needed.

[Github](https://github.com/about) is powered by the version control system [git](https://git-scm.com/), which has many users and uses. In Github, it is used to track versions of code and other documents.


### Examples Taster

1. [Click here](https://github.com/TIA-Lab/tiatoolbox/tree/develop/examples) for [jupyter notebooks](https://jupyter.org/), hosted on the web, with demos of `tiatoolbox`. All necessary resources to run the notebooks are remotely provided, so you don't need to have Python installed on your computer.
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
1. Open a `bash` terminal window<br/>
`cd <future-home-of-tiatoolbox-directory>` <br/>
2. Download a complete copy of the `tiatoolbox`.<br/>`conda clone https://github.com/TIA-Lab/tiatoolbox.git`<br/>
3. `cd tiatoolbox`
4. `conda env create -f requirements.conda.yml`<br/>
creates the virtual environment `tiatoolbox`. Details in the text file `requirements.conda.yml`
5. To use the packages installed in the environment, run the command:<br/>`conda activate tiatoolbox`

### License

The source code TIA Toolbox (tiatoolbox) as hosted on GitHub is released under the [GNU General Public License (Version 3)].

The full text of the licence is included in [LICENSE.md](https://raw.githubusercontent.com/TIA-Lab/tiatoolbox/develop/LICENSE.md).

[gnu general public license (version 3)]: https://www.gnu.org/licenses/gpl-3.0.html

### Auxiliary Files

Auxiliary files, such as pre-trained model weights downloaded from the TIA Centre webpage (https://warwick.ac.uk/fac/cross_fac/tia/), are provided under the [Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Dual License

If you would like to use any of the source code or auxiliary files (e.g. pre-trained model weights) under a different license agreement please contact the Tissue Image Analytics (TIA) Centre at the University of Warwick (tia@dcs.warwick.ac.uk).
