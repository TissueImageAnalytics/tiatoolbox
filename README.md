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

Computational Pathology Toolbox developed by TIA Centre

# All Users

This package is for those interested in digital pathology: including graduate students, medical staff, members of the Tia Centre and of Pathlake, and anyone, anywhere, who may find it useful. We will continue to improve this package,people taking account of developments in pathology, microscopy, computing and related disciplines. Please send comments and criticisms to **[tialab@dcs.warwick.ac.uk](mailto:tialab@dcs.warwick.ac.uk)**.

**`tiatoolbox`** is a multipurpose name that we use for 1) a certain computer program, 2) a collection of related programs, created by us at the Tia Centre to help peopleon get started in Digital Pathology, 3) this repository, 4) a certain virtual environment.


# Developers

Anyone wanting to contribute to this repository, please consult **XXX**, for our naming conventions and other useful information.

# Links, if needed
The [bash](https://www.gnu.org/software/bash) shell is available on all commonly encountered platforms. Windows users look [here](https://docs.microsoft.com/en-us/windows/wsl/about). Commands in this README are in bash.

[`conda`](https://github.com/conda/conda) is a management system for software packages and [virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html). To get `conda`, download [Anaconda](https://www.anaconda.com/), which includes hundreds of the most useful Python packages, using 2GB disk space. Alternatively, [miniconda](https://docs.conda.io/en/latest/miniconda.html) uses 400MB, and one adds packages as needed.

[Github](https://github.com/about) is powered by the version control system [git](https://git-scm.com/), which has many uses, for example to keep track of different drafts of a PhD thesis.


# Examples Taster

1. *__Without a single prior download or file edit__* [click here](https://github.com/TIA-Lab/tiatoolbox/tree/develop/examples) for [jupyter notebooks](https://jupyter.org/) providing `tiatoolbox` demos.
2. Click on a filename with suffix `.ipynb` and the notebook will open in your browser.
3.  Click on one of the two blue checkboxes in your browser window labelled either <span style="color:blue"> **Open in Colab**</span> or <span style="color:blue"> **Open in Kaggle**</span>. **Note** [colab](https://colab.research.google.com/notebooks/intro.ipynb#) and [kaggle](https://www.kaggle.com/) are websites providing free-of-charge platforms for running jupyter notebooks.
4. Operate the notebook in your browser, editing, inserting or deleting cells as desired.
5. Changes last no longer than your colab or kaggle session.

# Prepare computer for `tiatoolbox`

Preparation is needed to run the example jupyter notebooks (see above) on your computer. 
1. Open a `bash` terminal window and `cd`
to the directory where `tiatoolbox` will be located.
2. `conda clone https://github.com/TIA-Lab/tiatoolbox.git`<br/>
downloads a complete copy of the `tiatoolbox` repository to your computer.
3. `cd tiatoolbox`
4. `conda env create -f requirements.conda.yml`<br/>
creates the virtual environment `tiatoolbox`. Take a quick look at the contents of the text file `requirements.conda.yml` to get an idea of what the command installed.
4. To use the packages installed in the environment run the command:<br/>`conda activate tiatoolbox`<br/>
Subsequently `(tiatoolbox)` is prefixed to the bash prompt, until the command `conda deactivate`.
5. `tiatoolbox --help` gives info about how to use the program.




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
