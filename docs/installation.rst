.. highlight:: shell

************
Installation
************

Prerequisites
--------------
The prerequisites for tiatoolbox installation are OpenSlide binaries and OpenJpeg version 2.3.0 or above.
Please follow the instructions below to install prerequisite software according to the platform you are using.

Using Anaconda (Recommended)
============================

After `installing Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_ (or miniconda), you can create a virtual environment for TIA toolbox using the following command:

.. code-block:: console

    $ conda env create --file requirements.conda.yml


Alternative Method
==================

If you cannot use Anaconda or are having trouble with it, you can try an alternative install method. We will install prerequisite binary packages and then use pip (the Python package manager) to install python dependencies.

Windows
^^^^^^^
1. Download OpenSlide binaries from `this page <https://openslide.org/download/>`_. Extract the folder and add `bin` and `lib` subdirectories to
Windows `system path <https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)>`_.

2. Install OpenJPEG. The easiest way is to install OpenJpeg is through conda
using

.. code-block:: console

    C:\> conda install -c conda-forge openjpeg>=2.3.0

Linux (Ubuntu)
^^^^^^^^^^^^^^
On Linux the prerequisite software can be installed using the command

.. code-block:: console

    $ apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

The same command is used when working on the Colab or Kaggle platforms.
When working on Google Colab, we remove the packages `datascience` and `albumentations` because they conflict
and produce an error message.

macOS
^^^^^

On macOS there are two popular package managers, `homebrew`_ and `macports`_.

.. _homebrew: https://brew.sh/
.. _macports: https://www.macports.org/

Homebrew
""""""""

.. code-block:: console

    $ brew install openjpeg openslide

MacPorts
""""""""

.. code-block:: console

    $ port install openjpeg openslide


Stable release
--------------

Please note that TIAToolbox is tested for python version 3.7, 3.8 and 3.9.
To install TIA Toolbox, run this command in your terminal after you have installed the pre-requisite software:

.. code-block:: console

    $ pip install tiatoolbox

This is the preferred method to install TIA Toolbox, as it will always install the most recent stable release.

To upgrade an existing version of tiatoolbox to the latest stable release, run this command in your terminal:

.. code-block:: console

    $ pip install --ignore-installed --upgrade tiatoolbox

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for TIA Toolbox can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/tialab/tiatoolbox

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/tialab/tiatoolbox/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/tialab/tiatoolbox
.. _tarball: https://github.com/tialab/tiatoolbox/tarball/master
