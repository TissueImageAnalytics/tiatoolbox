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

After `installing Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_ (or miniconda), you can install TIA toolbox using the following command:

.. code-block:: console

    $ conda install -c conda-forge tiatoolbox

Please note that conda-forge installation support is limited on Windows as openslide binaries are not supported on official conda channels. An alternate way to install using conda on Windows could be to install it in `WSL2 with CUDA support <https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl>`_.

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

    C:\> conda install -c conda-forge openjpeg

3. Install
TIAToolbox.

.. code-block:: console

    C:\> pip install tiatoolbox

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

Please note that TIAToolbox is tested for python version 3.8, 3.9 and 3.10.
To install TIA Toolbox, run this command in your terminal after you have installed the prerequisite software:

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

Using Docker
==================

To run TIA toolbox in an isolated environment, use our `Docker image <https://github.com/tissueimageanalytics/tiatoolbox-docker/pkgs/container/tiatoolbox>`_ . We host different Dockerfiles in our github repository `tiatoolbox-docker <https://github.com/TissueImageAnalytics/tiatoolbox-docker>`_. Please report any issues related to the docker image in the repository `tiatoolbox-docker <https://github.com/TissueImageAnalytics/tiatoolbox-docker>`.

After `installing Docker <https://docs.docker.com/get-docker/>`_ (or Docker Desktop), you can use our TIA toolbox image in 3 different ways.

Use the pre-built docker image
^^^^^^^^^^^^^^^^^^^^
1. Pull the image from the Github Container Registry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: console

    $ docker pull ghcr.io/tissueimageanalytics/tiatoolbox:latest

2. Use the pre-built Docker image as a base image in a Dockerfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: console

    $ FROM ghcr.io/tissueimageanalytics/tiatoolbox:latest

Build the image locally
^^^^^^^^^^^^^^^^^^^^^^^
1. Navigate to the Dockerfile that you want to use,
based on the Python version and Operating System that you prefer

2. Build the
Docker image

.. code-block:: console

    $ docker build -t <IMAGE_NAME> .

3. Check that the image
has been created

.. code-block:: console

    $ docker images

4. Deploy the image
as a Docker container

.. code-block:: console

    $ docker run -it --rm --name <CONTAINER_NAME> <IMAGE_NAME>

5. Connect to the
running container

.. code-block:: console

    $ docker exec -it <CONTAINER_NAME> bash

To add your own script and run it through the Docker container, first copy your script into the docker environment and then execute it.

.. code-block:: console

    $ COPY /path/to/<script>.py .
    $ CMD ["python3", "<script>.py"]
