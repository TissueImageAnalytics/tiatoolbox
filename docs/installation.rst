.. highlight:: shell

.. _installation:

Installation
************

Prerequisites
=============

The prerequisites for TIAToolbox installation are OpenSlide binaries and OpenJPEG version 2.3.0 or above. Please follow the instructions below to install the prerequisite software according to your platform.

Linux (Ubuntu)
--------------

On Linux, the prerequisite software can be installed using the following command:

.. code-block:: console

    $ apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

The same command is used when working on the Colab or Kaggle platforms. When working on Google Colab, we remove the packages ``datascience`` and ``albumentations`` because they conflict and produce an error message.

Windows (10+)
-------------

1. Download OpenSlide binaries from `this page <https://openslide.org/download/>`_. Extract the folder and add the ``bin`` and ``lib`` subdirectories to the Windows `system path <https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)>`_. If you are using a conda environment, you can also copy the ``bin`` and ``lib`` subdirectories to ``[Anaconda Installation Path]/envs/[tiatoolbox-environment]/Library/``.

2. Install OpenJPEG. The easiest way to install OpenJPEG is through conda:

.. code-block:: console

    C:\> conda install -c conda-forge openjpeg

macOS
-----

On macOS, there are two popular package managers, `homebrew`_ and `macports`_.

.. _homebrew: https://brew.sh/
.. _macports: https://www.macports.org/

Homebrew
^^^^^^^^

.. code-block:: console

    $ brew install openjpeg openslide

MacPorts
^^^^^^^^

.. code-block:: console

    $ port install openjpeg openslide

Installing Stable Release
=========================

Please note that TIAToolbox is tested for Python versions 3.9, 3.10, 3.11, and 3.12.

Recommended
-----------

To install TIAToolbox, run this command in your terminal after you have installed the prerequisite software:

.. code-block:: console

    $ pip install tiatoolbox

This is the preferred method to install TIAToolbox, as it will always install the most recent stable release.

Upgrade
-------

To upgrade an existing version of TIAToolbox to the latest stable release, run this command in your terminal:

.. code-block:: console

    $ pip install --ignore-installed --upgrade tiatoolbox

Without Dependencies
--------------------

If you already have a Python environment set up with all the prerequisite software and dependencies installed and you would like to keep the existing versions of these dependencies, run this command in your terminal:

.. code-block:: console

    $ pip install --no-deps tiatoolbox

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Alternative Methods
===================

Using Anaconda
--------------

After installing `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_ (or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ / `mamba <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#mamba-vs-conda-clis>`_), you can install TIAToolbox using the following command:

.. code-block:: console

    $ conda install -c conda-forge tiatoolbox

or

.. code-block:: console

    $ mamba install tiatoolbox

Please note that conda-forge installation support is limited on Windows as OpenSlide binaries are not supported on official conda channels. An alternate way to install using conda on Windows could be to install it in `WSL2 with CUDA support <https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl>`_. In some cases, WSL2 runs faster on Python code, and therefore we **recommend** this option.

From Sources
------------

The sources for TIAToolbox can be downloaded from the `GitHub repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/TissueImageAnalytics/tiatoolbox.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/TissueImageAnalytics/tiatoolbox/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

.. _GitHub repo: https://github.com/TissueImageAnalytics/tiatoolbox.git
.. _tarball: https://github.com/TissueImageAnalytics/tiatoolbox/tarball/master

Using Docker
------------

To run TIAToolbox in an isolated environment, use our `Docker image <https://github.com/tissueimageanalytics/tiatoolbox-docker/pkgs/container/tiatoolbox>`_. We host different Dockerfiles in our GitHub repository `tiatoolbox-docker <https://github.com/TissueImageAnalytics/tiatoolbox-docker>`_. Please report any issues related to the Docker image in the repository `tiatoolbox-docker <https://github.com/TissueImageAnalytics/tiatoolbox-docker>`_.

After `installing Docker <https://docs.docker.com/get-docker/>`_ (or Docker Desktop), you can use our TIAToolbox image in three different ways.

Use the Pre-Built Docker Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TIAToolbox provides pre-built Docker containers which can be downloaded using the instructions below:

1. Pull the Image From GitHub Container Registry:

.. code-block:: console

    $ docker pull ghcr.io/tissueimageanalytics/tiatoolbox:latest

2. Use the Pre-Built Docker Image as a Base Image in a Dockerfile:

.. code-block:: dockerfile

    FROM ghcr.io/tissueimageanalytics/tiatoolbox:latest

Build the Image Locally
^^^^^^^^^^^^^^^^^^^^^^^

1. Navigate to the Dockerfile that you want to use, based on the Python version and Operating System that you prefer.

2. Build the Docker image:

.. code-block:: console

    $ docker build -t <IMAGE_NAME> .

3. Check that the image has been created:

.. code-block:: console

    $ docker images

4. Deploy the image as a Docker container:

.. code-block:: console

    $ docker run -it --rm --name <CONTAINER_NAME> <IMAGE_NAME>

5. Connect to the running container:

.. code-block:: console

    $ docker exec -it <CONTAINER_NAME> bash

To add your own script and run it through the Docker container, first copy your script into the Docker environment and then execute it:

.. code-block:: dockerfile

    COPY /path/to/<script>.py .
    CMD ["python3", "<script>.py"]
