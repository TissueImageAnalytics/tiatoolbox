.. highlight:: shell

============
Installation
============


Prerequisites
--------------
The prerequisites for tiatoolbox installation are OpenSlide binaries and OpenJpeg version 2.3.0 or above.
Please follow the instructions below to install prerequisite software on your respective platform.

Windows
^^^^^^^
1. Download OpenSlide binaries from `this page <https://openslide.org/download/>`_ and add it to
Windows `system path <https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)>`_.

2. Install OpenJPEG. The easiest way is to install OpenJpeg is through conda
using

.. code-block:: console

    C:\> conda install -c conda-forge openjpeg>=2.3.0

Linux (Ubuntu)
^^^^^^^^^^^^^^
On Linux the preprequiste software can be installed using the command

.. code-block:: console

    $ apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

The same command is used when working on the Colab or Kaggle platforms.
In addition, we remove the packages`datascience`and`albumentations`from Colab because they conflict
and produce an error message.

MacOS
^^^^^

TBA

Stable release
--------------

Please note that TIAToolbox is tested for python version 3.6, 3.7 and 3.8.
To install TIA Toolbox, run this command in your terminal:

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
