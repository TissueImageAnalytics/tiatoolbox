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
On Linux the preprequiste software can be installed using a single command

.. code-block:: console

    $ apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

This is the same command used in Google Colab examples. Although Google Colab examples remove `datascience`
and `albumentations` due to compatibility issues.

MacOS
^^^^^


Stable release
--------------

To install TIA Toolbox, run this command in your terminal:

.. code-block:: console

    $ pip install tiatoolbox

This is the preferred method to install TIA Toolbox, as it will always install the most recent stable release.

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
