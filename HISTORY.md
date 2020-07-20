History
=======

0.3.0 (2020-07-19)
------------------

### Major and Feature Improvements

- Adds feature `read_region` to read a small region from whole slide images
- Adds feature `save_tiles` to save image tiles from whole slide images
- Adds feature `imresize` to resize images
- Adds feature `transforms.background_composite` to avoid creation of black tiles from whole slide images.

### Changes to API

- None

### Bug Fixes and Other Changes

- Adds `pandas` as dependency

------------------
0.2.2 (2020-07-12)
------------------

### Major and Feature Improvements

-   None

### Changes to API

-   None

### Bug Fixes and Other Changes

-   Fix command line interface for `slide-info` feature and travis pypi deployment

------------------
0.2.1 (2020-07-10)
------------------

### Major and Feature Improvements

-   None

### Changes to API

-   None

### Bug Fixes and Other Changes

-   Minor changes to configuration files.

------------------
0.2.0 (2020-07-10)
------------------

### Major and Feature Improvements

-   Adds feature slide\_info to read whole slide images and display meta
    data information
-   Adds multiprocessing decorator TIAMultiProcess to allow running
    toolbox functions using multiprocessing.

### Changes to API

-   None

### Bug Fixes and Other Changes

-   Adds Sphinx Readthedocs support
    <https://readthedocs.org/projects/tia-toolbox/> for stable and
    develop branches
-   Adds code coverage tools to test the pytest coverage of the package
-   Adds deepsource integration to highlight and fix bug risks,
    performance issues etc.
-   Adds README to allow users to setup the environment.
-   Adds conda and pip requirements instructions

------------------
0.1.0 (2020-05-28)
------------------

-   First release on PyPI.

