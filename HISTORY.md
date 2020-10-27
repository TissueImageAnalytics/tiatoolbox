History
=======

0.4.0 (2020-10-25)
------------------

### Major and Feature Improvements

- Adds `OpenSlideWSIReader` to read Openslide image formats
- Adds support to read Omnyx jp2 images using `OmnyxJP2WSIReader`.
- New feature added to perform stain normalisation using `Ruifork`, `Reinhard`, `Vahadane`, `Macenko` methods and using custom stain matrices.
- Adds example notebook to read whole slide images via the toolbox.
- Adds `WSIMeta` class to save meta data for whole slide images. `WSIMeta` casts properties to python types. Properties from OpenSlide are returned as string. raw values can always be accessed via `slide.raw`. Adds data validation e.g., checking that level_count matches up with the length of the `level_dimensions` and `level_downsamples`. Adds type hints to `WSIMeta`.
- Adds exceptions `FileNotSupported` and `MethodNotSupported`


### Changes to API

-  Restructures `WSIReader` as parent class to allow support to read whole slide images in other formats.
- Adds `slide_info` as a property of `WSIReader`
- Updates `slide_info` type to `WSIMeta` from `dict`
- Depericiates support for multiprocessing from within the toolbox. The toolbox is focussed on processing single whole slide and standard images. External libraries can be used to run using multi processing on multiple files.

### Bug Fixes and Other Changes

- Adds `scikit-learn`, `glymur` as a dependency
- Adds licence information
- Removes `pathos` as a dependency
- Updates `openslide-python` requirement to 1.1.2

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

