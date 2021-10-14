History
=======

0.7.0 (2021-09-16)
------------------
### Major and Feature Improvements
- Drops support for python 3.6
- Update minimum requirement to python 3.7
- Adds support for python 3.9
- Adds `models` base to the repository. Currently, PyTorch models are supported. New custom models can be added. The tiatoolbox also supports using custom weights to pre-existing built-in models.
  - Adds `classification` package and CNNPatchPredictor which takes predefined model architecture and pre-trained weights as input. The pre-trained weights for classification using kather100k data set is automatically downloaded if no weights are provided as input.
- Adds mask-based patch extraction functionality to extract patches based on the regions that are highlighted in the `input_mask`. If `'auto'` option is selected, a tissue mask is automatically generated for the `input_image` using tiatoolbox `TissueMasker` functionality.
- Adds visualisation module to overlay the results of an algorithm.

### Changes to API
- Command line interface for stain normalisation can be called using the keyword `stain-norm` instead of `stainnorm`
- Replaces `FixedWindowPatchExtractor` with `SlidingWindowPatchExtractor` .
- get_patchextractor takes the `slidingwindow` as an argument.
- Depreciates `VariableWindowPatchExtractor`

### Bug Fixes and Other Changes
- Significantly improved python notebook documentation for clarity, consistency and ease of use for non-experts.
- Adds detailed installation instructions for Windows, Linux and Mac

### Development related changes
- Moves flake8 above pytest in the `travis.yml` script stage.
- Adds `set -e` at the start of the script stage in `travis.yml` to cause it to exit on error and (hopefully) not run later parts of the stage.
- Readthedocs related changes
  - Uses `requirements.txt` in `.readthedocs.yml`
  - Uses apt-get installation for openjpeg and openslide
  - Removes conda build on readthedocs build
- Adds extra checks to pre-commit, e.g., import sorting, spellcheck etc. Detailed list can be found on this [commit](https://github.com/TIA-Lab/tiatoolbox/commit/662a143e915fa55416badd992d8e7358211730a6).


0.6.0 (2021-05-11)
------------------
### Major and Feature Improvements
- Add `TissueMasker` class to allow tissue masking using `Otsu` and `Morphological` processing.
- Add helper/convenience method to WSIReader(s) to produce a mask. Add reader object to allow reading a mask conveniently as if it were a WSI i.e., use same location and resolution to read tissue area and mask area.
- Add `PointsPatchExtractor` returns patches that can be used by classification models. Takes `csv`, `json` or `pd.DataFrame` and returns patches corresponding to each pixel location.
- Add feature `FixedWindowPatchExtractor` to run sliding window deep learning algorithms.
- Add example notebooks for patch extraction and tissue masking.
- Update readme with improved instructions to use the toolbox. Make the README file somewhat more comprehensible to beginners, particularly those with not much background or experience.

### Changes to API
- `tiatoolbox.dataloader` replaced by `tiatoolbox.wsicore`

### Bug Fixes and Other Changes
- Minor bug fixes

### Development-related changes
- Improve unit test coverage.
- Move test data to tiatoolbox server.


------------------

0.5.2 (2021-03-12)
------------------
### Bug Fixes and Other Changes
- Fix URL for downloading test JP2 image.
- Update readme with new logo.

------------------

0.5.1 (2020-12-31)
------------------
### Bug Fixes and Other Changes
- Add `scikit-image` as dependency in `setup.py`
- Update notebooks to add instructions to install dependencies

------------------
0.5.0 (2020-12-30)
------------------
### Major and Feature Improvements

- Adds `get_wsireader()` to return appropriate WSIReader.
- Adds new functions to allow reading of regions using WSIReader at different resolutions given in units of:
  - microns per-pixel (mpp)
  - objective lens power (power)
  - pixels-per baseline (baseline)
  - resolution level (level)
- Adds functions for reading regions are `read_bounds` and `read_rect`.
  - `read_bounds` takes a tuple (left, top, right, bottom) of coordinates in baseline (level 0) reference frame and returns a region bounded by those.
  - `read_rect` takes one coordinate in baseline reference frame and an output size in pixels.
- Adds `VirtualWSIReader` as a subclass of WSIReader which can be used to read visual fields (tiles).
  - `VirtualWSIReader`  accepts ndarray or image path as input.
- Adds MPP fall back to standard TIFF resolution tags  with warning.
  - If OpenSlide cannot determine microns per pixel (`mpp`) from the metadata, checks the TIFF resolution units (TIFF tags: `ResolutionUnit`, `XResolution` and  `YResolution`) to calculate MPP. Additionally, add function to estimate missing objective power if MPP is known of derived from TIFF resolution tags.
- Estimates missing objective power from MPP with warning.
- Adds example notebooks for stain normalisation and WSI reader.
- Adds caching to slide info property. This is done by checking if a private `self._m_info` exists and returning it if so, otherwise `self._info` is called to create the info for the first time (or to force regenerating) and the result is assigned to `self._m_info`. This could in future be made much simpler with the `functools.cached_property` decorator in Python 3.8+.
- Adds pre processing step to stain normalisation where stain matrix encodes colour information from tissue region only.

### Changes to API
- `read_region` refactored to be backwards compatible with openslide arguments.
- `slide_info` changed to `info`
- Updates WSIReader which only takes one input
- `WSIReader` `input_path` variable changed to `input_img`
- Adds `tile_read_size`, `tile_objective_value` and `output_dir` to WSIReader.save_tiles()
- Adds `tile_read_size` as a tuple
- `transforms.imresize` takes additional arguments `output_size` and interpolation method 'optimise' which selects `cv2.INTER_AREA` for `scale_factor<1` and `cv2.INTER_CUBIC` for `scale_factor>1`

### Bug Fixes and Other Changes
- Refactors glymur code to use index slicing instead of deprecated read function.
- Refactors thumbnail code to use `read_bounds` and be a member of the WSIReader base class.
- Updates `README.md` to clarify installation instructions.
- Fixes slide_info.py for changes in WSIReader API.
- Fixes save_tiles.py for changes in WSIReader API.
- Updates `example_wsiread.ipynb` to reflect the changes in WSIReader.
- Adds Google Colab and Kaggle links to allow user to run notebooks directly on colab or kaggle.
- Fixes a bug in taking directory input for stainnorm operation for command line interface.
- Pins `numpy<=1.19.3` to avoid compatibility issues with opencv.
- Adds `scikit-image` or `jupyterlab` as a dependency.

### Development related changes
- Moved `test_wsireader_jp2_save_tiles` to test_wsireader.py.
- Change recipe in Makefile for coverage to use pytest-cov instead of coverage.
- Runs travis only on PR.
- Adds [pre-commit](https://pre-commit.com/#install) for easy setup of client-side git [hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) for [black code formatting](https://github.com/psf/black#version-control-integration) and flake8 linting.
- Adds [flake8-bugbear](https://github.com/PyCQA/flake8-bugbear) to pre-commit for catching potential deepsource errors.
- Adds constants for test regions in `test_wsireader.py`.
- Rearranges `usage.rst` for better readability.
- Adds `pre-commit`, `flake8`, `flake8-bugbear`, `black`, `pytest-cov` and `recommonmark` as dependency.


------------------
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
- Depreciates support for multiprocessing from within the toolbox. The toolbox is focused on processing single whole slide and standard images. External libraries can be used to run using multiprocessing on multiple files.

### Bug Fixes and Other Changes

- Adds `scikit-learn`, `glymur` as a dependency
- Adds licence information
- Removes `pathos` as a dependency
- Updates `openslide-python` requirement to 1.1.2

------------------
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

