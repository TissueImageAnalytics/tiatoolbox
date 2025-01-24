# History

## TIAToolbox v1.6.0 (2024-12-12)

### Major Updates and Feature Improvements

- **Foundation Models Support via `timm` API** (#856, contributed by @GeorgeBatch)
  - Introduced `TimmBackbone` for running additional PyTorch Image Models.
  - Tested models include `UNI`, `Prov-GigaPath`, and `H-optimus-0`.
  - Added an example notebook demonstrating feature extraction with foundation models.
  - `timm` added as a dependency.
- **Performance Enhancements with `torch.compile`** (#716)
  - Improved performance on newer GPUs using `torch.compile`.
- **Multichannel Input Support in `WSIReader`** (#742)
- **AnnotationStore Filtering for Patch Extraction** (#822)
- **Python 3.12 Support**
- **Deprecation of Python 3.8 Support**
- **CLI Response Time Improvements** (#795)

### API Changes

- **Device Specification Update** (#882)
  - Replaced `has_gpu` with `device` for specifying GPU or CPU usage, aligning with PyTorch's `Model.to()` functionality.
- **Windows Compatibility Enhancement** (#769)
  - Replaced `POWER` with explicit multiplication.

### Bug Fixes and Other Changes

- **TIFFWSIReader Bound Reading Adjustment** (#777)
  - Fixed `read_bound` to use adjusted bounds.
  - Reduced code complexity in `WSIReader` (#814).
- **Annotation Rendering Fixes** (#813)
  - Corrected rendering of annotations with holes.
- **Non-Tiled TIFF Support in `WSIReader`** (#807, contributed by @GeorgeBatch)
- **HoVer-Net Documentation Update** (#751)
  - Corrected class output information.
- **Citation File Fix for `cffconvert`** (#869, contributed by @Alon-Alexander)
- **Bokeh Compatibility Updates**
  - Updated `bokeh_app` for compatibility with `bokeh>=3.5.0`.
  - Switched from `size` to `radius` for `bokeh>3.4.0` compatibility (#796).
- **JSON Extraction Fixes** (#772)
  - Restructured SQL expression construction for JSON properties with dots in keys.
- **VahadaneExtractor Warning** (#871)
  - Added warning due to changes in `scikit-learn>0.23.0` dictionary learning (#382).
- **PatchExtractor Error Message Refinement** (#883)
- **Immutable Output Fix in `WSIReader`** (#850)

### Development-Related Changes

- **Mypy Checks Added**
  - Applied to `utils`, `tools`, `data`, `annotation`, and `cli/common`.
- **ReadTheDocs PDF Build Deprecation**
- **Formatter Update**
  - Replaced `black` with `ruff-format`.
- **Dependency Removal**
  - Removed `jinja2`.
- **Test Environment Update**
  - Updated to `Ubuntu 24.04`.
- **Conda Environment Workflow Update**
  - Implemented `micromamba` setup.
- **Codecov Reporting Fix** (#811)
  **Full Changelog:** https://github.com/TissueImageAnalytics/tiatoolbox/compare/v1.5.1...v1.6.0

______________________________________________________________________

## TIAToolbox v1.5.1 (2023-12-16)

### Development related changes

- Specifies compatible Python versions
  - Fixes `tiatoolbox-feedstock` build for conda-forge release #763

**Full Changelog:** https://github.com/TissueImageAnalytics/tiatoolbox/compare/v1.5.0...v1.5.1

______________________________________________________________________

## TIAToolbox v1.5.0 (2023-12-15)

### Major Updates and Feature Improvements

- Adds the bokeh visualization tool. #684
  - The tool allows a user to launch a server on their machine to visualise whole slide images, overlay the results of deep learning algorithms or to select a patch from whole slide image and run TIAToolbox deep learning engines.
  - This tool powers the TIA demos server. For details please see https://tiademos.dcs.warwick.ac.uk/.
- Extends Annotation to Support Init from WKB #639
- Adds `IOConfig` for NuClick in `pretrained_model.yaml` #709
- Adds functions to save the TIAToolbox Engine outputs to Zarr and AnnotationStore files. #724
- Adds Support for QuPath Annotation Imports #721

### Changes to API

- Adds `model.to(device)` and `model.load_model_from_file()` functionality to make it compatible with PyTorch API. #733
- Replaces `pretrained` with `weights` to make the engines compatible with the new PyTorch API. #621
- Adds support for high-level imports for various utility functions and classes such as `WSIReader`, `PatchPredictor` and `imread` #606, #607,
- Adds `tiatoolbox.typing` for type hints. #619
- Fixes incorrect file size saved by `save_tiles`, issue with certain WSIs raised by @TomastpPereira
- TissueMasker transform now returns mask instead of a list. #748
  - Fixes #732

### Bug Fixes and Other Changes

- Fixes `pixman` incompability error on Colab #601
- Removes `shapely.speedups`. The module no longer has any affect in Shapely >=2.0. #622
- Fixes errors in the slidegraph example notebook #608
- Fixes bugs in WSI Registration #645, #670, #693
- Fixes the situation where PatchExtractor.get_coords() can return patch coords which lie fully outside the bounds of a slide. #712
  - Fixes #710
- Fixes #738 raised by @xiachenrui

### Development related changes

- Replaces `flake8` and `isort` with `ruff` #625, #666
- Adds `mypy` checks to `root` and `utils` package. This will be rolled out in phases to other modules. #723
- Adds a module to detect file types using magic number/signatures #616
- Uses `poetry` for version updates instead of `bump2version`. #638
- Removes `setup.cfg` and uses `pyproject.toml` for project configurations.
- Reduces runtime for some unit tests e.g., #627, #630, #631, #629
- Reuses models and datasets in tests on GitHub actions by utilising cache #641, #644
- Set up parallel tests locally #671

**Full Changelog:** https://github.com/TissueImageAnalytics/tiatoolbox/compare/v1.4.0...v1.5.0

______________________________________________________________________

## TIAToolbox v1.4.1 (2023-07-25)

### Bug Fixes and Other Changes

- Fix dictionary changed size Error #626 (#605)

______________________________________________________________________

## TIAToolbox v1.4.0 (2023-04-24)

### Major Updates and Feature Improvements

- Adds Python 3.11 support [experimental] #500
  - Python 3.11 is not fully supported by `pytorch` https://github.com/pytorch/pytorch/issues/86566 and `openslide` https://github.com/openslide/openslide-python/pull/188
- Removes Python 3.7 support
  - This allows upgrading all the dependencies which were dependent on an older version of Python.
- Adds Neighbourhood Querying Support To AnnotationStore #540
  - This enables easy and efficient querying of annotations within a neighbourhood of other annotations.
- Adds `MultiTaskSegmentor` engine #424
- Fixes an issue with stain augmentation to apply augmentation to only tissue regions.
  - #546 contributed by @navidstuv
- Filters logger output to stdout instead of stderr.
  - Fixes #255
- Allows import of some modules at higher level for improved usability
  - `WSIReader` can now be imported as `from tiatoolbox.wsicore import WSIReader`
  - `WSIMeta` can now be imported as `from tiatoolbox.wsicore import WSIMeta`
  - `HoVerNet`, `HoVerNetPlus`, `IDaRS`, `MapDe`, `MicroNet`, `NuClick`, `SCCNN` can now be imported as \`from tiatoolbox.models import HoVerNet, HoVerNetPlus, IDaRS, MapDe, MicroNet, NuClick, SCCNN
- Improves `PatchExtractor` performance. Updates `WSIPatchDataset` to be consistent. #571
- Updates documentation for `License` for clarity on source code and model weights license.

### Changes to API

- Updates SCCNN architecture to make it consistent with other models. #544

### Bug Fixes and Other Changes

- Fixes Parsing Missing Omero Version NGFF Metadata #568
  - Fixes #535 raised by @benkamphaus
- Fixes reading of DICOM WSIs at the correct level #564
  - Fixes #529
- Fixes `scipy`, `matplotlib`, `scikit-image` deprecated code
- Fixes breaking changes in `DICOMWSIReader` to make it compatible with latest `wsidicom` version. #539, #580
- Updates `shapely` dependency to version >=2.0.0 and fixes any breaking changes.
- Fixes bug with `DictionaryStore.bquery` and `geometry=None`, i.e. only a where predicate given.
  - Partly Fixes #532 raised by @blaginin
- Fixes local tests for Windows/Linux
- Fixes `flake8`, `deepsource` errors.
- Uses `logger` instead of `warnings` and `print` statements to properly log runs.

### Development related changes

- Upgrades dependencies which are dependent on Python 3.7
- Moves `requirements*.txt` files to `requirements` folder
- Removes `tox`
- Uses `pyproject.toml` for `bdist_wheel`, `pytest` and `isort`
- Adds `joblib` and `numba` as dependencies.

______________________________________________________________________

## TIAToolbox v1.3.3 (2023-03-02)

### Major Updates and Feature Improvements

- Restricts dependency versions for long term stability of the current version

### Changes to API

None

### Bug Fixes and Other Changes

- Fix bug related to reading scikit-image

### Development related changes

- Restricts dependency versions for compatibility

______________________________________________________________________

## TIAToolbox v1.3.2 (2023-02-17)

### Major Updates and Feature Improvements

None

### Changes to API

None

### Bug Fixes and Other Changes

- Fix bug related to reading DICOM files

### Development related changes

- Restricts wsidicom version to \<0.7.0 for compatibility

______________________________________________________________________

## TIAToolbox v1.3.1 (2022-12-20)

### Major Updates and Feature Improvements

- Adds [NuClick](https://arxiv.org/abs/2005.14511) architecture #449
- Adds Annotation Store Reader #476
- Adds [DFBR](https://arxiv.org/abs/2202.09971) method for registering pair of images #510

### Changes to API

- Adds a sample SVS loading function `tiatoolbox.data.small_svs()` to the data module #517

### Bug Fixes and Other Changes

- Simplifies example notebook for image reading for better readability
- Restricts Shapely version to \<2.0.0 for compatibility

### Development related changes

- Adds GitHub workflow for automatic generation of docker image
- Updates dependencies
- Updates bump2version config
- Enables flake8 E800 checks for commented codes.
- Fixes several errors generated by DeepSource.
- Prevent test dumping file to root
- Removes duplicate functions to generate parameterized test scenarios

______________________________________________________________________

## TIAToolbox v1.3.0 (2022-10-20)

### Major Updates and Feature Improvements

- Adds an AnnotationTileGenerator and AnnotationRenderer which allows serving of tiles rendered directly from an annotation store.
- Adds [DFBR](https://arxiv.org/abs/2202.09971) registration model and jupyter notebook example
  - Adds DICE metric
- Adds [SCCNN](https://doi.org/10.1109/tmi.2016.2525803) architecture. \[[read the docs](https://tia-toolbox.readthedocs.io/en/develop/_autosummary/tiatoolbox.models.architecture.sccnn.SCCNN.html)\]
- Adds [MapDe](https://arxiv.org/abs/1806.06970) architecture. \[[read the docs](https://tia-toolbox.readthedocs.io/en/develop/_autosummary/tiatoolbox.models.architecture.mapde.MapDe.html)\]
- Adds support for reading MPP metadata from NGFF v0.4
- Adds enhancements to tiatoolbox.annotation.storage that are useful when using an AnnotationStore for visualization purposes.

### Changes to API

- None

### Bug Fixes and Other Changes

- Fixes colorbar_params #410
- Fixes Jupyter notebooks for better read the docs rendering
  - Fixes typos, metadata and links
- Fixes nucleus_segmentor_engine for boundary artefacts
- Fixes the colorbar cropping in tests
- Adds citation in README.md and CITATION.cff to Nature Communications Medicine paper
- Fixes a bug #452 raised by @rogertrullo where only the numerator of the TIFF resolution tags was being read.
- Fixes HoVer-Net+ post-processing to be inline with original work.
- Fixes a bug where an exception would be raised if the OME XML is missing objective power.

### Development related changes

- Uses Furo theme for readthedocs
- Replaces nbgallery and nbsphinx with myst-nb for jupyter notebook rendering
- Uses myst for markdown parsing
- Uses requirements.txt to define dependencies for requirements consistency
- Adds notebook AST pre-commit hook
- Adds check to validate python examples in the code
- Adds check to resolve imports
- Fixes an error in a docstring which triggered the failing test.
- Adds pre-commit hooks to format markdown and notebook markdown
- Adds pip install workflow to resolve dependencies when requirements file is updated
- Improves tiatoolbox import using LazyLoader

______________________________________________________________________

## TIAToolbox v1.2.1 (2022-07-07)

### Major Updates and Feature Improvements

- None

### Changes to API

- None

### Bug Fixes and Other Changes

- Fixes issues with dependencies.
  - Adds flask to dependencies.
- Fixes missing file in the python package.
- Clarifies help string for show-wsi option.

### Development related changes

- Removes Travis CI.
  - GitHub Actions will be used instead.
- Adds pre-commit hooks to check requirements consistency.
- Adds GitHub Action to resolve conda environment checks on Windows and Ubuntu.

______________________________________________________________________

## TIAToolbox v1.2.0 (2022-07-05)

### Major Updates and Feature Improvements

- Adds support for Python 3.10
- Adds short description for IDARS algorithm #383
- Adds support for NGFF v0.4 [OME-ZARR](https://ngff.openmicroscopy.org/latest/).
- Adds CLI for launching tile server.

### Changes to API

- Renames `stainnorm_target()` function to `stain_norm_target()`.
- Removes `get_wsireader`
- Replaces the custom PlattScaler in `tools/scale.py` with the regular Scikit-Learn [LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

### Bug Fixes and Other Changes

- Fixes bugs in UNET architecture.
  - Number of channels in Batchnorm argument in the decoding path to match with the input channels.
  - Padding `0` creates feature maps in the decoder part with the same size as encoder.
- Fixes linter issues and typos
- Fixes incorrect output with overlap in `predictor.merge_predictions()` and `return_raw=True`
  - Thanks to @paulhacosta for raising #356, Fixed by #358.
- Fixes errors with JP2 read. Checks input path exists.
- Fixes errors with torch upgrade to 1.12.

### Development related changes

- Adds pre-commit hooks for consistency across the repo.
- Sets up GitHub Actions Workflow.
  - Travis CI will be removed in future release.

______________________________________________________________________

## TIAToolbox v1.1.0 (2022-05-07)

### Major Updates and Feature Improvements

- Adds DICOM Support.
- Updates license to more permissive BSD 3-clause.
- Adds `micronet` model.
- Improves support for `tiff` files.
  - Adds a check for tiles in a TIFF file when opening.
  - Uses OpenSlide to read a TIFF if it has tiles instead of OpenCV (VirtualWSIReader).
  - Adds a fallback to tifffile if it is tiled but openslide cannot read it
    (e.g. jp2k or jpegxl tiles).
- Adds support for multi-channel images (HxWxC).
- Fixes performance issues in `semantic_segmentor.py`.
  - Performance gain measurement: 21.67s (new) vs 45.564 (old) using a 4k x 4k WSI.
  - External Contribution from @ByteHexler.
- Adds benchmark for Annotations Store.

### Changes to API

- None

### Bug Fixes and Other Changes

- Enhances the error messages to be more informative.
- Fixes Flake8 Errors, typos.
  - Fixes patch predictor models based after fixing a typo.
- Bug fixes in Graph functions.
- Adds documentation for docker support.
- General tidying up of docstrings.
- Adds metrics to readthedocs/docstrings for pretrained models.

### Development related changes

- Adds `pydicom` and `wsidicom` as dependency.
- Updates dependencies.
- Fixes Travis detection and makes improvements to run tests faster on Travis.
- Adds Dependabot to automatically update dependencies.
- Improves CLI definitions to make it easier to integrate new functions.
- Fixes compile options for test_annotation_stores.py

______________________________________________________________________

## TIAToolbox v1.0.1 (2022-01-31)

### Major Updates and Feature Improvements

- Updates dependencies for conda recipe #262

### Changes to API

- None

### Bug Fixes and Other Changes

- Adds User Warning For Missing SQLite Functions
- Fixes Pixman version check errors
- Fixes empty query in instance segmentor

### Development related changes

- Fixes flake8 linting issues and typos
- Conditional pytest.skipif to skip GPU tests on travis while running them locally or elsewhere

______________________________________________________________________

## TIAToolbox v1.0.0 (2021-12-23)

### Major Updates and Feature Improvements

- Adds nucleus instance segmentation base class
  - Adds [HoVerNet](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045) architecture
- Adds multi-task segmentor [HoVerNet+](https://arxiv.org/abs/2108.13904) model
- Adds <a href="https://www.thelancet.com/journals/landig/article/PIIS2589-7500(2100180-1/fulltext">IDaRS</a> pipeline
- Adds [SlideGraph](https://arxiv.org/abs/2110.06042) pipeline
- Adds PCam patch classification models
- Adds support for stain augmentation feature
- Adds classes and functions under `tiatoolbox.tools.graph` to enable construction of graphs in a format which can be used with PyG (PyTorch Geometric).
- Add classes which act as a mutable mapping (dictionary like) structure and enables efficient management of annotations. (#135)
- Adds example notebook for adding advanced models
- Adds classes which can generate zoomify tiles from a WSIReader object.
- Adds WSI viewer using Zoomify/WSIReader API (#212)
- Adds README to example page for clarity
- Adds support to override or specify mpp and power

### Changes to API

- Replaces `models.controller` API with `models.engine`
- Replaces `CNNPatchPredictor` with `PatchPredictor`

### Bug Fixes and Other Changes

- Fixes `filter_coordinates` read wrong resolutions for patch extraction
- For `PatchPredictor`
  - `ioconfig` will supersede everything
  - if `ioconfig` is not provided
    - If `model` is pretrained (defined in `pretrained_model.yaml` )
      - Use the yaml ioconfig
      - Any other input patch reading arguments will overwrite the yaml ioconfig (at the same keyword).
    - If `model` is not defined, all input patch reading arguments must be provided else exception will be thrown.
- Improves performance of mask based patch extraction

### Development related changes

- Improve tests performance for Travis runs
- Adds feature detection mechanism to detect the platform and installed packages etc.
- On demand imports for some libraries for performance
- Improves performance of mask based patch extraction

______________________________________________________________________

## TIAToolbox v0.8.0 (2021-10-27)

### Major Updates and Feature Improvements

- Adds `SemanticSegmentor` which is Predictor equivalent for semantic segmentation.
- Add `TIFFWSIReader` class to support OMETiff reading.
- Adds `FeatureExtractor` API to controller.
- Adds WSI Serialization Dataset which support changing parallel workers on the fly. This would reduce the time spent to create new worker for every WSI/Tile (costly).
- Adds IOState data class to contain IO information for loading input to model and assembling model output back to WSI/Tile.
- Minor updates for `get_coordinates` to pave the way for getting patch IO for segmentation.
- Migrates old code to new variable names (patch extraction, patch wsi model).
- Change in API from `pretrained_weight` to `pretrained_weights`.
- Adds cli for semantic segmentation.
- Update python notebooks to add `read_rect` and `read_bounds` examples with `mpp` read.

### Changes to API

- Adds `WSIReader.open`. `get_wsireader` will deprecate in the next release. Please use `WSIReader.open` instead.
- CLI is now POSIX compatible
  - Replaces underscores in variable names with hyphens
- Models API updated to use `pretrained_weights` instead of `pretrained_weight`.
- Move string_to_tuple to tiatoolbox/utils/misc.py

### Bug Fixes and Other Changes

- Fixes README git clone instructions.
- Fixes stain normalisation due to changes in sklearn.
- Fixes a test in tests/test_slide_info
- Fixes readthedocs documentation issues

### Development related changes

- Adds dependencies for tiffile, imagecodecs, zarr.
- Adds more stringent pre-commit checks
- Moved local test files into `tiatoolbox/data`.
- Fixed `Manifest.ini` and added `tiatoolbox/data`. This means that this directory will be downloaded with the package.
- Using `pkg_resources` to properly load bundled resources (e.g. `target_image.png`) in `tiatoolbox.data`.
- Removed duplicate code in `conftest.py` for downloading remote files. This is now in `tiatoolbox.data._fetch_remote_file`.
- Fixes errors raised by new flake8 rules.
  - Remove leading underscores from fixtures.
- Rename some remote sample files to make more sense.
- Moves all cli commands/options from cli.py to cli_commands to make it clean and easier to add new commands
- Removes redundant tests
- Updates to new GitHub organisation name in the repo
  - Fixes related links

______________________________________________________________________

## TIAToolbox v0.7.0 (2021-09-16)

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
- Adds extra checks to pre-commit, e.g., import sorting, spellcheck etc. Detailed list can be found on this [commit](https://github.com/TissueImageAnalytics/tiatoolbox/commit/662a143e915fa55416badd992d8e7358211730a6).

______________________________________________________________________

## TIAToolbox v0.6.0 (2021-05-11)

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

______________________________________________________________________

## TIAToolbox v0.5.2 (2021-03-12)

### Bug Fixes and Other Changes

- Fix URL for downloading test JP2 image.
- Update readme with new logo.

______________________________________________________________________

## TIAToolbox v0.5.1 (2020-12-31)

### Bug Fixes and Other Changes

- Add `scikit-image` as dependency in `setup.py`
- Update notebooks to add instructions to install dependencies

______________________________________________________________________

## TIAToolbox v0.5.0 (2020-12-30)

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
  - `VirtualWSIReader` accepts ndarray or image path as input.
- Adds MPP fall back to standard TIFF resolution tags with warning.
  - If OpenSlide cannot determine microns per pixel (`mpp`) from the metadata, checks the TIFF resolution units (TIFF tags: `ResolutionUnit`, `XResolution` and `YResolution`) to calculate MPP. Additionally, add function to estimate missing objective power if MPP is known of derived from TIFF resolution tags.
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

______________________________________________________________________

## TIAToolbox v0.4.0 (2020-10-25)

### Major and Feature Improvements

- Adds `OpenSlideWSIReader` to read Openslide image formats
- Adds support to read Omnyx jp2 images using `OmnyxJP2WSIReader`.
- New feature added to perform stain normalisation using `Ruifork`, `Reinhard`, `Vahadane`, `Macenko` methods and using custom stain matrices.
- Adds example notebook to read whole slide images via the toolbox.
- Adds `WSIMeta` class to save meta data for whole slide images. `WSIMeta` casts properties to python types. Properties from OpenSlide are returned as string. raw values can always be accessed via `slide.raw`. Adds data validation e.g., checking that level_count matches up with the length of the `level_dimensions` and `level_downsamples`. Adds type hints to `WSIMeta`.
- Adds exceptions `FileNotSupported` and `MethodNotSupported`

### Changes to API

- Restructures `WSIReader` as parent class to allow support to read whole slide images in other formats.
- Adds `slide_info` as a property of `WSIReader`
- Updates `slide_info` type to `WSIMeta` from `dict`
- Depreciates support for multiprocessing from within the toolbox. The toolbox is focused on processing single whole slide and standard images. External libraries can be used to run using multiprocessing on multiple files.

### Bug Fixes and Other Changes

- Adds `scikit-learn`, `glymur` as a dependency
- Adds licence information
- Removes `pathos` as a dependency
- Updates `openslide-python` requirement to 1.1.2

______________________________________________________________________

## TIAToolbox v0.3.0 (2020-07-19)

### Major and Feature Improvements

- Adds feature `read_region` to read a small region from whole slide images
- Adds feature `save_tiles` to save image tiles from whole slide images
- Adds feature `imresize` to resize images
- Adds feature `transforms.background_composite` to avoid creation of black tiles from whole slide images.

### Changes to API

- None

### Bug Fixes and Other Changes

- Adds `pandas` as dependency

______________________________________________________________________

## TIAToolbox v0.2.2 (2020-07-12)

### Major and Feature Improvements

- None

### Changes to API

- None

### Bug Fixes and Other Changes

- Fix command line interface for `slide-info` feature and travis pypi deployment

______________________________________________________________________

## TIAToolbox v0.2.1 (2020-07-10)

### Major and Feature Improvements

- None

### Changes to API

- None

### Bug Fixes and Other Changes

- Minor changes to configuration files.

______________________________________________________________________

## TIAToolbox v0.2.0 (2020-07-10)

### Major and Feature Improvements

- Adds feature slide_info to read whole slide images and display meta
  data information
- Adds multiprocessing decorator TIAMultiProcess to allow running
  toolbox functions using multiprocessing.

### Changes to API

- None

### Bug Fixes and Other Changes

- Adds Sphinx Readthedocs support
  <https://readthedocs.org/projects/tia-toolbox/> for stable and
  develop branches
- Adds code coverage tools to test the pytest coverage of the package
- Adds deepsource integration to highlight and fix bug risks,
  performance issues etc.
- Adds README to allow users to setup the environment.
- Adds conda and pip requirements instructions

______________________________________________________________________

## TIAToolbox v0.1.0 (2020-05-28)

- First release on PyPI.
