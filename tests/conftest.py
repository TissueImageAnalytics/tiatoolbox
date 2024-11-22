"""pytest fixtures."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable

import pytest
import torch

import tiatoolbox
from tiatoolbox import logger
from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.utils.env_detection import has_gpu, running_on_ci

# -------------------------------------------------------------------------------------
# Generate Parameterized Tests
# -------------------------------------------------------------------------------------


def pytest_configure() -> None:
    """Perform initial configuration for TIAToolbox tests."""
    logger.info(
        "🏁 Starting tests. TIAToolbox Version: %s. CI: %s",
        tiatoolbox.__version__,
        running_on_ci(),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate (parameterize) test scenarios.

    Adapted from pytest documentation. For more information on
    parameterized tests see:
    https://docs.pytest.org/en/6.2.x/example/parametrize.html#a-quick-port-of-testscenarios

    """
    # Return if the test is not part of a class or if the class does not
    # have a scenarios attribute.
    if metafunc.cls is None or not hasattr(metafunc.cls, "scenarios"):
        return
    idlist = []
    argvalues = []
    argnames = None
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


# -------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def root_path(request: pytest.FixtureRequest) -> Path:
    """Return the root path of the project."""
    return Path(request.config.rootdir) / "tiatoolbox"


@pytest.fixture(scope="session")
def tmp_samples_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a temporary path."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def remote_sample(tmp_samples_path: str) -> Callable:
    """Factory fixture for fetching sample files."""

    def __remote_sample(key: str) -> Path:
        """Wrapper around tiatoolbox.data._fetch_remote_sample for tests."""
        return _fetch_remote_sample(key, tmp_samples_path)

    return __remote_sample


@pytest.fixture(scope="session")
def sample_ndpi(remote_sample: Callable) -> Path:
    """Sample pytest fixture for ndpi images.

    Download ndpi image for pytest.

    """
    return remote_sample("ndpi-1")


@pytest.fixture(scope="session")
def sample_ndpi2(remote_sample: Callable) -> Path:
    """Sample pytest fixture for ndpi images.

    Download ndpi image for pytest.
    Collected from doi:10.5072/zenodo.219445

    """
    return remote_sample("ndpi-2")


@pytest.fixture(scope="session")
def sample_svs(remote_sample: Callable) -> Path:
    """Sample pytest fixture for svs images.

    Download svs image for pytest.

    """
    return remote_sample("svs-1-small")


@pytest.fixture(scope="session")
def sample_ome_tiff(remote_sample: Callable) -> Path:
    """Sample pytest fixture for ome-tiff (brightfield pyramid) images.

    Download ome-tiff image for pytest.

    """
    return remote_sample("ome-brightfield-small-pyramid")


@pytest.fixture(scope="session")
def sample_ome_tiff_level_0(remote_sample: Callable) -> Path:
    """Sample pytest fixture for ome-tiff image with one level.

    Download ome-tiff image for pytest.

    """
    return remote_sample("ome-brightfield-small-level-0")


@pytest.fixture(scope="session")
def sample_ventana_tif(remote_sample: Callable) -> Path:
    """Sample pytest fixture for non-tiled tif Ventana images.

    Download Ventana tif image for pytest.

    """
    return remote_sample("ventana-tif")


@pytest.fixture(scope="session")
def sample_regular_tif(remote_sample: Callable) -> Path:
    """Sample pytest fixture for non-tiled tif Ventana images.

    Download Ventana tif image for pytest.

    """
    return remote_sample("regular-tif")


@pytest.fixture(scope="session")
def sample_jp2(remote_sample: Callable) -> Path:
    """Sample pytest fixture for JP2 images.

    Download jp2 image for pytest.

    """
    return remote_sample("jp2-omnyx-small")


@pytest.fixture(scope="session")
def sample_all_wsis(
    sample_ndpi: Path,
    sample_svs: Path,
    sample_jp2: Path,
    tmpdir_factory: pytest.TempdirFactory,
) -> Path:
    """Sample wsi(s) of all types supported by tiatoolbox."""
    dir_path = Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(sample_ndpi.name).symlink_to(sample_ndpi)
        dir_path.joinpath(sample_svs.name).symlink_to(sample_svs)
        dir_path.joinpath(sample_jp2.name).symlink_to(sample_jp2)
    except OSError:
        shutil.copy(sample_ndpi, dir_path.joinpath(sample_ndpi.name))
        shutil.copy(sample_svs, dir_path.joinpath(sample_svs.name))
        shutil.copy(sample_jp2, dir_path.joinpath(sample_jp2.name))

    return dir_path


@pytest.fixture(scope="session")
def sample_all_wsis2(
    sample_ndpi2: Path,
    sample_svs: Path,
    sample_jp2: Path,
    tmpdir_factory: pytest.TempdirFactory,
) -> Path:
    """Sample wsi(s) of all types supported by tiatoolbox.

    Adds sample fluorescence ndpi image.

    """
    dir_path = Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(sample_ndpi2.name).symlink_to(sample_ndpi2)
        dir_path.joinpath(sample_svs.name).symlink_to(sample_svs)
        dir_path.joinpath(sample_jp2.name).symlink_to(sample_jp2)
    except OSError:
        shutil.copy(sample_ndpi2, dir_path.joinpath(sample_ndpi2.name))
        shutil.copy(sample_svs, dir_path.joinpath(sample_svs.name))
        shutil.copy(sample_jp2, dir_path.joinpath(sample_jp2.name))

    return dir_path


@pytest.fixture(scope="session")
def sample_svs_ndpi_wsis(
    sample_ndpi2: Path,
    sample_svs: Path,
    tmpdir_factory: pytest.TempdirFactory,
) -> Path:
    """Sample SVS and NDPI wsi(s).

    Uses sample fluorescence ndpi image.

    """
    dir_path = Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(sample_ndpi2.name).symlink_to(sample_ndpi2)
        dir_path.joinpath(sample_svs.name).symlink_to(sample_svs)
    except OSError:
        shutil.copy(sample_ndpi2, dir_path.joinpath(sample_ndpi2.name))
        shutil.copy(sample_svs, dir_path.joinpath(sample_svs.name))

    return dir_path


@pytest.fixture(scope="session")
def source_image(remote_sample: Callable) -> Path:
    """Sample pytest fixture for source image.

    Download stain normalization source image for pytest.

    """
    return remote_sample("stainnorm-source")


@pytest.fixture(scope="session")
def norm_macenko(remote_sample: Callable) -> Path:
    """Sample pytest fixture for norm_macenko image.

    Download norm_macenko image for pytest.

    """
    return remote_sample("stainnorm-target-macenko")


@pytest.fixture(scope="session")
def norm_reinhard(remote_sample: Callable) -> Path:
    """Sample pytest fixture for norm_reinhard image.

    Download norm_reinhard image for pytest.

    """
    return remote_sample("stainnorm-target-reinhard")


@pytest.fixture(scope="session")
def norm_ruifrok(remote_sample: Callable) -> Path:
    """Sample pytest fixture for norm_ruifrok image.

    Download norm_ruifrok image for pytest.

    """
    return remote_sample("stainnorm-target-ruifrok")


@pytest.fixture(scope="session")
def norm_vahadane(remote_sample: Callable) -> Path:
    """Sample pytest fixture for norm_vahadane image.

    Download norm_vahadane image for pytest.

    """
    return remote_sample("stainnorm-target-vahadane")


@pytest.fixture(scope="session")
def sample_visual_fields(
    source_image: Path,
    norm_ruifrok: Path,
    norm_reinhard: Path,
    norm_macenko: Path,
    norm_vahadane: Path,
    tmpdir_factory: pytest.TempdirFactory,
) -> Path:
    """Sample visual fields(s) of all types supported by tiatoolbox."""
    dir_path = Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(source_image.name).symlink_to(source_image)
        dir_path.joinpath(norm_ruifrok.name).symlink_to(norm_ruifrok)
        dir_path.joinpath(norm_reinhard.name).symlink_to(norm_reinhard)
        dir_path.joinpath(norm_macenko.name).symlink_to(norm_macenko)
        dir_path.joinpath(norm_vahadane.name).symlink_to(norm_vahadane)
    except OSError:
        shutil.copy(source_image, dir_path.joinpath(source_image.name))
        shutil.copy(norm_ruifrok, dir_path.joinpath(norm_ruifrok.name))
        shutil.copy(norm_reinhard, dir_path.joinpath(norm_reinhard.name))
        shutil.copy(norm_macenko, dir_path.joinpath(norm_macenko.name))
        shutil.copy(norm_vahadane, dir_path.joinpath(norm_vahadane.name))

    return dir_path


@pytest.fixture(scope="session")
def patch_extr_vf_image(remote_sample: Callable) -> Path:
    """Sample pytest fixture for a visual field image.

    Download TCGA-HE-7130-01Z-00-DX1 image for pytest.

    """
    return remote_sample("patch-extraction-vf")


@pytest.fixture(scope="session")
def patch_extr_csv(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction csv.

    Download sample patch extraction csv for pytest.

    """
    return remote_sample("patch-extraction-csv")


@pytest.fixture(scope="session")
def patch_extr_json(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction json.

    Download sample patch extraction json for pytest.

    """
    return remote_sample("patch-extraction-csv")


@pytest.fixture(scope="session")
def patch_extr_npy(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction npy.

    Download sample patch extraction npy for pytest.

    """
    return remote_sample("patch-extraction-npy")


@pytest.fixture(scope="session")
def patch_extr_csv_noheader(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction noheader csv.

    Download sample patch extraction noheader csv for pytest.

    """
    return remote_sample("patch-extraction-csv-noheader")


@pytest.fixture(scope="session")
def patch_extr_2col_json(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction 2col json.

    Download sample patch extraction 2col json for pytest.

    """
    return remote_sample("patch-extraction-2col-json")


@pytest.fixture(scope="session")
def patch_extr_2col_npy(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction 2col npy.

    Download sample patch extraction 2col npy for pytest.

    """
    return remote_sample("patch-extraction-2col-npy")


@pytest.fixture(scope="session")
def patch_extr_jp2_csv(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction jp2 csv.

    Download sample patch extraction jp2 csv for pytest.

    """
    return remote_sample("patch-extraction-jp2-csv")


@pytest.fixture(scope="session")
def patch_extr_jp2_read(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction jp2 read npy.

    Download sample patch extraction jp2 read npy for pytest.

    """
    return remote_sample("patch-extraction-jp2-read-npy")


@pytest.fixture(scope="session")
def patch_extr_npy_read(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction read npy.

    Download sample patch extraction read npy for pytest.

    """
    return remote_sample("patch-extraction-read-npy")


@pytest.fixture(scope="session")
def patch_extr_svs_csv(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction svs csv.

    Download sample patch extraction svs csv for pytest.

    """
    return remote_sample("patch-extraction-svs-csv")


@pytest.fixture(scope="session")
def patch_extr_svs_header(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction svs_header csv.

    Download sample patch extraction svs_header csv for pytest.

    """
    return remote_sample("patch-extraction-svs-header-csv")


@pytest.fixture(scope="session")
def patch_extr_svs_npy_read(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch extraction svs_read npy.

    Download sample patch extraction svs_read npy for pytest.

    """
    return remote_sample("patch-extraction-svs-read-npy")


@pytest.fixture(scope="session")
def sample_patch1(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch 1.

    Download sample patch 1 (Kather100K) for pytest.

    """
    return remote_sample("sample-patch-1")


@pytest.fixture(scope="session")
def sample_patch2(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch 2.

    Download sample patch 2 (Kather100K) for pytest.

    """
    return remote_sample("sample-patch-2")


@pytest.fixture(scope="session")
def sample_patch3(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch 3.

    Download sample patch 3 (PCam) for pytest.

    """
    return remote_sample("sample-patch-3")


@pytest.fixture(scope="session")
def sample_patch4(remote_sample: Callable) -> Path:
    """Sample pytest fixture for sample patch 4.

    Download sample patch 4 (PCam) for pytest.

    """
    return remote_sample("sample-patch-4")


@pytest.fixture(scope="session")
def dir_sample_patches(
    sample_patch1: Path,
    sample_patch2: Path,
    tmpdir_factory: pytest.TempdirFactory,
) -> Path:
    """Directory of sample image patches for testing."""
    dir_path = Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(sample_patch1.name).symlink_to(sample_patch2)
        dir_path.joinpath(sample_patch2.name).symlink_to(sample_patch2)
    except OSError:
        shutil.copy(sample_patch1, dir_path.joinpath(sample_patch1.name))
        shutil.copy(sample_patch2, dir_path.joinpath(sample_patch2.name))

    return dir_path


@pytest.fixture(scope="session")
def sample_wsi_dict(remote_sample: Callable) -> dict:
    """Sample pytest fixture for torch wsi dataset.

    Download svs image for pytest.

    """
    file_names = [
        "wsi1_8k_8k_svs",
        "wsi1_8k_8k_jp2",
        "wsi1_8k_8k_jpg",
        "wsi2_4k_4k_svs",
        "wsi2_4k_4k_jp2",
        "wsi2_4k_4k_jpg",
        "wsi2_4k_4k_msk",
        "wsi2_4k_4k_pred",
        "wsi3_20k_20k_svs",
        "wsi4_4k_4k_svs",
        "wsi3_20k_20k_pred",
        "wsi4_4k_4k_pred",
    ]
    return {name: remote_sample(name) for name in file_names}


@pytest.fixture(scope="session")
def fixed_image(remote_sample: Callable) -> Path:
    """Sample pytest fixture for fixed image.

    Download fixed image for pytest.
    """
    return remote_sample("fixed_image")


@pytest.fixture(scope="session")
def moving_image(remote_sample: Callable) -> Path:
    """Sample pytest fixture for moving image.

    Download moving image for pytest.
    """
    return remote_sample("moving_image")


@pytest.fixture(scope="session")
def dfbr_features(remote_sample: Callable) -> Path:
    """Sample pytest fixture for DFBR features.

    Download features used by Deep Feature Based
    Registration (DFBR) method for pytest.
    """
    return remote_sample("dfbr_features")


@pytest.fixture(scope="session")
def fixed_mask(remote_sample: Callable) -> Path:
    """Sample pytest fixture for fixed mask.

    Download fixed mask for pytest.
    """
    return remote_sample("fixed_mask")


@pytest.fixture(scope="session")
def moving_mask(remote_sample: Callable) -> Path:
    """Sample pytest fixture for moving mask.

    Download moving mask for pytest.
    """
    return remote_sample("moving_mask")


@pytest.fixture(scope="session")
def chdir() -> Callable:
    """Return a context manager to change the current working directory.

    Todo: switch to chdir from contextlib when Python 3.11 is required

    """
    try:
        from contextlib import chdir
    except ImportError:
        from contextlib import AbstractContextManager

        class chdir(AbstractContextManager):  # noqa: N801
            """Non thread-safe context manager to change the current working directory.

            See Also: https://github.com/python/cpython/blob/main/Lib/contextlib.py.

            """

            def __init__(self: chdir, path: Path) -> None:
                self.path = path
                self._old_cwd = []

            def __enter__(self: chdir) -> None:
                self._old_cwd.append(os.getcwd())  # noqa: PTH109
                os.chdir(self.path)

            def __exit__(self: chdir, *excinfo: object) -> None:
                os.chdir(self._old_cwd.pop())

    return chdir


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Set up a temporary data directory for testing visualization UI."""
    tmp_path = tmp_path_factory.mktemp("data")
    (tmp_path / "slides").mkdir()
    (tmp_path / "overlays").mkdir()
    return {"base_path": tmp_path}


# -------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------


def timed(fn: Callable, *args: object) -> (Callable, float):
    """A decorator that times the execution of a function.

    Args:
        fn (Callable): The function to be timed.
        args (object): Arguments to be passed to the function.

    Returns:
        A tuple containing the result of the function
        and the time taken to execute it in seconds.

    """
    compile_time = 0.0
    if has_gpu():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args)
        end.record()
        torch.cuda.synchronize()
        compile_time = start.elapsed_time(end) / 1000
    else:
        start = time.time()
        result = fn(*args)
        end = time.time()
        compile_time = end - start
    return result, compile_time
