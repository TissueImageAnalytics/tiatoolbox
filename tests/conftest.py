import pytest
import requests
import shutil
import pathlib

# -------------------------------------------------------------------------------------
# Pytest Fixtures
# -------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _sample_ndpi(tmpdir_factory):
    """Sample pytest fixture for ndpi images. Download ndpi image for pytest."""
    ndpi_file_path = tmpdir_factory.mktemp("data").join("CMU-1.ndpi")
    if not pathlib.Path(ndpi_file_path).is_file():
        print("\nDownloading NDPI")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Hamamatsu/CMU-1.ndpi"
        )
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping NDPI")

    return ndpi_file_path


@pytest.fixture(scope="session")
def _sample_svs(tmpdir_factory):
    """Sample pytest fixture for svs images. Download ndpi image for pytest."""
    svs_file_path = tmpdir_factory.mktemp("data").join("CMU-1-Small-Region.svs")
    if not pathlib.Path(svs_file_path).is_file():
        print("\nDownloading SVS")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Aperio/CMU-1-Small-Region.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping SVS")

    return svs_file_path


@pytest.fixture(scope="session")
def _sample_jp2(tmpdir_factory):
    """Sample pytest fixture for JP2 images. Download ndpi image for pytest."""
    jp2_file_path = tmpdir_factory.mktemp("data").join("test1.jp2")
    if not pathlib.Path(jp2_file_path).is_file():
        print("\nDownloading JP2")
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/people/csundo/test2.jp2"
        )
        with open(jp2_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping JP2")

    return jp2_file_path


@pytest.fixture(scope="session")
def _sample_all_wsis(_sample_ndpi, _sample_svs, _sample_jp2, tmpdir_factory):
    """Sample wsi(s) of all types supported by tiatoolbox."""
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_sample_ndpi.basename).symlink_to(_sample_ndpi)
        dir_path.joinpath(_sample_svs.basename).symlink_to(_sample_svs)
        dir_path.joinpath(_sample_jp2.basename).symlink_to(_sample_jp2)
    except OSError:
        shutil.copy(_sample_ndpi, dir_path.joinpath(_sample_ndpi.basename))
        shutil.copy(_sample_svs, dir_path.joinpath(_sample_svs.basename))
        shutil.copy(_sample_jp2, dir_path.joinpath(_sample_jp2.basename))

    return dir_path
