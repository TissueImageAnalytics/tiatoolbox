"""pytest fixtures."""

import pytest
import requests
import shutil
import pathlib


@pytest.fixture(scope="session")
def _sample_ndpi(tmpdir_factory):
    """Sample pytest fixture for ndpi images.
    Download ndpi image for pytest.

    """
    ndpi_file_path = tmpdir_factory.mktemp("data").join("CMU-1.ndpi")
    if not pathlib.Path(ndpi_file_path).is_file():
        print("\nDownloading NDPI")
        r = requests.get("https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1.ndpi")
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping NDPI")

    return ndpi_file_path


@pytest.fixture(scope="session")
def _sample_svs(tmpdir_factory):
    """Sample pytest fixture for svs images.
    Download svs image for pytest.

    """
    svs_file_path = tmpdir_factory.mktemp("data").join("CMU-1-Small-Region.svs")
    if not pathlib.Path(svs_file_path).is_file():
        print("\nDownloading SVS")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1-Small-Region.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping SVS")

    return svs_file_path


@pytest.fixture(scope="session")
def _sample_jp2(tmpdir_factory):
    """Sample pytest fixture for JP2 images.
    Download jp2 image for pytest.

    """
    jp2_file_path = tmpdir_factory.mktemp("data").join("test1.jp2")

    if not pathlib.Path(jp2_file_path).is_file():
        print("\nDownloading JP2")
        r = requests.get("https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/test1.jp2")
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


@pytest.fixture(scope="session")
def _source_image(tmpdir_factory):
    """Sample pytest fixture for source image.
    Download stain normalisation source image for pytest.

    """
    src_image_file_path = tmpdir_factory.mktemp("data").join("source_image.png")

    if not pathlib.Path(src_image_file_path).is_file():
        print("\nDownloading Source Image")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/common/source_image.png"
        )
        with open(src_image_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping Source Image")

    return src_image_file_path


@pytest.fixture(scope="session")
def _norm_macenko(tmpdir_factory):
    """Sample pytest fixture for norm_macenko image.
    Download norm_macenko image for pytest.

    """
    norm_macenko_file_path = tmpdir_factory.mktemp("data").join("norm_macenko.png")

    if not pathlib.Path(norm_macenko_file_path).is_file():
        print("\nDownloading Norm_macenko image")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/stainnorm/norm_macenko.png"
        )
        with open(norm_macenko_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping Norm_macenko image")

    return norm_macenko_file_path


@pytest.fixture(scope="session")
def _norm_reinhard(tmpdir_factory):
    """Sample pytest fixture for norm_reinhard image.
    Download norm_reinhard image for pytest.

    """
    norm_reinhard_file_path = tmpdir_factory.mktemp("data").join("norm_reinhard.png")

    if not pathlib.Path(norm_reinhard_file_path).is_file():
        print("\nDownloading norm_reinhard image")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/stainnorm/norm_reinhard.png"
        )
        with open(norm_reinhard_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping norm_reinhard image")

    return norm_reinhard_file_path


@pytest.fixture(scope="session")
def _norm_ruifrok(tmpdir_factory):
    """Sample pytest fixture for norm_ruifrok image.
    Download norm_ruifrok image for pytest.

    """
    norm_ruifrok_file_path = tmpdir_factory.mktemp("data").join("norm_ruifrok.png")

    if not pathlib.Path(norm_ruifrok_file_path).is_file():
        print("\nDownloading norm_ruifrok image")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/stainnorm/norm_ruifrok.png"
        )
        with open(norm_ruifrok_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping norm_ruifrok image")

    return norm_ruifrok_file_path


@pytest.fixture(scope="session")
def _norm_vahadane(tmpdir_factory):
    """Sample pytest fixture for norm_vahadane image.
    Download norm_vahadane image for pytest.

    """
    norm_vahadane_file_path = tmpdir_factory.mktemp("data").join("norm_vahadane.png")

    if not pathlib.Path(norm_vahadane_file_path).is_file():
        print("\nDownloading norm_vahadane image")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/stainnorm/norm_vahadane.png"
        )
        with open(norm_vahadane_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping norm_vahadane image")

    return norm_vahadane_file_path


@pytest.fixture(scope="session")
def _sample_visual_fields(
    _source_image,
    _norm_ruifrok,
    _norm_reinhard,
    _norm_macenko,
    _norm_vahadane,
    tmpdir_factory,
):
    """Sample visual fields(s) of all types supported by tiatoolbox."""
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_source_image.basename).symlink_to(_source_image)
        dir_path.joinpath(_norm_ruifrok.basename).symlink_to(_norm_ruifrok)
        dir_path.joinpath(_norm_reinhard.basename).symlink_to(_norm_reinhard)
        dir_path.joinpath(_norm_macenko.basename).symlink_to(_norm_macenko)
        dir_path.joinpath(_norm_vahadane.basename).symlink_to(_norm_vahadane)
    except OSError:
        shutil.copy(_source_image, dir_path.joinpath(_source_image.basename))
        shutil.copy(_norm_ruifrok, dir_path.joinpath(_norm_ruifrok.basename))
        shutil.copy(_norm_reinhard, dir_path.joinpath(_norm_reinhard.basename))
        shutil.copy(_norm_macenko, dir_path.joinpath(_norm_macenko.basename))
        shutil.copy(_norm_vahadane, dir_path.joinpath(_norm_vahadane.basename))

    return dir_path


@pytest.fixture(scope="session")
def _patch_extr_vf_image(tmpdir_factory):
    """Sample pytest fixture for a visual field image.
    Download TCGA-HE-7130-01Z-00-DX1 image for pytest.

    """
    tcga_he_7130_file_path = tmpdir_factory.mktemp("data").join(
        "TCGA-HE-7130-01Z-00-DX1.png"
    )

    if not pathlib.Path(tcga_he_7130_file_path).is_file():
        print("\nDownloading TCGA-HE-7130-01Z-00-DX1 image")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/TCGA-HE-7130-01Z-00-DX1.png"
        )
        with open(tcga_he_7130_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping TCGA-HE-7130-01Z-00-DX1 image")

    return tcga_he_7130_file_path


@pytest.fixture(scope="session")
def _patch_extr_csv(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction csv.
    Download sample patch extraction csv for pytest.

    """
    csv_file_path = tmpdir_factory.mktemp("data").join("sample_patch_extraction.csv")

    if not pathlib.Path(csv_file_path).is_file():
        print("\nDownloading sample patch extraction csv file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction.csv"
        )
        with open(csv_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction csv file")

    return csv_file_path


@pytest.fixture(scope="session")
def _patch_extr_json(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction json.
    Download sample patch extraction json for pytest.

    """
    json_file_path = tmpdir_factory.mktemp("data").join("sample_patch_extraction.json")

    if not pathlib.Path(json_file_path).is_file():
        print("\nDownloading sample patch extraction json file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction.json"
        )
        with open(json_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction json file")

    return json_file_path


@pytest.fixture(scope="session")
def _patch_extr_npy(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction npy.
    Download sample patch extraction npy for pytest.

    """
    npy_file_path = tmpdir_factory.mktemp("data").join("sample_patch_extraction.npy")

    if not pathlib.Path(npy_file_path).is_file():
        print("\nDownloading sample patch extraction npy file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction.npy"
        )
        with open(npy_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction npy file")

    return npy_file_path


@pytest.fixture(scope="session")
def _patch_extr_csv_noheader(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction noheader csv.
    Download sample patch extraction noheader csv for pytest.

    """
    noheader_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction-noheader.csv"
    )

    if not pathlib.Path(noheader_file_path).is_file():
        print("\nDownloading sample patch extraction no header csv file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction-noheader.csv"
        )
        with open(noheader_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction no header csv file")

    return noheader_file_path


@pytest.fixture(scope="session")
def _patch_extr_2col_json(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction 2col json.
    Download sample patch extraction 2col json for pytest.

    """
    col2_json_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_2col.json"
    )

    if not pathlib.Path(col2_json_file_path).is_file():
        print("\nDownloading sample patch extraction 2col json file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_2col.json"
        )
        with open(col2_json_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction 2col json file")

    return col2_json_file_path


@pytest.fixture(scope="session")
def _patch_extr_2col_npy(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction 2col npy.
    Download sample patch extraction 2col npy for pytest.

    """
    col_npy_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_2col.npy"
    )

    if not pathlib.Path(col_npy_file_path).is_file():
        print("\nDownloading sample patch extraction 2col npy file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_2col.npy"
        )
        with open(col_npy_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction 2col npy file")

    return col_npy_file_path


@pytest.fixture(scope="session")
def _patch_extr_jp2_csv(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction jp2 csv.
    Download sample patch extraction jp2 csv for pytest.

    """
    jp2_csv_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_jp2.csv"
    )

    if not pathlib.Path(jp2_csv_file_path).is_file():
        print("\nDownloading sample patch extraction jp2 csv file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_jp2.csv"
        )
        with open(jp2_csv_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction jp2 csv file")

    return jp2_csv_file_path


@pytest.fixture(scope="session")
def _patch_extr_jp2_read(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction jp2 read npy.
    Download sample patch extraction jp2 read npy for pytest.

    """
    jp2_read_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_jp2read.npy"
    )

    if not pathlib.Path(jp2_read_file_path).is_file():
        print("\nDownloading sample patch extraction jp2 read npy file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_jp2read.npy"
        )
        with open(jp2_read_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction jp2 read npy file")

    return jp2_read_file_path


@pytest.fixture(scope="session")
def _patch_extr_npy_read(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction read npy.
    Download sample patch extraction read npy for pytest.

    """
    read_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_read.npy"
    )

    if not pathlib.Path(read_file_path).is_file():
        print("\nDownloading sample patch extraction read npy file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_read.npy"
        )
        with open(read_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction read npy file")

    return read_file_path


@pytest.fixture(scope="session")
def _patch_extr_svs_csv(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction svs csv.
    Download sample patch extraction svs csv for pytest.

    """
    svs_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_svs.csv"
    )

    if not pathlib.Path(svs_file_path).is_file():
        print("\nDownloading sample patch extraction svs csv file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_svs.csv"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction svs csv file")

    return svs_file_path


@pytest.fixture(scope="session")
def _patch_extr_svs_header(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction svs_header csv.
    Download sample patch extraction svs_header csv for pytest.

    """
    svs_header_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_svs_header.csv"
    )

    if not pathlib.Path(svs_header_file_path).is_file():
        print("\nDownloading sample patch extraction svs_header csv file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_svs_header.csv"
        )
        with open(svs_header_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction svs_header csv file")

    return svs_header_file_path


@pytest.fixture(scope="session")
def _patch_extr_svs_npy_read(tmpdir_factory):
    """Sample pytest fixture for sample patch extraction svs_read npy.
    Download sample patch extraction svs_read npy for pytest.

    """
    svs_read_file_path = tmpdir_factory.mktemp("data").join(
        "sample_patch_extraction_svsread.npy"
    )

    if not pathlib.Path(svs_read_file_path).is_file():
        print("\nDownloading sample patch extraction svs_read npy file")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/"
            "patchextraction/sample_patch_extraction_svsread.npy"
        )
        with open(svs_read_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping sample patch extraction svs_read npy file")

    return svs_read_file_path


@pytest.fixture(scope="session")
def _sample_patch1(tmpdir_factory):
    """Sample pytest fixture for sample patch 1.
    Download sample patch 1 for pytest.

    """
    patch_file_path = tmpdir_factory.mktemp("data").join("kather_patch1.tif")

    if not pathlib.Path(patch_file_path).is_file():
        print("\nDownloading sample patch 1")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/models/kather_patch1.tif"
        )
        with open(patch_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping Source Image")

    return patch_file_path


@pytest.fixture(scope="session")
def _sample_patch2(tmpdir_factory):
    """Sample pytest fixture for sample patch 2.
    Download sample patch 2 for pytest.

    """
    patch_file_path = tmpdir_factory.mktemp("data").join("kather_patch2.tif")

    if not pathlib.Path(patch_file_path).is_file():
        print("\nDownloading sample patch 2")
        r = requests.get(
            "https://tiatoolbox.dcs.warwick.ac.uk/testdata/models/kather_patch2.tif"
        )
        with open(patch_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping Source Image")

    return patch_file_path


@pytest.fixture(scope="session")
def _dir_sample_patches(_sample_patch1, _sample_patch2, tmpdir_factory):
    """Directory of sample image patches for testing."""
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_sample_patch1.basename).symlink_to(_sample_patch2)
        dir_path.joinpath(_sample_patch2.basename).symlink_to(_sample_patch2)
    except OSError:
        shutil.copy(_sample_patch1, dir_path.joinpath(_sample_patch1.basename))
        shutil.copy(_sample_patch2, dir_path.joinpath(_sample_patch2.basename))

    return dir_path


@pytest.fixture(scope="session")
def _sample_wsi_dict(tmpdir_factory):
    """Sample pytest fixture for torch wsi dataset.
    Download svs image for pytest.

    """
    file_name_dict = {
        "wsi1_8k_8k_svs": "wsi1_8k_8k.svs",
        "wsi1_8k_8k_jp2": "wsi1_8k_8k.jp2",
        "wsi1_8k_8k_jpg": "wsi1_8k_8k.jpg",
        "wsi2_4k_4k_svs": "wsi2_4k_4k.svs",
        "wsi2_4k_4k_jp2": "wsi2_4k_4k.jp2",
        "wsi2_4k_4k_jpg": "wsi2_4k_4k.jpg",
        "wsi2_4k_4k_msk": "wsi2_4k_4k.mask.png",
        "wsi2_4k_4k_pred": "wsi2_4k_4k.pred.dat",
    }

    info_dict = {}
    URL_HOME = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/models/new"
    for file_code, file_name in file_name_dict.items():
        file_path = tmpdir_factory.mktemp("data").join(file_name)
        if not pathlib.Path(file_path).is_file():
            print("\nDownloading %s" % file_path)
            r = requests.get("%s/%s" % (URL_HOME, file_name))
            with open(file_path, "wb") as f:
                f.write(r.content)
        else:
            print("\nSkipping %s" % file_path)
        info_dict[file_code] = file_path
    return info_dict
