"""Unit test package for HoVerNet+."""

import copy

# ! The garbage collector
import gc
import multiprocessing
import shutil
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pytest

from tiatoolbox.models import IOSegmentorConfig, MultiTaskSegmentor, SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imwrite
from tiatoolbox.utils.metrics import f1_detection
from tiatoolbox.utils.misc import select_device

ON_GPU = toolbox_env.has_gpu()
BATCH_SIZE = 1 if not ON_GPU else 8  # 16
try:
    NUM_POSTPROC_WORKERS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_POSTPROC_WORKERS = 2

# ----------------------------------------------------


def _crash_func(_: object) -> None:
    """Helper to induce crash."""
    msg = "Propagation Crash."
    raise ValueError(msg)


def semantic_postproc_func(raw_output: np.ndarray) -> np.ndarray:
    """Function to post process semantic segmentations.

    Post processes semantic segmentation to form one map output.

    """
    return np.argmax(raw_output, axis=-1)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_functionality_local(remote_sample: Callable, tmp_path: Path) -> None:
    """Local functionality test for multi task segmentor."""
    gc.collect()
    root_save_dir = Path(tmp_path)
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    save_dir = root_save_dir / "multitask"
    shutil.rmtree(save_dir, ignore_errors=True)

    # * generate full output w/o parallel post-processing worker first
    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict_a = joblib.load(f"{output[0][1]}.0.dat")

    # * then test run when using workers, will then compare results
    # * to ensure the predictions are the same
    shutil.rmtree(save_dir, ignore_errors=True)
    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    assert multi_segmentor.num_postproc_workers == NUM_POSTPROC_WORKERS
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict_b = joblib.load(f"{output[0][1]}.0.dat")
    layer_map_b = np.load(f"{output[0][1]}.1.npy")
    assert len(inst_dict_b) > 0, "Must have some nuclei"
    assert layer_map_b is not None, "Must have some layers."

    inst_coords_a = np.array([v["centroid"] for v in inst_dict_a.values()])
    inst_coords_b = np.array([v["centroid"] for v in inst_dict_b.values()])
    score = f1_detection(inst_coords_b, inst_coords_a, radius=1.0)
    assert score > 0.95, "Heavy loss of precision!"


def test_functionality_hovernetplus(remote_sample: Callable, tmp_path: Path) -> None:
    """Functionality test for multitask segmentor."""
    root_save_dir = Path(tmp_path)
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))
    required_dims = (258, 258)
    # above image is 512 x 512 at 0.252 mpp resolution. This is 258 x 258 at 0.500 mpp.

    save_dir = f"{root_save_dir}/multi/"
    shutil.rmtree(save_dir, ignore_errors=True)

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")
    layer_map = np.load(f"{output[0][1]}.1.npy")

    assert len(inst_dict) > 0, "Must have some nuclei."
    assert layer_map is not None, "Must have some layers."
    assert (
        layer_map.shape == required_dims
    ), "Output layer map dimensions must be same as the expected output shape"


def test_functionality_hovernet(remote_sample: Callable, tmp_path: Path) -> None:
    """Functionality test for multitask segmentor."""
    root_save_dir = Path(tmp_path)
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    save_dir = root_save_dir / "multi"
    shutil.rmtree(save_dir, ignore_errors=True)

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")

    assert len(inst_dict) > 0, "Must have some nuclei."


def test_masked_segmentor(remote_sample: Callable, tmp_path: Path) -> None:
    """Test segmentor when image is masked."""
    root_save_dir = Path(tmp_path)
    sample_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = tmp_path.joinpath("small_svs_tissue_mask.jpg")

    save_dir = root_save_dir / "instance"

    # resolution for travis testing, not the correct ones
    resolution = 4.0
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        margin=128,
        tile_shape=[512, 512],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )
    multi_segmentor = MultiTaskSegmentor(
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
        pretrained_model="hovernet_fast-pannuke",
    )

    output = multi_segmentor.predict(
        [sample_wsi_svs],
        masks=[sample_wsi_msk],
        mode="wsi",
        ioconfig=ioconfig,
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")

    assert len(inst_dict) > 0, "Must have some nuclei."


def test_functionality_process_instance_predictions(
    remote_sample: Callable,
    tmp_path: Path,
) -> None:
    """Test the functionality of instance predictions processing."""
    root_save_dir = Path(tmp_path)
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    save_dir = root_save_dir / "semantic"
    shutil.rmtree(save_dir, ignore_errors=True)

    semantic_segmentor = SemanticSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
    )
    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
    )

    output = semantic_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )
    raw_maps = [np.load(f"{output[0][1]}.raw.{head_idx}.npy") for head_idx in range(4)]

    dummy_reference = [{i: {"box": np.array([0, 0, 32, 32])} for i in range(1000)}]

    dummy_tiles = [np.zeros((512, 512))]
    dummy_bounds = np.array([0, 0, 512, 512])

    multi_segmentor.wsi_layers = [np.zeros_like(raw_maps[0][..., 0])]
    multi_segmentor._wsi_inst_info = copy.deepcopy(dummy_reference)
    multi_segmentor._futures = [
        [dummy_reference, [dummy_reference[0].keys()], dummy_tiles, dummy_bounds],
    ]
    multi_segmentor._merge_post_process_results()
    assert len(multi_segmentor._wsi_inst_info[0]) == 0


def test_empty_image(tmp_path: Path) -> None:
    """Test MultiTaskSegmentor for an empty image."""
    root_save_dir = Path(tmp_path)
    sample_patch = np.ones((256, 256, 3), dtype="uint8") * 255
    sample_patch_path = root_save_dir / "sample_tile.png"
    imwrite(sample_patch_path, sample_patch)

    save_dir = root_save_dir / "hovernetplus"

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
    )

    _ = multi_segmentor.predict(
        [sample_patch_path],
        mode="tile",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    save_dir = root_save_dir / "hovernet"

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
    )

    _ = multi_segmentor.predict(
        [sample_patch_path],
        mode="tile",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
    )

    save_dir = root_save_dir / "semantic"

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="fcn_resnet50_unet-bcss",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
        output_types=["semantic"],
    )

    bcc_wsi_ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        output_resolutions=[{"units": "mpp", "resolution": 0.25}],
        tile_shape=2048,
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        margin=128,
        save_resolution={"units": "mpp", "resolution": 2},
    )

    _ = multi_segmentor.predict(
        [sample_patch_path],
        mode="tile",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
        ioconfig=bcc_wsi_ioconfig,
    )


def test_functionality_semantic(remote_sample: Callable, tmp_path: Path) -> None:
    """Functionality test for multitask segmentor."""
    root_save_dir = Path(tmp_path)

    save_dir = root_save_dir / "multi"
    shutil.rmtree(save_dir, ignore_errors=True)
    with pytest.raises(
        ValueError,
        match=r"Output type must be specified for instance or semantic segmentation.",
    ):
        MultiTaskSegmentor(
            pretrained_model="fcn_resnet50_unet-bcss",
            batch_size=BATCH_SIZE,
            num_postproc_workers=NUM_POSTPROC_WORKERS,
        )

    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))
    save_dir = f"{root_save_dir}/multi/"

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="fcn_resnet50_unet-bcss",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
        output_types=["semantic"],
    )

    bcc_wsi_ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        output_resolutions=[{"units": "mpp", "resolution": 0.25}],
        tile_shape=2048,
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        margin=128,
        save_resolution={"units": "mpp", "resolution": 2},
    )

    multi_segmentor.model.postproc_func = semantic_postproc_func

    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        crash_on_exception=True,
        save_dir=save_dir,
        ioconfig=bcc_wsi_ioconfig,
    )

    layer_map = np.load(f"{output[0][1]}.0.npy")

    assert layer_map is not None, "Must have some segmentations."


def test_crash_segmentor(remote_sample: Callable, tmp_path: Path) -> None:
    """Test engine crash when given malformed input."""
    root_save_dir = Path(tmp_path)
    sample_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = tmp_path.joinpath("small_svs_tissue_mask.jpg")

    save_dir = f"{root_save_dir}/multi/"

    # resolution for travis testing, not the correct ones
    resolution = 4.0
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        margin=128,
        tile_shape=[512, 512],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )
    multi_segmentor = MultiTaskSegmentor(
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
        pretrained_model="hovernetplus-oed",
    )

    # * Test crash propagation when parallelize post-processing
    shutil.rmtree(save_dir, ignore_errors=True)
    multi_segmentor.model.postproc_func = _crash_func
    with pytest.raises(ValueError, match=r"Crash."):
        multi_segmentor.predict(
            [sample_wsi_svs],
            masks=[sample_wsi_msk],
            mode="wsi",
            ioconfig=ioconfig,
            device=select_device(on_gpu=ON_GPU),
            crash_on_exception=True,
            save_dir=save_dir,
        )
