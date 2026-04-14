"""Unit test package for KongNet Model."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from tiatoolbox import rcParam
from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture.kongnet import (
    CenterBlock,
    DecoderBlock,
    KongNet,
    KongNetInstancePostProcResult,
    KongNetDecoder,
    SubPixelUpsample,
    TimmEncoderFixed,
)
from tiatoolbox.models.engine.io_config import IOSegmentorConfig
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"
ON_GPU = toolbox_env.has_gpu()


def test_timm_encoder_fixed_with_drop_path() -> None:
    """Test TimmEncoderFixed encoder with drop_path_rate."""
    encoder = TimmEncoderFixed(
        name="resnet18",
        pretrained=False,
        in_channels=3,
        depth=5,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.2,
    )
    assert encoder is not None

    # Test forward pass
    input_tensor = torch.randn(2, 3, 64, 64)
    features = encoder(input_tensor)
    assert len(features) == 6  # input + 5 levels
    assert features[0].shape == (2, 3, 64, 64)  # First is input

    # Test properties
    out_channels = encoder.out_channels
    assert len(out_channels) == 6
    assert out_channels[0] == 3

    output_stride = encoder.output_stride
    assert output_stride == 32


def test_timm_encoder_fixed_without_drop_path() -> None:
    """Test TimmEncoderFixed encoder without drop_path_rate (None)."""
    encoder = TimmEncoderFixed(
        name="resnet18",
        pretrained=False,
        in_channels=3,
        depth=5,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=None,
    )
    assert encoder is not None

    # Test forward pass
    input_tensor = torch.randn(2, 3, 64, 64)
    features = encoder(input_tensor)
    assert len(features) == 6


def test_timm_encoder_fixed_output_stride_limit() -> None:
    """Test TimmEncoderFixed output_stride calculation."""
    encoder = TimmEncoderFixed(
        name="resnet18",
        pretrained=False,
        in_channels=3,
        depth=3,
        output_stride=32,
        drop_rate=0.5,
    )
    # With depth=3, max output_stride is 2^3 = 8
    assert encoder.output_stride == 8


def test_sub_pixel_upsample() -> None:
    """Test SubPixelUpsample module."""
    upsample = SubPixelUpsample(
        in_channels=32,
        out_channels=16,
        upscale_factor=2,
    )
    assert upsample is not None

    # Test forward pass
    input_tensor = torch.randn(1, 32, 8, 8)
    output = upsample(input_tensor)
    assert output.shape == (1, 16, 16, 16)  # 2x upsampling


def test_decoder_block_with_skip() -> None:
    """Test DecoderBlock with skip connection."""
    decoder_block = DecoderBlock(
        in_channels=64,
        skip_channels=32,
        out_channels=32,
        attention_type="scse",
    )
    assert decoder_block is not None

    # Test forward pass with skip
    input_tensor = torch.randn(1, 64, 4, 4)
    skip_tensor = torch.randn(1, 32, 8, 8)
    output = decoder_block(input_tensor, skip_tensor)
    assert output.shape == (1, 32, 8, 8)


def test_decoder_block_without_skip() -> None:
    """Test DecoderBlock without skip connection."""
    decoder_block = DecoderBlock(
        in_channels=64,
        skip_channels=0,
        out_channels=32,
        attention_type="scse",
    )

    # Test forward pass without skip
    input_tensor = torch.randn(1, 64, 4, 4)
    output = decoder_block(input_tensor, skip=None)
    assert output.shape == (1, 32, 8, 8)


def test_center_block() -> None:
    """Test CenterBlock module."""
    center_block = CenterBlock(in_channels=64)
    assert center_block is not None

    # Test forward pass
    input_tensor = torch.randn(1, 64, 4, 4)
    output = center_block(input_tensor)
    assert output.shape == (1, 64, 4, 4)


def test_kongnet_decoder() -> None:
    """Test KongNetDecoder module."""
    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_channels = (256, 128, 64, 32, 16)

    decoder = KongNetDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        n_blocks=5,
        attention_type="scse",
        center=True,
    )
    assert decoder is not None

    # Create mock encoder features
    features = [
        torch.randn(2, 3, 256, 256),
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 1024, 8, 8),
    ]

    output = decoder(*features)
    assert output.shape == (2, 16, 256, 256)


def test_kongnet_decoder_without_center() -> None:
    """Test KongNetDecoder module without center block."""
    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_channels = (256, 128, 64, 32, 16)

    decoder = KongNetDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        n_blocks=5,
        attention_type="scse",
        center=False,
    )
    assert decoder is not None

    # Create mock encoder features
    features = [
        torch.randn(2, 3, 256, 256),
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 1024, 8, 8),
    ]

    output = decoder(*features)
    assert output.shape == (2, 16, 256, 256)


def test_kongnet_decoder_mismatch_error() -> None:
    """Test KongNetDecoder raises error when n_blocks doesn't match decoder_channels."""
    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_channels = (256, 128, 64, 32, 16)

    with pytest.raises(
        ValueError,
        match=r"The number of blocks 3 must match the length of decoder_channels 5.",
    ):
        KongNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=3,  # Mismatch: decoder_channels has 5 elements
            attention_type="scse",
            center=True,
        )


def test_kongnet_head_mismatch_error() -> None:
    """Test KongNet head_mismatch_error.

    Raise error when num_channels_per_head length doesn't match num_heads.

    """
    with pytest.raises(
        ValueError, match=r"Number of decoders 3 must match number of heads 6."
    ):
        KongNet(
            num_heads=6,
            num_channels_per_head=[3, 3, 3],  # Only 3 elements
            target_channels=[2, 5, 8, 11, 14, 17],
            min_distance=5,
            threshold_abs=0.5,
        )


def test_kongnet_target_channel_validation() -> None:
    """KongNet should reject target channels outside the concatenated output."""
    with pytest.raises(ValueError, match="target_channels"):
        KongNet(
            num_heads=2,
            num_channels_per_head=[2, 2],
            target_channels=[4],
            min_distance=5,
            threshold_abs=0.5,
        )


def test_kongnet_training_output_spec_uses_named_targeted_heads() -> None:
    """KongNet should expose stable named output metadata for training."""
    model = KongNet(
        num_heads=3,
        num_channels_per_head=[3, 3, 3],
        target_channels=[5, 8],
        min_distance=5,
        threshold_abs=0.5,
        class_dict={0: "Tumour Cell", 1: "Lymphocyte"},
    )

    specs = model.training_output_spec

    assert [spec.name for spec in specs] == ["head_0", "tumour_cell", "lymphocyte"]
    assert specs[0].channel_slice == slice(0, 3)
    assert specs[0].target_channels == ()
    assert specs[1].channel_slice == slice(3, 6)
    assert specs[1].target_channels == (5,)
    assert specs[1].target_channel_offsets == (2,)
    assert specs[2].channel_slice == slice(6, 9)
    assert specs[2].target_channels == (8,)
    assert specs[2].target_channel_offsets == (2,)


def test_kongnet_preproc() -> None:
    """Test KongNet preproc static method."""
    # Create a random uint8 image
    rng = np.random.default_rng(1337)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    # Apply preprocessing
    processed = KongNet.preproc(image)

    # Check shape is preserved
    assert processed.shape == (64, 64, 3)

    # Check dtype is float
    assert processed.dtype in [np.float32, np.float64]

    # Check normalization (values should be roughly in range of normalized ImageNet)
    assert processed.min() >= -3.0  # Roughly min after normalization
    assert processed.max() <= 3.0  # Roughly max after normalization


def test_kongnet_postproc() -> None:
    """Test KongNet postproc method."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[2, 2],
        target_channels=[0, 2],
        min_distance=5,
        threshold_abs=0.5,
    )

    # Create a mock probability map
    rng = np.random.default_rng(1337)
    block = rng.random((64, 64, 2), dtype=np.float32)

    # Add some peaks
    block[15, 15, 0] = 0.9
    block[45, 45, 1] = 0.9

    # Apply postprocessing
    output = model.postproc(block)

    # Check shape is preserved
    assert output.shape == (64, 64, 2)

    # Output should contain detected peaks
    assert output.max() > 0


def _disk_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Create a filled disk mask for synthetic KongNet tests."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return ((rows - center[0]) ** 2 + (cols - center[1]) ** 2) <= radius**2


def test_kongnet_extract_component_maps_requires_full_head_output() -> None:
    """KongNet instance post-processing should reject centroid-only outputs."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[3, 3],
        target_channels=[2, 5],
        min_distance=5,
        threshold_abs=0.35,
    )

    centroid_only_output = np.zeros((32, 32, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="full-head channels"):
        model.extract_component_maps(centroid_only_output, from_logits=False)


def test_kongnet_postproc_instance_class_maps() -> None:
    """KongNet should build deterministic CoNIC-style maps from full-head output."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[3, 3],
        target_channels=[2, 5],
        min_distance=4,
        threshold_abs=0.35,
        class_dict={1: "Tumour", 2: "Lymphocyte"},
    )

    logits = np.full((6, 64, 64), -8.0, dtype=np.float32)
    left_nucleus = _disk_mask((64, 64), (20, 18), 7)
    left_boundary = left_nucleus & ~_disk_mask((64, 64), (20, 18), 5)
    right_nucleus = _disk_mask((64, 64), (20, 46), 7)
    right_boundary = right_nucleus & ~_disk_mask((64, 64), (20, 46), 5)

    logits[0][left_nucleus] = 8.0
    logits[1][left_boundary] = 8.0
    logits[2, 20, 18] = 10.0

    logits[3][right_nucleus] = 8.0
    logits[4][right_boundary] = 8.0
    logits[5, 20, 46] = 10.0

    result = model.postproc_instance_class_maps(
        logits,
        from_logits=True,
        mask_threshold=0.5,
        boundary_weight=1.0,
        min_instance_size=10,
    )

    assert isinstance(result, KongNetInstancePostProcResult)
    assert result.instance_map.shape == (64, 64)
    assert result.class_map.shape == (64, 64)
    assert result.conic_map.shape == (64, 64, 2)
    assert set(np.unique(result.instance_map)) == {0, 1, 2}
    assert result.class_map[20, 18] == 1
    assert result.class_map[20, 46] == 2
    assert result.instance_map[20, 18] != 0
    assert result.instance_map[20, 46] != 0
    assert result.marker_map[20, 18] != 0
    assert result.marker_map[20, 46] != 0
    assert np.count_nonzero(result.peak_map) == 2
    assert result.instance_classes == {1: 1, 2: 2}


def test_kongnet_load_state_dict() -> None:
    """Test KongNet load_state_dict method."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[3, 3],
        target_channels=[0, 3],
        min_distance=5,
        threshold_abs=0.5,
    )

    original_state = model.state_dict()
    mock_state_dict = {"model": original_state}

    # Load state dict
    model.load_state_dict(mock_state_dict, strict=True)

    # Verify it loaded successfully
    new_state = model.state_dict()
    assert len(new_state) == len(original_state)


def test_kongnet_wide_decoder() -> None:
    """Test KongNet with wide_decoder option."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[2, 2],
        target_channels=[0, 2],
        min_distance=5,
        threshold_abs=0.5,
        wide_decoder=True,
    )
    assert model is not None

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 128, 128).to(device)
        output = model(input_tensor)
        assert output.shape == (1, 4, 128, 128)


def test_kongnet_modeling() -> None:
    """Test for KongNet model."""
    # test creation
    model = KongNet(
        num_heads=3,
        num_channels_per_head=[2, 2, 2],
        target_channels=[1, 3, 5],
        min_distance=5,
        threshold_abs=0.5,
        wide_decoder=False,
        class_dict=None,
        tile_shape=(512, 512),
    )
    assert model is not None

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 128, 128).to(device)
        output = model(input_tensor)
        assert output.shape == (1, 6, 128, 128)

        batch_tensor = torch.randn(1, 128, 128, 3).to(device)
        output = KongNet.infer_batch(model, batch_tensor, device=device)
        assert output.shape == (1, 128, 128, 3)


def test_pretrained_model_creation() -> None:
    """Test for get_pretrained_model function."""
    pretrained_info = rcParam["pretrained_model_info"]
    pretrained_model_names = [
        "KongNet_CoNIC_1",
        "KongNet_MONKEY_1",
        "KongNet_PanNuke_1",
        "KongNet_PUMA_T1_3",
        "KongNet_PUMA_T2_3",
        "KongNet_Det_MIDOG_1",
    ]

    for model_name in pretrained_model_names:
        info = pretrained_info[model_name]
        arch_info = info["architecture"]
        model = KongNet(**arch_info["kwargs"])

        assert (
            model.target_channels == info["architecture"]["kwargs"]["target_channels"]
        )
        assert model.min_distance == info["architecture"]["kwargs"]["min_distance"]
        assert model.threshold_abs == info["architecture"]["kwargs"]["threshold_abs"]
        assert model.tile_shape == info["architecture"]["kwargs"]["tile_shape"]

        io_info = info["ioconfig"]
        ioconfig = IOSegmentorConfig(**io_info["kwargs"])
        assert ioconfig.input_resolutions == io_info["kwargs"]["input_resolutions"]
        assert ioconfig.output_resolutions == io_info["kwargs"]["output_resolutions"]
        assert ioconfig.patch_input_shape == io_info["kwargs"]["patch_input_shape"]
        assert ioconfig.patch_output_shape == io_info["kwargs"]["patch_output_shape"]
        assert ioconfig.stride_shape == io_info["kwargs"]["stride_shape"]
        assert ioconfig.save_resolution == io_info["kwargs"]["save_resolution"]


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_kongnet_wsi_inference(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for KongNet model WSI inference."""
    sample_wsi = Path(remote_sample("wsi1_2k_2k_svs"))

    detector = NucleusDetector(model="KongNet_CoNIC_1")
    out = detector.run(
        images=[sample_wsi],
        patch_mode=False,
        device="cuda",
        save_dir=track_tmp_path,
        overwrite=True,
        output_type="annotationstore",
        auto_get_mask=True,
        memory_threshold=50,
        batch_size=4,
    )

    annotation_store_path = out[sample_wsi]
    assert Path(annotation_store_path).exists()
    store = SQLiteStore.open(annotation_store_path)
    assert 900 < len(store) < 1100

    detector = NucleusDetector(model="KongNet_Det_MIDOG_1")
    out = detector.run(
        images=[sample_wsi],
        patch_mode=False,
        device="cuda",
        save_dir=track_tmp_path,
        overwrite=True,
        output_type="annotationstore",
        auto_get_mask=True,
        memory_threshold=50,
        batch_size=4,
    )

    annotation_store_path = out[sample_wsi]
    assert Path(annotation_store_path).exists()
    store = SQLiteStore.open(annotation_store_path)
    assert len(store) == 0
