"""Unit test package for EfficientNet-UNet Tissue Mask Model."""

from collections.abc import Callable
from pathlib import Path

import dask.array as da
import numpy as np
import torch
from torch import nn

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.architecture.tissue_mask_model import (
    Conv2dReLU,
    Conv2dStaticSamePadding,
    EfficientNetEncoder,
    EfficientNetUnet,
    MBConvBlock,
    SegmentationHead,
    SiLU,
    UnetDecoder,
    UnetDecoderBlock,
)
from tiatoolbox.models.engine.io_config import IOSegmentorConfig
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_functional_efficientnetunet() -> None:
    """Test for EfficientNetUnet model."""
    # test fetch pretrained weights
    pretrained_weights = fetch_pretrained_weights("efficientunet-tissue_mask")
    assert pretrained_weights is not None

    # test creation
    model = EfficientNetUnet(num_classes=1)
    assert model is not None

    # load pretrained weights
    pretrained = torch.load(pretrained_weights, map_location=device)
    model.load_state_dict(pretrained)

    # test get pretrained model
    model, ioconfig = get_pretrained_model("efficientunet-tissue_mask")
    assert isinstance(model, EfficientNetUnet)
    assert isinstance(ioconfig, IOSegmentorConfig)

    # test inference
    generator = np.random.default_rng(1337)
    test_image = generator.integers(0, 256, size=(2048, 2048, 3), dtype=np.uint8)
    reader = VirtualWSIReader.open(test_image)
    read_kwargs = {"resolution": 0, "units": "level", "coord_space": "resolution"}
    batch = np.array(
        [
            reader.read_bounds((0, 0, 512, 512), **read_kwargs),
            reader.read_bounds((512, 512, 1024, 1024), **read_kwargs),
        ],
    )
    batch = torch.from_numpy(batch)
    output = model.infer_batch(model, batch, device=device)
    assert output.shape == (2, 512, 512, 1)


def test_efficientnetunet_with_semantic_segmentor(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test EfficientNetUnet tissue mask generation."""
    segmentor = SemanticSegmentor(model="efficientunet-tissue_mask")

    sample_image = remote_sample("svs-1-small")
    inputs = [str(sample_image)]

    output = segmentor.run(
        images=inputs,
        device=device,
        patch_mode=False,
        output_type="annotationstore",
        save_dir=track_tmp_path / "efficientnetunet_test_outputs",
        overwrite=True,
        class_dict={0: "background", 1: "tissue"},
    )

    assert len(output) == 1
    assert Path(output[sample_image]).exists()

    store = SQLiteStore.open(output[sample_image])
    assert len(store) > 0

    unique_types = set()

    for annotation in store.values():
        unique_types.add(annotation.properties["type"])

    assert unique_types == {"background", "tissue"}

    tissue_area_px = 0.0
    for annotation in store.values():
        unique_types.add(annotation.properties["type"])
        if annotation.properties["type"] == "tissue":
            tissue_area_px += annotation.geometry.area
    assert 1800000 < tissue_area_px < 20000000

    store.close()


def test_silu_activation() -> None:
    """Test SiLU activation function."""
    activation = SiLU()
    x = torch.randn(1, 64, 32, 32)
    output = activation(x)
    assert output.shape == (1, 64, 32, 32)

    # Test that SiLU computes x * sigmoid(x)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(output, expected)


def test_conv2d_static_same_padding() -> None:
    """Test Conv2dStaticSamePadding layer."""
    # Test with stride=1
    conv = Conv2dStaticSamePadding(32, 64, kernel_size=3, stride=1)
    x = torch.randn(1, 32, 128, 128)
    output = conv(x)
    assert output.shape == (1, 64, 128, 128)

    # Test with stride=2
    conv = Conv2dStaticSamePadding(32, 64, kernel_size=3, stride=2)
    x = torch.randn(1, 32, 128, 128)
    output = conv(x)
    assert output.shape == (1, 64, 64, 64)

    # Test with tuple kernel_size and stride
    conv = Conv2dStaticSamePadding(32, 64, kernel_size=(3, 3), stride=(2, 2))
    x = torch.randn(1, 32, 128, 128)
    output = conv(x)
    assert output.shape == (1, 64, 64, 64)

    # Test with groups parameter
    conv = Conv2dStaticSamePadding(32, 32, kernel_size=3, stride=1, groups=32)
    x = torch.randn(1, 32, 64, 64)
    output = conv(x)
    assert output.shape == (1, 32, 64, 64)

    # Test with dilation
    conv = Conv2dStaticSamePadding(32, 64, kernel_size=3, stride=1, dilation=2)
    x = torch.randn(1, 32, 64, 64)
    output = conv(x)
    assert output.shape[1] == 64

    # Test static_padding attribute exists
    assert isinstance(conv.static_padding, nn.Identity)


def test_mbconv_block_with_residual() -> None:
    """Test MBConvBlock with residual connection."""
    # Test with residual (same in/out channels, stride=1)
    block = MBConvBlock(
        in_planes=32, out_planes=32, expand_ratio=6, kernel_size=3, stride=1
    )
    x = torch.randn(1, 32, 64, 64)
    output = block(x)
    assert output.shape == (1, 32, 64, 64)
    assert block.use_residual is True


def test_mbconv_block_without_residual() -> None:
    """Test MBConvBlock without residual connection."""
    # Test without residual (different in/out channels)
    block = MBConvBlock(
        in_planes=32, out_planes=64, expand_ratio=6, kernel_size=3, stride=2
    )
    x = torch.randn(1, 32, 64, 64)
    output = block(x)
    assert output.shape == (1, 64, 32, 32)
    assert block.use_residual is False


def test_mbconv_block_no_expansion() -> None:
    """Test MBConvBlock with expand_ratio=1 (no expansion)."""
    # Test with expand_ratio=1 (no expansion phase)
    block = MBConvBlock(
        in_planes=32, out_planes=64, expand_ratio=1, kernel_size=3, stride=1
    )
    x = torch.randn(1, 32, 64, 64)
    output = block(x)
    assert output.shape == (1, 64, 64, 64)

    # Check that expand layers are Identity
    assert isinstance(block._expand_conv, nn.Identity)
    assert isinstance(block._bn0, nn.Identity)


def test_mbconv_block_different_reduction_ratios() -> None:
    """Test MBConvBlock with different reduction ratios."""
    # Test with different reduction_ratio
    block = MBConvBlock(
        in_planes=32,
        out_planes=64,
        expand_ratio=6,
        kernel_size=5,
        stride=1,
        reduction_ratio=8,
    )
    x = torch.randn(1, 32, 64, 64)
    output = block(x)
    assert output.shape == (1, 64, 64, 64)


def test_efficientnet_encoder() -> None:
    """Test EfficientNetEncoder forward pass and output shapes."""
    encoder = EfficientNetEncoder()
    x = torch.randn(1, 3, 224, 224)
    features = encoder(x)

    assert len(features) == 5
    assert features[0].shape == (1, 32, 112, 112)
    assert features[1].shape == (1, 24, 56, 56)
    assert features[2].shape == (1, 40, 28, 28)
    assert features[3].shape == (1, 112, 14, 14)
    assert features[4].shape == (1, 320, 7, 7)


def test_conv2d_relu() -> None:
    """Test Conv2dReLU block."""
    block = Conv2dReLU(in_channels=32, out_channels=64)
    x = torch.randn(1, 32, 128, 128)
    output = block(x)
    assert output.shape == (1, 64, 128, 128)

    # Test with custom kernel size and padding
    block = Conv2dReLU(in_channels=32, out_channels=64, kernel_size=5, padding=2)
    x = torch.randn(1, 32, 128, 128)
    output = block(x)
    assert output.shape == (1, 64, 128, 128)


def test_unet_decoder_block_with_skip() -> None:
    """Test UnetDecoderBlock with skip connection."""
    block = UnetDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
    input_tensor = torch.randn(1, 128, 32, 32)
    skip = torch.randn(1, 64, 64, 64)
    output = block(input_tensor, skip)
    assert output.shape == (1, 64, 64, 64)


def test_unet_decoder_block_without_skip() -> None:
    """Test UnetDecoderBlock without skip connection."""
    block = UnetDecoderBlock(in_channels=128, skip_channels=0, out_channels=64)
    input_tensor = torch.randn(1, 128, 32, 32)
    output = block(input_tensor, skip=None)
    assert output.shape == (1, 64, 64, 64)


def test_unet_decoder_block_mismatched_shapes() -> None:
    """Test UnetDecoderBlock with mismatched input and skip shapes."""
    block = UnetDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
    input_tensor = torch.randn(1, 128, 32, 32)
    # Skip with slightly different spatial dimensions
    skip = torch.randn(1, 64, 65, 65)
    output = block(input_tensor, skip)
    assert output.shape == (1, 64, 65, 65)


def test_unet_decoder() -> None:
    """Test UnetDecoder forward pass."""
    decoder = UnetDecoder()

    # Generate dummy feature maps matching encoder output
    features = [
        torch.randn(1, 32, 112, 112),
        torch.randn(1, 24, 56, 56),
        torch.randn(1, 40, 28, 28),
        torch.randn(1, 112, 14, 14),
        torch.randn(1, 320, 7, 7),
    ]

    output = decoder(features)

    # Output should be upsampled back to close to original resolution
    assert output.shape == (1, 16, 224, 224)


def test_segmentation_head() -> None:
    """Test SegmentationHead layer."""
    head = SegmentationHead(in_channels=16, out_channels=1)
    x = torch.randn(1, 16, 224, 224)
    output = head(x)
    assert output.shape == (1, 1, 224, 224)

    # Test with different kernel size
    head = SegmentationHead(in_channels=16, out_channels=2, kernel_size=1)
    x = torch.randn(1, 16, 224, 224)
    output = head(x)
    assert output.shape == (1, 2, 224, 224)


def test_efficientnetunet_preproc_imagenet_norm() -> None:
    """Test EfficientNetUnet preprocessing with ImageNet normalization."""
    # Create a test image with known values
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    processed = EfficientNetUnet.preproc(img)

    # Check that normalization is applied correctly
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    expected = (128 / 255.0 - mean) / std

    assert np.allclose(processed[0, 0, :], expected, rtol=1e-5)


def test_efficientnetunet_postproc_threshold() -> None:
    """Test EfficientNetUnet postprocessing with different thresholds."""
    model = EfficientNetUnet(num_classes=1, threshold=0.5)

    # Create probability map with values above and below threshold
    probs = np.array([[[0.3]], [[0.7]], [[0.5]], [[0.9]]])
    probs = probs.reshape(2, 2, 1).astype(np.float32)

    mask = model.postproc(probs)
    assert mask.shape == (2, 2)
    assert mask.dtype == np.uint8

    # Note: morphological operations may change exact values,
    # so we just check shape and dtype


def test_efficientnetunet_postproc_with_dask() -> None:
    """Test EfficientNetUnet postprocessing with dask arrays."""
    model = EfficientNetUnet(num_classes=1)

    # Create dask array
    probs = da.random.random((256, 256, 1), chunks=(128, 128, 1))
    mask = model.postproc(probs)

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8


def test_efficientnetunet_infer_batch() -> None:
    """Test EfficientNetUnet batch inference."""
    model = EfficientNetUnet(num_classes=1)

    # Create batch of images
    batch = torch.randn(4, 256, 256, 3)
    probs = model.infer_batch(model, batch, device="cpu")

    assert probs.shape == (4, 256, 256, 1)
    assert isinstance(probs, np.ndarray)

    # Check that probabilities are in valid range [0, 1]
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_efficientnet_encoder_block_args() -> None:
    """Test EfficientNetEncoder block configuration."""
    encoder = EfficientNetEncoder()

    # Verify block_args configuration
    assert len(encoder.block_args) == 7

    # Check first block config [in_c, out_c, expand, k, s, repeats]
    assert encoder.block_args[0] == [32, 16, 1, 3, 1, 1]

    # Verify total number of blocks
    total_blocks = sum(args[5] for args in encoder.block_args)
    assert len(encoder._blocks) == total_blocks
