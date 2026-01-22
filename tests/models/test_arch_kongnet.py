"""Unit test package for KongNet Model."""

from collections.abc import Callable
from pathlib import Path
import pathlib

import numpy as np
import pytest
import torch
from torch import nn

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture.kongnet import (
    KongNet,
    TimmEncoderFixed,
    SubPixelUpsample,
    DecoderBlock,
    CenterBlock,
    KongNetDecoder,
)
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"
ON_GPU = toolbox_env.has_gpu()

def test_TimmEncoderFixed_with_drop_path() -> None:
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

def test_TimmEncoderFixed_without_drop_path() -> None:
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

def test_TimmEncoderFixed_output_stride_limit() -> None:
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

def test_SubPixelUpsample() -> None:
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

def test_DecoderBlock_with_skip() -> None:
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

def test_DecoderBlock_without_skip() -> None:
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

def test_CenterBlock() -> None:
    """Test CenterBlock module."""
    center_block = CenterBlock(in_channels=64)
    assert center_block is not None
    
    # Test forward pass
    input_tensor = torch.randn(1, 64, 4, 4)
    output = center_block(input_tensor)
    assert output.shape == (1, 64, 4, 4)

def test_KongNetDecoder() -> None:
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

def test_KongNetDecoder_without_center() -> None:
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

def test_KongNetDecoder_mismatch_error() -> None:
    """Test KongNetDecoder raises error when n_blocks doesn't match decoder_channels."""
    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_channels = (256, 128, 64, 32, 16)
    
    with pytest.raises(ValueError, match="Model depth is 3, but you provide"):
        KongNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=3,  # Mismatch: decoder_channels has 5 elements
            attention_type="scse",
            center=True,
        )

def test_KongNet_head_mismatch_error() -> None:
    """Test KongNet raises error when num_channels_per_head length doesn't match num_heads."""
    with pytest.raises(ValueError, match="Number of decoders .* must match"):
        KongNet(
            num_heads=6,
            num_channels_per_head=[3, 3, 3],  # Only 3 elements
            target_channels=[2, 5, 8, 11, 14, 17],
            min_distance=5,
            threshold_abs=0.5,
        )

def test_KongNet_preproc() -> None:
    """Test KongNet preproc static method."""
    # Create a random uint8 image
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Apply preprocessing
    processed = KongNet.preproc(image)
    
    # Check shape is preserved
    assert processed.shape == (64, 64, 3)
    
    # Check dtype is float
    assert processed.dtype in [np.float32, np.float64]
    
    # Check normalization (values should be roughly in range of normalized ImageNet)
    assert processed.min() >= -3.0  # Roughly min after normalization
    assert processed.max() <= 3.0   # Roughly max after normalization

def test_KongNet_postproc() -> None:
    """Test KongNet postproc method."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[2, 2],
        target_channels=[0, 2],
        min_distance=5,
        threshold_abs=0.5,
    )
    
    # Create a mock probability map
    block = np.random.rand(64, 64, 2).astype(np.float32)
    
    # Add some peaks
    block[15, 15, 0] = 0.9
    block[45, 45, 1] = 0.9
    
    # Apply postprocessing
    output = model.postproc(block)
    
    # Check shape is preserved
    assert output.shape == (64, 64, 2)
    
    # Output should contain detected peaks
    assert output.max() > 0

def test_KongNet_load_state_dict() -> None:
    """Test KongNet load_state_dict method."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[3, 3],
        target_channels=[0, 3],
        min_distance=5,
        threshold_abs=0.5,
    )
    
    original_state = model.state_dict()
    mock_state_dict = {'model': original_state}
    
    # Load state dict
    model.load_state_dict(mock_state_dict, strict=True)
    
    # Verify it loaded successfully
    new_state = model.state_dict()
    assert len(new_state) == len(original_state)

def test_KongNet_wide_decoder() -> None:
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


def test_KongNet_Modeling() -> None:
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
        postproc_tile_shape=(512, 512)
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

@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_KongNet_WSI_Inference(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for KongNet model WSI inference."""
    
    sample_wsi = Path(remote_sample("wsi1_2k_2k_svs"))

    detector = NucleusDetector(model='KongNet_CoNIC_1')
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
    assert 1000 < len(store)  < 1100


    detector = NucleusDetector(model='KongNet_Det_MIDOG_1')
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
    

    detector = NucleusDetector(model='KongNet_Det_MIDOG_1')
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


