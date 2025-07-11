"""Define vanilla CNNs with torch backbones, mainly for patch classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import timm
import torch
import torchvision.models as torch_models
from timm.layers import SwiGLUPacked
from torch import nn

from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from torchvision.models import WeightsEnum


def _get_architecture(
    arch_name: str,
    weights: str or WeightsEnum = None,
    **kwargs: dict,
) -> list[nn.Sequential, ...] | nn.Sequential:
    """Retrieve a CNN model architecture.

    This function fetches a Convolutional Neural Network (CNN) model architecture,
    either predefined in torchvision or custom-made within tiatoolbox, for
    patch classification tasks.

    Args:
        arch_name (str):
            Name of the architecture (e.g. 'resnet50', 'alexnet').
        weights (str or WeightsEnum):
            Pretrained torchvision model weights to use (get_model_weights).
            Defaults to "DEFAULT".
        **kwargs (dict):
            Key-word arguments.

    Returns:
        list[nn.Sequential, ...] | nn.Sequential:
            A list of PyTorch network layers wrapped with `nn.Sequential`.

    Raises:
        ValueError:
            If `arch_name` is not supported.

    Example:
        >>> model = _get_architecture("resnet18")
        >>> print(model)

    """
    backbone_dict = {
        "alexnet": torch_models.alexnet,
        "resnet18": torch_models.resnet18,
        "resnet34": torch_models.resnet34,
        "resnet50": torch_models.resnet50,
        "resnet101": torch_models.resnet101,
        "resnext50_32x4d": torch_models.resnext50_32x4d,
        "resnext101_32x8d": torch_models.resnext101_32x8d,
        "wide_resnet50_2": torch_models.wide_resnet50_2,
        "wide_resnet101_2": torch_models.wide_resnet101_2,
        "densenet121": torch_models.densenet121,
        "densenet161": torch_models.densenet161,
        "densenet169": torch_models.densenet169,
        "densenet201": torch_models.densenet201,
        "inception_v3": torch_models.inception_v3,
        "googlenet": torch_models.googlenet,
        "mobilenet_v2": torch_models.mobilenet_v2,
        "mobilenet_v3_large": torch_models.mobilenet_v3_large,
        "mobilenet_v3_small": torch_models.mobilenet_v3_small,
    }
    if arch_name not in backbone_dict:
        msg = f"Backbone `{arch_name}` is not supported."
        raise ValueError(msg)

    creator = backbone_dict[arch_name]
    if "inception_v3" in arch_name or "googlenet" in arch_name:
        model = creator(weights=weights, aux_logits=False, num_classes=1000)
        return nn.Sequential(*list(model.children())[:-3])

    model = creator(weights=weights, **kwargs)
    # Unroll all the definition and strip off the final GAP and FCN
    if "resnet" in arch_name or "resnext" in arch_name:
        return nn.Sequential(*list(model.children())[:-2])
    if "densenet" in arch_name:
        return model.features
    if "alexnet" in arch_name:
        return model.features

    return model.features


def _get_timm_architecture(
    arch_name: str,
    *,
    pretrained: bool,
) -> list[nn.Sequential, ...] | nn.Sequential:
    """Retrieve a timm model architecture.

    This function fetches a model architecture from the timm library, specifically for
    pathology-related tasks.

    Args:
        arch_name (str):
            Name of the architecture (e.g. 'UNI', 'UN2', 'H-optimus-0',
            'efficientnet_b0', etc.).
        pretrained (bool, keyword-only):
            Whether to load pretrained weights.

    Returns:
        list[nn.Sequential, ...] | nn.Sequential:
            A ready-to-use timm model.

    Raises:
        ValueError:
            If the backbone architecture `arch_name` is not supported.

    Example:
        >>> model = _get_timm_architecture("UNI", pretrained=True)
        >>> print(model)

    """
    if arch_name in [f"efficientnet_b{i}" for i in range(8)]:
        model = timm.create_model(arch_name, pretrained=pretrained)
        return nn.Sequential(*list(model.children())[:-1])

    arch_map = {
        # UNI tile encoder: https://huggingface.co/MahmoodLab/UNI
        "UNI": {
            "model": "hf-hub:MahmoodLab/UNI",
            "init_values": 1e-5,
            "dynamic_img_size": True,
        },
        # Prov-GigaPath tile encoder: https://huggingface.co/prov-gigapath/prov-gigapath
        "prov-gigapath": {"model": "hf_hub:prov-gigapath/prov-gigapath"},
        # H-Optimus-0 tile encoder: https://huggingface.co/bioptimus/H-optimus-0
        "H-optimus-0": {
            "model": "hf-hub:bioptimus/H-optimus-0",
            "init_values": 1e-5,
            "dynamic_img_size": False,
        },
        # H-Optimus-1 tile encoder: https://huggingface.co/bioptimus/H-optimus-1
        "H-optimus-1": {
            "model": "hf-hub:bioptimus/H-optimus-1",
            "init_values": 1e-5,
            "dynamic_img_size": False,
        },
        # HO-mini tile encoder: https://huggingface.co/bioptimus/H0-mini
        "H0-mini": {
            "model": "hf-hub:bioptimus/H0-mini",
            "init_values": 1e-5,
            "dynamic_img_size": False,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        },
        # UNI2-h tile encoder: https://huggingface.co/MahmoodLab/UNI2-h
        "UNI2": {
            "model": "hf-hub:MahmoodLab/UNI2-h",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        # Virchow tile encoder: https://huggingface.co/paige-ai/Virchow
        "Virchow": {
            "model": "hf_hub:paige-ai/Virchow",
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        },
        # Virchow2 tile encoder: https://huggingface.co/paige-ai/Virchow2
        "Virchow2": {
            "model": "hf_hub:paige-ai/Virchow2",
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        },
        # Kaiko tile encoder:
        # https://huggingface.co/1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms
        "kaiko": {
            "model": (
                "hf_hub:1aurent/"
                "vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms"
            ),
            "dynamic_img_size": True,
        },
    }

    if arch_name in arch_map:  # pragma: no cover
        # Coverage skipped timm API is tested using efficient U-Net.
        return timm.create_model(
            arch_map[arch_name].pop("model"),
            pretrained=pretrained,
            **arch_map[arch_name],
        )

    msg = f"Backbone {arch_name} not supported. "
    raise ValueError(msg)


def _postproc(image: np.ndarray) -> np.ndarray:
    """Define the post-processing of this class of model.

    This simply applies argmax along last axis of the input.

    Args:
        image (np.ndarray):
            The input image array.

    Returns:
        np.ndarray:
            The post-processed image array.

    """
    return np.argmax(image, axis=-1)


def _infer_batch(
    model: nn.Module,
    batch_data: torch.Tensor,
    device: str,
) -> dict[str, np.ndarray]:
    """Run inference on an input batch.

    Contains logic for forward operation as well as i/o aggregation.

    Args:
        model (nn.Module):
            PyTorch defined model.
        batch_data (torch.Tensor):
            A batch of data generated by
            `torch.utils.data.DataLoader`.
        device (str):
            Transfers model to the specified device. Default is "cpu".

    Returns:
        dict[str, np.ndarray]:
            The model predictions as a NumPy array.

    Example:
        >>> output = _infer_batch(model, batch_data, "cuda")
        >>> print(output)

    """
    img_patches_device = batch_data.to(device=device).type(
        torch.float32,
    )  # to NCHW
    img_patches_device = img_patches_device.permute(0, 3, 1, 2).contiguous()

    # Inference mode
    model.eval()
    # Do not compute the gradient (not training)
    with torch.inference_mode():
        output = model(img_patches_device)
    # Output should be a single tensor or scalar
    return output.cpu().numpy()


class CNNModel(ModelABC):
    """Retrieve the model backbone and attach an extra FCN to perform classification.

    This class initializes a Convolutional Neural Network (CNN) model with a specified
    backbone and attaches a fully connected layer for classification tasks.

    Args:
        backbone (str):
            Name of the CNN model backbone (e.g., "resnet18", "densenet121").
        num_classes (int):
            Number of classes output by model. Defaults to 1.

    Attributes:
        num_classes (int):
            Number of classes output by the model.
        feat_extract (nn.Module):
            Backbone CNN model.
        pool (nn.Module):
            Type of pooling applied after feature extraction.
        classifier (nn.Module):
            Linear classifier module used to map the features to the output.

    Example:
        >>> model = CNNModel("resnet18", num_classes=2)
        >>> output = model(torch.randn(1, 3, 224, 224))
        >>> print(output.shape)

    """

    def __init__(self: CNNModel, backbone: str, num_classes: int = 1) -> None:
        """Initialize :class:`CNNModel`."""
        super().__init__()
        self.num_classes = num_classes

        # set num_classes to 100 to avoid downloading weights
        self.feat_extract = _get_architecture(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Best way to retrieve channel dynamically is passing a small forward pass
        prev_num_ch = self.feat_extract(torch.rand([2, 3, 96, 96])).shape[1]
        self.classifier = nn.Linear(prev_num_ch, num_classes)

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self: CNNModel, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        Returns:
            torch.Tensor:
                The output logits after passing through the model.

        """
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        gap_feat = torch.flatten(gap_feat, 1)
        logit = self.classifier(gap_feat)
        return torch.softmax(logit, -1)

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Define the post-processing of this class of model.

        This simply applies argmax along last axis of the input.

        Args:
            image (np.ndarray):
                The input image array.

        Returns:
            np.ndarray:
                The post-processed image array.

        """
        return _postproc(image=image)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Example:
            >>> output = _infer_batch(model, batch_data, "cuda")
            >>> print(output)

        """
        return _infer_batch(model=model, batch_data=batch_data, device=device)


class TimmModel(ModelABC):
    """Retrieve the tile encoder from timm.

    This is a wrapper for pretrained models within timm.

    Args:
        backbone (str):
            Model name. Currently, the tool supports following
             model names and their default associated weights from timm.
             - "efficientnet_b{i}" for i in [0, 1, ..., 7]
             - "UNI"
             - "prov-gigapath"
             - "UNI2"
             - "Virchow"
             - "Virchow2"
             - "kaiko"
             - "H-optimus-0"
             - "H-optimus-1"
             - "H0-mini"
        num_classes (int):
            Number of classes output by model.
        pretrained (bool, keyword-only):
            Whether to load pretrained weights.

    Attributes:
        num_classes (int):
            Number of classes output by the model.
        pretrained (bool):
            Whether to load pretrained weights.
        feat_extract (nn.Module):
            Backbone Timm model.
        classifier (nn.Module):
            Linear classifier module used to map the features to the
            output.

    Example:
        >>> model = TimmModel("UNI", pretrained=True)
        >>> output = model(torch.randn(1, 3, 224, 224))
        >>> print(output.shape)

    """

    def __init__(
        self: TimmModel,
        backbone: str,
        num_classes: int = 1,
        *,
        pretrained: bool,
    ) -> None:
        """Initialize :class:`TimmModel`."""
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.feat_extract = _get_timm_architecture(
            arch_name=backbone, pretrained=pretrained
        )

        # Best way to retrieve channel dynamically is passing a small forward pass
        prev_num_ch = self.feat_extract(torch.rand([2, 3, 224, 224])).shape[1]
        self.classifier = nn.Linear(prev_num_ch, num_classes)

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self: TimmModel, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        Returns:
            torch.Tensor:
                The output logits after passing through the model.

        """
        feat = self.feat_extract(imgs)
        feat = torch.flatten(feat, 1)
        logit = self.classifier(feat)
        return torch.softmax(logit, -1)

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Define the post-processing of this class of model.

        This simply applies argmax along last axis of the input.

        Args:
            image (np.ndarray):
                The input image array.

        Returns:
            np.ndarray:
                The post-processed image array.

        """
        return _postproc(image=image)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        device: str,
    ) -> dict[str, np.ndarray]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            dict[str, np.ndarray]:
                The model predictions as a NumPy array.

        Example:
            >>> output = _infer_batch(model, batch_data, "cuda")
            >>> print(output)

        """
        return _infer_batch(model=model, batch_data=batch_data, device=device)


class CNNBackbone(ModelABC):
    """Retrieve the model backbone and strip the classification layer.

    This is a wrapper for pretrained models within pytorch.

    Args:
        backbone (str):
            Model name. Currently, the tool supports following
             model names and their default associated weights from pytorch.
                - "alexnet"
                - "resnet18"
                - "resnet34"
                - "resnet50"
                - "resnet101"
                - "resnext50_32x4d"
                - "resnext101_32x8d"
                - "wide_resnet50_2"
                - "wide_resnet101_2"
                - "densenet121"
                - "densenet161"
                - "densenet169"
                - "densenet201"
                - "inception_v3"
                - "googlenet"
                - "mobilenet_v2"
                - "mobilenet_v3_large"
                - "mobilenet_v3_small"

    Attributes:
        feat_extract (nn.Module):
            Backbone CNN model.
        pool (nn.Module):
            Type of pooling applied after feature extraction.

    Examples:
        >>> # Creating resnet50 architecture from default pytorch
        >>> # without the classification layer with its associated
        >>> # weights loaded
        >>> model = CNNBackbone(backbone="resnet50")
        >>> model.eval()  # set to evaluation mode
        >>> # dummy sample in NHWC form
        >>> samples = torch.rand(4, 3, 512, 512)
        >>> features = model(samples)
        >>> features.shape  # features after global average pooling
        torch.Size([4, 2048])

    """

    def __init__(self: CNNBackbone, backbone: str) -> None:
        """Initialize :class:`CNNBackbone`."""
        super().__init__()
        # set num_classes=1000 to avoid downloading weights
        self.feat_extract = _get_architecture(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self: CNNBackbone, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        Returns:
            torch.Tensor:
                The extracted features.

        """
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        return torch.flatten(gap_feat, 1)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        device: str,
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list[dict[str, np.ndarray]]:
                list of dictionary values with numpy arrays.

        Example:
            >>> output = CNNBackbone.infer_batch(model, batch_data, "cuda")
            >>> print(output)

        """
        return [_infer_batch(model=model, batch_data=batch_data, device=device)]


class TimmBackbone(ModelABC):
    """Retrieve tile encoders from timm.

    This is a wrapper for pretrained models within timm.

    Args:
        backbone (str):
            Model name. Supported model names include:
                - "efficientnet_b{i}" for i in [0, 1, ..., 7]
                - "UNI"
                - "prov-gigapath"
                - "UNI2"
                - "Virchow"
                - "Virchow2"
                - "kaiko"
                - "H-optimus-0"
                - "H-optimus-1"
                - "H0-mini"
        pretrained (bool, keyword-only):
            Whether to load pretrained weights.

    Attributes:
        feat_extract (nn.Module):
            Backbone timm model.

    Examples:
        >>> # Creating UNI tile encoder
        >>> model = TimmBackbone(backbone="UNI", pretrained=True)
        >>> model.eval()  # set to evaluation mode
        >>> # dummy sample in NHWC form
        >>> samples = torch.rand(4, 3, 224, 224)
        >>> features = model(samples)
        >>> features.shape  # feature vector
        torch.Size([4, 1024])
    """

    def __init__(self: TimmBackbone, backbone: str, *, pretrained: bool) -> None:
        """Initialize :class:`TimmBackbone`."""
        super().__init__()
        self.pretrained = pretrained
        self.feat_extract = _get_timm_architecture(
            arch_name=backbone, pretrained=pretrained
        )

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self: TimmBackbone, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        Returns:
            torch.Tensor:
                The extracted features.

        """
        feats = self.feat_extract(imgs)
        return torch.flatten(feats, 1)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        device: str,
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list[dict[str, np.ndarray]]:
                list of dictionary values with numpy arrays.

        Example:
            >>> output = TimmBackbone.infer_batch(model, batch_data, "cuda")
            >>> print(output)

        """
        return [_infer_batch(model=model, batch_data=batch_data, device=device)]
