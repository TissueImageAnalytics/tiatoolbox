"""Define vanilla CNNs with torch backbones, mainly for patch classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import timm
import torch
import torchvision.models as torch_models
from torch import nn

from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils.misc import select_device

if TYPE_CHECKING:  # pragma: no cover
    from torchvision.models import WeightsEnum


def _get_architecture(
    arch_name: str,
    weights: str or WeightsEnum = "DEFAULT",
    **kwargs: dict,
) -> list[nn.Sequential, ...] | nn.Sequential:
    """Get a model.

    Model architectures are either already defined within torchvision or
    they can be custom-made within tiatoolbox.

    Args:
        arch_name (str):
            Architecture name.
        weights (str or WeightsEnum):
            torchvision model weights (get_model_weights).
        kwargs (dict):
            Key-word arguments.

    Returns:
        List of PyTorch network layers wrapped with `nn.Sequential`.
        https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

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
    model = creator(weights=weights, **kwargs)

    # Unroll all the definition and strip off the final GAP and FCN
    if "resnet" in arch_name or "resnext" in arch_name:
        return nn.Sequential(*list(model.children())[:-2])
    if "densenet" in arch_name:
        return model.features
    if "alexnet" in arch_name:
        return model.features
    if "inception_v3" in arch_name or "googlenet" in arch_name:
        return nn.Sequential(*list(model.children())[:-3])

    return model.features


def _get_timm_architecture(
    arch_name: str,
    *,
    pretrained: bool,
) -> list[nn.Sequential, ...] | nn.Sequential:
    """Get architecture and weights for pathology-specific timm models.

    Args:
        arch_name (str):
            Architecture name.
        pretrained (bool, keyword-only):
            Whether to load pretrained weights.

    Returns:
        A ready-to-use timm model.

    Raises:
        ValueError:
            If the backbone architecture is not supported.

    """
    if arch_name in [f"efficientnet_b{i}" for i in range(8)]:
        model = timm.create_model(arch_name, pretrained=pretrained)
        return nn.Sequential(*list(model.children())[:-1])

    if arch_name == "UNI":  # pragma: no cover
        # UNI tile encoder: https://huggingface.co/MahmoodLab/UNI
        # Coverage skipped timm API is tested using efficient U-Net.
        return timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=pretrained,
            init_values=1e-5,
            dynamic_img_size=True,
        )

    if arch_name == "prov-gigapath":  # pragma: no cover
        # Prov-GigaPath tile encoder: https://huggingface.co/prov-gigapath/prov-gigapath
        # Coverage skipped timm API is tested using efficient U-Net.
        return timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=pretrained,
        )

    if arch_name == "H-optimus-0":  # pragma: no cover
        # H-Optimus-0 tile encoder: https://huggingface.co/bioptimus/H-optimus-0
        # Coverage skipped timm API is tested using efficient U-Net.
        return timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=pretrained,
            init_values=1e-5,
            dynamic_img_size=False,
        )

    msg = f"Backbone {arch_name} not supported. "
    raise ValueError(msg)


def _postproc(image: np.ndarray) -> np.ndarray:
    """Define the post-processing of this class of model.

    This simply applies argmax along last axis of the input.

    """
    return np.argmax(image, axis=-1)


def _infer_batch(
    model: nn.Module,
    batch_data: torch.Tensor,
    *,
    on_gpu: bool,
) -> np.ndarray:
    """Run inference on an input batch.

    Contains logic for forward operation as well as i/o aggregation.

    Args:
        model (nn.Module):
            PyTorch defined model.
        batch_data (torch.Tensor):
            A batch of data generated by
            `torch.utils.data.DataLoader`.
        on_gpu (bool):
            Whether to run inference on a GPU.

    """
    img_patches_device = batch_data.to(select_device(on_gpu=on_gpu)).type(
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

    Args:
        backbone (str):
            Model name.
        num_classes (int):
            Number of classes output by model.

    Attributes:
        num_classes (int):
            Number of classes output by the model.
        feat_extract (nn.Module):
            Backbone CNN model.
        pool (nn.Module):
            Type of pooling applied after feature extraction.
        classifier (nn.Module):
            Linear classifier module used to map the features to the
            output.

    """

    def __init__(self: CNNModel, backbone: str, num_classes: int = 1) -> None:
        """Initialize :class:`CNNModel`."""
        super().__init__()
        self.num_classes = num_classes

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

        """
        return _postproc(image=image)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        *,
        on_gpu: bool,
    ) -> np.ndarray:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        return _infer_batch(model=model, batch_data=batch_data, on_gpu=on_gpu)


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

        """
        feat = self.feat_extract(imgs)
        feat = torch.flatten(feat, 1)
        logit = self.classifier(feat)
        return torch.softmax(logit, -1)

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Define the post-processing of this class of model.

        This simply applies argmax along last axis of the input.

        """
        return _postproc(image=image)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        *,
        on_gpu: bool,
    ) -> np.ndarray:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        return _infer_batch(model=model, batch_data=batch_data, on_gpu=on_gpu)


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
        self.feat_extract = _get_architecture(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self: CNNBackbone, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        """
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        return torch.flatten(gap_feat, 1)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        *,
        on_gpu: bool,
    ) -> list[np.ndarray]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        Returns:
            list[np.ndarray]:
                list of numpy arrays.

        """
        return [_infer_batch(model=model, batch_data=batch_data, on_gpu=on_gpu)]


class TimmBackbone(ModelABC):
    """Retrieve tile encoders from timm.

    This is a wrapper for pretrained models within timm.

    Args:
        backbone (str):
            Model name. Currently, the tool supports following
             model names and their default associated weights from timm.
             - "efficientnet_b{i}" for i in [0, 1, ..., 7]
             - "UNI"
             - "prov-gigapath"
        pretrained (bool, keyword-only):
            Whether to load pretrained weights.

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

        """
        feats = self.feat_extract(imgs)
        return torch.flatten(feats, 1)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        *,
        on_gpu: bool,
    ) -> list[np.ndarray]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        Returns:
            list[np.ndarray]:
                list of numpy arrays.

        """
        return [_infer_batch(model=model, batch_data=batch_data, on_gpu=on_gpu)]
