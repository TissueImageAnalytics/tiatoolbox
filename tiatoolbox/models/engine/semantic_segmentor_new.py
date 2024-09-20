"""Defines SemanticSegmentor Engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .patch_predictor import PatchPredictor

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from tiatoolbox.models.models_abc import ModelABC


class SemanticSegmentor(PatchPredictor):
    """Pixel-wise segmentation predictor.

    The tiatoolbox model should produce the following results on the BCSS dataset
    using fcn_resnet50_unet-bcss.

    .. list-table:: Semantic segmentation performance on the BCSS dataset
       :widths: 15 15 15 15 15 15 15
       :header-rows: 1

       * -
         - Tumour
         - Stroma
         - Inflammatory
         - Necrosis
         - Other
         - All
       * - Amgad et al.
         - 0.851
         - 0.800
         - 0.712
         - 0.723
         - 0.666
         - 0.750
       * - TIAToolbox
         - 0.885
         - 0.825
         - 0.761
         - 0.765
         - 0.581
         - 0.763

    Note, if `model` is supplied in the arguments, it will ignore the
    `pretrained_model` and `pretrained_weights` arguments.

    Args:
        model (nn.Module):
            Use externally defined PyTorch model for prediction with
            weights already loaded. Default is `None`. If provided,
            `pretrained_model` argument is ignored.
        pretrained_model (str):
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_.
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case-insensitive.
        pretrained_weights (str):
            Path to the weight of the corresponding `pretrained_model`.
        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data. Take note that they will
            also perform preprocessing.
        num_postproc_workers (int):
            This value is there to maintain input compatibility with
            `tiatoolbox.models.classification` and is not used.
        verbose (bool):
            Whether to output logging information.
        dataset_class (obj):
            Dataset class to be used instead of default.
        auto_generate_mask (bool):
            To automatically generate tile/WSI tissue mask if is not
            provided.

    Attributes:
        process_prediction_per_batch (bool):
            A flag to denote whether post-processing for inference
            output is applied after each batch or after finishing an entire
            tile or WSI.

    Examples:
        >>> # Sample output of a network
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> predictor = SemanticSegmentor(model='fcn-tissue_mask')
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0.raw') , ('B/wsi.svs', 'output/1.raw')]
        >>> # if a network have 2 output heads, each head output of 'A/wsi.svs'
        >>> # will be respectively stored in 'output/0.raw.0', 'output/0.raw.1'

    """

    def __init__(
        self: SemanticSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`SemanticSegmentor`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_post_proc_workers=num_post_proc_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )
