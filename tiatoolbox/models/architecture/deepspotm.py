"""Define DeepSpot-M for virtual spatial transcriptomics from H&E tiles.

DeepSpot-M (Nonchev et al., medRxiv 2026) is a multimodal foundation model that
maps a 224x224 H&E histology tile to transcriptome-wide spatial gene expression.
This module wraps the standalone ``deepspotm`` package as a tiatoolbox
:class:`ModelABC`, so the model can be run through the
:class:`tiatoolbox.models.engine.deep_feature_extractor.DeepFeatureExtractor`
engine over a whole-slide image. Each output "feature" column is a predicted
gene, so the engine writes an ``(n_tiles, n_genes)`` expression matrix together
with the matching tile coordinates.

The model weights live in the gated ``ratschlab/DeepSpotM`` Hugging Face repo:
accept the terms and authenticate (``huggingface-cli login``) before use. The
``deepspotm`` package is an optional dependency and is imported lazily, so it is
only required when this model is instantiated::

    pip install tiatoolbox[deepspotm]

Example:
    >>> from tiatoolbox.models.architecture.deepspotm import DeepSpotM
    >>> from tiatoolbox.models.engine.deep_feature_extractor import (
    ...     DeepFeatureExtractor,
    ... )
    >>> # A marker panel keeps the output lean; omit ``genes`` for the full
    >>> # transcriptome-wide (~19k gene) panel.
    >>> model = DeepSpotM(source="scgpt", genes=["EPCAM", "CD3D", "PTPRC"])
    >>> extractor = DeepFeatureExtractor(model=model, batch_size=32)
    >>> output = extractor.run(
    ...     ["slide.svs"],
    ...     patch_mode=False,
    ...     patch_input_shape=(224, 224),
    ...     input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    ...     save_dir="deepspotm_output",
    ... )

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

DEEPSPOTM_HF_REPO = "ratschlab/DeepSpotM"

#: Gene-embedding sources shipped with DeepSpot-M; one is selected at load time.
DEEPSPOTM_SOURCES = ("evo2", "orthrus", "prott5", "scgpt", "apertus")


class DeepSpotM(ModelABC):
    """DeepSpot-M model for virtual spatial transcriptomics from H&E.

    Wraps the standalone ``deepspotm`` package as a tiatoolbox model. A forward
    pass maps a batch of 224x224 H&E tiles to predicted spatial gene expression;
    run it through
    :class:`tiatoolbox.models.engine.deep_feature_extractor.DeepFeatureExtractor`
    to score a whole-slide image tile by tile.

    Args:
        repo_id_or_path (str):
            Hugging Face repo id or a local directory holding the exported model
            (``config.json``, ``model.safetensors``, ``tokens.csv``). Default is
            ``"ratschlab/DeepSpotM"`` (gated).
        source (str):
            Gene-embedding source to activate, one of
            :data:`DEEPSPOTM_SOURCES`. Default is ``"scgpt"``.
        genes (str | Sequence[str] | None):
            Restrict the prediction to these gene symbols (order preserved). Only
            the selected gene queries are computed, which is faster and keeps the
            output matrix small. Default ``None`` predicts the full panel
            (~19k genes). Unknown symbols raise :class:`KeyError`.
        device (str):
            Device passed to the underlying loader. Default ``"cpu"``. The
            :class:`DeepFeatureExtractor` engine manages device placement at run
            time, so this mainly matters when the model is used directly.
        revision (str | None):
            Optional Hugging Face revision (tag, branch, or commit).

    Attributes:
        model (torch.nn.Module):
            The wrapped ``deepspotm`` model.
        image_processor (Callable):
            DeepSpot-M's evaluation image transform (resize to 224, center-crop,
            normalize). Applied per tile in :meth:`infer_batch`.
        gene_names (list[str]):
            Ordered gene symbols; ``gene_names[i]`` labels output column ``i``.
        gene_indices (torch.Tensor | None):
            Column indices of the selected genes, or ``None`` for the full panel.

    Example:
        >>> import numpy as np
        >>> model = DeepSpotM(source="scgpt", genes=["EPCAM"])
        >>> tiles = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
        >>> features = DeepSpotM.infer_batch(model, tiles, device="cpu")[0]
        >>> features.shape
        (4, 1)

    """

    def __init__(
        self: DeepSpotM,
        repo_id_or_path: str = DEEPSPOTM_HF_REPO,
        source: str = "scgpt",
        genes: str | Sequence[str] | None = None,
        *,
        device: str = "cpu",
        revision: str | None = None,
    ) -> None:
        """Initialize :class:`DeepSpotM`."""
        super().__init__()

        try:
            from deepspotm import DeepSpotM as _DeepSpotMModel  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - optional dependency
            msg = (
                "DeepSpot-M requires the `deepspotm` package, which is not "
                "installed. Install it with `pip install "
                "tiatoolbox[deepspotm]` or `pip install deepspotm`."
            )
            raise ImportError(msg) from err

        model, image_processor = _DeepSpotMModel.from_pretrained(
            repo_id_or_path,
            source=source,
            device=device,
            revision=revision,
        )
        self.model = model
        self.image_processor = image_processor

        if genes is None:
            self.gene_indices = None
            self.gene_names = list(model.gene_names)
        else:
            gene_list = [genes] if isinstance(genes, str) else list(genes)
            # Validates every symbol against the model panel (raises KeyError).
            self.gene_indices = model.genes_to_indices(gene_list)
            self.gene_names = gene_list

    # pylint: disable=W0221
    # because ModelABC.forward is generic, this is the concrete definition.
    def forward(self: DeepSpotM, imgs: torch.Tensor) -> torch.Tensor:
        """Predict gene expression for a batch of preprocessed tiles.

        Args:
            imgs (torch.Tensor):
                Preprocessed tiles, shape ``(N, 3, 224, 224)``.

        Returns:
            torch.Tensor:
                Predicted expression, shape ``(N, len(gene_names))``.

        """
        gene_indices = self.gene_indices
        if gene_indices is not None:
            gene_indices = gene_indices.to(imgs.device)
        # deepspotm returns (expression, pooled, attention); keep expression.
        expression, _, _ = self.model(imgs, gene_indices=gene_indices)
        return expression

    @staticmethod
    def infer_batch(
        model: ModelABC,
        batch_data: np.ndarray | torch.Tensor,
        device: str = "cpu",
    ) -> list[np.ndarray]:
        """Run inference on a batch of raw H&E tiles.

        Applies DeepSpot-M's own image transform to each tile before predicting,
        and returns a single-element list so the
        :class:`DeepFeatureExtractor` engine stores the predictions as an
        ``(n_tiles, n_genes)`` matrix (mirroring the feature-extractor contract).

        Args:
            model (ModelABC):
                A :class:`DeepSpotM` instance.
            batch_data (np.ndarray | torch.Tensor):
                A batch of tiles in NHWC (channel-last) RGB layout, as produced
                by the tiatoolbox WSI patch loader.
            device (str):
                Device to run inference on. Default ``"cpu"``.

        Returns:
            list[np.ndarray]:
                Single-element list holding the ``(n_tiles, n_genes)``
                expression array.

        """
        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cpu().numpy()
        tiles = np.asarray(batch_data)
        # WSI patches are read as uint8 [0, 255]; coerce defensively so the
        # transform never rescales a float array as if it were in [0, 1].
        if tiles.dtype != np.uint8:
            tiles = np.clip(np.rint(tiles), 0, 255).astype(np.uint8)

        # DeepSpot-M's transform accepts a channel-last uint8 array directly and
        # returns a normalized (3, 224, 224) tensor.
        pixel_values = torch.stack(
            [model.image_processor(tile[..., :3]) for tile in tiles]
        ).to(device)

        model.eval()
        with torch.inference_mode():
            expression = model(pixel_values)

        return [expression.cpu().numpy()]
