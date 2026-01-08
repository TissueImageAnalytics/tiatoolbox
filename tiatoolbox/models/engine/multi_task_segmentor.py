"""This module enables multi-task segmentor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from .semantic_segmentor import SemanticSegmentor, SemanticSegmentorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from pathlib import Path

    import numpy as np

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOSegmentorConfig


class MultiTaskSegmentor(SemanticSegmentor):
    """An engine specifically designed to handle tiles or WSIs inference.

    Note, if `model` is supplied in the arguments, it will ignore the
    `pretrained_model` and `pretrained_weights` arguments. Each WSI's instance
    predictions (e.g. nuclear instances) will be store under a `.dat` file and
    the semantic segmentation predictions will be stored in a `.npy` file. The
    `.dat` files contains a dictionary of form:

    .. code-block:: yaml

        inst_uid:
            # top left and bottom right of bounding box
            box: (start_x, start_y, end_x, end_y)
            # centroid coordinates
            centroid: (x, y)
            # array/list of points
            contour: [(x1, y1), (x2, y2), ...]
            # the type of nuclei
            type: int
            # the probabilities of being this nuclei type
            prob: float

    Args:
        model (nn.Module): Use externally defined PyTorch model for prediction with.
          weights already loaded. Default is `None`. If provided,
          `pretrained_model` argument is ignored.
        pretrained_model (str): Name of the existing models support by tiatoolbox
          for processing the data. Refer to [URL] for details.
          By default, the corresponding pretrained weights will also be
          downloaded. However, you can override with your own set of weights
          via the `pretrained_weights` argument. Argument is case insensitive.
        pretrained_weights (str): Path to the weight of the corresponding
          `pretrained_model`.
        batch_size (int) : Number of images fed into the model each time.
        num_loader_workers (int) : Number of workers to load the data.
          Take note that they will also perform preprocessing.
        num_postproc_workers (int) : Number of workers to post-process
          predictions.
        verbose (bool): Whether to output logging information.
        dataset_class (obj): Dataset class to be used instead of default.
        auto_generate_mask (bool): To automatically generate tile/WSI tissue mask
          if is not provided.
        output_types (list): Ordered list describing what sort of segmentation the
            output from the model postproc gives for a two-task model this may be:
            ['instance', 'semantic']

    Examples:
        >>> # Sample output of a network
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> predictor = MultiTaskSegmentor(
        ...     model='hovernetplus-oed',
        ...     output_type=['instance', 'semantic'],
        ... )
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0') , ('B/wsi.svs', 'output/1')]
        >>> # Each output of 'A/wsi.svs'
        >>> # will be respectively stored in 'output/0.0.dat', 'output/0.1.npy'
        >>> # Here, the second integer represents the task number
        >>> # e.g. between 0 or 1 for a two task model

    """

    def __init__(
        self: MultiTaskSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`NucleusInstanceSegmentor`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def run(
        self: MultiTaskSegmentor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        *,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: IntPair | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the semantic segmentation engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both
        patch-level and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. Can be a list of file paths, WSIReader objects,
                or a NumPy array of image patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing. Only used when `patch_mode` is False.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`). Default
                is True.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore". Default
                is "dict".
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    auto_get_mask (bool):
                        Automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches per forward pass.
                    class_dict (dict):
                        Mapping of classification outputs to class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    output_resolutions (Resolution):
                        Resolution used for writing output predictions.
                    patch_output_shape (tuple[int, int]):
                        Shape of output patches (height, width).
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
            >>> output = segmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"

            >>> output = segmentor.run(
            ...     image_patches,
            ...     patch_mode=True,
            ...     output_type="zarr"
            ... )
            >>> output
            ... "/path/to/Output.zarr"

            >>> output = segmentor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... "/path/to/wsi1.db"

        """
        return super().run(
            images=images,
            masks=masks,
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )
