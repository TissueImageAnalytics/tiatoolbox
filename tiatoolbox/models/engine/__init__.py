from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent

nucleus_instance_segmentor = _lazy_import("nucleus_instance_segmentor", location)
patch_predictor = _lazy_import("patch_predictor", location)
semantic_segmentor = _lazy_import("semantic_segmentor", location)
