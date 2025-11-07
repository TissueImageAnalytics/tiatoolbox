import pathlib
import shutil

import pandas as pd
import pytest

from tiatoolbox.models.engine.nucleus_detector import (
    NucleusDetector,
)
from tiatoolbox.utils import env_detection as toolbox_env

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()


if __name__ == "__main__":
    detector = NucleusDetector(model="mapde-conic", batch_size=8, num_workers=2)
    detector.run(
        images=[pathlib.Path("/media/u1910100/data/slides/CMU-1-Small-Region.svs")], 
        patch_mode=False,
        device="cuda",
        save_dir=pathlib.Path("/media/u1910100/data/overlays/test"),
        overwrite=True,
    )