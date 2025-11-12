import pathlib

from tiatoolbox.models.engine.nucleus_detector import (
    NucleusDetector,
)
from tiatoolbox.utils import env_detection as toolbox_env

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()


if __name__ == "__main__":

    # model_name = "sccnn-crchisto"
    model_name = "mapde-conic"

    detector = NucleusDetector(model=model_name, batch_size=16, num_workers=8)
    detector.run(
        images=[pathlib.Path("/media/u1910100/data/slides/CMU-1-Small-Region.svs")],
        patch_mode=False,
        device="cuda",
        save_dir=pathlib.Path("/media/u1910100/data/overlays/test"),
        overwrite=True,
        output_type="annotationstore",
        class_dict={0: "nucleus"},
        auto_get_mask=True,
        memory_threshold=80
    )
