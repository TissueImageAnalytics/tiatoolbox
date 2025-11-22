from pathlib import Path

from tiatoolbox.models.engine.nucleus_detector import (
    NucleusDetector,
)
from tiatoolbox.utils import env_detection as toolbox_env

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()

if __name__ == "__main__":
    # model_name = "sccnn-crchisto"
    model_name = "mapde-conic"

    # test_image_path = "/media/u1910100/data/slides/CMU-1-Small-Region.svs"
    # reader = WSIReader.open(test_image_path)

    # patch_1 = reader.read_region((1500, 1500), level=0, size=(31, 31))

    # imwrite(Path("/media/u1910100/data/slides/patch_1.png"), patch_1)

    # patch_2 = reader.read_region((1000, 1000), level=0, size=(31, 31))
    # imwrite(Path("/media/u1910100/data/slides/patch_2.png"), patch_2)

    # patches = [
    #     Path("/media/u1910100/data/slides/patch_1.png"),
    #     Path("/media/u1910100/data/slides/patch_2.png"),
    # ]

    detector = NucleusDetector(model=model_name, batch_size=16, num_workers=8)
    detector.run(
        images=[Path("/media/u1910100/data/slides/wsi1_2k_2k.svs")],
        # images=patches,
        patch_mode=False,
        device="cuda",
        save_dir=Path("/media/u1910100/data/overlays/test"),
        overwrite=True,
        output_type="annotationstore",
        auto_get_mask=True,
        memory_threshold=70,
    )
