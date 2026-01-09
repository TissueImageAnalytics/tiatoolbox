
import pathlib
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector


detector = NucleusDetector(model='KongNet_Det_MIDOG_1')

wsi_path = "/media/u1910100/data/slides/TUM1.svs"
# wsi_path = "/media/u1910100/data/slides/CMU-1-Small-Region.svs"

out = detector.run(
    images=[pathlib.Path(wsi_path)],
    patch_mode=False,
    device="cuda",
    save_dir=pathlib.Path("/media/u1910100/data/overlays/test"),
    overwrite=True,
    output_type="annotationstore",
    auto_get_mask=True,
    memory_threshold=50,
    num_workers=1,
    batch_size=8,
)

