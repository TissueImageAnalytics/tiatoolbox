
import pathlib
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector


detector = NucleusDetector(model='KongNet_CoNIC_1')

# wsi_path = "/media/u1910100/data/slides/TCGA-AO-A0J2-01Z-00-DX1.7C9FEC7B-6040-4C58-9563-D10C0D7AC72E.svs"
wsi_path = "/media/u1910100/data/slides/CMU-1-Small-Region.svs"

out = detector.run(
    images=[pathlib.Path(wsi_path)],
    patch_mode=False,
    device="cuda",
    save_dir=pathlib.Path("/media/u1910100/data/overlays/test"),
    overwrite=True,
    output_type="annotationstore",
    auto_get_mask=True,
    memory_threshold=30,
    num_workers=1,
    batch_size=8,
    cache_dir=pathlib.Path("/media/u1910100/data/cache")
)

