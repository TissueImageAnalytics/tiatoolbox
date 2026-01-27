import sys
sys.path.append("..")  # to import from parent directory
from pathlib import Path
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

wsi_path = "/media/u1910100/data/GitHub/tiatoolbox/examples/tmp/sample_wsis/D_P000019_PAS_CPG.tif"
tmp_dir = Path("./tmp/")

# detector = NucleusDetector(model="KongNet_MONKEY_1")

# out = detector.run(
#     images=[Path(wsi_path)],
#     patch_mode=False,
#     device="cuda",
#     save_dir=tmp_dir / "sample_wsi_results/",
#     overwrite=True,
#     output_type="annotationstore",
#     auto_get_mask=True,
#     memory_threshold=5,
#     num_workers=1,
#     batch_size=8,
# )

segmentor = SemanticSegmentor(model="fcn_resnet50_unet-bcss")

out = segmentor.run(
    images=[Path(wsi_path)],
    patch_mode=False,
    device="cuda",
    save_dir=tmp_dir / "sample_wsi_results/",
    overwrite=True,
    output_type="annotationstore",
    auto_get_mask=True,
    memory_threshold=75,
    num_workers=0,
    batch_size=16,
)

