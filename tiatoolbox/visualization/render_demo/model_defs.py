import torch
from tiatoolbox.models.engine.nucleus_instance_segmentor import IOSegmentorConfig, NucleusInstanceSegmentor
from tiatoolbox.models.architecture.nuclick import NuClick
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.engine.interactive_segmentor import InteractiveSegmentor, IOInteractiveSegmentorConfig


class InteractiveModel:

    def __init__(self, input_type, ioconfig, run_function, name) -> None:
        self.input_type = input_type
        self.ioconfig = ioconfig
        self.run_function = run_function
        self.name = name

    def run(self, slide, input):
        return self.run_function(slide, input, self.ioconfig)



models={}

"""define models to be available in visualization tool here. To add a model,
the following must be provided.
Input type - can be one of the following:
-'mask': a binary mask of the same size as the thumbnail of the input image. 
The selected region will be true in the mask.
-'points': a set of points selected by user
-'patch': an image patch of the input image. 
The selceted region will be sent as an ndarray
-'bounds': a bounding box of the selected region.
An ioconfig should be provided defining the input shapes and resolutions etc
that the model will input and output.
see tiatoolbox.models.engine.nucleus_instance_segmentor.IOSegmentorConfig for
more details

the output is expected to be in
"""

#Add Hovernet
input_type = 'mask'
ioconfig = IOSegmentorConfig(input_resolutions=[{'resolution':0.25, 'units':'mpp'}], output_resolutions=[{'resolution':0.25, 'units':'mpp'}])
def run_hovernet(slide, input, ioconfig):
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=6,
        num_postproc_workers=12,
        batch_size=24,
    )

    output = inst_segmentor.predict(
        [slide],
        [input],
        save_dir="sample_tile_results/",
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
    )

    return output

models['hovernet']=InteractiveModel(input_type, ioconfig, run_hovernet, 'hovernet')


#Add nuclick
input_type = 'points'
ioconfig = IOInteractiveSegmentorConfig(input_resolutions=[{'resolution': 0.25, 'units': 'mpp'}], patch_size=(128, 128))
def run_nuclick(slide, input, ioconfig):   
    model = NuClick(5, 1)
    pretrained_weights=r'E:\TTB_vis_folder\NuClick_Nuclick_40xAll.pth'
    saved_state_dict = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)

    inst_segmentor = InteractiveSegmentor(
        num_loader_workers=0,
        batch_size=16,
        model=model,
    )

    nuclick_output = inst_segmentor.predict(
        [slide],
        [input],
        ioconfig=ioconfig,
        save_dir="sample_tile_results/",
        patch_size=(128,128),
        resolution=0.25,
        units='mpp',
        on_gpu=True,
        save_output=True,
    )

    return nuclick_output

models['nuclick']=InteractiveModel(input_type, ioconfig, run_nuclick, 'nuclick')