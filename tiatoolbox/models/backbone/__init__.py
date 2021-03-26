
import torchvision.models as torch_models
import torch.nn as nn

def get_model(backbone, **kwargs):
    """
    Create a model creator which can either alrd defined by torchvision
    or a custom-made within tiatoolbox
    """
    backbone_dict = {
        "resnet50"    : torch_models.resnet50,
        "densenet121" : torch_models.densenet121
    }
    creator = backbone_dict[backbone]
    model = creator(**kwargs) # ! abit too hacky

    # unroll all the definition and strip off the final GAP and FCN
    # different model will have diffent form, sample resnet and densenet atm
    if 'resnet' in backbone:
        feat_extract = nn.Sequential(*list(model.children())[:-2])
    elif 'densenet' in backbone:
        feat_extract = nn.Sequential(*list(model.children())[0])
    return feat_extract

