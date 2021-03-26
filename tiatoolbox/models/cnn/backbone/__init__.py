
import torchvision.models as torch_models

def get_model_creator(backbone):
    backbone_dict = {
        "resnet50"    : torch_models.resnet50,
        "densenet121" : torch_models.densenet121
    }
    return 