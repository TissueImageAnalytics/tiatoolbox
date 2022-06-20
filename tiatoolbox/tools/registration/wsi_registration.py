import torchvision
from torchvision.models._utils import IntermediateLayerGetter


class RigidRegistration:
    def __init__(self):
        self.patch_size = (224, 224)
        self.x_scale, self.y_scale = [], []
        model = torchvision.models.vgg16(True)
        return_layers = {"16": "block3_pool", "23": "block4_pool", "30": "block5_pool"}
        self.feature_extractor = IntermediateLayerGetter(
            model.features, return_layers=return_layers
        )
