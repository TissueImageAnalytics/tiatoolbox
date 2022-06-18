import torchvision
from torchvision.models._utils import IntermediateLayerGetter


class rigidRegistration:
    def __init__(self):
        self.patch_size = (224, 224)
        self.Xscale, self.Yscale = [], []
        model = torchvision.models.vgg16(True)
        return_layers = {"16": "block3_pool", "23": "block4_pool", "30": "block5_pool"}
        self.FeatureExtractor = IntermediateLayerGetter(
            model.features, return_layers=return_layers
        )
