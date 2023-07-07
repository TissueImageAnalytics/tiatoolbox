import PIL
import torchvision.transforms as transforms


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list):
            List of torchvision transforms for preprocessing the image.
            The transforms will be applied in the order that they are
            given in the list. For more information, visit the following
            link: https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self, preprocs):
        self.func = transforms.Compose(preprocs)

    def __call__(self, img):
        img = PIL.Image.fromarray(img)
        img = self.func(img)
        return img.permute(1, 2, 0)


def predefined_preproc_func(dataset_name):
    """Get the preprocessing information used for the pretrained model.

    Args:
        dataset_name (str):
            Dataset name used to determine what preprocessing was used.
    Returns:
        _TorchPreprocCaller:
            Preprocessing function for transforming the input data.

    """
    preproc_dict = {
        "kather100k": [
            transforms.ToTensor(),
        ],
        "pcam": [
            transforms.ToTensor(),
        ],
    }

    if dataset_name not in preproc_dict:
        raise ValueError(
            f"Predefined preprocessing for dataset `{dataset_name}` does not exist."
        )

    preprocs = preproc_dict[dataset_name]
    return _TorchPreprocCaller(preprocs)
