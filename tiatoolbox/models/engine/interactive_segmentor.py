import copy
import os
import pathlib
import warnings
from collections import OrderedDict
from typing import Tuple, List
import joblib
import matplotlib.pyplot as plt

import numpy as np
import torch
import tqdm
from tiatoolbox.models.architecture.nuclick import NuClick

from tiatoolbox.models.dataset.interactive_segmentation import InteractiveSegmentorDataset
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.abc import IOConfigABC
from tiatoolbox.utils import misc


class IOInteractiveSegmentorConfig(IOConfigABC):
    """Contain interactive segmentor input and output information.

    input_resolutions (list): Resolution of each input head of model
        inference, must be in the same order as target model.forward().
    
    """
    input_resolutions = None
    output_resolutions = None

    def __init__(
        self,
        input_resolutions: List[dict],
        patch_size: Tuple[int, int],
        **kwargs,
    ):
        
        self.input_resolutions = input_resolutions
        self.patch_size = patch_size


class InteractiveSegmentor:

    def __init__(
        self,
        batch_size=8,
        num_loader_workers=0,
        model=None,
        pretrained_model=None,
        pretrained_weights=None,
        verbose=True,
    ):
        """
        Note, if `model` is supplied in the arguments, it will ignore the
        `pretrained_model` and `pretrained_weights` arguments.

        Args:
            batch_size (int) : Number of images fed into the model each time.
            num_loader_workers (int) : Number of workers to load the data.
                Take note that they will also perform preprocessing.
            model (nn.Module): Use externally defined PyTorch model for prediction with.
                weights already loaded. Default is `None`. If provided,
                `pretrained_model` argument is ignored.
            pretrained_model (str): Name of the existing models support by tiatoolbox
                for processing the data. For a full list of pretrained models, refer to the
                `docs <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_.
                By default, the corresponding pretrained weights will also be
                downloaded. However, you can override with your own set of weights
                via the `pretrained_weights` argument. Argument is case insensitive.
            pretrained_weights (str): Path to the weight of the corresponding
                `pretrained_model`.
            verbose (bool): Whether to output logging information.
            
        """
        super().__init__()

        self.imgs = None
        self.mode = None

        if model is None and pretrained_model is None:
            raise ValueError("Must provide either of `model` or `pretrained_model`")

        if model is not None:
            self.model = model
            ioconfig = None  # retrieve iostate from provided model ?
        else:
            model, ioconfig = get_pretrained_model(pretrained_model, pretrained_weights)

        self.ioconfig = ioconfig  # for storing original
        self._ioconfig = None  # for storing runtime
        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_workers
        self.verbose = verbose

    def _predict_engine(
        self,
        dataset,
        on_gpu=True,
    ):
        """Make a prediction on a dataset. The dataset may be mutated.
            The dataset is constructed from all the clicks on one input image.

        Args:
            dataset (torch.utils.data.Dataset): PyTorch dataset object created using
              tiatoolbox.models.data.interactive_segmentation.InteractiveSegmentorDataset.
            on_gpu (bool): whether to run model on the GPU.

        Returns:
            inst_info_dict (dict): A dictionary containing a mapping of each nucleus instance
                    within the image. It has following form

                    inst_info = {
                            box: number[],
                            centroids: number[],
                            contour: number[][],
                            type: number,
                            prob: number,
                    }
                    inst_info_dict = {[inst_uid: number] : inst_info}

                    and `inst_uid` is an integer corresponds to the instance
                    having the same pixel value within `pred_inst`.

        """

        # preprocessing must be defined with the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        if self.verbose:
            pbar = tqdm.tqdm(
                total=int(len(dataloader)), leave=True, ncols=80, ascii=True, position=0
            )

        # use external for testing
        model = misc.model_to(on_gpu, self.model)

        cum_output = []
        cum_bounding_boxes = []
        for _, batch_data in enumerate(dataloader):

            input = batch_data["input"]
            #for patch in dataset:
            #plt.imshow(np.transpose(np.squeeze(input[0,0:3,:,:]),(1,2,0)))
            #plt.show()
            nuc_points = batch_data["input"][:, 3, :, :].numpy()
            bounding_boxes = batch_data["bounding_box"].tolist()

            batch_output_probabilities = self.model.infer_batch(
                model, input, on_gpu
            )

            # Nuclick post-processing:
            batch_output_predictions = NuClick.postproc(
                preds = batch_output_probabilities, thresh=0.5, minSize=10, 
                minHole=30, doReconstruction=True, nucPoints=nuc_points
            )

            # tolist might be very expensive
            cum_output.extend(batch_output_predictions.tolist())
            cum_bounding_boxes.extend(bounding_boxes)

            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()

        # Generate inst_dict
        cum_output = np.asarray(cum_output)
        cum_bounding_boxes = np.asarray(cum_bounding_boxes)
        inst_dict = NuClick.generate_inst_dict(cum_output, cum_bounding_boxes)
        return inst_dict

    def predict(
        self,
        imgs,
        points,
        labels=None,
        on_gpu=True,
        ioconfig: IOInteractiveSegmentorConfig=None,
        patch_size: Tuple[int, int]=None,
        resolution=None,
        units=None,
        save_dir=None,
        save_output=False,
    ):
        """Make a prediction for a list of input data.

        Args:
            imgs (list): List of inputs to process. The input must be a list of file paths.
            points (list): List of path to files containing points('clicks') for each image. 
            labels: List of labels for each image, Optional.
            on_gpu (bool): whether to run model on the GPU.
            patch_input_shape (tuple): Size of patches input to the model. Patches
              are at requested read resolution, not with respect to level 0, and must be
              positive. For nuclei segmentation using NuClick, patch_size should be (128,128).
            resolution (float): Resolution used for reading the image. Please see
                :obj:`WSIReader` for details.
            units (str): Units of resolution used for reading the image. Choose from
              either `level`, `power` or `mpp`. Please see
                :obj:`WSIReader` for details.
            save_dir (str or pathlib.Path): Output directory when processing
              multiple tiles and whole-slide images. By default, it is folder `output`
              where the running script is invoked.
            save_output (bool): Whether to save output for a single file. default=False

        Returns:
            output (dict): A dictionary containing a mapping of each nucleus instance
                    within the image. It has following form

                    inst_info = {
                            box: number[],
                            centroids: number[],
                            contour: number[][],
                            type: number,
                            prob: number,
                    }
                    inst_info_dict = {[inst_uid: number] : inst_info}

                    and `inst_uid` is an integer corresponds to the instance
                    having the same pixel value within `pred_inst`.

                If multiple image tiles or whole-slide images are provided as input,
                or save_output is True, then results are saved to `save_dir` and a
                dictionary indicating save location for each input is return.
        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> points = ['points1.csv', 'points2.csv']
            >>> nuclick_model = NuClick(num_input_channels=5, num_output_channels=1)
            >>> predictor = InteractiveSegmemtor(model = nuclick_model)
            >>> output = predictor.predict(imgs=wsis, points=points)
            >>> output.keys()
            ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            {'dat': '0.dat'}
            >>> output['wsi2.svs']
            {'dat': '1.dat'}

        """

        if labels is not None:
            # if a labels is provided, then return with the prediction
            return_labels = bool(labels)
            if len(labels) != len(imgs):
                raise ValueError(
                    f"len(labels) != len(imgs) : " f"{len(labels)} != {len(imgs)}"
                )
        if  len(points) != len(imgs):
            raise ValueError(
                f"len(points) != len(imgs) : " f"{len(points)} != {len(imgs)}"
            )

        # ! not sure if there is any way to make this nicer
        make_config_flag = (
            patch_size is None,
            resolution is None,
            units is None,
        )

        if ioconfig is None and self.ioconfig is None and any(make_config_flag):
            raise ValueError(
                "Must provide either `ioconfig` or "
                "`patch_input_shape`, `resolution`, and `units`."
            )
        if ioconfig is None and self.ioconfig:
            ioconfig = copy.deepcopy(self.ioconfig)
            # ! not sure if there is a nicer way to set this
            if patch_size is not None:
                ioconfig.patch_size = patch_size
            if resolution is not None:
                ioconfig.input_resolutions[0]["resolution"] = resolution
            if units is not None:
                ioconfig.input_resolutions[0]["units"] = units
        elif ioconfig is None and all(not v for v in make_config_flag):
            ioconfig = IOInteractiveSegmentorConfig(
                input_resolutions=[{"resolution": resolution, "units": units}],
                patch_size=patch_size,
            )


        if len(imgs) > 1:
            warnings.warn(
                "When providing multiple whole-slide images / tiles, "
                "we save the outputs and return the locations "
                "to the corresponding files."
            )
            if save_dir is None:
                warnings.warn(
                    "> 1 WSIs detected but there is no save directory set."
                    "All subsequent output will be saved to current runtime"
                    "location under folder 'output'. Overwriting may happen!"
                )
                save_dir = pathlib.Path(os.getcwd()).joinpath("output")

            save_dir = pathlib.Path(save_dir)

        if save_dir is not None:
            save_dir = pathlib.Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=False)



        # None if no output
        outputs = None

        self._ioconfig = ioconfig
        # generate a list of output file paths if number of input images > 1
        file_dict = OrderedDict()

        for idx, img_path in enumerate(imgs):
            img_path = pathlib.Path(img_path)
            img_label = None if labels is None else labels[idx]
            points_path = points[idx]

            dataset = InteractiveSegmentorDataset(
                img_path = img_path, points = points_path,
                resolution = self._ioconfig.input_resolutions[0]["resolution"], 
                units = self._ioconfig.input_resolutions[0]["units"],
                patch_size = self._ioconfig.patch_size
            )

            output_inst_dict = self._predict_engine(
                dataset,
                on_gpu=on_gpu,
            )

            if len(imgs) > 1 or save_output:
                # dynamic 0 padding
                img_code = f"{idx:0{len(str(len(imgs)))}d}"

                save_info = {}
                save_path = os.path.join(str(save_dir), img_code)
                dat_save_path = f"{save_path}.dat"
                save_info["dat"] = dat_save_path
                joblib.dump(output_inst_dict, dat_save_path)
                file_dict[str(img_path)] = save_info


        output = file_dict if len(imgs) > 1 or save_output else output_inst_dict

        return output