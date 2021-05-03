# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****


import tiatoolbox.models.abc as tia_model_abc

import torch
import torch.nn as nn


class Model_Base(tia_model_abc.Model_Base):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(
        self,
        batch_size,
        infer_input_shape=None,
        infer_output_shape=None,
        nr_loader_worker=0,
        nr_posproc_worker=0,
        preproc_args=None,
        postproc_args=None,
    ):
        super().__init__()
        raise NotImplementedError

    def load_model(self, checkpoint_path, *args, **kwargs):
        """Load model checkpoint."""
        raise NotImplementedError

    @staticmethod
    def __infer_batch(model, batch_data):
        """Contain logic for forward operation as well as i/o aggregation.

        image_list: Torch.Tensor (N,...)
        info_list : A list of (N,...), each item is metadata correspond to
                    image at same index in `image_list`

        """
        raise NotImplementedError

    @staticmethod
    def postprocess(image, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def preprocess(image, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X, *args, **kwargs):
        """The most basic and is in line with sklearn model.predict(X)
        where X is an image list (np.array). Internally, this will
        create an internal dataset and call predict_dataset.

        ! Should X alrd be in-compliance shape wrt to infer,
        ! or we pad it on the fly?

        Return the prediction after being post process.

        """
        raise NotImplementedError

    def predict_dataset(self, dataset, *args, **kwargs):
        """Apply the prediction on a dataset object. Dataset object is Torch compliance
        and return output should be compatible with input of __infer_batch.

        """
        raise NotImplementedError

    def predict_wsi(self, wsi_path, *args, **kwargs):
        """Contain dedicated functionality to run inference on an entire WSI."""
        raise NotImplementedError
