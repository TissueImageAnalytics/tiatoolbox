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
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Unit test package for ABC and __init__ ."""

import pytest

from tiatoolbox import rcParam
from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.architecture import get_pretrained_model


@pytest.mark.skip(reason="Local test, not applicable for travis.")
def test_get_pretrained_model():
    """Test for downloading and creating pretrained models."""
    pretrained_info = rcParam["pretrained_model_info"]
    for pretrained_name in pretrained_info.keys():
        get_pretrained_model(pretrained_name, overwrite=True)


def test_model_abc():
    """Test API in model ABC."""
    # test missing definition for abstract
    with pytest.raises(TypeError):
        # crash due to not defining forward, infer_batch, postproc
        ModelABC()  # skipcq

    # intentionally created to check error
    # skipcq
    class Proto(ModelABC):
        # skipcq
        def __init__(self):
            super().__init__()

        @staticmethod
        # skipcq
        def infer_batch():
            pass

    # skipcq
    with pytest.raises(TypeError):
        # crash due to not defining forward and postproc
        Proto()  # skipcq

    # intentionally create to check inheritance
    # skipcq
    class Proto(ModelABC):
        # skipcq
        def forward(self):
            pass

        @staticmethod
        # skipcq
        def infer_batch():
            pass

    model = Proto()
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(1) == 1, "Must be unchanged!"

    # intentionally created to check error
    # skipcq
    class Proto(ModelABC):
        # skipcq
        def __init__(self):
            super().__init__()

        @staticmethod
        # skipcq
        def postproc(image):
            return image - 2

        # skipcq
        def forward(self):
            pass

        @staticmethod
        # skipcq
        def infer_batch():
            pass

    model = Proto()  # skipcq
    # test assign un-callable to preproc_func/postproc_func
    with pytest.raises(ValueError, match=r".*callable*"):
        model.postproc_func = 1
    with pytest.raises(ValueError, match=r".*callable*"):
        model.preproc_func = 1

    # test setter/getter/initial of preproc_func/postproc_func
    assert model.preproc_func(1) == 1
    model.preproc_func = lambda x: x - 1
    assert model.preproc_func(1) == 0
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(1) == -1, "Must be unchanged!"
    model.preproc_func = None
    assert model.preproc_func(2) == 2

    # repeat the setter test for postproc
    assert model.postproc_func(2) == 0
    model.postproc_func = lambda x: x - 1
    assert model.postproc_func(1) == 0
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(2) == 0, "Must be unchanged!"
    # coverage setter check
    model.postproc_func = None
    assert model.postproc_func(2) == 0
