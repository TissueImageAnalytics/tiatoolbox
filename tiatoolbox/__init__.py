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

"""Top-level package for TIA Toolbox."""

import os

import pkg_resources
import yaml

__author__ = """TIA Lab"""
__email__ = "tialab@dcs.warwick.ac.uk"
__version__ = "1.0.0"

# This will set the tiatoolbox external data
# default to be the user home folder, should work on both Window and Unix/Linux
# C:\Users\USER\.tiatoolbox
# /home/USER/.tiatoolbox

# Initialize internal logging facilities, such that models and etc.
# can have reporting mechanism, may need to change protocol
import logging

# We only create a logger if root has no handler to prevent overwriting use existing
# logging
logging.captureWarnings(True)
if not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()  # get root logger
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
else:
    logger = logging.getLogger()

# runtime context parameters
rcParam = {"TIATOOLBOX_HOME": os.path.join(os.path.expanduser("~"), ".tiatoolbox")}

# Load a dictionary of sample files data (names and urls)
PRETRAINED_FILES_REGISTRY_PATH = pkg_resources.resource_filename(
    "tiatoolbox", "data/pretrained_model.yaml"
)
with open(PRETRAINED_FILES_REGISTRY_PATH) as registry_handle:
    PRETRAINED_INFO = yaml.safe_load(registry_handle)
rcParam["pretrained_model_info"] = PRETRAINED_INFO


from tiatoolbox import models, tiatoolbox, tools, utils, wsicore

if __name__ == "__main__":
    print("tiatoolbox version:" + str(__version__))
