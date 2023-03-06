"""Top-level package for TIA Toolbox."""

import importlib.util
import os
import sys
from pathlib import Path

import pkg_resources
import yaml

__author__ = """TIA Lab"""
__email__ = "tialab@dcs.warwick.ac.uk"
__version__ = "1.3.3"

# This will set the tiatoolbox external data
# default to be the user home folder, should work on both Window and Unix/Linux
# C:\Users\USER\.tiatoolbox
# /home/USER/.tiatoolbox

# Initialize internal logging facilities, such that models etc.
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


def _lazy_import(name: str):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

def _lazy_import_class(module_name: str, class_name: str):
    module = _lazy_import(module_name)
    return getattr(module, class_name)


annotation = _lazy_import("tiatoolbox.annotation")
models = _lazy_import("tiatoolbox.models")
tools = _lazy_import("tiatoolbox.tools")
utils = _lazy_import("tiatoolbox.utils")
wsicore = _lazy_import("tiatoolbox.wsicore")

HoVerNet = _lazy_import_class("tiatoolbox.models.architecture.hovernet", "HoVerNet")


if __name__ == "__main__":
    print("tiatoolbox version:" + str(__version__))

    #
