"""Top-level package for TIA Toolbox."""

from __future__ import annotations

import importlib.resources as importlib_resources
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import yaml

if TYPE_CHECKING:  # pragma: no cover
    from logging import LogRecord
    from types import ModuleType

__author__ = """TIA Centre"""
__email__ = "tialab@dcs.warwick.ac.uk"
__version__ = "1.5.1"

# This will set the tiatoolbox external data
# default to be the user home folder, should work on both Window and Unix/Linux
# C:\Users\USER\.tiatoolbox
# /home/USER/.tiatoolbox

# Initialize internal logging facilities, such that models etc.
# can have reporting mechanism, may need to change protocol
import logging

# We only create a logger if root has no handler to prevent overwriting use existing
# logging
logging.captureWarnings(capture=True)
if not logging.getLogger().hasHandlers():
    formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)

    logger = logging.getLogger()  # get root logger
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
else:
    logger = logging.getLogger()


class DuplicateFilter(logging.Filter):
    """Define an object to filter duplicate logs.

    The DuplicateFilter filters logs to avoid printing them multiple times
    while running code in a loop.

    """

    def filter(self: DuplicateFilter, record: LogRecord) -> bool:
        """Filter input record."""
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False


class _RcParam(TypedDict):
    """All the parameters in the rcParam dictionary should be defined here."""

    TIATOOLBOX_HOME: Path
    pretrained_model_info: dict[str, dict]
    torch_compile_mode: str


def read_registry_files(path_to_registry: str | Path) -> dict:
    """Reads registry files using importlib_resources.

    Args:
        path_to_registry (str or Path):
            Path to registry files from tiatoolbox root.

    Returns:
        Contents of yaml file.


    """
    pretrained_files_registry_path = importlib_resources.as_file(
        importlib_resources.files("tiatoolbox") / str(path_to_registry),
    )

    with pretrained_files_registry_path as registry_file_path:
        registry_handle = Path.open(registry_file_path)
        return yaml.safe_load(registry_handle)


# runtime context parameters
rcParam: _RcParam = {  # noqa: N816
    "TIATOOLBOX_HOME": Path.home() / ".tiatoolbox",
    "pretrained_model_info": read_registry_files(
        "data/pretrained_model.yaml",
    ),  # Load a dictionary of sample files data (names and urls)
    "torch_compile_mode": "default",
    # Set `torch-compile` mode to `default`
    # Options: `disable`, `default`, `reduce-overhead`, `max-autotune`
    # or “max-autotune-no-cudagraphs”
}


def _lazy_import(name: str, module_location: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, module_location)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(name=name, path=str(module_location))
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


if __name__ == "__main__":
    print("tiatoolbox version:" + str(__version__))  # noqa: T201
    location = Path(__file__).parent
    annotation = _lazy_import("annotation", location)
    models = _lazy_import("models", location)
    tiatoolbox = _lazy_import("tiatoolbox", location)
    tools = _lazy_import("tools", location)
    utils = _lazy_import("utils", location)
    visualization = _lazy_import("visualization", location)
    wsicore = _lazy_import("wsicore", location)
