"""Utils package for toolbox utilities."""
from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent

env_detection = _lazy_import("env_detection", location)
exceptions = _lazy_import("exceptions", location)
image = _lazy_import("image", location)
metrics = _lazy_import("metrics", location)
misc = _lazy_import("misc", location)
transforms = _lazy_import("transforms", location)
visualization = _lazy_import("visualization", location)
