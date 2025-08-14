"""Module for processing SVS metadata and generating fsspec zarr JSON file.

The fsspec zarr json file is meant to be used in case SVS or TIFF files
can be accessed using byte range HTTP API.

The fsspec zarr json file can be opened using FsspecJsonWSIReader.

"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tifffile import TiffFile, tiff2fsspec

from tiatoolbox.wsicore.wsireader import TIFFWSIReaderDelegate

# Constants
EXPECTED_KEY_VALUE_PAIRS = 2
EXPECTED_ARG_COUNT = 4
URL_PLACEHOLDER = "https://replace.me/"


def convert_metadata(metadata: dict) -> dict:
    """Convert metadata to JSON-compatible format."""
    if isinstance(metadata, dict):
        return {key: convert_metadata(value) for key, value in metadata.items()}
    if isinstance(metadata, list):
        return [convert_metadata(item) for item in metadata]
    if isinstance(metadata, datetime):
        return metadata.isoformat()  # Convert datetime to ISO 8601 string
    return metadata


def replace_url(
    data: dict[str, Any], output_path: Path, old_url: str, new_url: str
) -> None:
    """Replace URL in the JSON file."""
    for value in data.values():
        if isinstance(value, list) and value[0] == old_url:
            value[0] = new_url

    with output_path.open("w") as json_file:
        json.dump(data, json_file, indent=2)


def main(svs_file_path: str, json_file_path: str, final_url: str) -> None:
    """Main function to process an SVS file.

    Args:
        svs_file_path (str): The local file path of the SVS file to be processed.
        json_file_path (str): The file path where the output JSON will be saved.
        final_url (str): The URL where the SVS file is stored online
        and can be accessed via HTTP byte range API.

    Example:
        main('/path/to/CMU-1-Small-Region.svs', '/path/to/CMU-1-Small-Region.json', 'https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1-Small-Region.svs')

    """
    url_to_replace = f"{URL_PLACEHOLDER}{Path(svs_file_path).name}"

    tiff = TiffFile(svs_file_path)

    tiff_file_pages = tiff.pages

    # Generate fsspec JSON
    tiff2fsspec(svs_file_path, url=URL_PLACEHOLDER, out=json_file_path)

    if tiff.is_svs:
        metadata = TIFFWSIReaderDelegate.parse_svs_metadata(tiff_file_pages)
    else:  # pragma: no cover
        metadata = TIFFWSIReaderDelegate.parse_generic_tiff_metadata(tiff_file_pages)

    # Convert metadata to JSON-compatible format
    metadata_serializable = convert_metadata(metadata)

    # Read the JSON data from the file
    json_path = Path(json_file_path)
    with json_path.open() as file:
        json_data = json.load(file)

    # Decode `.zattrs` JSON string into a dictionary
    zattrs = json.loads(json_data[".zattrs"])

    # Ensure "multiscales" exists and is a list
    if "multiscales" not in zattrs or not isinstance(
        zattrs["multiscales"], list
    ):  # pragma: no cover
        zattrs["multiscales"] = [{}]  # Initialize as a list with an empty dictionary

    # Update metadata into `.zattrs`
    zattrs["multiscales"][0]["metadata"] = metadata_serializable

    # Convert back to a JSON string
    json_data[".zattrs"] = json.dumps(zattrs)

    # Replace URLs in the JSON file
    replace_url(json_data, json_path, url_to_replace, final_url)


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARG_COUNT:
        msg = " Usage: python script.py <svs_file_path> <json_file_path> <final_url>"
        raise ValueError(msg)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
