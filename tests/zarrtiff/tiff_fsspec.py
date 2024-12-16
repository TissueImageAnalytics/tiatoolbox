"""Module for processing SVS metadata and generating fsspec JSON file."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tifffile import TiffFile, TiffPages, tiff2fsspec

if TYPE_CHECKING:
    from numbers import Number

# Constants
EXPECTED_KEY_VALUE_PAIRS = 2
EXPECTED_ARG_COUNT = 4
URL_PLACEHOLDER = "https://replace.me/"


def _parse_svs_metadata(pages: TiffPages) -> dict[str, Any]:
    # Copy/paste from TIFFWSIReader._parse_svs_metadata, extract to the util method.
    """Extract SVS-specific metadata."""
    raw = {}
    mpp: list[float] | None = None
    objective_power: float | None = None
    vendor = "Aperio"

    description = pages[0].description
    raw["Description"] = description
    parts = description.split("|")
    description_headers, key_value_pairs = parts[0], parts[1:]
    description_headers = description_headers.split(";")

    software, photometric_info = description_headers[0].splitlines()
    raw["Software"] = software
    raw["Photometric Info"] = photometric_info

    def parse_svs_tag(string: str) -> tuple[str, Number | str | datetime]:
        """Parse SVS key-value string."""
        pair = string.split("=")
        if len(pair) != EXPECTED_KEY_VALUE_PAIRS:
            invalid_metadata_msg = (
                "Invalid metadata. Expected string of the format 'key=value'."
            )
            raise ValueError(invalid_metadata_msg)

        key, value_string = pair
        key = key.strip()
        value_string = value_string.strip()

        def us_date(string: str) -> datetime:
            """Return datetime parsed according to US date format."""
            return datetime.strptime(string, r"%m/%d/%y").astimezone()

        def time(string: str) -> datetime:
            """Return datetime parsed according to HMS format."""
            return datetime.strptime(string, r"%H:%M:%S").astimezone()

        casting_precedence = [us_date, time, int, float]
        value: Number | str | datetime = value_string
        for cast in casting_precedence:
            try:
                value = cast(value_string)
                break
            except ValueError:
                continue

        return key, value

    svs_tags = dict(parse_svs_tag(string) for string in key_value_pairs)
    raw["SVS Tags"] = svs_tags
    mpp = [svs_tags.get("MPP")] * 2 if svs_tags.get("MPP") is not None else None
    objective_power = svs_tags.get("AppMag")

    return {
        "objective_power": objective_power,
        "vendor": vendor,
        "mpp": mpp,
        "raw": raw,
    }


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
    """Main function to handle SVS file processing."""
    url_to_replace = f"{URL_PLACEHOLDER}{Path(svs_file_path).name}"

    tiff_file_pages = TiffFile(svs_file_path).pages

    # Generate fsspec JSON
    tiff2fsspec(svs_file_path, url=URL_PLACEHOLDER, out=json_file_path)

    # Parse SVS metadata
    metadata = _parse_svs_metadata(pages=tiff_file_pages)

    # Convert metadata to JSON-compatible format
    metadata_serializable = convert_metadata(metadata)

    # Read the JSON data from the file
    json_path = Path(json_file_path)
    with json_path.open() as file:
        json_data = json.load(file)

    # Decode `.zattrs` JSON string into a dictionary
    zattrs = json.loads(json_data[".zattrs"])

    # Update metadata into `.zattrs`
    if "multiscales" in zattrs and isinstance(zattrs["multiscales"], list):
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
