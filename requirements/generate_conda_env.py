"""Generate a conda environment YAML from requirements.txt.

Converts pip-style dependencies into conda-forge compatible dependencies.

Python version is controlled via the PYTHON_VERSION environment variable.

"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

# ================================
# Config
# ================================
REQ_FILE = "requirements/requirements.txt"
OUT_FILE = "requirements/requirements.conda.generated.yml"

PYTHON_VERSION = os.environ.get("PYTHON_VERSION", "3.14")

# pip → conda mapping
PIP_TO_CONDA = {
    "opencv-python": "opencv",
    "opencv-python-headless": "opencv",
    "simpleitk": "simpleitk",
    "torch": "pytorch",
    "torchvision": "torchvision",
    "pillow": "pillow",
    "scikit-learn": "scikit-learn",
    "scikit-image": "scikit-image",
    "pyyaml": "pyyaml",
    "openslide-bin": "openslide",
}


# ================================
# Helpers
# ================================
def parse_line(line: str) -> tuple[str, str] | None:
    """Parse a requirements.txt line into (name, version)."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None

    # Remove extras
    line = re.split(r"\[", line)[0]

    match = re.split(r"(>=|<=|==|~=|>|<)", line, maxsplit=1)

    name = match[0].strip().lower()
    version = "".join(match[1:]) if len(match) > 1 else ""

    return name, version


def merge_versions(existing: str, new: str) -> str:
    """Keep the most specific version constraint."""
    if not existing:
        return new
    if not new:
        return existing

    # Prefer stricter constraint (simple heuristic)
    return new if len(new) > len(existing) else existing


# ================================
# Main
# ================================
def main() -> None:
    """Generate conda environment YAML."""
    req_path = Path(REQ_FILE)

    if not req_path.exists():
        msg = f"{REQ_FILE} not found"
        raise FileNotFoundError(msg)

    deps: dict[str, str] = {}

    for raw_line in req_path.read_text().splitlines():
        parsed = parse_line(raw_line)
        if not parsed:
            continue

        name, version = parsed

        # ✅ Apply pip → conda mapping
        mapped = PIP_TO_CONDA.get(name, name)

        # ✅ Merge versions if duplicate appears
        previous = deps.get(mapped, "")
        deps[mapped] = merge_versions(previous, version)

    # ✅ Final deterministic ordering
    sorted_deps = sorted(f"{pkg}{ver}" for pkg, ver in deps.items())

    env = {
        "name": "tiatoolbox",
        "channels": ["conda-forge"],
        "channel_priority": "strict",
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            *sorted_deps,
        ],
    }

    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(env, sort_keys=False))


# ================================
# Entry
# ================================
if __name__ == "__main__":
    main()
