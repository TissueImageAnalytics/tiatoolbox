"""Generate a conda environment YAML from requirements.txt.

This script converts pip-style dependencies into conda-forge compatible
dependencies and outputs a fully solvable conda environment file.

The Python version is controlled via the PYTHON_VERSION environment variable.

"""

import os
import re
from pathlib import Path

import yaml

# ================================
# Configuration
# ================================
REQ_FILE = "requirements/requirements.txt"
OUT_FILE = "requirements/requirements.conda.generated.yml"

# Python version injected from CI (default = 3.14)
PYTHON_VERSION = os.environ.get("PYTHON_VERSION", "3.14")

# Mapping: pip package name → conda-forge name
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
}


# ================================
# Helpers
# ================================
def parse_requirement(line: str) -> tuple[str, str] | None:
    """Parse a single requirements.txt line into (name, version_spec).

    Removes inline comments and extras, and maps pip names to conda equivalents.
    """
    # Remove inline comments
    line = line.split("#", 1)[0].strip()

    if not line:
        return None

    # Remove extras (e.g. package[extra])
    line = re.split(r"\[", line)[0]

    # Split version spec
    match = re.split(r"(>=|<=|==|~=|>|<)", line, maxsplit=1)

    name = match[0].strip().lower()

    # Map pip → conda name
    name = PIP_TO_CONDA.get(name, name)

    if len(match) > 1:
        version = "".join(match[1:])
        return name, version

    return name, ""


# ================================
# Main logic
# ================================
def main() -> None:
    """Generate a conda environment YAML from requirements.txt."""
    req_path = Path(REQ_FILE)

    if not req_path.exists():
        msg = f"{REQ_FILE} not found"
        raise FileNotFoundError(msg)

    conda_deps: set[str] = set()

    for line in req_path.read_text().splitlines():
        parsed = parse_requirement(line)
        if not parsed:
            continue

        name, version = parsed
        dep = name + version

        conda_deps.add(dep)

    # Ensure deterministic ordering
    sorted_deps = sorted(conda_deps)

    env = {
        "name": "tiatoolbox",
        "channels": ["conda-forge"],
        "channel_priority": "strict",
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",  # installed but not used
            *sorted_deps,
        ],
    }

    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(env, sort_keys=False))


# ================================
# Entry point
# ================================
if __name__ == "__main__":
    main()
