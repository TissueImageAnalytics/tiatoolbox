"""Generate a conda environment YAML from pyproject.toml.

Converts pip-style dependencies into conda-forge compatible dependencies.

Python version is controlled via the PYTHON_VERSION environment variable.

"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

# ================================
# Config
# ================================
PYPROJECT_FILE = "pyproject.toml"
OUT_FILE = "requirements/requirements.conda.generated.yml"

PYTHON_VERSION = os.environ.get("PYTHON_VERSION", "3.15")

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
    """Parse requirement string into (name, version)."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None

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
    return new if len(new) > len(existing) else existing


def to_yaml(env: dict) -> str:
    """Convert environment dict to YAML string."""
    lines = [
        f"name: {env['name']}",
        "channels:",
        *[f"  - {ch}" for ch in env["channels"]],
    ]

    if "channel_priority" in env:
        lines.append(f"channel_priority: {env['channel_priority']}")

    lines.extend(
        [
            "dependencies:",
            *[f"  - {dep}" for dep in env["dependencies"]],
        ]
    )

    return "\n".join(lines) + "\n"


# ================================
# Main
# ================================
def main() -> None:
    """Generate conda environment YAML."""
    pyproject_path = Path(PYPROJECT_FILE)

    if not pyproject_path.exists():
        msg = f"{PYPROJECT_FILE} not found"
        raise FileNotFoundError(msg)

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    raw_deps = pyproject["project"]["dependencies"]

    deps: dict[str, str] = {}

    for raw_line in raw_deps:
        parsed = parse_line(raw_line)
        if not parsed:
            continue

        name, version = parsed

        # ✅ special case: openslide-bin → openslide (ignore version)
        # Due to different version numbering between conda-forge and pip
        if name == "openslide-bin":
            deps["openslide"] = ""
            continue

        mapped = PIP_TO_CONDA.get(name, name)

        previous = deps.get(mapped, "")
        deps[mapped] = merge_versions(previous, version)

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
    out_path.write_text(to_yaml(env))


# ================================
# Entry
# ================================
if __name__ == "__main__":
    main()
