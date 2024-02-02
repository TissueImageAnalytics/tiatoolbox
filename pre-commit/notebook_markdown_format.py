"""Check markdown cells in notebooks for common mistakes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import mdformat


def format_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Format a notebook in MyST style.

    Args:
        notebook (dict):
            A notebook dictionary (parsed from JSON).

    Returns:
        dict:
            The notebook dictionary with the markdown cells formatted.

    """
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        cell["source"] = [
            f"{line}\n"
            for line in mdformat.text(
                "".join(cell["source"]),
                extensions={"myst"},
                codeformatters={"python"},
            ).split("\n")
        ]
    return notebook


def main(files: list[Path]) -> None:
    """Check markdown cells in notebooks for common mistakes.

    Args:
        files (list):
            A list of notebook files to check.

    Returns:
        bool:
            True if all notebooks pass, False otherwise.

    """
    for path in files:
        notebook = json.loads(path.read_text())
        formatted_notebook = format_notebook(copy.deepcopy(notebook))
        changed = any(
            cell != formatted_cell
            for cell, formatted_cell in zip(
                notebook["cells"],
                formatted_notebook["cells"],
            )
        )
        if not changed:
            continue
        print("Formatting notebook", path)
        with Path.open(path, "w") as fh:
            json.dump(formatted_notebook, fh, indent=1, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lint notebook markdown files.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Notebook markdown files to lint.",
        type=Path,
    )
    args = parser.parse_args()
    main(sorted(args.files))
