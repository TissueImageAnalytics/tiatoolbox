"""Static analysis of requirements files and import statements.

Imports which are not found in the requirements files are considered bad.
Any found bad imports will be printed and the script will exit with a non-zero
status.

"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys
import tokenize
from pathlib import Path
from typing import NoReturn

from requirements_consistency import parse_requirements

# Mapping from package name (pip/anaconda) to import name
# This is to avoid pinging PyPI
KNOWN_ALIASES = {
    "pillow": ["PIL"],
    "umap-learn": ["umap"],
    "pyyaml": ["yaml"],
    "openslide-python": ["openslide"],
    "opencv-python": ["cv2"],
    "opencv": ["cv2"],
    "setuptools": ["setuptools", "pkg_resources"],
    "scikit-learn": ["sklearn"],
    "scikit-image": ["skimage"],
    "pytorch": ["torch"],
    "ipython": ["IPython"],
}

REQUIREMENTS_FILES = (
    "requirements/requirements.txt",
    "requirements/requirements_dev.txt",
    "requirements/requirements.conda.yml",
    "requirements/requirements.dev.conda.yml",
    "requirements/requirements.win64.conda.yml",
    "setup.py",
)


def find_source_files(base_dir: Path) -> list[Path]:
    """Recursively find all source files in the given directory.

    Args:
        base_dir (Path):
            Path to the directory to find source files in.

    Returns:
        list:
            List of paths to source files.

    """
    ignore = ["venv", "build", "dist", "__pycache__"]
    source_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not (d in ignore or d[0] in (".", "_"))]
        files = [  # noqa: PLW2901
            f for f in files if f.endswith(".py") and f[0] not in (".",)
        ]
        source_files.extend(Path(root) / f for f in files)
    return source_files


def find_imports(py_source_path: Path) -> list[str]:
    """Find all imports in the given Python source file.

    Args:
        py_source_path (Path):
            Path to the Python source file.

    Returns:
        list:
            List of AST import nodes (ast.Import or ast.ImportFrom) in
            the file.

    """
    with Path.open(  # This file could be any python file anywhere, skipcq
        py_source_path,
    ) as fh:
        source = fh.read()
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def std_spec(fullname: str) -> str:
    """Return True if in the standard library or a built-in.

    Args:
        fullname (str):
            Full name of the module.

    Returns:
        str:
            True if the name is in the standard library or a built-in.

    """
    if fullname in sys.builtin_module_names:
        return True
    path_finder = importlib.machinery.PathFinder()
    spec = path_finder.find_spec(fullname)
    if spec is None:
        return False
    origin = Path(spec.origin)
    return "site-packages" not in origin.parts and "dist-packages" not in origin.parts


def stems(node: ast.Import | ast.ImportFrom) -> list[tuple[str, str]]:
    """Return the stem of each alias in the given import node.

    Args:
        node (ast.Import or ast.ImportFrom):
            Import node to get stems from.

    Returns:
        list:
            List of tuples of the alias name and the stem.

    """
    if isinstance(node, ast.Import):
        return [(alias.name, alias.name.split(".")[0]) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module, node.module.split(".")[0])]
    msg = f"Unexpected node type: {type(node)}. Should be ast.Import or ast.ImportFrom."
    raise TypeError(
        msg,
    )


def main() -> NoReturn:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Static analysis of requirements files and import statements.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Paths to source files to check. If not specified, all files in"
            " the tiatoolbox directory are checked."
        ),
        type=Path,
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    source_root = root / "tiatoolbox"
    requirements_paths = [root / name for name in REQUIREMENTS_FILES]
    source_files = args.files or find_source_files(source_root)
    passed = True
    for req_path in requirements_paths:
        bad_imports = find_bad_imports(root, source_files, req_path)
        if bad_imports:
            passed = False
    sys.exit(1 - passed)


def find_bad_imports(
    root: Path,
    source_files: list[Path],
    requirements_path: Path,
) -> list[tuple[ast.Import | ast.ImportFrom, ast.alias]]:
    """Find bad imports in the given requirements file.

    Args:
        root (Path):
            Root directory of the project.
        source_files (list(Path)):
            Root directory of the source code.
        requirements_path (Path):
            Path to the requirements file.

    Returns:
        list:
            List of bad imports as tuples of the import AST node and the
            alias.

    """
    result = []
    # Parse the requirements file
    reqs = parse_requirements(requirements_path)
    # Apply the mapping from known package names to import names
    req_imports = {subkey for key in reqs for subkey in KNOWN_ALIASES.get(key, [key])}

    for path in source_files:
        file_import_nodes = find_imports(path)
        # Mapping of import alias names and stems to nodes
        stem_to_node_alias: dict[tuple[ast.alias, str], ast.Import] = {
            stem: (node, alias)
            for node in file_import_nodes
            for alias, stem in stems(node)
            if not std_spec(stem)
        }
        bad_imports = {
            stem: (node, alias)
            for stem, (node, alias) in stem_to_node_alias.items()
            if stem not in req_imports.union({"tiatoolbox"})
        }
        if bad_imports:
            for stem, (node, alias) in bad_imports.items():
                # Tokenize the line to check for noqa comments
                comments = find_comments(path, node.lineno)
                if "# noqa" in comments:
                    continue
                result.append((node, alias))
                print(
                    f"{path.relative_to(root)}:{node.lineno}:"
                    f" Import not in {requirements_path.name}:"
                    f" {stem}" + (f" ({alias})" if alias != stem else ""),
                )
    return result


def find_comments(path: str | Path, line_num: int) -> list:
    """Find comments on the given line.

    Args:
        path (str | Path):
            Path to the file.
        line_num (int):
            Line number to find comments on.

    Returns:
        list:
            List of comments on the line.

    """
    with Path.open(path, "rb") as fh:
        # This file could be any python file anywhere.
        tokens = tokenize.tokenize(fh.readline)
        return [
            t.string
            for t in tokens
            if t.type == tokenize.COMMENT and t.start[0] == line_num
        ]


if __name__ == "__main__":
    main()
