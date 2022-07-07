import ast
import importlib
import os
import sys
import tokenize
from pathlib import Path
from typing import Dict, List, Tuple, Union

from requirements_consistency import REQUIREMENTS_FILES, parse_requirements

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
    "scikit-learn": ["sklearn", "joblib"],
    "scikit-image": ["skimage"],
    "pytorch": ["torch"],
}


def find_source_files(base_dir: Path) -> List[Path]:
    """Recursively find all source files in the given directory."""
    ignore = ["venv", "build", "dist", "__pycache__"]
    source_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not (d in ignore or d[0] in (".", "_"))]
        files = [f for f in files if f.endswith(".py") and f[0] not in (".",)]
        source_files.extend(Path(root) / f for f in files)
    return source_files


def find_imports(py_source_path: Path) -> List[str]:
    """Find all imports in the given Python source file."""
    with open(py_source_path, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def std_spec(fullname: str) -> str:
    """Return True if in the standard library or a built-in."""
    if fullname in sys.builtin_module_names:
        return True
    path_finder = importlib.machinery.PathFinder()
    spec = path_finder.find_spec(fullname)
    if spec is None:
        return False
    origin = Path(spec.origin)
    return "site-packages" not in origin.parts and "dist-packages" not in origin.parts


def stem(alias: ast.alias) -> str:
    return alias.name.split(".")[0]


def stems(node: Union[ast.Import, ast.ImportFrom]) -> List[Tuple[ast.alias, str]]:
    if isinstance(node, ast.Import):
        return [(alias.name, stem(alias)) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module, node.module.split(".")[0])]
    raise TypeError(
        f"Unexpected node type: {type(node)}. Should be ast.Import or ast.ImportFrom."
    )


def main():
    root = Path(__file__).parent.parent
    source_root = root / "tiatoolbox"
    requirements_paths = [
        root / p
        for tup in REQUIREMENTS_FILES
        for p in tup
        if p and p != "docs/requirements.txt"
    ]
    passed = True
    for req_path in requirements_paths:
        bad_imports = find_bad_imports(root, source_root, req_path)
        if bad_imports:
            passed = False
    sys.exit(1 - passed)


def find_bad_imports(
    root: Path, source_root: Path, requirements_path: Path
) -> List[Tuple[Union[ast.Import, ast.ImportFrom], ast.alias]]:
    result = []
    # Parse the requirements file
    reqs = parse_requirements(requirements_path)
    # Apply the mapping from known package names to import names
    req_imports = {subkey for key in reqs for subkey in KNOWN_ALIASES.get(key, [key])}

    for path in find_source_files(source_root):
        file_import_nodes = find_imports(path)
        # Mapping of import alias names and stems to nodes
        stem_to_node_alias: Dict[Tuple[ast.alias, str], ast.Import] = {
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
                    f" {stem}" + (f" ({alias})" if alias != stem else "")
                )
    return result


def find_comments(path, line_num: int):
    with open(path, "rb") as fh:
        tokens = tokenize.tokenize(fh.readline)
        return [
            t.string
            for t in tokens
            if t.type == tokenize.COMMENT and t.start[0] == line_num
        ]


if __name__ == "__main__":
    main()
