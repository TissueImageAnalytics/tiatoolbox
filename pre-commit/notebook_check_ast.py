"""Simple check to ensure each code cell in a notebook is valid Python."""
import argparse
import ast
import json
from pathlib import Path
from typing import List


def main(files: List[Path]) -> bool:
    """Check each file in the list of files for valid Python."""
    passed = True
    for path in files:
        with open(path) as fh:
            notebook = json.load(fh)
        for n, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join([x for x in cell["source"] if not x.startswith("!")])
            try:
                ast.parse(source)
            except SyntaxError as e:
                passed = False
                print(f"{path.name}: {e.msg} (cell {n}, line {e.lineno})")
                break
    return passed  # noqa: R504


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check notebook AST")
    parser.add_argument("files", nargs="+", help="Path to notebook(s)", type=Path)
    args = parser.parse_args()
    exit(1 - main(args.files))
