"""Simple check to ensure each code cell in a notebook is valid Python."""
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple


def git_branch_name() -> str:
    """Get the current branch name."""
    return (
        subprocess.check_output(["/usr/bin/git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )


def git_branch_modified_paths(from_ref: str, to_ref: str) -> Set[Path]:
    """Get a set of file paths modified on this branch vs develop."""
    from_to = f"{from_ref}...{to_ref}"
    return {
        Path(p)
        for p in subprocess.check_output(
            [
                "/usr/bin/git",
                "diff",
                "--name-only",
                from_to,
            ]
        )
        .decode()
        .strip()
        .splitlines()
    }


def git_previous_commit_modified_paths() -> Set[Path]:
    """Get a set of file paths modified in the previous commit."""
    return {
        Path(p)
        for p in subprocess.check_output(
            ["/usr/bin/git", "diff", "--name-only", "HEAD~"]
        )
        .decode()
        .strip()
        .splitlines()
    }


@dataclass(frozen=True)
class Replacement:
    """Replacement dataclass.

    Attributes:
        pattern:
            Regex pattern to match.
        replace:
            Replacement string.
        main_replace:
            Replacement string for main branch.

    """

    pattern: str
    replace: str
    main_replace: str = None


MAIN_BRANCHES = ("master", "main")
GIT_REV = git_branch_name()
REPLACEMENTS = [
    Replacement(
        (
            r"(^\s*[!%]\s*)pip install "
            "(git+https://github.com/TissueImageAnalytics/tiatoolbox.git@.*|tiatoolbox)"
        ),
        (
            r"\1pip install "
            f"git+https://github.com/TissueImageAnalytics/tiatoolbox.git@{GIT_REV}"
        ),
        r"\1pip install tiatoolbox",
    ),
]


def main(files: List[Path], from_ref: str, to_ref: str) -> bool:
    """Check that URLs in the notebook are relative to the current branch."""
    passed = True
    print(f"From ref '{from_ref}' to ref '{to_ref}'")
    modified_paths = git_branch_modified_paths(from_ref, to_ref)
    for path in files:
        if path not in modified_paths:
            print(f"Skipping {path} (modified but not in given file list)")
            continue
        changed, notebook = check_notebook(path)
        passed = passed and not changed
        # Write the file if it has changed
        if changed:
            print(f"Updating {path}")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(notebook, fh, indent=1, ensure_ascii=False)
                fh.write("\n")
        else:
            print(f"Skipping {path}")
    return passed


def check_notebook(path: Path) -> Tuple[bool, dict]:
    """Check the notebook for URL replacements.

    Args:
        path:
            Path to notebook.

    Returns:
        Tuple of whether the file was changed and the notebook object.

    """
    file_changed = False
    # Load the notebook
    with open(path, encoding="utf-8") as fh:
        notebook = json.load(fh)
        # Check each cell
    for cell_num, cell in enumerate(notebook["cells"]):
        # Check each line
        for line_num, line in enumerate(cell["source"]):
            new_line = replace_line(line)
            if new_line != line:
                print(f"{path.name}: Changed (cell {cell_num+1}, line {line_num+1})")
                file_changed = True
                cell["source"][line_num] = new_line
    return file_changed, notebook


def replace_line(line: str) -> str:
    """Perform pattern replacements in the line."""
    for r in REPLACEMENTS:
        if re.match(r.pattern, line):
            # Replace matches
            if GIT_REV in MAIN_BRANCHES:
                line = re.sub(r.pattern, r.main_replace, line)
            else:
                line = re.sub(r.pattern, r.replace, line)
            print(line)
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check notebook URLs")
    parser.add_argument(
        "files",
        nargs="+",
        help="Path to notebook(s)",
        type=Path,
        default=list(Path.cwd().rglob("*.ipynb")),
    )
    parser.add_argument(
        "-f", "--from-ref", help="Reference to diff from", type=str, default="develop"
    )
    parser.add_argument(
        "-t",
        "--to-ref",
        help="Reference to diff to",
        type=str,
        default=git_branch_name(),
    )
    args = parser.parse_args()
    sys.exit(1 - main(args.files, args.from_ref, args.to_ref))
