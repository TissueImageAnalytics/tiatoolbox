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
class PatternReplacement:
    """Replacement dataclass.

    Attributes:
        pattern:
            Regex pattern to match.
        replacement:
            Replacement string.
        main_replacement:
            Replacement string for main branch.

    """

    pattern: str
    replacement: str
    main_replacement: str = None


MAIN_BRANCHES = ("master", "main")


def main(files: List[Path], from_ref: str, to_ref: str) -> bool:
    """Check that URLs in the notebook are relative to the current branch.

    Args:
        files:
            List of files to check.
        from_ref:
            Reference to diff from.
        to_ref:
            Reference to diff to.

    """
    replacements = [
        PatternReplacement(
            pattern=(
                r"(^\s*[!%]\s*)pip install "
                r"(git\+https://github\.com/TissueImageAnalytics/tiatoolbox\.git@[\S]*|tiatoolbox)"  # noqa: E501
            ),
            replacement=(
                r"\1pip install "
                f"git+https://github.com/TissueImageAnalytics/tiatoolbox.git@{to_ref}"
            ),
            main_replacement=r"\1pip install tiatoolbox",
        ),
        PatternReplacement(
            pattern=(
                r"https://github\.com/TissueImageAnalytics/tiatoolbox/(blob|tree)/"
                r"(.*)/examples/(.*)\.ipynb"
            ),
            replacement=(
                "https://github.com/TissueImageAnalytics/tiatoolbox/\1/"
                f"{to_ref}/examples/\3.ipynb"
            ),
        ),
    ]
    passed = True
    print(f"From ref '{from_ref}' to ref '{to_ref}'")
    for path in files:
        changed, notebook = check_notebook(path, to_ref, replacements)
        passed = passed and not changed
        # Write the file if it has changed
        if changed:
            print(f"Updating {path}")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(notebook, fh, indent=1, ensure_ascii=False)
                fh.write("\n")
        else:
            print(f"Skipping {path} (no changes)")
    return passed


def check_notebook(
    path: Path, to_ref: str, replacements: List[PatternReplacement]
) -> Tuple[bool, dict]:
    """Check the notebook for URL replacements.

    Args:
        path:
            Path to notebook.
        to_ref:
            Reference to diff to.
        replacements:
            List of replacements to perform.

    Returns:
        Tuple of whether the file was changed and the notebook object.

    """
    project_root = Path(__file__).parent.parent
    changed = False
    # Check if the path is inside the project root
    if path.resolve().parents[1] != project_root.resolve():
        print(f"Skipping {path} (not inside the project directory)")
        return changed, None
    # Load the notebook
    with open(path, encoding="utf-8") as fh:
        notebook = json.load(fh)
        # Check each cell
    for cell_num, cell in enumerate(notebook["cells"]):
        # Check each line
        for line_num, line in enumerate(cell["source"]):
            new_line = replace_line(line, to_ref, replacements)
            if new_line != line:
                print(f"{path.name}: Changed (cell {cell_num+1}, line {line_num+1})")
                changed = True
                cell["source"][line_num] = new_line
    return changed, notebook


def replace_line(line: str, to_ref: str, replacements: List[PatternReplacement]) -> str:
    """Perform pattern replacements in the line.

    Args:
        line:
            Line to replace.
        to_ref:
            Reference to diff to.
        replacements:
            List of replacements to perform.

    """
    for rep in replacements:
        if re.match(rep.pattern, line):
            # Replace matches
            if to_ref in MAIN_BRANCHES:
                line = re.sub(rep.pattern, rep.main_replacement, line)
            else:
                line = re.sub(rep.pattern, rep.replacement, line)
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
