import ast
import doctest
import importlib
import os
import sys
from pathlib import Path
from typing import List, Type, Union

import pytest


@pytest.fixture()
def source_files(root_path):
    """Recursively yield source files from the project."""
    ignore = {"__pycache__"}

    def generator():
        for root, dirs, files in os.walk(root_path):
            files = [f for f in files if f.endswith(".py") and f[0] != "."]
            dirs[:] = [d for d in dirs if d not in ignore and d[0] != "."]
            for file in files:
                yield Path(root, file)

    return generator()


def extract_examples(docstring: str) -> str:
    """Extract doctest examples from a docstring.

    Args:
        docstring (str):
            The docstring to extract examples from.

    Returns:
        str:
            The doctest examples.

    """
    return


def test_iter_sources(source_files, root_path):
    """Test that all source files exist."""
    for file in source_files:

        with open(file) as fh:
            source = fh.read()
        tree = ast.parse(source)
        parser = doctest.DocTestParser()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            doc = parser.get_doctest(docstring, [], node.name, file.name, node.lineno)
            if not doc.examples:
                continue

            # Find how many lines the class/function signature spans
            # (for accurate line numbers)
            signature_lines = (
                node.body[0].lineno
                - (node.lineno + len(doc.docstring.splitlines()))
                - 1
            )
            doc.lineno += signature_lines - 1

            # Check syntax is valid
            source_tree = check_ast(doc, file.relative_to(root_path))

            check_imports(source_tree, root_path, file, doc)


def check_imports(
    source_tree: ast.AST, root_path: Path, file: Path, doc: doctest.DocTest
) -> None:
    """Check that imports in the source AST are valid."""
    imports = [
        node
        for node in ast.walk(source_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    sys.modules["tiatoolbox"] = importlib.import_module("tiatoolbox", root_path)
    for import_node in imports:
        names = import_node_names(import_node)
        # Resolve the import
        for name in names:
            lineno = doc.lineno + doc.examples[0].lineno + import_node.lineno
            try:
                spec = importlib.util.find_spec(name, package=root_path)
            except ModuleNotFoundError as e:
                pytest.fail(
                    f"{file.relative_to(root_path)}:{lineno}:"
                    f" ModuleNotFoundError: {e.msg}"
                )
            if not (spec or name in sys.modules):
                pytest.fail(
                    f"{file.relative_to(root_path)}:{lineno}: "
                    f"ImportError: No module named '{name}'"
                )


def import_node_names(import_node: Union[ast.Import, ast.ImportFrom]) -> List[str]:
    """Get the names being imported by import nodes."""
    if isinstance(import_node, ast.ImportFrom):
        return [import_node.module]
    if isinstance(import_node, ast.Import):
        return [name.name for name in import_node.names]
    raise Type("Unknown node type")


def check_ast(doc, rel_path) -> ast.AST:
    """Check that the source syntax is valid."""
    source = "".join(eg.source for eg in doc.examples)
    try:
        return ast.parse(source)
    except SyntaxError as e:
        lineno = doc.lineno + doc.examples[0].lineno + e.lineno
        pytest.fail(f"{rel_path}:{lineno}: SyntaxError: {e.msg}")
