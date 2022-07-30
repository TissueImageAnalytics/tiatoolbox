import ast
import doctest
import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

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


def test_validate_docstring_examples(source_files, root_path):
    """Test that all docstring examples are valid.

    Validity checks are:
    1. The docstring examples are syntactically valid (can parse an AST).
    2. That the imports can be resolved.

    """
    for file in source_files:

        source = Path(file).read_text()
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
            rel_path = file.relative_to(root_path)
            source_tree = check_ast(doc, rel_path)
            check_imports(source_tree, doc, rel_path)


def check_imports(source_tree: ast.AST, doc: doctest.DocTest, rel_path: Path) -> None:
    """Check that imports in the source AST are valid."""
    if source_tree is None:
        return
    imports = [
        node
        for node in ast.walk(source_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for import_node in imports:
        names = import_node_names(import_node)
        # Resolve the import
        for name in names:
            lineno = doc.lineno + doc.examples[0].lineno + import_node.lineno
            source = "\n".join(eg.source.strip() for eg in doc.examples)
            try:
                spec = importlib.util.find_spec(name)
            except ModuleNotFoundError as e:
                raise_source_exception(
                    source,
                    rel_path,
                    import_node.lineno,
                    lineno,
                    import_node.col_offset,
                    e,
                )
            if not (spec or name in sys.modules):
                raise_source_exception(
                    source,
                    rel_path,
                    import_node.lineno,
                    lineno,
                    import_node.col_offset,
                )


def raise_source_exception(
    source: str,
    rel_path: Path,
    source_lineno: int,
    file_lineno: int,
    source_offset: Optional[int] = None,
    exception: Optional[Exception] = None,
) -> None:
    """Raise an exception with the source code and line number highlighted.

    Args:
        source (str):
            The source code.
        rel_path (Path):
            The path to the file.
        source_lineno (int):
            The line number in the source code snippet.
        file_lineno (int):
            The line number in the file.
        source_offset (int):
            The offset in the source code snippet. Optional.
        exception (Exception):
            The parent exception which was caught. Optional.

    Raises:
        SyntaxError: If the source code is invalid.
        ModuleNotFoundError: If the module cannot be found.

    """
    message = exception.msg if exception else ""
    source_lines = [
        ("...." if n != source_lineno - 1 else "   >") + line
        for n, line in enumerate(source.splitlines())
    ]
    if source_offset:
        source_lines.insert(source_lineno, f"{' '*(source_offset+3)}^ {message}")
    annotated_source = "\n".join(source_lines)
    exception = type(exception) if exception else SyntaxError
    raise exception(
        f"{rel_path}:{file_lineno}: {message}\n{annotated_source}"
    ) from None


def import_node_names(import_node: Union[ast.Import, ast.ImportFrom]) -> List[str]:
    """Get the names being imported by import nodes."""
    if isinstance(import_node, ast.ImportFrom):
        return [import_node.module]
    if isinstance(import_node, ast.Import):
        return [name.name for name in import_node.names]
    raise TypeError("Unknown node type")


def check_ast(doc, rel_path) -> ast.AST:
    """Check that the source syntax is valid."""
    source = "".join(eg.source for eg in doc.examples)
    try:
        return ast.parse(source, rel_path)
    except SyntaxError as e:
        lineno = doc.lineno + doc.examples[0].lineno + e.lineno
        raise_source_exception(source, rel_path, e.lineno, lineno, e.offset, e)
    return None
