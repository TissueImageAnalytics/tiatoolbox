"""Tests for toolbox global workspace."""

import importlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import tiatoolbox
from tiatoolbox import DuplicateFilter, logger


def test_set_root_dir():
    """Test for setting new root dir."""
    # skipcq
    importlib.reload(tiatoolbox)
    from tiatoolbox import rcParam

    old_root_dir = rcParam["TIATOOLBOX_HOME"]
    test_dir_path = os.path.join(os.getcwd(), "tmp_check/")
    # clean up previous test
    if os.path.exists(test_dir_path):
        os.rmdir(test_dir_path)
    rcParam["TIATOOLBOX_HOME"] = test_dir_path
    # reimport to see if it overwrites
    # silence Deep Source because this is an intentional check
    # skipcq
    from tiatoolbox import rcParam

    os.makedirs(rcParam["TIATOOLBOX_HOME"])
    if not os.path.exists(test_dir_path):
        pytest.fail(f"`{rcParam['TIATOOLBOX_HOME']}` != `{test_dir_path}`")
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"], ignore_errors=True)
    rcParam["TIATOOLBOX_HOME"] = old_root_dir  # reassign for subsequent test


def test_set_logger():
    """Test for setting new logger."""
    logger = logging.getLogger()
    logger.handlers = []  # reset first to overwrite import
    handler_1 = logging.StreamHandler()
    handler_2 = logging.StreamHandler()
    handler_3 = logging.StreamHandler()
    logger.addHandler(handler_1)
    logger.addHandler(handler_2)
    logger.addHandler(handler_3)
    assert len(logger.handlers) == 3
    # skipcq
    importlib.reload(tiatoolbox)
    # should not overwrite, so still have 2 handler
    assert len(logger.handlers) == 3
    logger.handlers = []  # remove all handler
    # skipcq
    importlib.reload(tiatoolbox)
    assert len(logger.handlers) == 2


def helper_logger_test(level: str):
    """Helper for logger tests."""
    if level.lower() in ["debug", "info"]:
        output = "out"
        order = (0, 1)
    else:
        output = "err"
        order = (1, 0)
    run_statement = (
        f"from tiatoolbox import logger; "
        f"import logging; "
        f"logger.setLevel(logging.{level.upper()}); "
        f'logger.{level.lower()}("Test if {level.lower()} is written to std{output}.")'
    )

    proc = subprocess.Popen(
        [
            "python",
            "-c",
            run_statement,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert (
        f"[{level.upper()}] Test if {level.lower()} is written to std{output}.".encode()
        in proc.communicate()[order[0]]
    )
    assert proc.communicate()[order[1]] == b""


def test_logger_output():
    """Tests if logger is writing output to correct value."""
    # Test DEBUG is written to stdout
    helper_logger_test(level="debug")

    # Test INFO is written to stdout
    helper_logger_test(level="info")

    # Test WARNING is written to stderr
    helper_logger_test(level="warning")

    # Test ERROR is written to stderr
    helper_logger_test(level="error")

    # Test CRITICAL is written to stderr
    helper_logger_test(level="critical")


def test_duplicate_filter(caplog):
    """Tests DuplicateFilter for warnings."""
    for _ in range(2):
        logger.warning("Test duplicate filter warnings.")
    assert "Test duplicate filter warnings." in caplog.text
    assert "\n" in caplog.text[:-2]

    caplog.clear()

    duplicate_filter = DuplicateFilter()
    logger.addFilter(duplicate_filter)
    for _ in range(2):
        logger.warning("Test duplicate filter warnings.")
    logger.removeFilter(duplicate_filter)
    assert "Test duplicate filter warnings." in caplog.text
    assert "\n" not in caplog.text[:-2]


def test_lazy_import():
    import sys

    from tiatoolbox import _lazy_import

    assert "exceptions" not in sys.modules

    _lazy_import(
        "exceptions",
        Path(__file__).parent.parent / "tiatoolbox" / "utils" / "exceptions.py",
    )

    assert "exceptions" in sys.modules
