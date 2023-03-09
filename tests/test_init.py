"""Tests for toolbox global workspace."""

import importlib
import logging
import os
import shutil
import subprocess

import pytest

import tiatoolbox


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


def helper_logger_test(run_statement: str):
    """Helper for logger tests."""
    proc = subprocess.Popen(
        [
            "python",
            "-c",
            run_statement,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.communicate()[0], proc.communicate()[1]


def test_logger_output():
    """Tests if logger is writing output to correct value."""
    # Test DEBUG is written to stdout
    run_statement = (
        "from tiatoolbox import logger; "
        "import logging; "
        "logger.setLevel(logging.DEBUG); "
        'logger.debug("Test if debug is written to stdout.")'
    )
    out, err = helper_logger_test(run_statement=run_statement)
    assert b"[DEBUG] Test if debug is written to stdout." in out
    assert err == b""

    # Test INFO is written to stdout
    run_statement = (
        "from tiatoolbox import logger; "
        "import logging; "
        "logger.setLevel(logging.INFO); "
        'logger.info("Test if info is written to stdout.")'
    )
    out, err = helper_logger_test(run_statement=run_statement)
    assert b"[INFO] Test if info is written to stdout." in out
    assert err == b""

    # Test WARNING is written to stderr
    run_statement = (
        "from tiatoolbox import logger; "
        "import logging; "
        "logger.setLevel(logging.WARNING); "
        'logger.warning("Test if warning is written to stderr.")'
    )
    out, err = helper_logger_test(run_statement=run_statement)
    assert out == b""
    assert b"[WARNING] Test if warning is written to stderr." in err

    # Test ERROR is written to stderr
    run_statement = (
        "from tiatoolbox import logger; "
        "import logging; "
        "logger.setLevel(logging.ERROR); "
        'logger.error("Test if error is written to stderr.")'
    )
    out, err = helper_logger_test(run_statement=run_statement)
    assert out == b""
    assert b"[ERROR] Test if error is written to stderr." in err

    # Test ERROR is written to stderr
    run_statement = (
        "from tiatoolbox import logger; "
        "import logging; "
        "logger.setLevel(logging.ERROR); "
        'logger.error("Test if error is written to stderr.")'
    )
    out, err = helper_logger_test(run_statement=run_statement)
    assert out == b""
    assert b"[ERROR] Test if error is written to stderr." in err

    # Test CRITICAL is written to stderr
    run_statement = (
        "from tiatoolbox import logger; "
        "import logging; "
        "logger.setLevel(logging.CRITICAL); "
        'logger.critical("Test if critical is written to stderr.")'
    )
    out, err = helper_logger_test(run_statement=run_statement)
    assert out == b""
    assert b"[CRITICAL] Test if critical is written to stderr." in err
