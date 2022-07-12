"""Tests for toolbox global workspace."""

import importlib
import logging
import os
import shutil

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
    logger.addHandler(handler_1)
    logger.addHandler(handler_2)
    assert len(logger.handlers) == 2
    # skipcq
    importlib.reload(tiatoolbox)
    # should not overwrite, so still have 2 handler
    assert len(logger.handlers) == 2
    logger.handlers = []  # remove all handler
    # skipcq
    importlib.reload(tiatoolbox)
    assert len(logger.handlers) == 1
