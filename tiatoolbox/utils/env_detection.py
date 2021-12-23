# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****
"""Detection methods for the current environment.

This module contains methods for detecting aspects of the current
environment. Some things which this module can detect are:
 - Whether the current environment is interactive.
 - Whether the current environment is a conda environment.
 - Whether the current environment is running on travis, kaggle, or colab.

Note that these detections may not be correct 100% of the time but are
as accurate as can be reasonably be expected depending on what is being
detected.

"""

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
from numbers import Number
from typing import Tuple

from tiatoolbox import logger


def is_interactive() -> bool:
    """Detect if the current environment is interactive.

    This should return True for the following environments:
        - Python REPL (`$ python`)
        - IPython REPL (`$ ipython`)
        - Interactive Python shell (`$ python -i`)
        - Interactive IPython shell (`$ ipython -i`)
        - IPython passed a string (`$ ipython -c "print('Hello')"`)
        - Notebooks
            - Jupyter (`$ jupyter notebook`)
            - Google CoLab
            - Kaggle Notebooks
            - Jupyter lab (`$ jupyter lab`)
            - VSCode Python Interactive window (`# %%` cells)
            - VSCode Jupyter notebook environment
        - PyCharm Console

    This should return False for the following environments:
        - Python script (`$ python script.py`)
        - Python passed a string (`$ python -c "print('Hello')"`)
        - PyCharm Run
        - PyCharm Run (emulate terminal)

    Returns:
        bool: True if the current environment is interactive, False otherwise.

    """
    return hasattr(sys, "ps1")


def is_notebook() -> bool:
    """Detect if the current environment is a Jupyter notebook.

    Based on a method posted on StackOverflow:
     - Question at https://stackoverflow.com/questions/15411967
     - Question by Christoph
       (https://stackoverflow.com/users/498873/christoph)
     - Answer by Gustavo Bezerra
       (https://stackoverflow.com/users/2132753/gustavo-bezerra)

    Returns:
        bool: True if the current environment is a Jupyter notebook, False

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell == "TerminalInteractiveShell":  # noqa: PIE801
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def in_conda_env() -> bool:
    """Detect if the current environment is a conda environment.

    Returns:
        bool: True if the current environment is a conda environment, False otherwise.

    """
    return "CONDA_DEFAULT_ENV" in os.environ and "CONDA_PREFIX" in os.environ


def running_on_travis() -> bool:
    """Detect if the current environment is running on travis.

    Returns:
        bool: True if the current environment is on travis, False otherwise.

    """
    return os.environ.get("TRAVIS") == "true" and os.environ.get("CI") == "true"


def running_on_kaggle() -> bool:
    """Detect if the current environment is running on kaggle.

    Returns:
        bool: True if the current environment is on kaggle, False otherwise.

    """
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive"


def running_on_colab() -> bool:
    """Detect if the current environment is running on Google colab.

    Returns:
        bool: True if the current environment is on colab, False otherwise.

    """
    return "COLAB_GPU" in os.environ


def colab_has_gpu() -> bool:
    """Detect if the current environment is running on Google colab with a GPU.

    Returns:
        bool: True if the current environment is on colab with a GPU, False otherwise.

    """
    return bool(int(os.environ.get("COLAB_GPU", 0)))


def has_network(
    hostname="one.one.one.one", timeout: Number = 3
) -> bool:  # noqa: CCR001
    """Detect if the current environment has a network connection.

    Create a socket connection to the hostname and check if the connection
    is successful.

    Args:
        hostname (str): The hostname to ping. Defaults to "one.one.one.one".
        timeout (Number): Timeout in seconds for the fallback GET request.

    Returns:
        bool: True if the current environment has a network connection,
            False otherwise.

    """
    try:
        # Check DNS listing
        host = socket.gethostbyname(hostname)
        # Connect to host
        connection = socket.create_connection((host, 80), timeout=timeout)
        connection.close()
        return True
    except (socket.gaierror, socket.timeout):
        return False


def pixman_version() -> Tuple[int, int]:  # noqa: CCR001
    """The version of pixman that is installed.

    Returns:
        tuple: The version of pixman that is installed as a tuple of ints.

    Raises:
        Exception: If pixman is not installed or the version
        could not be determined.

    """
    version = None
    using = None

    if in_conda_env():
        # Using anaconda to check for pixman
        using = "conda"
        try:
            conda_list = subprocess.Popen(("conda", "list"), stdout=subprocess.PIPE)
            conda_pixman = subprocess.check_output(
                ("grep", "pixman"), stdin=conda_list.stdout
            )
            conda_list.wait()
        except subprocess.SubprocessError:
            conda_pixman = b""
        matches = re.search(
            r"^pixman\s*(\d+.\d+)*",
            conda_pixman.decode("utf-8"),
            flags=re.MULTILINE,
        )
        if matches:
            version = tuple(int(part) for part in matches.group(1).split("."))
    if shutil.which("dpkg") and version is None:
        # Using dpkg to check for pixman
        using = "dpkg"
        try:
            dkpg_output = subprocess.check_output(
                ["/usr/bin/dpkg", "-s", "libpixman-1-0"]
            )
        except subprocess.SubprocessError:
            dkpg_output = b""
        matches = re.search(
            r"^Version: (\d+.\d+)*",
            dkpg_output.decode("utf-8"),
            flags=re.MULTILINE,
        )
        if matches:
            version = tuple(int(part) for part in matches.group(1).split("."))
    if shutil.which("brew") and version is None:
        # Using homebrew to check for pixman
        using = "brew"
        try:
            brew_list = subprocess.Popen(
                ("brew", "list", "--versions"), stdout=subprocess.PIPE
            )
            brew_pixman = subprocess.check_output(
                ("grep", "pixman"), stdin=brew_list.stdout
            )
            brew_list.wait()
        except subprocess.SubprocessError:
            brew_pixman = b""
        matches = re.search(
            r"(\d+.\d+)*",
            brew_pixman.decode("utf-8"),
            flags=re.MULTILINE,
        )
        if matches:
            version = tuple(int(part) for part in matches.group(1).split("."))
    if platform.system() == "Darwin" and shutil.which("port") and version is None:
        # Using macports to check for pixman
        # Also checks the platform is Darwin as macports is only available on
        # MacOS.
        using = "port"
        port_list = subprocess.Popen(("port", "installed"), stdout=subprocess.PIPE)
        port_pixman = subprocess.check_output(
            ("grep", "pixman"), stdin=port_list.stdout
        )
        port_list.wait()
        matches = re.search(
            r"(\d+.\d+)*",
            port_pixman.decode("utf-8"),
            flags=re.MULTILINE,
        )
        if matches:
            version = tuple(int(part) for part in matches.group(1).split("."))
    if version:
        return version, using
    raise EnvironmentError("Unable to detect pixman version.")


def pixman_warning() -> None:  # pragma: no cover
    """Detect if pixman version 0.38 is being used.

    If so, warn the user that the pixman version may cause problems.
    Suggest a fix if possible.

    """

    def _show_warning() -> None:
        """Show a warning message if pixman is version 0.38."""
        try:
            version, using = pixman_version()
        except EnvironmentError:
            # Unable to determine the pixman version
            return

        # If the pixman version is bad, suggest some fixes
        fix = ""
        if using == "conda":
            fix = (
                "You may be able do this with the command: "
                'conda install -c conda-forge pixman">=0.39"'
            )
        if using == "dpkg":
            fix = (
                "To fix this you may need to set up an anaconda environment "
                "with pixman >=0.39 or install pixman >=0.39 from source. "
                "See the tiatoolbox documentation for more information on "
                "setting up a conda environment. "
                "Instructions to compile from source can be found at the GitLab "
                "mirror here: "
                "https://gitlab.freedesktop.org/pixman/pixman/-/blob/master/INSTALL"
            )
        if using == "brew":
            fix = "You may be able do this with the command: brew upgrade pixman"
        if using == "port":
            fix = "You may be able do this with the command: port upgrade pixman"
        # Log the warning
        if version[:2] == (0, 38):
            logger.warning(
                "It looks like you are using Pixman version 0.38 (via %s). "
                "This version is known to cause issues with OpenSlide. "
                "Please consider upgrading to Pixman version 0.39 or later. "
                "%s",
                using,
                fix,
            )

    thread = threading.Thread(target=_show_warning, args=(), kwargs={})
    thread.start()
