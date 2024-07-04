"""Detection methods for the current environment.

This module contains methods for detecting aspects of the current
environment.

Some things which this module can detect are:
    - Whether the current environment is interactive.
    - Whether the current environment is a conda environment.
    - Whether the current environment is running on Travis, Kaggle, or
      Colab.

Note that these detections may not be correct 100% of the time but are
as accurate as can be reasonably be expected depending on what is being
detected.

"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading

import torch

from tiatoolbox import logger


def has_gpu() -> bool:
    """Detect if the runtime has GPU.

    This function calls torch function underneath. To mask an
    environment to have no GPU, you can set "CUDA_VISIBLE_DEVICES"
    environment variable to empty before running the python script.

    Returns:
        bool:
            True if the current runtime environment has GPU, False
            otherwise.

    """
    return torch.cuda.is_available()


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
        bool:
            True if the current environment is interactive, False
            otherwise.

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
        bool:
            True if the current environment is a Jupyter notebook, False
            otherwise.

    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
    except (NameError, ImportError):
        return False  # Probably standard Python interpreter
    else:
        return False  # Other type (?)


def in_conda_env() -> bool:
    """Detect if the current environment is a conda environment.

    Returns:
        bool:
            True if the current environment is a conda environment,
            False otherwise.

    """
    return "CONDA_DEFAULT_ENV" in os.environ and "CONDA_PREFIX" in os.environ


def running_on_travis() -> bool:
    """Detect if the current environment is running on travis.

    Returns:
        bool:
            True if the current environment is on travis, False
            otherwise.

    """
    return os.environ.get("TRAVIS") == "true" and os.environ.get("CI") == "true"


def running_on_github() -> bool:
    """Detect if the current environment is running on GitHub Actions.

    Returns:
        bool:
            True if the current environment is on GitHub, False
            otherwise.

    """
    return os.environ.get("GITHUB_ACTIONS") == "true"


def running_on_circleci() -> bool:
    """Detect if the current environment is running on CircleCI.

    Returns:
        bool:
            True if the current environment is on CircleCI, False
            otherwise.

    """
    return os.environ.get("CIRCLECI") == "true"


def running_on_ci() -> bool:
    """Detect if the current environment is running on continuous integration (CI).

    Returns:
        bool:
            True if the current environment is on CI, False
            otherwise.

    """
    return any(
        (
            os.environ.get("CI", "").lower() == "true",
            running_on_travis(),
            running_on_github(),
            running_on_circleci(),
        ),
    )


def running_on_kaggle() -> bool:
    """Detect if the current environment is running on Kaggle.

    Returns:
        bool:
            True if the current environment is on Kaggle, False
            otherwise.

    """
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive"


def running_on_colab() -> bool:
    """Detect if the current environment is running on Google Colab.

    Returns:
        bool:
            True if the current environment is on Colab, False
            otherwise.

    """
    return "COLAB_GPU" in os.environ


def colab_has_gpu() -> bool:
    """Detect if the current environment is running on Google Colab with a GPU.

    Returns:
        bool:
            True if the current environment is on colab with a GPU,
            False otherwise.

    """
    return bool(int(os.environ.get("COLAB_GPU", 0)))


def has_network(
    hostname: str = "one.one.one.one",
    timeout: float = 3,
) -> bool:
    """Detect if the current environment has a network connection.

    Create a socket connection to the hostname and check if the connection
    is successful.

    Args:
        hostname (str):
            The hostname to ping. Defaults to "one.one.one.one".
        timeout (float):
            Timeout in seconds for the fallback GET request.

    Returns:
        bool:
            True if the current environment has a network connection,
            False otherwise.

    """
    try:
        # Check DNS listing
        host = socket.gethostbyname(hostname)
        # Connect to host
        connection = socket.create_connection((host, 80), timeout=timeout)
        connection.close()
    except (socket.gaierror, socket.timeout):
        return False
    else:
        return True


def check_pixman_using_anaconda(versions: list) -> tuple[list, str]:
    """Using anaconda to check for pixman."""
    using = "conda"
    try:
        conda_list = subprocess.Popen(  # noqa: S603
            ("conda", "list"),
            stdout=subprocess.PIPE,
        )
        conda_pixman = subprocess.check_output(  # noqa: S603
            ("grep", "pixman"),
            stdin=conda_list.stdout,
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
        versions = [version_to_tuple(matches.group(1))]

    return versions, using


def check_pixman_using_dpkg(versions: list) -> tuple[list, str]:
    """Using dpkg to check for pixman."""
    using = "dpkg"
    try:
        dkpg_output = subprocess.check_output(  # noqa: S603
            ["/usr/bin/dpkg", "-s", "libpixman-1-0"],
        )
    except subprocess.SubprocessError:
        dkpg_output = b""
    matches = re.search(
        r"^Version: ((?:\d+[._]+)+\d*)",
        dkpg_output.decode("utf-8"),
        flags=re.MULTILINE,
    )
    if matches:
        versions = [version_to_tuple(matches.group(1))]

    return versions, using


def check_pixman_using_brew(versions: list) -> tuple[list, str]:
    """Using homebrew to check for pixman."""
    using = "brew"
    try:
        brew_list = subprocess.Popen(  # noqa: S603
            ("brew", "list", "--versions"),
            stdout=subprocess.PIPE,
        )
        brew_pixman = subprocess.check_output(  # noqa: S603
            ("grep", "pixman"),
            stdin=brew_list.stdout,
        )
        brew_list.wait()
    except subprocess.SubprocessError:
        brew_pixman = b""
    matches = re.findall(
        r"((?:\d+[._]+)+\d*)",
        brew_pixman.decode("utf-8"),
        flags=re.MULTILINE,
    )
    if matches:
        versions = [version_to_tuple(match) for match in matches]

    return versions, using


def check_pixman_using_macports(versions: list) -> tuple[list, str]:
    """Using macports to check for pixman.

    Also checks the platform is Darwin,
    as macports is only available on macOS.

    """
    using = "port"
    port_list = subprocess.Popen(  # noqa: S603
        ("port", "installed"),
        stdout=subprocess.PIPE,
    )
    port_pixman = subprocess.check_output(  # noqa: S603
        ("grep", "pixman"),
        stdin=port_list.stdout,
    )
    port_list.wait()
    matches = re.findall(
        r"((?:\d+[._]+)+\d*)",
        port_pixman.decode("utf-8"),
        flags=re.MULTILINE,
    )
    if matches:
        versions = [version_to_tuple(match) for match in matches]

    return versions, using


def pixman_versions() -> tuple[list, str | None]:
    """The version(s) of pixman that are installed.

    Some package managers (brew) may report multiple versions of pixman
    installed as part of a dependency tree.

    Returns:
        list of tuple of int:
            The versions of pixman that are installed as tuples of ints.

    Raises:
        Exception:
            If pixman is not installed or the version could not be
            determined.

    """
    versions: list[int] = []
    using = None

    if in_conda_env():
        versions, using = check_pixman_using_anaconda(versions)
    if shutil.which("dpkg") and not versions:
        versions, using = check_pixman_using_dpkg(versions)
    if shutil.which("brew") and not versions:
        versions, using = check_pixman_using_brew(versions)
    if platform.system() == "Darwin" and shutil.which("port") and not versions:
        versions, using = check_pixman_using_macports(versions)
    if versions:
        return versions, using
    msg = "Unable to detect pixman version(s)."
    raise OSError(msg)


def version_to_tuple(match: str) -> tuple[int, ...]:
    """Convert a version string to a tuple of ints.

    Only supports versions containing integers and periods.

    Args:
        match (str): The version string to convert.

    Returns:
        tuple:
            The version string as a tuple of ints.

    """
    # Check that the string only contains integers and periods
    if not re.match(r"^\d+([._]\d+)*$", match):
        msg = f"{match} is not a valid version string."
        raise ValueError(msg)
    return tuple(int(part) for part in match.split("."))


def pixman_warning() -> None:  # pragma: no cover
    """Detect if pixman version 0.38 is being used.

    If so, warn the user that the pixman version may cause problems.
    Suggest a fix if possible.

    """

    def _show_warning() -> None:
        """Show a warning message if pixman is version 0.38."""
        try:
            versions, using = pixman_versions()
        except OSError:
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
                "To fix this you may need to do one of the following:\n"
                "  1. Install libpixman-1-dev from your package manager (e.g. apt).\n"
                "  2. Set up an anaconda environment with pixman >=0.39\n"
                "  3. Install pixman >=0.39 from source. "
                "Instructions to compile from source can be found at the GitLab "
                "mirror here: "
                "https://gitlab.freedesktop.org/pixman/pixman/-/blob/master/INSTALL"
            )
        if using == "brew":
            fix = "You may be able do this with the command: brew upgrade pixman"
        if using == "port":
            fix = "You may be able do this with the command: port upgrade pixman"
        # Log a warning if there is a pixman version in the range [0.38, 0.39)
        if any((0, 38) <= v < (0, 39) for v in versions):
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
