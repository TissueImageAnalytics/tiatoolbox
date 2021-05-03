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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Top-level package for TIA Toolbox."""

import os
from tiatoolbox import tiatoolbox
from tiatoolbox import wsicore
from tiatoolbox import utils
from tiatoolbox import tools

__author__ = """TIA Lab"""
__email__ = "tialab@dcs.warwick.ac.uk"
__version__ = "0.5.2"

# will set the tiatoolbox external data
# default to be the user home folder, should work on both Window and Unix/Linux
# C:\Users\USER\.tiatoolbox
# /home/USER/.tiatoolbox
TIATOOLBOX_HOME = os.path.join(os.path.expanduser("~"), ".tiatoolbox")


def set_home_dir(dir_path):
    """Set the home directory for saving data."""
    global TIATOOLBOX_HOME
    TIATOOLBOX_HOME = dir_path


if __name__ == "__main__":
    pass
