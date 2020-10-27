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

"""Custom Errors and Exceptions for TIAToolbox."""


class FileNotSupported(Exception):
    """Raise No supported file found error."""

    def __init__(self, message="File format is not supported"):
        self.message = message
        super().__init__(self.message)


class MethodNotSupported(Exception):
    """Raise No supported file found error."""

    def __init__(self, message="Method is not supported"):
        self.message = message
        super().__init__(self.message)
