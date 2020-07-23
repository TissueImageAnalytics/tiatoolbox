"""Custom Errors and Exceptions for TIAToolbox"""


class FileNotSupported(Exception):
    """Raises No supported file found Error"""
    def __init__(self, message="No supported file found."):
        self.message = message
        super().__init__(self.message)
