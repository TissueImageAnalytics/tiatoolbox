"""Custom Errors and Exceptions for TIAToolbox."""


class FileNotSupportedError(Exception):
    """Raise No supported file found error."""

    def __init__(self, message="File format is not supported"):
        self.message = message
        super().__init__(self.message)


class MethodNotSupportedError(Exception):
    """Raise No supported file found error."""

    def __init__(self, message="Method is not supported"):
        self.message = message
        super().__init__(self.message)
