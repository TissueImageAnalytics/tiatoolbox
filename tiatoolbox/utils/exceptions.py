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
