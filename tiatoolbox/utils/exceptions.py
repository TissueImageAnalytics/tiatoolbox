"""Custom Errors and Exceptions for TIAToolbox."""


class FileNotSupported(Exception):
    """Raise No supported file found error.

    Args:
        message (str) : Display message for the error.

    """

    def __init__(self, message: str = "File format is not supported"):
        self.message = message
        super().__init__(self.message)


class MethodNotSupported(Exception):
    """Raise No supported file found error.

    Args:
        message (str) : Display message for the error.

    """

    def __init__(self, message: str = "Method is not supported"):
        self.message = message
        super().__init__(self.message)
