"""Custom Errors and Exceptions for TIAToolbox."""


class FileNotSupportedError(Exception):
    """Raise No supported file found error.

    Args:
        message (str) : Display message for the error.

    """

    def __init__(self, message: str = "File format is not supported") -> None:
        """Initialize :class:`FileNotSupportedError`."""
        self.message = message
        super().__init__(self.message)


class MethodNotSupportedError(Exception):
    """Raise No supported file found error.

    Args:
        message (str) : Display message for the error.

    """

    def __init__(self, message: str = "Method is not supported") -> None:
        """Initialize :class:`MethodNotSupportedError`."""
        self.message = message
        super().__init__(self.message)
