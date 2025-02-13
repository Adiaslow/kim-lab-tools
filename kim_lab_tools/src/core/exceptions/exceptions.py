"""Custom exceptions for the Kim Lab Tools package."""


class KimLabToolsError(Exception):
    """Base exception for all Kim Lab Tools errors."""

    pass


class ValidationError(KimLabToolsError):
    """Raised when data validation fails."""

    pass


class ProcessingError(KimLabToolsError):
    """Raised when data processing fails."""

    pass


class LoadingError(KimLabToolsError):
    """Raised when data loading fails."""

    pass


class ReconstructionError(KimLabToolsError):
    """Raised when reconstruction fails."""

    pass
