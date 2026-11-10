"""Exception hierarchy for the LLZ codec."""


class LLZError(Exception):
    """Base exception for LLZ failures."""


class ConfigurationError(LLZError, ValueError):
    """Raised when a codec configuration is invalid."""


class ContainerFormatError(LLZError, ValueError):
    """Raised when an LLZ container is malformed or unsupported."""


class IntegrityError(LLZError):
    """Raised when decoded bytes fail the container integrity check."""
