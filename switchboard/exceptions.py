"""Custom exceptions for Switchboard."""


class SwitchboardError(Exception):
    """Base exception for all Switchboard errors."""

    pass


class ConfigurationError(SwitchboardError):
    """Raised when there's an error in configuration."""

    pass


class ModelNotFoundError(SwitchboardError):
    """Raised when a specified model is not found in configuration."""

    pass


class ProviderError(SwitchboardError):
    """Raised when there's an error with a model provider."""

    pass


class ProviderNotFoundError(SwitchboardError):
    """Raised when a specified provider is not supported."""

    pass


class APIKeyError(SwitchboardError):
    """Raised when API key is missing or invalid."""

    pass


class ModelResponseError(SwitchboardError):
    """Raised when model returns an invalid response."""

    pass


class FallbackExhaustedError(SwitchboardError):
    """Raised when all fallback models have failed."""

    pass
