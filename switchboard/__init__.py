"""Switchboard: Config-driven AI model switching made simple."""

from .__version__ import __author__, __description__, __email__, __version__
from .client import Client
from .exceptions import (
    APIKeyError,
    ConfigurationError,
    FallbackExhaustedError,
    ModelNotFoundError,
    ModelResponseError,
    ProviderError,
    ProviderNotFoundError,
    SwitchboardError,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "Client",
    "SwitchboardError",
    "ConfigurationError",
    "ModelNotFoundError",
    "ProviderError",
    "ProviderNotFoundError",
    "APIKeyError",
    "ModelResponseError",
    "FallbackExhaustedError",
]
