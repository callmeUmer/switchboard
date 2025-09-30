"""Switchboard: Config-driven AI model switching made simple."""

from .__version__ import __version__, __author__, __email__, __description__
from .client import Client
from .exceptions import (
    SwitchboardError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotFoundError,
    APIKeyError,
    ModelResponseError,
    FallbackExhaustedError,
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