"""Base provider interface for AI models."""

import ipaddress
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional
from urllib.parse import urlsplit

from ..exceptions import ConfigurationError, ProviderError
from ..utils import run_coroutine_sync


def _is_loopback_host(hostname: Optional[str]) -> bool:
    """Check whether a hostname refers to the local machine."""
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_base_url(url: str, allow_http: bool = False) -> str:
    """Validate a provider base URL before any credential is sent to it.

    Only https URLs are accepted, with two exceptions for plain http:
    loopback hosts (localhost / 127.x / ::1) and an explicit
    ``allow_http: true`` opt-in in the provider's extra_params.

    Args:
        url: The base URL to validate
        allow_http: Explicit opt-in for plain-http, non-loopback hosts

    Returns:
        The validated URL

    Raises:
        ConfigurationError: If the URL scheme/host is unacceptable
    """
    parts = urlsplit(url)

    if not parts.hostname:
        raise ConfigurationError(f"base_url has no host: {url!r}")

    if parts.scheme == "https":
        return url

    if parts.scheme == "http":
        if allow_http or _is_loopback_host(parts.hostname):
            return url
        raise ConfigurationError(
            f"Refusing plain-http base_url {url!r}: API keys would be sent "
            "unencrypted. Use https, or set 'allow_http: true' in "
            "extra_params to explicitly opt in."
        )

    raise ConfigurationError(
        f"base_url must use https (got {url!r}); plain http is allowed only "
        "for loopback hosts or with 'allow_http: true' in extra_params."
    )


@dataclass
class CompletionResponse:
    """Response from a model completion."""

    content: str
    model: str
    provider: str
    timestamp: datetime
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "usage": self.usage,
            "metadata": self.metadata,
        }


class BaseProvider(ABC):
    """Base class for all AI model providers.

    Subclasses must set the ``name`` class attribute and may extend
    ``ALLOWED_CONFIG_KEYS`` to accept provider-specific config keys.
    """

    #: Provider name identifier (must be set by subclasses)
    name: ClassVar[str]

    #: Config keys (from YAML extra_params) this provider accepts.
    #: Anything else is rejected so untrusted config cannot smuggle
    #: unexpected settings into the provider.
    ALLOWED_CONFIG_KEYS: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize provider.

        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration

        Raises:
            ConfigurationError: If kwargs contain keys outside
                ALLOWED_CONFIG_KEYS
        """
        unknown = set(kwargs) - self.ALLOWED_CONFIG_KEYS
        if unknown:
            provider_name = getattr(self, "name", self.__class__.__name__)
            raise ConfigurationError(
                f"Unsupported extra_params for {provider_name}: "
                f"{sorted(unknown)}; allowed: {sorted(self.ALLOWED_CONFIG_KEYS)}"
            )
        self.api_key = api_key
        self.config = kwargs
        self._validate_configuration()

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported model names."""
        pass

    def _validate_configuration(self) -> None:
        """Validate provider configuration.

        Raises:
            ProviderError: If configuration is invalid
        """
        if self.requires_api_key() and not self.api_key:
            provider_name = getattr(
                self,
                "name",
                self.__class__.__name__.replace("Provider", "").lower(),
            )
            raise ProviderError(f"{provider_name} provider requires an API key")

    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key.

        Returns:
            True if API key is required, False otherwise
        """
        return True

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate completion for the given prompt.

        Args:
            prompt: Input prompt
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            **kwargs: Additional model-specific parameters

        Returns:
            CompletionResponse with the generated content

        Raises:
            ProviderError: If completion fails
        """
        pass

    def complete_sync(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Synchronous completion wrapper.

        Args:
            prompt: Input prompt
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            **kwargs: Additional model-specific parameters

        Returns:
            CompletionResponse with the generated content
        """
        return run_coroutine_sync(
            lambda: self.complete(
                prompt, model, max_tokens, temperature, timeout, **kwargs
            ),
            timeout=timeout or 60,
        )

    def is_model_supported(self, model: str) -> bool:
        """Check if model is supported by this provider.

        Args:
            model: Model name to check

        Returns:
            True if model is supported, False otherwise
        """
        return model in self.supported_models

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name

        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.name,
            "model": model,
            "supported": self.is_model_supported(model),
        }

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Simple test completion
            response = await self.complete(
                prompt="Hello",
                model=self.supported_models[0] if self.supported_models else "",
                max_tokens=1,
                timeout=10,
            )
            return bool(response.content)
        except Exception:
            return False

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.name}Provider"

    def __repr__(self) -> str:
        """Detailed string representation of the provider."""
        return f"{self.name}Provider(models={len(self.supported_models)})"
