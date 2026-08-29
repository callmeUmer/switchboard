"""Provider registry for managing available providers."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Type

from ..exceptions import ProviderError, ProviderNotFoundError
from .base import BaseProvider


class ProviderRegistry:
    """Registry for managing AI model providers."""

    def __init__(self):
        """Initialize empty provider registry."""
        self._providers: Dict[str, Type[BaseProvider]] = {}
        self._instances: Dict[str, BaseProvider] = {}

    def register(self, provider_class: Type[BaseProvider]) -> None:
        """Register a provider class.

        Args:
            provider_class: Provider class to register

        Raises:
            ProviderError: If provider name already exists
        """
        if not issubclass(provider_class, BaseProvider):
            raise ProviderError(
                f"Provider class must inherit from BaseProvider, got {provider_class}"
            )

        # Get provider name from the class attribute; never instantiate the
        # class here (doing so at import time triggered SDK side effects and
        # bogus API-key-format warnings)
        provider_name = getattr(provider_class, "name", None)
        if not isinstance(provider_name, str):
            provider_name = provider_class.__name__.lower().replace("provider", "")

        if provider_name in self._providers:
            raise ProviderError(f"Provider '{provider_name}' is already registered")

        self._providers[provider_name] = provider_class

    def get_provider_class(self, provider_name: str) -> Type[BaseProvider]:
        """Get provider class by name.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider is not registered
        """
        if provider_name not in self._providers:
            available = list(self._providers.keys())
            raise ProviderNotFoundError(
                f"Provider '{provider_name}' not found. "
                f"Available providers: {available}"
            )

        return self._providers[provider_name]

    def create_provider(
        self, provider_name: str, api_key: Optional[str] = None, **kwargs
    ) -> BaseProvider:
        """Create and configure a provider instance.

        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            **kwargs: Additional provider configuration

        Returns:
            Configured provider instance

        Raises:
            ProviderNotFoundError: If provider is not registered
            ProviderError: If provider configuration is invalid
        """
        provider_class = self.get_provider_class(provider_name)

        try:
            return provider_class(api_key=api_key, **kwargs)
        except Exception as e:
            raise ProviderError(
                f"Failed to create provider '{provider_name}': {e}"
            ) from e

    def get_or_create_provider(
        self, provider_name: str, api_key: Optional[str] = None, **kwargs
    ) -> BaseProvider:
        """Get cached provider instance or create new one.

        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            **kwargs: Additional provider configuration

        Returns:
            Provider instance
        """
        # Create cache key from provider name, api_key, and config.
        # Full SHA-256 digests: truncated hashes risk collisions that would
        # hand one caller another caller's cached provider (and API key).
        # The cache is unbounded but keyed by (provider, key, config) tuples
        # that come from a finite YAML config, so growth is bounded in practice.
        api_key_hash = ""
        if api_key:
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        config_str = json.dumps(kwargs, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        # Cache key includes the API key hash so providers with different
        # keys are never shared
        cache_key = f"{provider_name}:{api_key_hash}:{config_hash}"

        if cache_key not in self._instances:
            self._instances[cache_key] = self.create_provider(
                provider_name, api_key, **kwargs
            )

        return self._instances[cache_key]

    def list_providers(self) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def is_provider_registered(self, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: Provider name to check

        Returns:
            True if provider is registered, False otherwise
        """
        return provider_name in self._providers

    def clear_cache(self) -> None:
        """Clear cached provider instances."""
        self._instances.clear()

    def unregister(self, provider_name: str) -> None:
        """Unregister a provider.

        Args:
            provider_name: Name of provider to unregister

        Raises:
            ProviderNotFoundError: If provider is not registered
        """
        if provider_name not in self._providers:
            raise ProviderNotFoundError(f"Provider '{provider_name}' not found")

        del self._providers[provider_name]

        # Clear related cached instances
        keys_to_remove = [
            key for key in self._instances.keys() if key.split(":")[0] == provider_name
        ]
        for key in keys_to_remove:
            del self._instances[key]

    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a registered provider.

        Args:
            provider_name: Provider name

        Returns:
            Dictionary with provider information
        """
        provider_class = self.get_provider_class(provider_name)

        # Supported models generally require a configured instance (live API
        # fetch); report an empty list rather than instantiating with a fake
        # API key just to gather info.
        requires_api_key = True
        models: List[str] = []
        try:
            temp_instance = provider_class()
            requires_api_key = temp_instance.requires_api_key()
            models = temp_instance.supported_models
        except Exception:
            pass

        return {
            "name": provider_name,
            "class": provider_class.__name__,
            "supported_models": models,
            "requires_api_key": requires_api_key,
        }


# Global registry instance
_registry = ProviderRegistry()


def register_provider(provider_class: Type[BaseProvider]) -> None:
    """Register a provider class globally.

    Args:
        provider_class: Provider class to register
    """
    _registry.register(provider_class)


def get_provider(
    provider_name: str, api_key: Optional[str] = None, **kwargs
) -> BaseProvider:
    """Get a provider instance from the global registry.

    Args:
        provider_name: Name of the provider
        api_key: API key for the provider
        **kwargs: Additional provider configuration

    Returns:
        Provider instance
    """
    return _registry.get_or_create_provider(provider_name, api_key, **kwargs)


def list_providers() -> List[str]:
    """Get list of registered provider names."""
    return _registry.list_providers()


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return _registry
