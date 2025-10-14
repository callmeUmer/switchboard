"""Provider registry for managing available providers."""

from typing import Dict, Type, Optional, List, Any
from .base import BaseProvider
from ..exceptions import ProviderNotFoundError, ProviderError


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

        # Get provider name from class
        if hasattr(provider_class, 'name') and isinstance(getattr(provider_class, 'name'), property):
            # Create temporary instance to get the name property value
            try:
                # Try creating with dummy API key first for providers that require it
                temp_instance = provider_class(api_key="dummy-key")
                provider_name = temp_instance.name
            except Exception:
                try:
                    # Try without API key
                    temp_instance = provider_class()
                    provider_name = temp_instance.name
                except Exception:
                    # Fallback to class name if instantiation fails
                    provider_name = provider_class.__name__.lower().replace('provider', '')
        elif hasattr(provider_class, 'name'):
            provider_name = provider_class.name
        else:
            provider_name = provider_class.__name__.lower().replace('provider', '')

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
                f"Provider '{provider_name}' not found. Available providers: {available}"
            )

        return self._providers[provider_name]

    def create_provider(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
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
            raise ProviderError(f"Failed to create provider '{provider_name}': {e}") from e

    def get_or_create_provider(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseProvider:
        """Get cached provider instance or create new one.

        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            **kwargs: Additional provider configuration

        Returns:
            Provider instance
        """
        # Create cache key from provider name, api_key, and config
        # Use frozenset for stable, hashable representation
        import hashlib

        # Include API key in cache key (hash it for security)
        api_key_hash = ""
        if api_key:
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        config_items = sorted(kwargs.items())
        config_str = str(config_items)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Cache key now includes API key hash to prevent sharing providers with different keys
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

        # Clear related cached instances (handles new cache key format with api_key_hash)
        keys_to_remove = [key for key in self._instances.keys() if key.split(':')[0] == provider_name]
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

        # Try to get supported models (might need instantiation)
        try:
            # For providers that require API key, use a dummy key for info gathering
            if hasattr(provider_class, 'requires_api_key'):
                try:
                    temp_requires_api_key = provider_class().requires_api_key()
                except Exception:
                    temp_requires_api_key = True
            else:
                temp_requires_api_key = True

            if temp_requires_api_key:
                temp_instance = provider_class(api_key="dummy-key")
            else:
                temp_instance = provider_class()

            models = temp_instance.supported_models
        except Exception:
            models = []
            temp_requires_api_key = True

        return {
            "name": provider_name,
            "class": provider_class.__name__,
            "supported_models": models,
            "requires_api_key": temp_requires_api_key,
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
    provider_name: str,
    api_key: Optional[str] = None,
    **kwargs
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