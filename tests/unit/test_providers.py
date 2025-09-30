"""Unit tests for provider system."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from switchboard.providers.base import BaseProvider, CompletionResponse
from switchboard.providers.registry import ProviderRegistry, register_provider, get_provider
from switchboard.exceptions import ProviderError, ProviderNotFoundError
from ..conftest import MockProvider


class TestCompletionResponse:
    """Test CompletionResponse functionality."""

    def test_completion_response_creation(self):
        """Test creating CompletionResponse."""
        timestamp = datetime.now()
        response = CompletionResponse(
            content="Test response",
            model="test-model",
            provider="test-provider",
            timestamp=timestamp,
            usage={"tokens": 10},
            metadata={"id": "test-id"}
        )

        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        assert response.timestamp == timestamp
        assert response.usage == {"tokens": 10}
        assert response.metadata == {"id": "test-id"}

    def test_completion_response_to_dict(self):
        """Test converting CompletionResponse to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        response = CompletionResponse(
            content="Test response",
            model="test-model",
            provider="test-provider",
            timestamp=timestamp
        )

        result = response.to_dict()

        expected = {
            "content": "Test response",
            "model": "test-model",
            "provider": "test-provider",
            "timestamp": "2024-01-01T12:00:00",
            "usage": None,
            "metadata": None
        }

        assert result == expected


class TestBaseProvider:
    """Test BaseProvider abstract class."""

    class ConcreteProvider(BaseProvider):
        """Concrete implementation for testing."""

        @property
        def name(self):
            return "test"

        @property
        def supported_models(self):
            return ["test-model-1", "test-model-2"]

        async def complete(self, prompt, model, max_tokens=None, temperature=None, timeout=None, **kwargs):
            return CompletionResponse(
                content=f"Response to: {prompt}",
                model=model,
                provider=self.name,
                timestamp=datetime.now()
            )

    def test_base_provider_creation(self):
        """Test creating BaseProvider instance."""
        provider = self.ConcreteProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.config == {}
        assert provider.name == "test"

    def test_base_provider_with_config(self):
        """Test creating BaseProvider with additional config."""
        provider = self.ConcreteProvider(
            api_key="test-key",
            custom_param="value",
            base_url="https://api.example.com"
        )

        assert provider.api_key == "test-key"
        assert provider.config["custom_param"] == "value"
        assert provider.config["base_url"] == "https://api.example.com"

    def test_base_provider_requires_api_key(self):
        """Test provider that requires API key."""
        with pytest.raises(ProviderError, match="test provider requires an API key"):
            self.ConcreteProvider()

    def test_base_provider_no_api_key_required(self):
        """Test provider that doesn't require API key."""
        class NoAPIKeyProvider(self.ConcreteProvider):
            def requires_api_key(self):
                return False

        provider = NoAPIKeyProvider()
        assert provider.api_key is None

    def test_is_model_supported(self):
        """Test checking if model is supported."""
        provider = self.ConcreteProvider(api_key="test")

        assert provider.is_model_supported("test-model-1") is True
        assert provider.is_model_supported("test-model-2") is True
        assert provider.is_model_supported("unsupported-model") is False

    def test_get_model_info(self):
        """Test getting model information."""
        provider = self.ConcreteProvider(api_key="test")
        info = provider.get_model_info("test-model-1")

        expected = {
            "provider": "test",
            "model": "test-model-1",
            "supported": True
        }

        assert info == expected

    def test_complete_sync(self):
        """Test synchronous completion wrapper."""
        provider = self.ConcreteProvider(api_key="test")
        response = provider.complete_sync("Test prompt", "test-model-1")

        assert isinstance(response, CompletionResponse)
        assert response.content == "Response to: Test prompt"
        assert response.model == "test-model-1"
        assert response.provider == "test"

    @pytest.mark.asyncio
    async def test_complete_async(self):
        """Test asynchronous completion."""
        provider = self.ConcreteProvider(api_key="test")
        response = await provider.complete("Test prompt", "test-model-1")

        assert isinstance(response, CompletionResponse)
        assert response.content == "Response to: Test prompt"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        provider = self.ConcreteProvider(api_key="test")
        health = await provider.health_check()

        assert health is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        class FailingProvider(self.ConcreteProvider):
            async def complete(self, prompt, model, **kwargs):
                raise Exception("Provider failure")

        provider = FailingProvider(api_key="test")
        health = await provider.health_check()

        assert health is False

    def test_str_representation(self):
        """Test string representation of provider."""
        provider = self.ConcreteProvider(api_key="test")

        assert str(provider) == "testProvider"
        assert repr(provider) == "testProvider(models=2)"


class TestProviderRegistry:
    """Test ProviderRegistry functionality."""

    def test_registry_creation(self):
        """Test creating empty registry."""
        registry = ProviderRegistry()

        assert len(registry._providers) == 0
        assert len(registry._instances) == 0

    def test_register_provider(self, clean_registry):
        """Test registering a provider."""
        class TestProvider(BaseProvider):
            @property
            def name(self):
                return "test"

            @property
            def supported_models(self):
                return ["test-model"]

            async def complete(self, prompt, model, max_tokens=None, temperature=None, timeout=None, **kwargs):
                pass

        clean_registry.register(TestProvider)

        assert "test" in clean_registry._providers
        assert clean_registry._providers["test"] == TestProvider

    def test_register_invalid_provider(self, clean_registry):
        """Test registering invalid provider."""
        class NotAProvider:
            pass

        with pytest.raises(ProviderError, match="Provider class must inherit from BaseProvider"):
            clean_registry.register(NotAProvider)

    def test_register_duplicate_provider(self, clean_registry):
        """Test registering duplicate provider."""
        class TestProvider(BaseProvider):
            @property
            def name(self):
                return "test"

            @property
            def supported_models(self):
                return ["test-model"]

            async def complete(self, prompt, model, max_tokens=None, temperature=None, timeout=None, **kwargs):
                pass

        clean_registry.register(TestProvider)

        with pytest.raises(ProviderError, match="Provider 'test' is already registered"):
            clean_registry.register(TestProvider)

    def test_get_provider_class(self, clean_registry):
        """Test getting provider class."""
        class TestProvider(BaseProvider):
            @property
            def name(self):
                return "test"

            @property
            def supported_models(self):
                return ["test-model"]

            async def complete(self, prompt, model, max_tokens=None, temperature=None, timeout=None, **kwargs):
                pass

        clean_registry.register(TestProvider)
        provider_class = clean_registry.get_provider_class("test")

        assert provider_class == TestProvider

    def test_get_nonexistent_provider_class(self, clean_registry):
        """Test getting nonexistent provider class."""
        with pytest.raises(ProviderNotFoundError, match="Provider 'nonexistent' not found"):
            clean_registry.get_provider_class("nonexistent")

    def test_create_provider(self, clean_registry, mock_provider):
        """Test creating provider instance."""
        clean_registry.register(MockProvider)
        provider = clean_registry.create_provider("test", api_key="test-key")

        assert isinstance(provider, MockProvider)
        assert provider.api_key == "test-key"

    def test_create_provider_with_config(self, clean_registry):
        """Test creating provider with additional config."""
        clean_registry.register(MockProvider)
        provider = clean_registry.create_provider(
            "test",
            api_key="test-key",
            custom_param="value"
        )

        assert provider.config["custom_param"] == "value"

    def test_get_or_create_provider_caching(self, clean_registry):
        """Test provider instance caching."""
        clean_registry.register(MockProvider)

        provider1 = clean_registry.get_or_create_provider("test", api_key="test-key")
        provider2 = clean_registry.get_or_create_provider("test", api_key="test-key")

        assert provider1 is provider2

    def test_get_or_create_provider_different_config(self, clean_registry):
        """Test creating different providers with different config."""
        clean_registry.register(MockProvider)

        provider1 = clean_registry.get_or_create_provider("test", api_key="key1")
        provider2 = clean_registry.get_or_create_provider("test", api_key="key2")

        assert provider1 is not provider2

    def test_list_providers(self, clean_registry):
        """Test listing registered providers."""
        clean_registry.register(MockProvider)

        providers = clean_registry.list_providers()
        assert providers == ["test"]

    def test_is_provider_registered(self, clean_registry):
        """Test checking if provider is registered."""
        assert clean_registry.is_provider_registered("test") is False

        clean_registry.register(MockProvider)
        assert clean_registry.is_provider_registered("test") is True

    def test_clear_cache(self, clean_registry):
        """Test clearing provider cache."""
        clean_registry.register(MockProvider)
        clean_registry.get_or_create_provider("test", api_key="test-key")

        assert len(clean_registry._instances) > 0

        clean_registry.clear_cache()
        assert len(clean_registry._instances) == 0

    def test_unregister_provider(self, clean_registry):
        """Test unregistering provider."""
        clean_registry.register(MockProvider)
        clean_registry.get_or_create_provider("test", api_key="test-key")

        assert "test" in clean_registry._providers
        assert len(clean_registry._instances) > 0

        clean_registry.unregister("test")

        assert "test" not in clean_registry._providers
        assert len(clean_registry._instances) == 0

    def test_unregister_nonexistent_provider(self, clean_registry):
        """Test unregistering nonexistent provider."""
        with pytest.raises(ProviderNotFoundError, match="Provider 'nonexistent' not found"):
            clean_registry.unregister("nonexistent")

    def test_get_provider_info(self, clean_registry):
        """Test getting provider information."""
        clean_registry.register(MockProvider)
        info = clean_registry.get_provider_info("test")

        assert info["name"] == "test"
        assert info["class"] == "MockProvider"
        assert "test-model-1" in info["supported_models"]


class TestGlobalRegistryFunctions:
    """Test global registry functions."""

    def test_register_provider_global(self, clean_registry):
        """Test global provider registration."""
        register_provider(MockProvider)

        assert clean_registry.is_provider_registered("test")

    def test_get_provider_global(self, clean_registry):
        """Test global get_provider function."""
        register_provider(MockProvider)
        provider = get_provider("test", api_key="test-key")

        assert isinstance(provider, MockProvider)
        assert provider.api_key == "test-key"