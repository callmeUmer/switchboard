"""Unit tests for Switchboard client."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from switchboard.client import Client
from switchboard.exceptions import (
    APIKeyError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderNotFoundError,
    SwitchboardError,
)
from switchboard.providers.base import CompletionResponse


class TestClient:
    """Test Client functionality."""

    def test_client_creation(self, config_file):
        """Test creating Client instance."""
        client = Client(config_file)

        assert client.config_manager.config_path == config_file
        assert client._config is None

    def test_client_creation_no_config(self):
        """Test creating Client without config path."""
        with patch("switchboard.client.get_config_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            client = Client()

            mock_get_manager.assert_called_once_with(None)
            assert client.config_manager == mock_manager

    def test_ensure_config_loaded(self, config_file, sample_config_data):
        """Test ensuring config is loaded."""
        client = Client(config_file)

        # Config should be None initially
        assert client._config is None

        # After calling _ensure_config_loaded, config should be loaded
        client._ensure_config_loaded()
        assert client._config is not None
        assert client._config.default_model == "test-model-1"

    def test_resolve_model_explicit(self, config_file):
        """Test resolving model with explicit model specified."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model = client._resolve_model("explicit-model", None)
        assert model == "explicit-model"

    def test_resolve_model_task_based(self, config_file):
        """Test resolving model with task-based routing."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model = client._resolve_model(None, "test-task")
        assert model == "test-model-1"  # Primary model for test-task

    def test_resolve_model_default(self, config_file):
        """Test resolving model with default fallback."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model = client._resolve_model(None, None)
        assert model == "test-model-1"  # Default model

    def test_resolve_model_nonexistent_task(self, config_file):
        """Test resolving model with nonexistent task."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model = client._resolve_model(None, "nonexistent-task")
        assert model == "test-model-1"  # Falls back to default

    def test_get_provider_success(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test successfully getting provider."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model_config = client.config_manager.get_model_config("test-model-1")
        provider = client._get_provider(model_config)

        assert provider is not None
        assert provider.name == "test"

    def test_get_provider_missing_api_key(self, config_file, registered_mock_provider):
        """Test getting provider with missing API key."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model_config = client.config_manager.get_model_config("test-model-1")

        with pytest.raises(APIKeyError):
            client._get_provider(model_config)

    def test_get_provider_not_found(self, config_file, mock_env_vars):
        """Test getting nonexistent provider."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model_config = client.config_manager.get_model_config("test-model-1")

        with pytest.raises(ProviderNotFoundError):
            client._get_provider(model_config)

    def test_prepare_completion_params(self, config_file):
        """Test preparing completion parameters."""
        client = Client(config_file)
        client._ensure_config_loaded()

        model_config = client.config_manager.get_model_config("test-model-1")
        kwargs = {"temperature": 0.9, "custom_param": "value"}

        params = client._prepare_completion_params(model_config, kwargs)

        expected = {
            "max_tokens": 100,
            "temperature": 0.9,  # kwargs override config
            "timeout": 30,
            "custom_param": "value",
        }

        assert params == expected

    def test_complete_success(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test successful completion."""
        client = Client(config_file)

        response = client.complete("Test prompt", model="test-model-1")

        assert isinstance(response, CompletionResponse)
        assert response.content == "Mock response"
        assert response.provider == "test"

    def test_complete_with_task(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test completion with task routing."""
        client = Client(config_file)

        response = client.complete("Test prompt", task="test-task")

        assert isinstance(response, CompletionResponse)
        # Should use test-task's primary model (test-model-1)

    def test_complete_with_kwargs(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test completion with additional parameters."""
        client = Client(config_file)

        response = client.complete(
            "Test prompt", model="test-model-1", temperature=0.5, max_tokens=50
        )

        assert isinstance(response, CompletionResponse)

        # Check that provider received the parameters
        provider = registered_mock_provider.get_or_create_provider(
            "test", api_key="test-api-key-12345"
        )
        assert provider.last_params["temperature"] == 0.5
        assert provider.last_params["max_tokens"] == 50

    def test_complete_provider_failure(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test completion with provider failure."""
        client = Client(config_file)

        # Make provider fail
        provider = registered_mock_provider.get_or_create_provider(
            "test", api_key="test-api-key-12345"
        )
        provider.should_fail = True

        with pytest.raises(SwitchboardError, match="Completion failed"):
            client.complete("Test prompt", model="test-model-1")

    @pytest.mark.asyncio
    async def test_complete_async_success(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test successful async completion."""
        client = Client(config_file)

        response = await client.complete_async("Test prompt", model="test-model-1")

        assert isinstance(response, CompletionResponse)
        assert response.content == "Mock response"

    @pytest.mark.asyncio
    async def test_complete_async_failure(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test async completion with failure."""
        client = Client(config_file)

        # Make provider fail
        provider = registered_mock_provider.get_or_create_provider(
            "test", api_key="test-api-key-12345"
        )
        provider.should_fail = True

        with pytest.raises(SwitchboardError, match="Async completion failed"):
            await client.complete_async("Test prompt", model="test-model-1")

    def test_list_models(self, config_file):
        """Test listing available models."""
        client = Client(config_file)

        models = client.list_models()

        expected = ["test-model-1", "test-model-2", "openai-model"]
        assert set(models) == set(expected)

    def test_list_tasks(self, config_file):
        """Test listing available tasks."""
        client = Client(config_file)

        tasks = client.list_tasks()

        expected = ["test-task", "coding"]
        assert set(tasks) == set(expected)

    def test_get_model_info(self, config_file, mock_env_vars, registered_mock_provider):
        """Test getting model information."""
        client = Client(config_file)

        info = client.get_model_info("test-model-1")

        assert info["provider"] == "test"
        assert info["model"] == "test-model-1"
        assert info["supported"] is True

    def test_get_model_info_nonexistent(self, config_file):
        """Test getting info for nonexistent model."""
        client = Client(config_file)

        with pytest.raises(ConfigurationError):
            client.get_model_info("nonexistent-model")

    def test_reload_config(self, config_file):
        """Test reloading configuration."""
        client = Client(config_file)

        # Load initial config
        client._ensure_config_loaded()
        original_config = client._config

        # Reload config
        client.reload_config()

        assert client._config is not original_config
        assert client._config.default_model == original_config.default_model

    def test_health_check_single_model(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test health check for single model."""
        client = Client(config_file)

        results = client.health_check("test-model-1")

        assert "test-model-1" in results
        assert results["test-model-1"] is True

    def test_health_check_all_models(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test health check for all models."""
        client = Client(config_file)

        results = client.health_check()

        assert "test-model-1" in results
        assert "test-model-2" in results
        # openai-model will fail health check due to missing provider

    def test_health_check_failing_model(
        self, config_file, mock_env_vars, registered_mock_provider
    ):
        """Test health check with failing model."""
        client = Client(config_file)

        # Make provider fail
        provider = registered_mock_provider.get_or_create_provider(
            "test", api_key="test-api-key-12345"
        )
        provider.should_fail = True

        results = client.health_check("test-model-1")

        assert results["test-model-1"] is False

    def test_health_check_nonexistent_model(self, config_file):
        """Test health check with nonexistent model."""
        client = Client(config_file)

        results = client.health_check("nonexistent-model")

        assert results["nonexistent-model"] is False
