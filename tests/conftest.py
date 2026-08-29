"""Pytest configuration and shared fixtures."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from switchboard.config import ConfigManager, SwitchboardConfig
from switchboard.exceptions import ProviderError
from switchboard.providers.base import BaseProvider, CompletionResponse

# Fixed timestamp for deterministic assertions
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "models": {
            "test-model-1": {
                "provider": "test",
                "model_name": "test-model-1",
                "api_key_env": "TEST_API_KEY",
                "max_tokens": 100,
                "temperature": 0.7,
                "timeout": 30,
            },
            "test-model-2": {
                "provider": "test",
                "model_name": "test-model-2",
                "api_key_env": "TEST_API_KEY",
                "max_tokens": 200,
                "temperature": 0.5,
                "timeout": 60,
            },
            "openai-model": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 30,
            },
        },
        "tasks": {
            "test-task": {
                "primary_model": "test-model-1",
                "fallback_models": ["test-model-2"],
                "description": "Test task",
            },
            "coding": {
                "primary_model": "openai-model",
                "fallback_models": ["test-model-1"],
                "description": "Coding tasks",
            },
        },
        "default_model": "test-model-1",
        "default_fallback": ["test-model-2"],
    }


@pytest.fixture
def config_file(temp_dir, sample_config_data):
    """Create a temporary config file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_data, f)
    return config_path


@pytest.fixture
def config_manager(config_file):
    """Create a ConfigManager instance with test config."""
    return ConfigManager(config_file)


@pytest.fixture
def switchboard_config(sample_config_data):
    """Create a SwitchboardConfig instance."""
    return SwitchboardConfig(**sample_config_data)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("TEST_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
def mock_completion_response():
    """Create a mock CompletionResponse."""
    return CompletionResponse(
        content="Test response content",
        model="test-model",
        provider="test",
        timestamp=FIXED_DATETIME,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        metadata={"test": "data"},
    )


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    name = "test"
    # Permissive allowlist so tests can pass arbitrary config keys
    ALLOWED_CONFIG_KEYS = frozenset(
        {"base_url", "custom_param", "allow_http", "organization", "config"}
    )

    def __init__(self, api_key=None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.call_count = 0
        self.last_params = {}
        self.response_content = "Mock response"
        self.should_fail = False

    @property
    def supported_models(self) -> list:
        return ["test-model-1", "test-model-2", "test-model-3"]

    def requires_api_key(self) -> bool:
        return True

    async def complete(
        self, prompt, model, max_tokens=None, temperature=None, timeout=None, **kwargs
    ):
        self.call_count += 1
        self.last_params = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
            **kwargs,
        }

        if self.should_fail:
            raise ProviderError("Mock provider failure")

        return CompletionResponse(
            content=self.response_content,
            model=model,
            provider=self.name,
            timestamp=datetime.now(),
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )


# Make MockProvider available at module level for registration
__all__ = ["MockProvider"]


@pytest.fixture
def mock_provider():
    """Create a mock provider instance."""
    return MockProvider(api_key="test-key")


@pytest.fixture
def clean_registry():
    """Clean provider registry before and after tests."""
    from switchboard.providers.registry import get_registry

    registry = get_registry()
    original_providers = registry._providers.copy()
    original_instances = registry._instances.copy()

    # Clear for test
    registry._providers.clear()
    registry._instances.clear()

    yield registry

    # Restore original state
    registry._providers = original_providers
    registry._instances = original_instances


@pytest.fixture
def registered_mock_provider(clean_registry, mock_provider):
    """Register mock provider for testing."""
    clean_registry.register(MockProvider)
    return clean_registry


# HTTP mocking fixtures
@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mock httpx client for API testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Mocked API response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
    }

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    return mock_client, mock_response


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from OpenAI",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "This is a test response from Anthropic"}],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 25, "output_tokens": 12},
    }


# Error simulation fixtures
@pytest.fixture
def failing_provider():
    """Provider that always fails for error testing."""
    provider = MockProvider(api_key="test-key")
    provider.should_fail = True
    return provider
