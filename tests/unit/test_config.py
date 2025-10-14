"""Unit tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from switchboard.config import (
    ConfigManager,
    ModelConfig,
    SwitchboardConfig,
    TaskConfig,
    get_config_manager,
    load_config,
)
from switchboard.exceptions import ConfigurationError


class TestModelConfig:
    """Test ModelConfig validation and functionality."""

    def test_model_config_creation(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
        )

        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig(provider="test", model_name="test-model")

        assert config.provider == "test"
        assert config.model_name == "test-model"
        assert config.api_key_env is None
        assert config.max_tokens is None
        assert config.temperature == 0.7
        assert config.timeout == 30
        assert config.extra_params == {}

    def test_model_config_invalid_temperature(self):
        """Test ModelConfig with invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(provider="test", model_name="test-model", temperature=3.0)

        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(provider="test", model_name="test-model", temperature=-0.1)

    def test_model_config_extra_params(self):
        """Test ModelConfig with extra parameters."""
        config = ModelConfig(
            provider="test",
            model_name="test-model",
            extra_params={"custom_param": "value"},
        )

        assert config.extra_params == {"custom_param": "value"}


class TestTaskConfig:
    """Test TaskConfig validation and functionality."""

    def test_task_config_creation(self):
        """Test creating a valid TaskConfig."""
        config = TaskConfig(
            primary_model="gpt-4",
            fallback_models=["gpt-3.5-turbo", "claude-3-haiku"],
            description="Test task",
        )

        assert config.primary_model == "gpt-4"
        assert config.fallback_models == ["gpt-3.5-turbo", "claude-3-haiku"]
        assert config.description == "Test task"

    def test_task_config_defaults(self):
        """Test TaskConfig with default values."""
        config = TaskConfig(primary_model="test-model")

        assert config.primary_model == "test-model"
        assert config.fallback_models == []
        assert config.description is None


class TestSwitchboardConfig:
    """Test SwitchboardConfig validation and functionality."""

    def test_switchboard_config_creation(self, sample_config_data):
        """Test creating a valid SwitchboardConfig."""
        config = SwitchboardConfig(**sample_config_data)

        assert len(config.models) == 3
        assert len(config.tasks) == 2
        assert config.default_model == "test-model-1"
        assert config.default_fallback == ["test-model-2"]
        assert config.enable_caching is True
        assert config.cache_ttl == 3600

    def test_switchboard_config_invalid_default_model(self):
        """Test SwitchboardConfig with invalid default model."""
        config_data = {
            "models": {"test-model": {"provider": "test", "model_name": "test"}},
            "default_model": "nonexistent-model",
        }

        with pytest.raises(
            ValueError, match='Default model "nonexistent-model" not found'
        ):
            SwitchboardConfig(**config_data)

    def test_switchboard_config_invalid_task_model(self):
        """Test SwitchboardConfig with invalid task model."""
        config_data = {
            "models": {"test-model": {"provider": "test", "model_name": "test"}},
            "tasks": {
                "test-task": {
                    "primary_model": "nonexistent-model",
                    "fallback_models": [],
                }
            },
            "default_model": "test-model",
        }

        with pytest.raises(
            ValueError,
            match='Primary model "nonexistent-model" for task "test-task" not found',
        ):
            SwitchboardConfig(**config_data)

    def test_switchboard_config_invalid_fallback_model(self):
        """Test SwitchboardConfig with invalid fallback model."""
        config_data = {
            "models": {"test-model": {"provider": "test", "model_name": "test"}},
            "tasks": {
                "test-task": {
                    "primary_model": "test-model",
                    "fallback_models": ["nonexistent-model"],
                }
            },
            "default_model": "test-model",
        }

        with pytest.raises(
            ValueError,
            match='Fallback model "nonexistent-model" for task "test-task" not found',
        ):
            SwitchboardConfig(**config_data)


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_creation(self, config_file):
        """Test creating ConfigManager with config file."""
        manager = ConfigManager(config_file)
        assert manager.config_path == config_file

    def test_config_manager_load_config(self, config_manager, sample_config_data):
        """Test loading configuration."""
        config = config_manager.load_config()

        assert isinstance(config, SwitchboardConfig)
        assert len(config.models) == 3
        assert config.default_model == "test-model-1"

    def test_config_manager_file_not_found(self):
        """Test ConfigManager with nonexistent file."""
        # When a specific path is provided but doesn't exist, it should fail during load_config
        manager = ConfigManager("nonexistent.yaml")
        with pytest.raises(ConfigurationError):
            manager.load_config()

    def test_config_manager_invalid_yaml(self, temp_dir):
        """Test ConfigManager with invalid YAML."""
        invalid_config = temp_dir / "invalid.yaml"
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: content:\n  - bad")

        manager = ConfigManager(invalid_config)
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            manager.load_config()

    def test_config_manager_get_model_config(self, config_manager):
        """Test getting model configuration."""
        config_manager.load_config()
        model_config = config_manager.get_model_config("test-model-1")

        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == "test"
        assert model_config.model_name == "test-model-1"

    def test_config_manager_get_nonexistent_model(self, config_manager):
        """Test getting nonexistent model configuration."""
        config_manager.load_config()

        with pytest.raises(ConfigurationError, match="Model 'nonexistent' not found"):
            config_manager.get_model_config("nonexistent")

    def test_config_manager_get_task_config(self, config_manager):
        """Test getting task configuration."""
        config_manager.load_config()
        task_config = config_manager.get_task_config("test-task")

        assert isinstance(task_config, TaskConfig)
        assert task_config.primary_model == "test-model-1"

    def test_config_manager_get_nonexistent_task(self, config_manager):
        """Test getting nonexistent task configuration."""
        config_manager.load_config()
        task_config = config_manager.get_task_config("nonexistent")

        assert task_config is None

    def test_config_manager_get_api_key(self, config_manager, mock_env_vars):
        """Test getting API key from environment."""
        config_manager.load_config()
        model_config = config_manager.get_model_config("test-model-1")

        api_key = config_manager.get_api_key(model_config)
        assert api_key == "test-api-key-12345"

    def test_config_manager_missing_api_key(self, config_manager):
        """Test getting missing API key."""
        config_manager.load_config()
        model_config = config_manager.get_model_config("test-model-1")

        with pytest.raises(
            ConfigurationError, match="API key not found in environment variable"
        ):
            config_manager.get_api_key(model_config)

    def test_config_manager_no_api_key_env(self, config_manager):
        """Test model with no API key environment variable."""
        config_manager.load_config()
        model_config = ModelConfig(provider="test", model_name="test")

        api_key = config_manager.get_api_key(model_config)
        assert api_key is None

    def test_config_manager_reload(self, config_manager):
        """Test reloading configuration."""
        config1 = config_manager.load_config()
        config2 = config_manager.reload_config()

        assert config1 is not config2
        assert config1.default_model == config2.default_model

    def test_find_config_path_standard_locations(self, temp_dir):
        """Test finding config in standard locations."""
        # Create config in current directory
        config_path = temp_dir / "switchboard.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"models": {}, "default_model": "test"}, f)

        with patch("pathlib.Path.cwd", return_value=temp_dir):
            manager = ConfigManager()
            assert manager.config_path.name == "switchboard.yaml"


class TestGlobalFunctions:
    """Test global configuration functions."""

    def test_get_config_manager(self, config_file):
        """Test getting global config manager."""
        manager1 = get_config_manager(config_file)
        manager2 = get_config_manager()

        assert manager1 is manager2
        assert manager1.config_path == config_file

    def test_get_config_manager_new_path(self, config_file, temp_dir):
        """Test getting config manager with new path."""
        # Create another config file
        config_file2 = temp_dir / "config2.yaml"
        with open(config_file2, "w") as f:
            yaml.dump({"models": {}, "default_model": "test"}, f)

        manager1 = get_config_manager(config_file)
        manager2 = get_config_manager(config_file2)

        assert manager1 is not manager2
        assert manager2.config_path == config_file2

    def test_load_config_function(self, config_file):
        """Test load_config convenience function."""
        config = load_config(config_file)

        assert isinstance(config, SwitchboardConfig)
        assert len(config.models) == 3
