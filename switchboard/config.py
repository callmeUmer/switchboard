"""Configuration management for Switchboard."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from .exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    model_name: str = Field(..., description="Model identifier")
    api_key_env: Optional[str] = Field(
        None, description="Environment variable for API key"
    )
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for completion")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    timeout: Optional[int] = Field(30, description="Request timeout in seconds")
    extra_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v


class TaskConfig(BaseModel):
    """Configuration for task-based model routing."""

    primary_model: str = Field(..., description="Primary model for this task")
    fallback_models: List[str] = Field(
        default_factory=list, description="Fallback models in order"
    )
    description: Optional[str] = Field(None, description="Task description")


class SwitchboardConfig(BaseModel):
    """Main configuration for Switchboard."""

    models: Dict[str, ModelConfig] = Field(..., description="Available models")
    tasks: Dict[str, TaskConfig] = Field(
        default_factory=dict, description="Task-based routing"
    )
    default_model: str = Field(..., description="Default model when no task specified")
    default_fallback: List[str] = Field(
        default_factory=list, description="Default fallback chain"
    )
    enable_caching: bool = Field(True, description="Enable response caching")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, v, info):
        if info.data and "models" in info.data and v not in info.data["models"]:
            raise ValueError(f'Default model "{v}" not found in models configuration')
        return v

    @field_validator("tasks")
    @classmethod
    def validate_task_models(cls, v, info):
        if not info.data or "models" not in info.data:
            return v

        available_models = set(info.data["models"].keys())
        for task_name, task_config in v.items():
            # Check primary model
            if task_config.primary_model not in available_models:
                raise ValueError(
                    f'Primary model "{task_config.primary_model}" for task "{task_name}" not found in models'
                )

            # Check fallback models
            for fallback_model in task_config.fallback_models:
                if fallback_model not in available_models:
                    raise ValueError(
                        f'Fallback model "{fallback_model}" for task "{task_name}" not found in models'
                    )

        return v


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize config manager.

        Args:
            config_path: Path to config file. If None, looks for config in standard locations.
        """
        self.config_path = self._find_config_path(config_path)
        self.config: Optional[SwitchboardConfig] = None

        # Load environment variables
        load_dotenv()

    def _find_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Find configuration file path."""
        if config_path:
            return Path(config_path)

        # Look for config in standard locations
        possible_paths = [
            Path("switchboard.yaml"),
            Path("switchboard.yml"),
            Path("config/switchboard.yaml"),
            Path("config/switchboard.yml"),
            Path.home() / ".switchboard.yaml",
            Path.home() / ".switchboard.yml",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        raise ConfigurationError(
            f"No configuration file found. Looked in: {', '.join(str(p) for p in possible_paths)}"
        )

    def load_config(self) -> SwitchboardConfig:
        """Load and validate configuration."""
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)

            self.config = SwitchboardConfig(**config_data)
            return self.config

        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if not self.config:
            self.load_config()

        if model_name not in self.config.models:
            raise ConfigurationError(f"Model '{model_name}' not found in configuration")

        return self.config.models[model_name]

    def get_task_config(self, task_name: str) -> Optional[TaskConfig]:
        """Get configuration for a specific task."""
        if not self.config:
            self.load_config()

        return self.config.tasks.get(task_name)

    def get_api_key(self, model_config: ModelConfig) -> Optional[str]:
        """Get API key for a model from environment variables."""
        if not model_config.api_key_env:
            return None

        api_key = os.getenv(model_config.api_key_env)
        if not api_key:
            raise ConfigurationError(
                f"API key not found in environment variable: {model_config.api_key_env}"
            )

        return api_key

    def reload_config(self) -> SwitchboardConfig:
        """Reload configuration from file."""
        return self.load_config()


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get global config manager instance."""
    global _config_manager

    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)

    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None) -> SwitchboardConfig:
    """Convenience function to load configuration."""
    config_manager = get_config_manager(config_path)
    return config_manager.load_config()
