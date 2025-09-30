
"""Main client for Switchboard AI model switching."""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path

from .config import ConfigManager, get_config_manager, ModelConfig
from .providers import get_provider, CompletionResponse
from .exceptions import (
    SwitchboardError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderNotFoundError,
    APIKeyError
)


class Client:
    """Main client for AI model switching and completion."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize Switchboard client.

        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.config_manager = get_config_manager(config_path)
        self._config = None

    def _ensure_config_loaded(self):
        """Ensure configuration is loaded."""
        if self._config is None:
            self._config = self.config_manager.load_config()

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using configured models.

        Args:
            prompt: Input prompt for completion
            model: Specific model to use (overrides task-based routing)
            task: Task type for automatic model selection
            **kwargs: Additional parameters passed to the provider

        Returns:
            CompletionResponse with generated content

        Raises:
            SwitchboardError: If completion fails
            ConfigurationError: If configuration is invalid
            ModelNotFoundError: If specified model is not found
        """
        self._ensure_config_loaded()

        # Determine which model to use
        target_model = self._resolve_model(model, task)

        # Get model configuration
        model_config = self.config_manager.get_model_config(target_model)

        # Get provider instance
        provider = self._get_provider(model_config)

        # Merge configuration with kwargs
        completion_params = self._prepare_completion_params(model_config, kwargs)

        try:
            # Execute completion
            return provider.complete_sync(
                prompt=prompt,
                model=model_config.model_name,
                **completion_params
            )

        except Exception as e:
            raise SwitchboardError(f"Completion failed for model '{target_model}': {e}")

    async def complete_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Async version of complete method.

        Args:
            prompt: Input prompt for completion
            model: Specific model to use (overrides task-based routing)
            task: Task type for automatic model selection
            **kwargs: Additional parameters passed to the provider

        Returns:
            CompletionResponse with generated content
        """
        self._ensure_config_loaded()

        # Determine which model to use
        target_model = self._resolve_model(model, task)

        # Get model configuration
        model_config = self.config_manager.get_model_config(target_model)

        # Get provider instance
        provider = self._get_provider(model_config)

        # Merge configuration with kwargs
        completion_params = self._prepare_completion_params(model_config, kwargs)

        try:
            # Execute async completion
            return await provider.complete(
                prompt=prompt,
                model=model_config.model_name,
                **completion_params
            )

        except Exception as e:
            raise SwitchboardError(f"Async completion failed for model '{target_model}': {e}")

    def _resolve_model(self, model: Optional[str], task: Optional[str]) -> str:
        """Resolve which model to use based on input parameters.

        Args:
            model: Explicitly specified model
            task: Task type for routing

        Returns:
            Model name to use
        """
        if model:
            return model

        if task:
            task_config = self.config_manager.get_task_config(task)
            if task_config:
                return task_config.primary_model

        # Fall back to default model
        return self._config.default_model

    def _get_provider(self, model_config: ModelConfig):
        """Get provider instance for the given model configuration.

        Args:
            model_config: Model configuration

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider is not available
            APIKeyError: If API key is missing
        """
        try:
            # Get API key from environment
            api_key = self.config_manager.get_api_key(model_config)

            # Get provider instance
            provider = get_provider(
                provider_name=model_config.provider,
                api_key=api_key,
                **model_config.extra_params
            )

            return provider

        except Exception as e:
            if "API key" in str(e):
                raise APIKeyError(str(e))
            elif "Provider" in str(e) and "not found" in str(e):
                raise ProviderNotFoundError(str(e))
            else:
                raise

    def _prepare_completion_params(
        self,
        model_config: ModelConfig,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for completion call.

        Args:
            model_config: Model configuration
            kwargs: User-provided parameters

        Returns:
            Merged parameters for provider call
        """
        params = {}

        # Add model config parameters
        if model_config.max_tokens is not None:
            params["max_tokens"] = model_config.max_tokens

        if model_config.temperature is not None:
            params["temperature"] = model_config.temperature

        if model_config.timeout is not None:
            params["timeout"] = model_config.timeout

        # Override with user-provided parameters
        params.update(kwargs)

        return params

    def list_models(self) -> List[str]:
        """Get list of available models.

        Returns:
            List of model names
        """
        self._ensure_config_loaded()
        return list(self._config.models.keys())

    def list_tasks(self) -> List[str]:
        """Get list of configured tasks.

        Returns:
            List of task names
        """
        self._ensure_config_loaded()
        return list(self._config.tasks.keys())

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name

        Returns:
            Dictionary with model information
        """
        self._ensure_config_loaded()

        model_config = self.config_manager.get_model_config(model)
        provider = self._get_provider(model_config)

        return provider.get_model_info(model_config.model_name)

    def reload_config(self):
        """Reload configuration from file."""
        self._config = self.config_manager.reload_config()

    def health_check(self, model: Optional[str] = None) -> Dict[str, bool]:
        """Check health of models or providers.

        Args:
            model: Specific model to check. If None, checks all configured models.

        Returns:
            Dictionary mapping model names to health status
        """
        self._ensure_config_loaded()

        models_to_check = [model] if model else self.list_models()
        results = {}

        for model_name in models_to_check:
            try:
                model_config = self.config_manager.get_model_config(model_name)
                provider = self._get_provider(model_config)

                # Simple sync health check
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    health = loop.run_until_complete(provider.health_check())
                    results[model_name] = health
                except Exception:
                    results[model_name] = False

            except Exception:
                results[model_name] = False

        return results