"""Main client for Switchboard AI model switching."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import ModelConfig, SwitchboardConfig, get_config_manager
from .exceptions import (
    APIKeyError,
    ConfigurationError,
    FallbackExhaustedError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotFoundError,
    SwitchboardError,
)
from .providers import CompletionResponse, get_provider
from .providers.base import BaseProvider
from .utils import run_coroutine_sync


class Client:
    """Main client for AI model switching and completion."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize Switchboard client.

        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.config_manager = get_config_manager(config_path)
        self._config: Optional[SwitchboardConfig] = None

    def _ensure_config_loaded(self) -> SwitchboardConfig:
        """Ensure configuration is loaded and return it."""
        if self._config is None:
            self._config = self.config_manager.load_config()
        return self._config

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
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
            FallbackExhaustedError: If every model in the fallback chain fails
            ConfigurationError: If configuration is invalid (including a
                missing API key for any model in the chain — configuration
                problems fail loudly rather than being skipped)
            ProviderNotFoundError: If a configured provider is not registered
        """
        self._ensure_config_loaded()

        # Determine which model to use
        target_model = self._resolve_model(model, task)

        # Attempt completion with fallback support
        models_to_try = self._get_fallback_chain(target_model, task)
        last_error: Optional[Exception] = None
        attempted: List[str] = []

        for attempt_model in models_to_try:
            attempt_config = self.config_manager.get_model_config(attempt_model)
            attempt_provider = self._get_provider(attempt_config)
            attempt_params = self._prepare_completion_params(attempt_config, kwargs)

            try:
                return attempt_provider.complete_sync(
                    prompt=prompt, model=attempt_config.model_name, **attempt_params
                )
            except (ProviderError, ModelNotFoundError) as e:
                # Provider/model failures are retryable with the next fallback;
                # configuration errors propagate immediately above
                last_error = e
                attempted.append(attempt_model)

        # All models failed
        raise FallbackExhaustedError(
            f"All fallback models failed. Attempted: {attempted}. "
            f"Last error: {last_error}"
        ) from last_error

    async def complete_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Async version of complete method.

        Args:
            prompt: Input prompt for completion
            model: Specific model to use (overrides task-based routing)
            task: Task type for automatic model selection
            **kwargs: Additional parameters passed to the provider

        Returns:
            CompletionResponse with generated content

        Raises:
            FallbackExhaustedError: If every model in the fallback chain fails
            ConfigurationError: If configuration is invalid (including a
                missing API key for any model in the chain)
            ProviderNotFoundError: If a configured provider is not registered
        """
        self._ensure_config_loaded()

        # Determine which model to use
        target_model = self._resolve_model(model, task)

        # Attempt completion with fallback support
        models_to_try = self._get_fallback_chain(target_model, task)
        last_error: Optional[Exception] = None
        attempted: List[str] = []

        for attempt_model in models_to_try:
            attempt_config = self.config_manager.get_model_config(attempt_model)
            attempt_provider = self._get_provider(attempt_config)
            attempt_params = self._prepare_completion_params(attempt_config, kwargs)

            try:
                return await attempt_provider.complete(
                    prompt=prompt, model=attempt_config.model_name, **attempt_params
                )
            except (ProviderError, ModelNotFoundError) as e:
                # Provider/model failures are retryable with the next fallback;
                # configuration errors propagate immediately above
                last_error = e
                attempted.append(attempt_model)

        # All models failed
        raise FallbackExhaustedError(
            f"All fallback models failed. Attempted: {attempted}. "
            f"Last error: {last_error}"
        ) from last_error

    def _resolve_model(self, model: Optional[str], task: Optional[str]) -> str:
        """Resolve which model to use based on input parameters.

        Args:
            model: Explicitly specified model
            task: Task type for routing

        Returns:
            Model name to use

        Raises:
            ConfigurationError: If the specified task is not configured
        """
        if model:
            return model

        if task:
            task_config = self.config_manager.get_task_config(task)
            if task_config is None:
                available = list(self._ensure_config_loaded().tasks.keys())
                raise ConfigurationError(
                    f"Task '{task}' not found in configuration. "
                    f"Available tasks: {available}"
                )
            return task_config.primary_model

        # Fall back to default model
        return self._ensure_config_loaded().default_model

    def _get_fallback_chain(self, model: str, task: Optional[str]) -> List[str]:
        """Get the fallback chain for a model.

        Args:
            model: Primary model name
            task: Task type (if specified)

        Returns:
            List of models to try in order (primary first, then fallbacks)
        """
        chain = [model]

        # If task is specified and has fallback models, use those
        if task:
            task_config = self.config_manager.get_task_config(task)
            if task_config and task_config.fallback_models:
                chain.extend(task_config.fallback_models)
                return list(dict.fromkeys(chain))

        # Otherwise use default fallback chain
        config = self._ensure_config_loaded()
        if config.default_fallback:
            chain.extend(config.default_fallback)

        # De-duplicate preserving order
        return list(dict.fromkeys(chain))

    def _get_provider(self, model_config: ModelConfig) -> BaseProvider:
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
                **(model_config.extra_params or {}),
            )

            return provider

        except (ConfigurationError, APIKeyError, ProviderNotFoundError):
            # Re-raise known exceptions as-is
            raise
        except Exception as e:
            # Wrap unknown exceptions
            raise SwitchboardError(
                f"Failed to get provider '{model_config.provider}': {e}"
            ) from e

    def _prepare_completion_params(
        self, model_config: ModelConfig, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for completion call.

        Args:
            model_config: Model configuration
            kwargs: User-provided parameters

        Returns:
            Merged parameters for provider call
        """
        params: Dict[str, Any] = {}

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
        return list(self._ensure_config_loaded().models.keys())

    def list_tasks(self) -> List[str]:
        """Get list of configured tasks.

        Returns:
            List of task names
        """
        self._ensure_config_loaded()
        return list(self._ensure_config_loaded().tasks.keys())

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
                results[model_name] = run_coroutine_sync(
                    provider.health_check, timeout=10
                )
            except Exception:
                results[model_name] = False

        return results
