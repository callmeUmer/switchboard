"""Base provider interface for AI models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ..exceptions import ProviderError


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
    """Base class for all AI model providers."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize provider.

        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
        self._validate_configuration()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

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
            # Get provider name safely without calling abstract property during init
            provider_name = self.__class__.__name__.replace('Provider', '').lower()
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
        **kwargs
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
        **kwargs
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
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # Already in an event loop, run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.complete(prompt, model, max_tokens, temperature, timeout, **kwargs)
                )
                return future.result(timeout=timeout or 60)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.complete(prompt, model, max_tokens, temperature, timeout, **kwargs)
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
                timeout=10
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