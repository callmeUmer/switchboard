"""Anthropic provider implementation."""

import httpx
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import BaseProvider, CompletionResponse
from ..exceptions import ProviderError, ModelNotFoundError

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider for Claude models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.base_url = kwargs.get('base_url', 'https://api.anthropic.com')
        self.anthropic_version = kwargs.get('anthropic_version', '2023-06-01')
        self._cached_models: Optional[List[str]] = None

        # Validate API key format
        if api_key and not self._is_valid_api_key_format(api_key):
            logger.warning("Anthropic API key format appears invalid. Expected format: sk-ant-...")

        # Initialize Anthropic client if library is available
        if ANTHROPIC_AVAILABLE:
            self._client = Anthropic(api_key=api_key, base_url=self.base_url)
        else:
            self._client = None
            logger.debug("Anthropic library not available, using httpx for API calls")

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "anthropic"

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate Anthropic API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid, False otherwise
        """
        return api_key.startswith("sk-ant-")

    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from Anthropic API."""
        # Static fallback list
        fallback_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic library not available. Using static model list. Install with: pip install anthropic")
            return fallback_models

        if not self._client:
            logger.warning("Anthropic client not initialized. Using static model list.")
            return fallback_models

        try:
            models_response = self._client.models.list()
            # Get all Claude models
            model_ids = [model.id for model in models_response.data]
            logger.debug(f"Fetched {len(model_ids)} models from Anthropic API")
            return sorted(model_ids)

        except Exception as e:
            # Log the error but return static fallback list
            logger.warning(f"Failed to fetch models from Anthropic API: {e}. Using static model list.")
            return fallback_models

    @property
    def supported_models(self) -> List[str]:
        """List of supported Anthropic models fetched dynamically from API."""
        if self._cached_models is None:
            self._cached_models = self._fetch_available_models()
        return self._cached_models

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Anthropic API."""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }

    def _prepare_request_data(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare request data for Anthropic API."""
        # Validate model is supported
        if not self.is_model_supported(model):
            available_models = ", ".join(self.supported_models[:5])
            logger.error(f"Model '{model}' not supported. Available models include: {available_models}...")
            raise ModelNotFoundError(
                f"Model '{model}' is not supported by Anthropic provider. "
                f"Available models include: {available_models}..."
            )

        # Build messages for the new Claude 3 format
        messages = [{"role": "user", "content": prompt}]

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or 4096,  # Required for Anthropic
        }

        # Add optional parameters
        if temperature is not None:
            data["temperature"] = temperature

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["api_key", "base_url", "anthropic_version"] and value is not None:
                data[key] = value

        return data

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse Anthropic API response."""
        try:
            # Handle Claude 3 message format
            if "content" in response_data:
                content_blocks = response_data["content"]
                if isinstance(content_blocks, list) and content_blocks:
                    content = content_blocks[0].get("text", "")
                else:
                    content = str(content_blocks)
            else:
                content = response_data.get("completion", "")

            usage = response_data.get("usage", {})

            return CompletionResponse(
                content=content,
                model=model,
                provider=self.name,
                timestamp=datetime.now(),
                usage=usage,
                metadata={
                    "id": response_data.get("id"),
                    "type": response_data.get("type"),
                    "role": response_data.get("role"),
                    "stop_reason": response_data.get("stop_reason"),
                    "stop_sequence": response_data.get("stop_sequence"),
                }
            )

        except (KeyError, IndexError) as e:
            raise ProviderError(f"Invalid response format from Anthropic: {e}")

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Anthropic API.

        Args:
            prompt: Input prompt
            model: Anthropic model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            **kwargs: Additional Anthropic parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If API request fails
            ModelNotFoundError: If model is not supported
        """
        logger.debug(f"Starting completion request for model: {model}")

        try:
            request_data = self._prepare_request_data(
                prompt, model, max_tokens, temperature, **kwargs
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=timeout or 30
                )

                if response.status_code == 401:
                    logger.error("Authentication failed with Anthropic API")
                    raise ProviderError("Invalid Anthropic API key")
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded for Anthropic API")
                    raise ProviderError("Anthropic rate limit exceeded")
                elif response.status_code == 400:
                    error_detail = response.json().get("error", {}).get("message", "Bad request")
                    logger.error(f"Bad request to Anthropic API: {error_detail}")
                    raise ProviderError(f"Anthropic API error: {error_detail}")
                elif response.status_code != 200:
                    logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                    raise ProviderError(f"Anthropic API error: {response.status_code} - {response.text}")

                response_data = response.json()
                logger.debug(f"Completion request successful for model: {model}")
                return self._parse_response(response_data, model)

        except httpx.TimeoutException as e:
            logger.error(f"Anthropic API request timed out after {timeout or 30} seconds")
            raise ProviderError("Anthropic API request timed out") from e
        except httpx.RequestError as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise ProviderError(f"Anthropic API request failed: {e}") from e
        except Exception as e:
            if isinstance(e, (ProviderError, ModelNotFoundError)):
                raise
            logger.error(f"Unexpected error in Anthropic provider: {e}")
            raise ProviderError(f"Unexpected error in Anthropic provider: {e}") from e

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an Anthropic model."""
        base_info = super().get_model_info(model)

        # Fetch model details from Anthropic API if available
        if ANTHROPIC_AVAILABLE and self._client:
            try:
                model_data = self._client.models.retrieve(model)
                base_info.update({
                    "id": model_data.id,
                    "display_name": model_data.display_name,
                    "created_at": model_data.created_at,
                })
            except Exception:
                # If we can't fetch details, just return base info
                pass

        return base_info

    async def health_check(self) -> bool:
        """Check if Anthropic provider is healthy."""
        try:
            # Use the first available model for health check
            if not self.supported_models:
                logger.warning("No supported models available for health check")
                return False

            # Try to use a known stable model, fallback to first available
            preferred_models = ["claude-3-haiku-20240307", "claude-3-5-haiku-20241022"]
            test_model = None

            for preferred in preferred_models:
                if preferred in self.supported_models:
                    test_model = preferred
                    break

            if not test_model:
                test_model = self.supported_models[0]

            logger.debug(f"Running health check with model: {test_model}")

            # Simple test with minimal tokens
            response = await self.complete(
                prompt="Hi",
                model=test_model,
                max_tokens=1,
                timeout=10
            )
            return bool(response.content)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False