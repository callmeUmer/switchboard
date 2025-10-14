"""OpenAI provider implementation."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..exceptions import ModelNotFoundError, ProviderError
from .base import BaseProvider, CompletionResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for GPT models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.organization = kwargs.get("organization")
        self._cached_models: Optional[List[str]] = None

        # Validate API key format
        if api_key and not self._is_valid_api_key_format(api_key):
            logger.warning(
                "OpenAI API key format appears invalid. Expected format: sk-..."
            )

        # Initialize OpenAI client if library is available
        if OPENAI_AVAILABLE:
            self._client = OpenAI(
                api_key=api_key, base_url=self.base_url, organization=self.organization
            )
        else:
            self._client = None
            logger.debug("OpenAI library not available, using httpx for API calls")

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "openai"

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate OpenAI API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid, False otherwise
        """
        return api_key.startswith("sk-") and len(api_key) > 20

    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from OpenAI API.

        Returns:
            List of available model IDs from OpenAI API

        Raises:
            ProviderError: If unable to fetch models from API
        """
        if not OPENAI_AVAILABLE or not self._client:
            logger.error(
                "OpenAI library not available. Install with: pip install openai"
            )
            raise ProviderError(
                "OpenAI library not available. Install it with: pip install openai"
            )

        try:
            models_response = self._client.models.list()
            # Get all model IDs - let the user decide which models to use
            model_ids = [model.id for model in models_response.data]
            logger.debug(f"Fetched {len(model_ids)} models from OpenAI API")
            return sorted(model_ids)

        except Exception as e:
            logger.error(f"Failed to fetch models from OpenAI API: {e}")
            raise ProviderError(f"Failed to fetch models from OpenAI API: {e}") from e

    @property
    def supported_models(self) -> List[str]:
        """List of supported OpenAI models fetched dynamically from API."""
        if self._cached_models is None:
            self._cached_models = self._fetch_available_models()
        return self._cached_models

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        return headers

    def _prepare_request_data(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare request data for OpenAI API.

        Note: Model validation is done by OpenAI API directly.
        This allows users to use any model name, including newly released models
        that may not be in the cached model list yet.
        """
        # Build messages for chat completion
        messages = [{"role": "user", "content": prompt}]

        data = {
            "model": model,
            "messages": messages,
        }

        # Add optional parameters
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        if temperature is not None:
            data["temperature"] = temperature

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["api_key", "base_url", "organization"] and value is not None:
                data[key] = value

        return data

    def _parse_response(
        self, response_data: Dict[str, Any], model: str
    ) -> CompletionResponse:
        """Parse OpenAI API response."""
        try:
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage", {})

            return CompletionResponse(
                content=content,
                model=model,
                provider=self.name,
                timestamp=datetime.now(),
                usage=usage,
                metadata={
                    "id": response_data.get("id"),
                    "object": response_data.get("object"),
                    "created": response_data.get("created"),
                    "finish_reason": response_data["choices"][0].get("finish_reason"),
                },
            )

        except (KeyError, IndexError) as e:
            raise ProviderError(f"Invalid response format from OpenAI: {e}")

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate completion using OpenAI API.

        Args:
            prompt: Input prompt
            model: OpenAI model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            timeout: Request timeout in seconds
            **kwargs: Additional OpenAI parameters

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
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=timeout or 30,
                )

                if response.status_code == 401:
                    logger.error("Authentication failed with OpenAI API")
                    raise ProviderError("Invalid OpenAI API key")
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded for OpenAI API")
                    raise ProviderError("OpenAI rate limit exceeded")
                elif response.status_code == 404:
                    error_data = response.json().get("error", {})
                    error_msg = error_data.get("message", f"Model '{model}' not found")
                    logger.error(f"Model not found: {error_msg}")
                    raise ModelNotFoundError(f"OpenAI: {error_msg}")
                elif response.status_code == 400:
                    error_data = response.json().get("error", {})
                    error_msg = error_data.get("message", "Bad request")
                    # Check if it's a model-related error
                    if (
                        "model" in error_msg.lower()
                        and "does not exist" in error_msg.lower()
                    ):
                        logger.error(f"Model does not exist: {error_msg}")
                        raise ModelNotFoundError(f"OpenAI: {error_msg}")
                    logger.error(f"Bad request to OpenAI API: {error_msg}")
                    raise ProviderError(f"OpenAI API error: {error_msg}")
                elif response.status_code != 200:
                    logger.error(
                        f"OpenAI API error: {response.status_code} - {response.text}"
                    )
                    raise ProviderError(
                        f"OpenAI API error: {response.status_code} - {response.text}"
                    )

                response_data = response.json()
                logger.debug(f"Completion request successful for model: {model}")
                return self._parse_response(response_data, model)

        except httpx.TimeoutException as e:
            logger.error(f"OpenAI API request timed out after {timeout or 30} seconds")
            raise ProviderError("OpenAI API request timed out") from e
        except httpx.RequestError as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise ProviderError(f"OpenAI API request failed: {e}") from e
        except Exception as e:
            if isinstance(e, (ProviderError, ModelNotFoundError)):
                raise
            logger.error(f"Unexpected error in OpenAI provider: {e}")
            raise ProviderError(f"Unexpected error in OpenAI provider: {e}") from e

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an OpenAI model."""
        base_info = super().get_model_info(model)

        # Fetch model details from OpenAI API if available
        if OPENAI_AVAILABLE and self._client:
            try:
                model_data = self._client.models.retrieve(model)
                base_info.update(
                    {
                        "id": model_data.id,
                        "created": model_data.created,
                        "owned_by": model_data.owned_by,
                    }
                )
            except Exception:
                # If we can't fetch details, just return base info
                pass

        return base_info

    async def health_check(self) -> bool:
        """Check if OpenAI provider is healthy."""
        try:
            # Use the first available model for health check
            if not self.supported_models:
                logger.warning("No supported models available for health check")
                return False

            # Try to use a known stable, fast model for health checks
            preferred_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"]
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
                prompt="Hi", model=test_model, max_tokens=1, timeout=10
            )
            return bool(response.content)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
