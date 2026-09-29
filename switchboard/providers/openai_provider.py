"""OpenAI provider implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpx

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..exceptions import ModelNotFoundError, ModelResponseError, ProviderError
from ..utils import summarize_error_body
from .base import (
    BaseProvider,
    CompletionResponse,
    is_not_found_error,
    validate_base_url,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for GPT models."""

    name = "openai"
    ALLOWED_CONFIG_KEYS = frozenset({"base_url", "organization", "allow_http"})

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration (allowed keys: base_url,
                organization, allow_http)
        """
        super().__init__(api_key, **kwargs)
        self.base_url = validate_base_url(
            kwargs.get("base_url", "https://api.openai.com/v1"),
            allow_http=bool(kwargs.get("allow_http", False)),
        ).rstrip("/")
        self.organization = kwargs.get("organization")
        self._is_official_api = urlsplit(self.base_url).hostname == "api.openai.com"

        # Validate API key format
        if api_key and not self._is_valid_api_key_format(api_key):
            logger.warning(
                "OpenAI API key format appears invalid. Expected format: sk-..."
            )

        # Initialize OpenAI client if library is available
        self._client: Optional[Any] = None
        if OPENAI_AVAILABLE:
            self._client = OpenAI(
                api_key=api_key, base_url=self.base_url, organization=self.organization
            )
        else:
            logger.debug("OpenAI library not available, using httpx for API calls")

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate OpenAI API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid, False otherwise
        """
        return api_key.startswith("sk-") and len(api_key) > 20

    def _require_client(self) -> Any:
        """Return the SDK client used for the models API.

        Raises:
            ProviderError: If the openai library is not installed
        """
        if not OPENAI_AVAILABLE or not self._client:
            logger.error(
                "OpenAI library not available. Install with: pip install openai"
            )
            raise ProviderError(
                "OpenAI library not available. Install it with: pip install openai"
            )
        return self._client

    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from OpenAI API.

        Returns:
            List of available model IDs from OpenAI API

        Raises:
            ProviderError: If unable to fetch models from API
        """
        client = self._require_client()

        try:
            # Get all model IDs - let the user decide which models to use.
            # Iterating the page (rather than .data) follows any pagination.
            model_ids = [model.id for model in client.models.list()]
            logger.debug(f"Fetched {len(model_ids)} models from OpenAI API")
            return sorted(model_ids)

        except Exception as e:
            logger.error(f"Failed to fetch models from OpenAI API: {e}")
            raise ProviderError(f"Failed to fetch models from OpenAI API: {e}") from e

    def _retrieve_model(self, model: str) -> Optional[Any]:
        """Look up a single model via the models API.

        Returns:
            The model object, or None if the API says it does not exist

        Raises:
            ProviderError: If the lookup itself fails
        """
        client = self._require_client()

        try:
            return client.models.retrieve(model)
        except Exception as e:
            if is_not_found_error(e):
                return None
            logger.error(f"Failed to look up OpenAI model {model!r}: {e}")
            raise ProviderError(f"Failed to look up OpenAI model {model!r}: {e}") from e

    @property
    def supported_models(self) -> List[str]:
        """All OpenAI models available to this API key, fetched live."""
        return self._fetch_available_models()

    def _check_model(self, model: str) -> bool:
        """Ask the models API about this one model.

        OpenAI-compatible servers behind a custom base_url may not implement
        the single-model endpoint (a 404 there would wrongly mean "no such
        model"), so those check the full list instead.
        """
        if not self._is_official_api:
            return super()._check_model(model)
        return self._retrieve_model(model) is not None

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

        data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Add optional parameters
        if max_tokens is not None:
            # OpenAI deprecated max_tokens and its reasoning models reject it;
            # OpenAI-compatible servers behind a custom base_url may only
            # understand the old name, so keep it for them
            if self._is_official_api:
                data["max_completion_tokens"] = max_tokens
            else:
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
            # content is null when a reasoning model spends the whole token
            # budget on reasoning
            content = response_data["choices"][0]["message"]["content"] or ""
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
            raise ModelResponseError(f"Invalid response format from OpenAI: {e}")

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
                    error_msg = (
                        summarize_error_body(response) or f"Model '{model}' not found"
                    )
                    logger.error(f"Model not found: {error_msg}")
                    raise ModelNotFoundError(f"OpenAI: {error_msg}")
                elif response.status_code == 400:
                    error_msg = summarize_error_body(response) or "Bad request"
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
                    error_msg = summarize_error_body(response)
                    logger.error(
                        f"OpenAI API error: {response.status_code} - {error_msg}"
                    )
                    raise ProviderError(
                        f"OpenAI API error: {response.status_code} - {error_msg}"
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
        """Get information about an OpenAI model.

        On api.openai.com, one models-API lookup answers both whether the
        model exists and its details.

        Raises:
            ProviderError: If the lookup fails
        """
        if not self._is_official_api:
            # No single-model endpoint to rely on; see _check_model
            return super().get_model_info(model)

        info: Dict[str, Any] = {
            "provider": self.name,
            "model": model,
            "supported": False,
        }

        model_data = self._retrieve_model(model)
        if model_data is not None:
            self._known_models.add(model)
            info.update(
                {
                    "supported": True,
                    "id": model_data.id,
                    "created": model_data.created,
                    "owned_by": model_data.owned_by,
                }
            )

        return info

    async def health_check(self, model: Optional[str] = None) -> bool:
        """Check if OpenAI provider is healthy.

        Args:
            model: Model to probe with a minimal completion. If omitted, the
                models endpoint is queried instead (no tokens spent), which
                checks connectivity and credentials without picking a model.
        """
        if model:
            return await super().health_check(model)

        try:
            models = await asyncio.to_thread(self._fetch_available_models)
            return bool(models)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
