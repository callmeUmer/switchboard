"""Anthropic provider implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..exceptions import ModelNotFoundError, ModelResponseError, ProviderError
from ..utils import summarize_error_body
from .base import (
    BaseProvider,
    CompletionResponse,
    is_not_found_error,
    validate_base_url,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider for Claude models."""

    name = "anthropic"
    ALLOWED_CONFIG_KEYS = frozenset({"base_url", "anthropic_version", "allow_http"})

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration (allowed keys: base_url,
                anthropic_version, allow_http)
        """
        super().__init__(api_key, **kwargs)
        self.base_url = validate_base_url(
            kwargs.get("base_url", "https://api.anthropic.com"),
            allow_http=bool(kwargs.get("allow_http", False)),
        ).rstrip("/")
        self.anthropic_version = kwargs.get("anthropic_version", "2023-06-01")

        # Validate API key format
        if api_key and not self._is_valid_api_key_format(api_key):
            logger.warning(
                "Anthropic API key format appears invalid. Expected format: sk-ant-..."
            )

        # Initialize Anthropic client if library is available
        self._client: Optional[Any] = None
        if ANTHROPIC_AVAILABLE:
            self._client = Anthropic(api_key=api_key, base_url=self.base_url)
        else:
            logger.debug("Anthropic library not available, using httpx for API calls")

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate Anthropic API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid, False otherwise
        """
        return api_key.startswith("sk-ant-")

    def _require_client(self) -> Any:
        """Return the SDK client used for the models API.

        Raises:
            ProviderError: If the anthropic library is not installed
        """
        if not ANTHROPIC_AVAILABLE or not self._client:
            logger.error(
                "Anthropic library not available. Install with: pip install anthropic"
            )
            raise ProviderError(
                "Anthropic library not available. Install it with: "
                "pip install anthropic"
            )
        return self._client

    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from Anthropic API.

        Returns:
            List of available model IDs from Anthropic API

        Raises:
            ProviderError: If unable to fetch models from API
        """
        client = self._require_client()

        try:
            # Iterating the page auto-paginates; .data would only hold the
            # first page (20 models by default)
            model_ids = [model.id for model in client.models.list()]
            logger.debug(f"Fetched {len(model_ids)} models from Anthropic API")
            return sorted(model_ids)

        except Exception as e:
            logger.error(f"Failed to fetch models from Anthropic API: {e}")
            raise ProviderError(
                f"Failed to fetch models from Anthropic API: {e}"
            ) from e

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
            logger.error(f"Failed to look up Anthropic model {model!r}: {e}")
            raise ProviderError(
                f"Failed to look up Anthropic model {model!r}: {e}"
            ) from e

    @property
    def supported_models(self) -> List[str]:
        """All Anthropic models available to this API key, fetched live."""
        return self._fetch_available_models()

    def _check_model(self, model: str) -> bool:
        """Ask the models API about this one model."""
        return self._retrieve_model(model) is not None

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Anthropic API."""
        if not self.api_key:
            raise ProviderError("Anthropic API key is not configured")
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
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare request data for Anthropic API.

        Note: Model validation is done by the Anthropic API directly.
        This allows users to use any model name, including newly released
        models that may not be in the cached model list yet.
        """
        # Build messages for the Messages API
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
            if (
                key not in ["api_key", "base_url", "anthropic_version"]
                and value is not None
            ):
                data[key] = value

        return data

    def _parse_response(
        self, response_data: Dict[str, Any], model: str
    ) -> CompletionResponse:
        """Parse Anthropic API response."""
        try:
            # Messages API format, with legacy Text Completions fallback
            if "content" in response_data:
                content_blocks = response_data["content"]
                if isinstance(content_blocks, list):
                    # Thinking models put thinking blocks before the text
                    # (and tool use can split text), so join every text block
                    content = "".join(
                        block.get("text", "")
                        for block in content_blocks
                        if block.get("type") == "text"
                    )
                else:
                    content = str(content_blocks)
            elif "completion" in response_data:
                content = response_data["completion"]
            else:
                raise KeyError("missing 'content' or 'completion'")

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
                },
            )

        except (KeyError, IndexError) as e:
            raise ModelResponseError(f"Invalid response format from Anthropic: {e}")

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs,
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
                    timeout=timeout or 30,
                )

                if response.status_code == 401:
                    logger.error("Authentication failed with Anthropic API")
                    raise ProviderError("Invalid Anthropic API key")
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded for Anthropic API")
                    raise ProviderError("Anthropic rate limit exceeded")
                elif response.status_code == 400:
                    error_detail = summarize_error_body(response) or "Bad request"
                    logger.error(f"Bad request to Anthropic API: {error_detail}")
                    raise ProviderError(f"Anthropic API error: {error_detail}")
                elif response.status_code != 200:
                    error_detail = summarize_error_body(response)
                    logger.error(
                        f"Anthropic API error: {response.status_code} - {error_detail}"
                    )
                    raise ProviderError(
                        f"Anthropic API error: {response.status_code} - {error_detail}"
                    )

                response_data = response.json()
                logger.debug(f"Completion request successful for model: {model}")
                return self._parse_response(response_data, model)

        except httpx.TimeoutException as e:
            logger.error(
                f"Anthropic API request timed out after {timeout or 30} seconds"
            )
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
        """Get information about an Anthropic model.

        One models-API lookup answers both whether the model exists and its
        details.

        Raises:
            ProviderError: If the lookup fails
        """
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
                    "display_name": model_data.display_name,
                    "created_at": model_data.created_at,
                }
            )

        return info

    async def health_check(self, model: Optional[str] = None) -> bool:
        """Check if Anthropic provider is healthy.

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
