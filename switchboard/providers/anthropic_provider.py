"""Anthropic provider implementation."""

import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base import BaseProvider, CompletionResponse
from ..exceptions import ProviderError, ModelNotFoundError


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

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "anthropic"

    @property
    def supported_models(self) -> List[str]:
        """List of supported Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

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
        if not self.is_model_supported(model):
            raise ModelNotFoundError(f"Model '{model}' is not supported by Anthropic provider")

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
                    raise ProviderError("Invalid Anthropic API key")
                elif response.status_code == 429:
                    raise ProviderError("Anthropic rate limit exceeded")
                elif response.status_code == 400:
                    error_detail = response.json().get("error", {}).get("message", "Bad request")
                    raise ProviderError(f"Anthropic API error: {error_detail}")
                elif response.status_code != 200:
                    raise ProviderError(f"Anthropic API error: {response.status_code} - {response.text}")

                response_data = response.json()
                return self._parse_response(response_data, model)

        except httpx.TimeoutException:
            raise ProviderError("Anthropic API request timed out")
        except httpx.RequestError as e:
            raise ProviderError("Anthropic API request failed")
        except Exception as e:
            if isinstance(e, (ProviderError, ModelNotFoundError)):
                raise
            raise ProviderError(f"Unexpected error in Anthropic provider: {e}")

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an Anthropic model."""
        base_info = super().get_model_info(model)

        # Add Anthropic-specific model information
        model_specs = {
            "claude-3-opus-20240229": {
                "context_length": 200000,
                "training_data": "Up to Aug 2023",
                "capabilities": ["reasoning", "math", "coding", "multilingual"]
            },
            "claude-3-sonnet-20240229": {
                "context_length": 200000,
                "training_data": "Up to Aug 2023",
                "capabilities": ["general", "reasoning", "coding"]
            },
            "claude-3-haiku-20240307": {
                "context_length": 200000,
                "training_data": "Up to Aug 2023",
                "capabilities": ["speed", "general", "summarization"]
            },
            "claude-2.1": {
                "context_length": 200000,
                "training_data": "Up to early 2023",
                "capabilities": ["reasoning", "analysis", "coding"]
            },
            "claude-2.0": {
                "context_length": 100000,
                "training_data": "Up to early 2023",
                "capabilities": ["reasoning", "analysis", "coding"]
            },
            "claude-instant-1.2": {
                "context_length": 100000,
                "training_data": "Up to early 2023",
                "capabilities": ["speed", "general"]
            },
        }

        if model in model_specs:
            base_info.update(model_specs[model])

        return base_info

    async def health_check(self) -> bool:
        """Check if Anthropic provider is healthy."""
        try:
            # Simple test with minimal tokens
            response = await self.complete(
                prompt="Hi",
                model="claude-3-haiku-20240307",
                max_tokens=1,
                timeout=10
            )
            return bool(response.content)
        except Exception:
            return False