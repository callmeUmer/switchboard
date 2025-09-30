"""OpenAI provider implementation."""

import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base import BaseProvider, CompletionResponse
from ..exceptions import ProviderError, ModelNotFoundError


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for GPT models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.base_url = kwargs.get('base_url', 'https://api.openai.com/v1')
        self.organization = kwargs.get('organization')

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        """List of supported OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct",
        ]

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
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare request data for OpenAI API."""
        if not self.is_model_supported(model):
            raise ModelNotFoundError(f"Model '{model}' is not supported by OpenAI provider")

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

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> CompletionResponse:
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
                }
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
        **kwargs
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
        try:
            request_data = self._prepare_request_data(
                prompt, model, max_tokens, temperature, **kwargs
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=timeout or 30
                )

                if response.status_code == 401:
                    raise ProviderError("Invalid OpenAI API key")
                elif response.status_code == 429:
                    raise ProviderError("OpenAI rate limit exceeded")
                elif response.status_code == 400:
                    error_detail = response.json().get("error", {}).get("message", "Bad request")
                    raise ProviderError(f"OpenAI API error: {error_detail}")
                elif response.status_code != 200:
                    raise ProviderError(f"OpenAI API error: {response.status_code} - {response.text}")

                response_data = response.json()
                return self._parse_response(response_data, model)

        except httpx.TimeoutException:
            raise ProviderError("OpenAI API request timed out")
        except httpx.RequestError as e:
            raise ProviderError("OpenAI API request failed")
        except Exception as e:
            if isinstance(e, (ProviderError, ModelNotFoundError)):
                raise
            raise ProviderError(f"Unexpected error in OpenAI provider: {e}")

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an OpenAI model."""
        base_info = super().get_model_info(model)

        # Add OpenAI-specific model information
        model_specs = {
            "gpt-4": {"context_length": 8192, "training_data": "Up to Sep 2021"},
            "gpt-4-turbo": {"context_length": 128000, "training_data": "Up to Apr 2023"},
            "gpt-4-turbo-preview": {"context_length": 128000, "training_data": "Up to Apr 2023"},
            "gpt-3.5-turbo": {"context_length": 4096, "training_data": "Up to Sep 2021"},
            "gpt-3.5-turbo-0125": {"context_length": 16385, "training_data": "Up to Sep 2021"},
        }

        if model in model_specs:
            base_info.update(model_specs[model])

        return base_info

    async def health_check(self) -> bool:
        """Check if OpenAI provider is healthy."""
        try:
            # Simple test with minimal tokens
            response = await self.complete(
                prompt="Hi",
                model="gpt-3.5-turbo",
                max_tokens=1,
                timeout=10
            )
            return bool(response.content)
        except Exception:
            return False