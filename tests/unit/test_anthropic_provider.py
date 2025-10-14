"""Unit tests for Anthropic provider."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from switchboard.exceptions import ModelNotFoundError, ProviderError
from switchboard.providers.anthropic_provider import AnthropicProvider
from switchboard.providers.base import CompletionResponse


class TestAnthropicProvider:
    """Test Anthropic provider functionality."""

    def test_anthropic_provider_creation(self):
        """Test creating Anthropic provider."""
        provider = AnthropicProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.name == "anthropic"
        assert provider.base_url == "https://api.anthropic.com"
        assert provider.anthropic_version == "2023-06-01"

    def test_anthropic_provider_with_config(self):
        """Test creating Anthropic provider with custom config."""
        provider = AnthropicProvider(
            api_key="test-key",
            base_url="https://custom.anthropic.com",
            anthropic_version="2024-01-01",
        )

        assert provider.base_url == "https://custom.anthropic.com"
        assert provider.anthropic_version == "2024-01-01"

    def test_supported_models(self):
        """Test Anthropic supported models."""
        provider = AnthropicProvider(api_key="test-key")
        models = provider.supported_models

        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-haiku-20240307" in models
        assert "claude-2.1" in models
        assert len(models) >= 6

    def test_requires_api_key(self):
        """Test that Anthropic provider requires API key."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.requires_api_key() is True

    def test_no_api_key_error(self):
        """Test error when no API key provided."""
        with pytest.raises(
            ProviderError, match="anthropic provider requires an API key"
        ):
            AnthropicProvider()

    def test_get_headers(self):
        """Test getting request headers."""
        provider = AnthropicProvider(api_key="test-key")
        headers = provider._get_headers()

        expected = {
            "x-api-key": "test-key",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        assert headers == expected

    def test_get_headers_custom_version(self):
        """Test getting headers with custom version."""
        provider = AnthropicProvider(api_key="test-key", anthropic_version="2024-01-01")
        headers = provider._get_headers()

        assert headers["anthropic-version"] == "2024-01-01"

    def test_prepare_request_data_basic(self):
        """Test preparing basic request data."""
        provider = AnthropicProvider(api_key="test-key")
        data = provider._prepare_request_data("Hello", "claude-3-haiku-20240307")

        expected = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096,
        }

        assert data == expected

    def test_prepare_request_data_with_params(self):
        """Test preparing request data with parameters."""
        provider = AnthropicProvider(api_key="test-key")
        data = provider._prepare_request_data(
            "Hello",
            "claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.5,
            custom_param="value",
        )

        expected = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1000,
            "temperature": 0.5,
            "custom_param": "value",
        }

        assert data == expected

    def test_prepare_request_data_unsupported_model(self):
        """Test preparing request data with unsupported model."""
        provider = AnthropicProvider(api_key="test-key")

        with pytest.raises(
            ModelNotFoundError, match="Model 'unsupported' is not supported"
        ):
            provider._prepare_request_data("Hello", "unsupported")

    def test_parse_response_success(self, mock_anthropic_response):
        """Test parsing successful Anthropic response."""
        provider = AnthropicProvider(api_key="test-key")
        response = provider._parse_response(
            mock_anthropic_response, "claude-3-haiku-20240307"
        )

        assert isinstance(response, CompletionResponse)
        assert response.content == "This is a test response from Anthropic"
        assert response.model == "claude-3-haiku-20240307"
        assert response.provider == "anthropic"
        assert response.usage["input_tokens"] == 25
        assert response.metadata["id"] == "msg_test123"

    def test_parse_response_legacy_format(self):
        """Test parsing legacy completion format."""
        provider = AnthropicProvider(api_key="test-key")
        legacy_response = {
            "completion": "Legacy response format",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        response = provider._parse_response(legacy_response, "claude-2.1")

        assert response.content == "Legacy response format"
        assert response.model == "claude-2.1"

    def test_parse_response_invalid_format(self):
        """Test parsing invalid response format."""
        provider = AnthropicProvider(api_key="test-key")
        invalid_response = {"invalid": "format"}

        with pytest.raises(
            ProviderError, match="Invalid response format from Anthropic"
        ):
            provider._parse_response(invalid_response, "claude-3-haiku-20240307")

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_anthropic_response):
        """Test successful completion."""
        provider = AnthropicProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_anthropic_response

            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            response = await provider.complete(
                "Test prompt",
                "claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.7,
            )

            assert isinstance(response, CompletionResponse)
            assert response.content == "This is a test response from Anthropic"

            # Verify API call
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == "https://api.anthropic.com/v1/messages"

    @pytest.mark.asyncio
    async def test_complete_api_errors(self):
        """Test completion with various API errors."""
        provider = AnthropicProvider(api_key="test-key")

        error_cases = [
            (401, "Invalid Anthropic API key"),
            (429, "Anthropic rate limit exceeded"),
            (400, "Anthropic API error"),
            (500, "Anthropic API error: 500"),
        ]

        for status_code, expected_error in error_cases:
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = "Error message"
                if status_code == 400:
                    mock_response.json.return_value = {
                        "error": {"message": "Bad request details"}
                    }

                mock_client_instance = Mock()
                mock_client_instance.post = AsyncMock(return_value=mock_response)
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock()
                mock_client.return_value = mock_client_instance

                with pytest.raises(ProviderError, match=expected_error):
                    await provider.complete("Test", "claude-3-haiku-20240307")

    @pytest.mark.asyncio
    async def test_complete_timeout(self):
        """Test completion with timeout."""
        provider = AnthropicProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="Anthropic API request timed out"):
                await provider.complete("Test", "claude-3-haiku-20240307", timeout=5)

    @pytest.mark.asyncio
    async def test_complete_request_error(self):
        """Test completion with request error."""
        provider = AnthropicProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="Anthropic API request failed"):
                await provider.complete("Test", "claude-3-haiku-20240307")

    def test_get_model_info(self):
        """Test getting model information."""
        provider = AnthropicProvider(api_key="test-key")

        # Test known model
        info = provider.get_model_info("claude-3-opus-20240229")
        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-3-opus-20240229"
        assert info["supported"] is True
        assert "context_length" in info
        assert "capabilities" in info

        # Test unknown model
        info = provider.get_model_info("unknown-model")
        assert info["supported"] is False

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_anthropic_response):
        """Test successful health check."""
        provider = AnthropicProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_anthropic_response

            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            health = await provider.health_check()
            assert health is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        provider = AnthropicProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(side_effect=Exception("API Error"))
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            health = await provider.health_check()
            assert health is False
