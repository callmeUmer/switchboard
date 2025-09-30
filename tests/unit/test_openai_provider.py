"""Unit tests for OpenAI provider."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from switchboard.providers.openai_provider import OpenAIProvider
from switchboard.providers.base import CompletionResponse
from switchboard.exceptions import ProviderError, ModelNotFoundError


class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    def test_openai_provider_creation(self):
        """Test creating OpenAI provider."""
        provider = OpenAIProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.name == "openai"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.organization is None

    def test_openai_provider_with_config(self):
        """Test creating OpenAI provider with custom config."""
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://custom.openai.com/v1",
            organization="org-123"
        )

        assert provider.base_url == "https://custom.openai.com/v1"
        assert provider.organization == "org-123"

    def test_supported_models(self):
        """Test OpenAI supported models."""
        provider = OpenAIProvider(api_key="test-key")
        models = provider.supported_models

        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "gpt-4-turbo" in models
        assert len(models) > 5

    def test_requires_api_key(self):
        """Test that OpenAI provider requires API key."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.requires_api_key() is True

    def test_no_api_key_error(self):
        """Test error when no API key provided."""
        with pytest.raises(ProviderError, match="openai provider requires an API key"):
            OpenAIProvider()

    def test_get_headers(self):
        """Test getting request headers."""
        provider = OpenAIProvider(api_key="test-key")
        headers = provider._get_headers()

        expected = {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }

        assert headers == expected

    def test_get_headers_with_organization(self):
        """Test getting headers with organization."""
        provider = OpenAIProvider(api_key="test-key", organization="org-123")
        headers = provider._get_headers()

        assert headers["OpenAI-Organization"] == "org-123"

    def test_prepare_request_data_basic(self):
        """Test preparing basic request data."""
        provider = OpenAIProvider(api_key="test-key")
        data = provider._prepare_request_data("Hello", "gpt-3.5-turbo")

        expected = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        assert data == expected

    def test_prepare_request_data_with_params(self):
        """Test preparing request data with parameters."""
        provider = OpenAIProvider(api_key="test-key")
        data = provider._prepare_request_data(
            "Hello",
            "gpt-4",
            max_tokens=100,
            temperature=0.5,
            custom_param="value"
        )

        expected = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.5,
            "custom_param": "value"
        }

        assert data == expected

    def test_prepare_request_data_unsupported_model(self):
        """Test preparing request data with unsupported model."""
        provider = OpenAIProvider(api_key="test-key")

        with pytest.raises(ModelNotFoundError, match="Model 'unsupported' is not supported"):
            provider._prepare_request_data("Hello", "unsupported")

    def test_parse_response_success(self, mock_openai_response):
        """Test parsing successful OpenAI response."""
        provider = OpenAIProvider(api_key="test-key")
        response = provider._parse_response(mock_openai_response, "gpt-3.5-turbo")

        assert isinstance(response, CompletionResponse)
        assert response.content == "This is a test response from OpenAI"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 30
        assert response.metadata["id"] == "chatcmpl-test123"

    def test_parse_response_invalid_format(self):
        """Test parsing invalid response format."""
        provider = OpenAIProvider(api_key="test-key")
        invalid_response = {"invalid": "format"}

        with pytest.raises(ProviderError, match="Invalid response format from OpenAI"):
            provider._parse_response(invalid_response, "gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_openai_response):
        """Test successful completion."""
        provider = OpenAIProvider(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_response

            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            response = await provider.complete(
                "Test prompt",
                "gpt-3.5-turbo",
                max_tokens=100,
                temperature=0.7
            )

            assert isinstance(response, CompletionResponse)
            assert response.content == "This is a test response from OpenAI"

            # Verify API call
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_complete_api_errors(self):
        """Test completion with various API errors."""
        provider = OpenAIProvider(api_key="test-key")

        error_cases = [
            (401, "Invalid OpenAI API key"),
            (429, "OpenAI rate limit exceeded"),
            (400, "OpenAI API error"),
            (500, "OpenAI API error: 500")
        ]

        for status_code, expected_error in error_cases:
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = "Error message"
                if status_code == 400:
                    mock_response.json.return_value = {
                        "error": {"message": "Bad request details"}
                    }

                mock_client_instance = Mock()
                mock_client_instance.post = AsyncMock(return_value=mock_response)
                mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                mock_client_instance.__aexit__ = AsyncMock()
                mock_client.return_value = mock_client_instance

                with pytest.raises(ProviderError, match=expected_error):
                    await provider.complete("Test", "gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_complete_timeout(self):
        """Test completion with timeout."""
        provider = OpenAIProvider(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="OpenAI API request timed out"):
                await provider.complete("Test", "gpt-3.5-turbo", timeout=5)

    @pytest.mark.asyncio
    async def test_complete_request_error(self):
        """Test completion with request error."""
        provider = OpenAIProvider(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(side_effect=httpx.RequestError("Network error"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="OpenAI API request failed"):
                await provider.complete("Test", "gpt-3.5-turbo")

    def test_get_model_info(self):
        """Test getting model information."""
        provider = OpenAIProvider(api_key="test-key")

        # Test known model
        info = provider.get_model_info("gpt-4")
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["supported"] is True
        assert "context_length" in info

        # Test unknown model
        info = provider.get_model_info("unknown-model")
        assert info["supported"] is False

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_openai_response):
        """Test successful health check."""
        provider = OpenAIProvider(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_response

            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            health = await provider.health_check()
            assert health is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        provider = OpenAIProvider(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(side_effect=Exception("API Error"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            health = await provider.health_check()
            assert health is False