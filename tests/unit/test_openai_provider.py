"""Unit tests for OpenAI provider."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from switchboard.exceptions import ProviderError
from switchboard.providers.base import CompletionResponse
from switchboard.providers.openai_provider import OpenAIProvider


class NotFound(Exception):
    """Stand-in for an SDK 404 error (SDK errors expose ``status_code``)."""

    status_code = 404


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
            organization="org-123",
        )

        assert provider.base_url == "https://custom.openai.com/v1"
        assert provider.organization == "org-123"

    def test_supported_models(self):
        """Test the full model list is fetched live on every access."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.list.side_effect = lambda: iter(
            [Mock(id="gpt-4"), Mock(id="gpt-3.5-turbo")]
        )

        assert provider.supported_models == ["gpt-3.5-turbo", "gpt-4"]

        # Nothing depends on the list implicitly, so it is never cached
        provider.supported_models
        assert provider._client.models.list.call_count == 2

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
            "Content-Type": "application/json",
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
            "messages": [{"role": "user", "content": "Hello"}],
        }

        assert data == expected

    def test_prepare_request_data_with_params(self):
        """Test preparing request data with parameters."""
        provider = OpenAIProvider(api_key="test-key")
        data = provider._prepare_request_data(
            "Hello", "gpt-4", max_tokens=100, temperature=0.5, custom_param="value"
        )

        expected = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 100,
            "temperature": 0.5,
            "custom_param": "value",
        }

        assert data == expected

    def test_prepare_request_data_custom_base_url_keeps_max_tokens(self):
        """Test OpenAI-compatible servers still get the legacy max_tokens."""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://llm.example.com/v1"
        )
        data = provider._prepare_request_data("Hello", "local-model", max_tokens=100)

        assert data["max_tokens"] == 100
        assert "max_completion_tokens" not in data

    def test_prepare_request_data_unknown_model(self):
        """Test that unknown models are not rejected client-side.

        Model validation is left to the OpenAI API so newly released models
        work without a library update.
        """
        provider = OpenAIProvider(api_key="test-key")
        data = provider._prepare_request_data("Hello", "some-brand-new-model")

        assert data["model"] == "some-brand-new-model"

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

    def test_parse_response_null_content(self, mock_openai_response):
        """Test a reasoning model's null content parses as an empty string."""
        provider = OpenAIProvider(api_key="test-key")
        mock_openai_response["choices"][0]["message"]["content"] = None
        mock_openai_response["choices"][0]["finish_reason"] = "length"

        response = provider._parse_response(mock_openai_response, "gpt-6-luna")

        assert response.content == ""
        assert response.metadata["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_openai_response):
        """Test successful completion."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_response

            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            response = await provider.complete(
                "Test prompt", "gpt-3.5-turbo", max_tokens=100, temperature=0.7
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
            (500, "OpenAI API error: 500"),
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
                mock_client_instance.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value = mock_client_instance

                with pytest.raises(ProviderError, match=expected_error):
                    await provider.complete("Test", "gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_complete_timeout(self):
        """Test completion with timeout."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="OpenAI API request timed out"):
                await provider.complete("Test", "gpt-3.5-turbo", timeout=5)

    @pytest.mark.asyncio
    async def test_complete_request_error(self):
        """Test completion with request error."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="OpenAI API request failed"):
                await provider.complete("Test", "gpt-3.5-turbo")

    def test_get_model_info(self):
        """Test one single-model lookup gives both support and details."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.retrieve.return_value = Mock(
            id="gpt-6-luna", created=1700000000, owned_by="openai"
        )

        info = provider.get_model_info("gpt-6-luna")

        assert info == {
            "provider": "openai",
            "model": "gpt-6-luna",
            "supported": True,
            "id": "gpt-6-luna",
            "created": 1700000000,
            "owned_by": "openai",
        }
        provider._client.models.retrieve.assert_called_once_with("gpt-6-luna")
        provider._client.models.list.assert_not_called()

    def test_get_model_info_unknown_model(self):
        """Test a 404 from the lookup means the model is not supported."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.retrieve.side_effect = NotFound("no such model")

        info = provider.get_model_info("unknown-model")

        assert info == {
            "provider": "openai",
            "model": "unknown-model",
            "supported": False,
        }

    def test_get_model_info_lookup_failure(self):
        """Test a failed lookup raises instead of claiming "not supported"."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.retrieve.side_effect = Exception("network down")

        with pytest.raises(ProviderError, match="Failed to look up OpenAI model"):
            provider.get_model_info("gpt-6-luna")

    def test_is_model_supported_uses_single_lookup(self):
        """Test support checks ask about the one model, not the full list."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()

        assert provider.is_model_supported("gpt-6-luna") is True
        provider._client.models.retrieve.assert_called_once_with("gpt-6-luna")
        provider._client.models.list.assert_not_called()

    def test_custom_base_url_checks_full_list(self):
        """Test OpenAI-compatible servers are checked against the list.

        They may not implement GET /models/{id}, where a 404 would wrongly
        read as "no such model".
        """
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://llm.example.com/v1"
        )
        provider._client = Mock()
        provider._client.models.list.side_effect = lambda: iter([Mock(id="local")])

        assert provider.is_model_supported("local") is True
        assert provider.get_model_info("other")["supported"] is False
        provider._client.models.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_openai_response):
        """Test health check probes the requested model."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_response

            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            health = await provider.health_check("gpt-3.5-turbo")
            assert health is True

            request_json = mock_client_instance.post.call_args[1]["json"]
            assert request_json["model"] == "gpt-3.5-turbo"
            assert request_json["max_completion_tokens"] == 1

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.post = AsyncMock(side_effect=Exception("API Error"))
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            health = await provider.health_check("gpt-3.5-turbo")
            assert health is False

    @pytest.mark.asyncio
    async def test_health_check_without_model_lists_models(self):
        """Test health check with no model hits the models endpoint only."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.list.return_value = iter([Mock(id="some-model")])

        with patch("httpx.AsyncClient") as mock_client:
            health = await provider.health_check()

            assert health is True
            mock_client.assert_not_called()  # no completion, no tokens spent

    @pytest.mark.asyncio
    async def test_health_check_without_model_failure(self):
        """Test health check with no model fails when discovery fails."""
        provider = OpenAIProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.list.side_effect = Exception("API Error")

        health = await provider.health_check()
        assert health is False
