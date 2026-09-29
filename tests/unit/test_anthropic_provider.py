"""Unit tests for Anthropic provider."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from switchboard.exceptions import ProviderError
from switchboard.providers.anthropic_provider import AnthropicProvider
from switchboard.providers.base import CompletionResponse


class NotFound(Exception):
    """Stand-in for an SDK 404 error (SDK errors expose ``status_code``)."""

    status_code = 404


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
        """Test the full model list (all pages) is fetched live."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        # Iterating the SDK page walks every page, not just .data
        provider._client.models.list.side_effect = lambda: iter(
            [Mock(id="model-b"), Mock(id="model-a"), Mock(id="model-c")]
        )

        assert provider.supported_models == ["model-a", "model-b", "model-c"]

        # Nothing depends on the list implicitly, so it is never cached
        provider.supported_models
        assert provider._client.models.list.call_count == 2

    def test_supported_models_without_client(self):
        """Test there is no hardcoded fallback list when discovery is unavailable."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = None

        with pytest.raises(ProviderError, match="Anthropic library not available"):
            provider.supported_models

    def test_supported_models_fetch_failure(self):
        """Test API errors during discovery surface as ProviderError."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.list.side_effect = Exception("boom")

        with pytest.raises(ProviderError, match="Failed to fetch models"):
            provider.supported_models

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

    def test_prepare_request_data_unknown_model(self):
        """Test that unknown models are not rejected client-side.

        Model validation is left to the Anthropic API so newly released
        models work without a library update.
        """
        provider = AnthropicProvider(api_key="test-key")
        data = provider._prepare_request_data("Hello", "some-brand-new-model")

        assert data["model"] == "some-brand-new-model"

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

    def test_parse_response_skips_thinking_blocks(self, mock_anthropic_response):
        """Test text is taken from text blocks, not a leading thinking block."""
        provider = AnthropicProvider(api_key="test-key")
        mock_anthropic_response["content"] = [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]

        response = provider._parse_response(mock_anthropic_response, "claude-opus-5-5")

        assert response.content == "Hello world"

    def test_parse_response_no_text_blocks(self, mock_anthropic_response):
        """Test a response with no text (e.g. max_tokens hit) parses as empty."""
        provider = AnthropicProvider(api_key="test-key")
        mock_anthropic_response["content"] = []

        response = provider._parse_response(mock_anthropic_response, "claude-opus-5-5")

        assert response.content == ""

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
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
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
                mock_client_instance.__aexit__ = AsyncMock(return_value=None)
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
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
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
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ProviderError, match="Anthropic API request failed"):
                await provider.complete("Test", "claude-3-haiku-20240307")

    def test_get_model_info(self):
        """Test one single-model lookup gives both support and details."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.retrieve.return_value = Mock(
            id="claude-opus-5-5",
            display_name="Claude Opus 5.5",
            created_at="2026-08-01T00:00:00Z",
        )

        info = provider.get_model_info("claude-opus-5-5")

        assert info == {
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "supported": True,
            "id": "claude-opus-5-5",
            "display_name": "Claude Opus 5.5",
            "created_at": "2026-08-01T00:00:00Z",
        }
        provider._client.models.retrieve.assert_called_once_with("claude-opus-5-5")
        provider._client.models.list.assert_not_called()

    def test_get_model_info_unknown_model(self):
        """Test a 404 from the lookup means the model is not supported."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.retrieve.side_effect = NotFound("no such model")

        info = provider.get_model_info("unknown-model")

        assert info == {
            "provider": "anthropic",
            "model": "unknown-model",
            "supported": False,
        }

    def test_get_model_info_lookup_failure(self):
        """Test a failed lookup raises instead of claiming "not supported"."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.retrieve.side_effect = Exception("network down")

        with pytest.raises(ProviderError, match="Failed to look up Anthropic model"):
            provider.get_model_info("claude-opus-5-5")

    def test_get_model_info_without_client(self):
        """Test there is no hardcoded answer when the SDK is unavailable."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = None

        with pytest.raises(ProviderError, match="Anthropic library not available"):
            provider.get_model_info("claude-opus-5-5")

    def test_is_model_supported_uses_single_lookup(self):
        """Test support checks ask about the one model, not the full list."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()

        assert provider.is_model_supported("claude-opus-5-5") is True
        provider._client.models.retrieve.assert_called_once_with("claude-opus-5-5")
        provider._client.models.list.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_anthropic_response):
        """Test health check probes the requested model."""
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
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            health = await provider.health_check("claude-3-haiku-20240307")
            assert health is True

            request_json = mock_client_instance.post.call_args[1]["json"]
            assert request_json["model"] == "claude-3-haiku-20240307"
            assert request_json["max_tokens"] == 1

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
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            health = await provider.health_check("claude-3-haiku-20240307")
            assert health is False

    @pytest.mark.asyncio
    async def test_health_check_without_model_lists_models(self):
        """Test health check with no model hits the models endpoint only."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.list.return_value = iter([Mock(id="some-model")])

        with patch("httpx.AsyncClient") as mock_client:
            health = await provider.health_check()

            assert health is True
            mock_client.assert_not_called()  # no completion, no tokens spent

    @pytest.mark.asyncio
    async def test_health_check_without_model_failure(self):
        """Test health check with no model fails when discovery fails."""
        provider = AnthropicProvider(api_key="test-key")
        provider._client = Mock()
        provider._client.models.list.side_effect = Exception("API Error")

        health = await provider.health_check()
        assert health is False
