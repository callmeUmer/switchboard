"""AI model providers for Switchboard."""

from .base import BaseProvider, CompletionResponse, ProviderError
from .registry import ProviderRegistry, register_provider, get_provider

# Import providers to auto-register them
try:
    from .openai_provider import OpenAIProvider
    register_provider(OpenAIProvider)
except ImportError:
    # OpenAI not available - that's okay
    pass

try:
    from .anthropic_provider import AnthropicProvider
    register_provider(AnthropicProvider)
except ImportError:
    # Anthropic not available - that's okay
    pass

__all__ = [
    "BaseProvider",
    "CompletionResponse",
    "ProviderError",
    "ProviderRegistry",
    "register_provider",
    "get_provider",
]