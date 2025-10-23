# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-23

### Added
- Initial stable release of Switchboard AI
- Core `Client` class with `complete()` and `complete_async()` methods
- Configuration system with YAML-based model and task definitions
- Provider support for OpenAI and Anthropic
- Dynamic model discovery from provider APIs
- Task-based routing for automatic model selection
- Comprehensive fallback chain system with automatic failover
- Health check functionality for models and providers
- Environment-aware configuration management
- Full type hints and Pydantic validation
- Comprehensive exception hierarchy for error handling
- Support for model-specific parameters (temperature, max_tokens, timeout)
- API key management via environment variables
- Configuration reload capability
- Model information and listing methods
- Example configurations for development and production environments

### Features
- **Unified API**: Single interface for multiple AI providers
- **Task-based Routing**: Automatically select models based on task type
- **Fallback Chains**: Automatic failover when primary models are unavailable
- **Dynamic Discovery**: Automatically fetches available models from provider APIs
- **Type Safety**: Full type hints with Pydantic validation
- **Async Support**: Both sync and async completion methods
- **Health Checks**: Monitor model and provider availability
- **Configuration Management**: Easy YAML-based configuration

### Documentation
- Comprehensive README with quickstart guide
- Example configurations for dev and prod environments
- API documentation with type hints
- Test architecture documentation

### Testing
- Unit tests for all core components
- Integration tests for full workflows
- Mock provider system for testing without API calls
- pytest-based test suite with fixtures

## [0.1.0a1] - 2025-10-22

### Added
- Alpha release with core functionality
- Initial package structure and setup

---

[0.1.0]: https://github.com/callmeumer/switchboard/releases/tag/v0.1.0
[0.1.0a1]: https://github.com/callmeumer/switchboard/releases/tag/v0.1.0a1
