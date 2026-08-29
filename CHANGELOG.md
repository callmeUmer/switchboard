# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-29

### Security
- `base_url` in `extra_params` must now use https; plain http is allowed only
  for loopback hosts or with an explicit `allow_http: true` opt-in. Previously
  a config file could silently redirect API keys to an arbitrary or plaintext
  endpoint.
- `extra_params` keys are now allowlisted per provider; unknown keys raise
  `ConfigurationError`.
- Upstream error-response bodies are collapsed and truncated (~200 chars)
  before being logged or embedded in exception messages, preventing log
  injection and prompt echo through error trackers.
- `.env` is now loaded from the caller's working directory instead of being
  resolved relative to the installed package (where it was never found for
  normal installs).
- Provider registration no longer instantiates provider classes with a dummy
  API key at import time (removed SDK side effects and spurious key-format
  warnings on `import switchboard`).
- Registry cache keys use full SHA-256 digests (previously truncated to 64
  bits) and canonical JSON serialization of provider config.
- Raised dependency floors past known CVEs: `setuptools>=70` (CVE-2024-6345),
  `pydantic>=2.4.2` (CVE-2024-3772), `black>=24.3.0` (CVE-2024-21503);
  `anthropic>=0.39.0` (older floors could not run the models API). Added
  pip-audit to CI and a Dependabot config.
- CI: added `permissions: contents: read`, removed `|| true` and the `-k`
  filter so the full test suite actually gates merges; added lint and
  dependency-audit jobs.

### Fixed
- Fallback chains with duplicate model names no longer abort before trying any
  fallback; chains are de-duplicated preserving order.
- `complete()`/`complete_async()` no longer require the primary model's API
  key when a fallback would be used (removed dead pre-computation).
- Provider failures during fallback are distinguished from configuration
  errors: `ConfigurationError`/`APIKeyError`/`ProviderNotFoundError` now
  propagate immediately instead of being flattened into
  `FallbackExhaustedError`.
- `complete_sync` and `health_check` no longer risk calling `asyncio.run`
  inside a running event loop on error paths; the thread-join timeout now has
  a margin over the HTTP timeout.
- Anthropic provider raises a clean `ProviderError` when the API key is
  missing instead of sending a `None` header.
- Anthropic responses with neither `content` nor `completion` now raise
  `ModelResponseError` instead of returning empty content.
- pytest configuration consolidated into `pyproject.toml` (the old
  `pytest.ini` used an invalid section header and was silently ignored).
- Test suite fixes: async context-manager mocks no longer suppress exceptions,
  no live network calls from unit tests, assertions updated to actual
  behavior.

### Changed
- **Breaking:** Python 3.10+ required (3.8/3.9 are EOL).
- **Breaking:** unknown `task=` names raise `ConfigurationError` instead of
  silently using the default model.
- **Breaking:** unknown `extra_params` keys raise `ConfigurationError`.
- **Breaking:** `enable_caching`/`cache_ttl` config fields removed (they were
  never implemented; existing configs containing them still load, the values
  are ignored).
- Missing API-key environment variables raise `APIKeyError` (a
  `ConfigurationError` subclass).
- Anthropic provider no longer pre-validates model names client-side; the API
  is the authority, so newly released models work without a library update
  (matching the OpenAI provider).
- `FallbackExhaustedError` message now lists the attempted models.
- Version is single-sourced from `switchboard/__version__.py`.

### Removed
- References to the unimplemented `local` provider, `SWITCHBOARD_ENV`, and
  `SWITCHBOARD_CONFIG_PATH` from docs, examples, and `.env.example`. These
  move to the README backlog.

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
- Mock provider system for testing without API calls
- pytest-based test suite with fixtures

## [0.1.0a1] - 2025-10-22

### Added
- Alpha release with core functionality
- Initial package structure and setup

---

[0.2.0]: https://github.com/callmeumer/switchboard/releases/tag/v0.2.0
[0.1.0]: https://github.com/callmeumer/switchboard/releases/tag/v0.1.0
[0.1.0a1]: https://github.com/callmeumer/switchboard/releases/tag/v0.1.0a1
