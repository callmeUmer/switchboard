# Switchboard

> Config-driven AI model switching made simple

Switchboard is a Python library that provides a unified API for switching between different AI models and providers with built-in fallback support. Configure once, switch seamlessly.

## Features

- **Unified API** - Single interface for OpenAI, Anthropic, and more
- **Task-based routing** - Automatically select models based on task type
- **Fallback chains** - Automatic failover when primary models are unavailable
- **Dynamic model discovery** - Automatically fetches available models from provider APIs
- **Configuration-driven** - YAML-based configuration for easy management
- **Type-safe** - Type hints and Pydantic validation

Requires Python 3.10+.

## Quick Start

### Installation

```bash
# Base installation
pip install switchboard-ai

# With OpenAI support
pip install switchboard-ai[openai]

# With Anthropic support
pip install switchboard-ai[anthropic]

# With all providers
pip install switchboard-ai[all]
```

### Basic Usage

1. Create a configuration file `switchboard.yaml`:

```yaml
models:
  openai-smart:
    provider: openai
    model_name: gpt-6-astra
    api_key_env: OPENAI_API_KEY
    max_tokens: 16000

  claude-smart:
    provider: anthropic
    model_name: claude-opus-5-5
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 16000

tasks:
  coding:
    primary_model: openai-smart
    fallback_models: [claude-smart]
    description: Code generation and programming

default_model: openai-smart
```

2. Set your API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

3. Use in your code:

```python
from switchboard import Client

# Initialize client
client = Client()

# Generate completion
response = client.complete("Write a Python function to calculate fibonacci numbers")
print(response.content)

# Use task-based routing
response = client.complete("Write a Python function", task="coding")
print(f"Used model: {response.model}")

# Override model for specific request
response = client.complete("Hello", model="claude-smart")
```

## Configuration

### Configuration discovery

`Client()` without an explicit path looks for a config file in this order:

1. `./switchboard.yaml`
2. `./switchboard.yml`
3. `./config/switchboard.yaml`
4. `./config/switchboard.yml`
5. `~/.switchboard.yaml`
6. `~/.switchboard.yml`

> **Security note:** the current working directory is searched first, so a
> `switchboard.yaml` in an untrusted directory (a cloned repo, an extracted
> archive) controls which endpoints your requests — and API keys — go to.
> When running in directories you don't control, pass an explicit path:
> `Client(config_path="/path/to/switchboard.yaml")`.

### Models

Define available models with their provider configurations:

```yaml
models:
  model-name:
    provider: openai|anthropic
    model_name: actual-model-id
    api_key_env: ENV_VAR_NAME
    max_tokens: 4096
    temperature: 0.7  # optional; omitted from requests when unset
    timeout: 30
    extra_params:
      base_url: https://custom-endpoint.example.com/v1
```

The key under `models:` is the name your code and tasks refer to; `model_name`
is the provider's model ID. Naming entries by role (`claude-smart`,
`openai-fast`) rather than by model ID means upgrading to a newer model is a
one-line `model_name` change.

`temperature` has no default. Newer models (current Claude models, OpenAI
reasoning models) reject non-default sampling parameters, so it is only sent
when you set it. For OpenAI, `max_tokens` is sent as `max_completion_tokens`
to `api.openai.com` (the old name is deprecated and rejected by reasoning
models) and as `max_tokens` to a custom `base_url`.

`extra_params` keys are allowlisted per provider:

- **openai**: `base_url`, `organization`, `allow_http`
- **anthropic**: `base_url`, `anthropic_version`, `allow_http`

Unknown keys are rejected with a `ConfigurationError`. `base_url` must use
**https**; plain `http` is accepted only for loopback hosts (`localhost`,
`127.0.0.1`, `::1`) or with an explicit `allow_http: true` opt-in — otherwise
your API key would cross the network unencrypted. API keys are never stored in
the config file itself; only the *name* of an environment variable
(`api_key_env`) is.

### Model discovery

Switchboard does not ship a list of model names, and it does not keep one up
to date in the background either. Completions never check whether a model
exists; the provider's API does that, so a model released today works as soon
as you put it in your config.

When you ask whether a model exists (`get_model_info()`'s `supported` field),
Switchboard asks the provider about that one model
(`GET /v1/models/{id}`), so the answer is always current. A model confirmed to
exist is remembered for the life of the provider; a missing one is asked about
again next time, in case it has just been released. For OpenAI-compatible
servers behind a custom `base_url`, which may not implement that endpoint,
Switchboard checks the server's full model list instead.

Model lookups use the provider's SDK: `pip install switchboard-ai[openai]` /
`[anthropic]`.

### Tasks

Configure task-based routing with fallback chains:

```yaml
tasks:
  task-name:
    primary_model: model-name
    fallback_models: [backup-model-1, backup-model-2]
    description: "Task description"
```

### Environment Variables

Set API keys in your environment:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Advanced Usage

### Async Support

```python
import asyncio

async def main():
    client = Client()
    response = await client.complete_async("Hello world")
    print(response.content)

asyncio.run(main())
```

### Health Checks

Each check sends a 1-token completion to the model configured for that entry
(its `model_name`), so it verifies that exact model is reachable with your key.

```python
# Check specific model
health = client.health_check("claude-smart")
print(f"Claude healthy: {health['claude-smart']}")

# Check all models
health = client.health_check()
for model, status in health.items():
    print(f"{model}: {'✓' if status else '✗'}")
```

### Model Information

```python
# List available models
models = client.list_models()
print(f"Available models: {models}")

# Get model details
info = client.get_model_info("claude-smart")
print(f"Provider: {info['provider']}, supported: {info['supported']}")
```

### Configuration Management

```python
# Reload configuration
client.reload_config()

# List configured tasks
tasks = client.list_tasks()
print(f"Available tasks: {tasks}")
```

### Fallback and error behavior

- `complete()` tries the resolved model, then its fallback chain in order
  (duplicates are skipped). If every model fails with a provider error, it
  raises `FallbackExhaustedError` naming the attempted models and the last
  error.
- Configuration problems fail loudly instead of being skipped: a missing API
  key for **any** model in the chain raises `APIKeyError` (a
  `ConfigurationError`), and an unknown `task=` name raises
  `ConfigurationError` rather than silently using the default model.

## Roadmap / Backlog

Planned but not yet implemented (contributions welcome):

- Local/self-hosted provider (Ollama, llama.cpp-style servers)
- Response caching
- Environment-based config selection (`SWITCHBOARD_ENV`)
- Retry/backoff on rate limits

## Examples

See the [examples/](examples/) directory for complete configuration examples:

- [`config-dev.yaml`](examples/config-dev.yaml) - Development configuration
- [`config-prod.yaml`](examples/config-prod.yaml) - Production configuration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development

```bash
# Clone repository
git clone https://github.com/callmeumer/switchboard.git
cd switchboard

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black . && flake8 . && mypy switchboard
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📫 Issues: [GitHub Issues](https://github.com/callmeumer/switchboard/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/callmeumer/switchboard/discussions)
- 🐦 Twitter: [@callmeumer](https://twitter.com/callmeumer)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.