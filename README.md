# Switchboard

> Config-driven AI model switching made simple

Switchboard is a Python library that provides a unified API for switching between different AI models and providers with built-in fallback support. Configure once, switch seamlessly.

## Features

- **Unified API** - Single interface for OpenAI, Anthropic, and more
- **Task-based routing** - Automatically select models based on task type
- **Fallback chains** - Automatic failover when primary models are unavailable
- **Configuration-driven** - YAML-based configuration for easy management
- **Environment-aware** - Support for development, staging, and production configs
- **Type-safe** - Full type hints and Pydantic validation

## Quick Start

### Installation

```bash
pip install switchboard
```

### Basic Usage

1. Create a configuration file `switchboard.yaml`:

```yaml
models:
  gpt-4:
    provider: openai
    model_name: gpt-4
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7

  claude-3:
    provider: anthropic
    model_name: claude-3-sonnet-20240229
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 4096
    temperature: 0.7

tasks:
  coding:
    primary_model: gpt-4
    fallback_models: [claude-3]
    description: Code generation and programming

default_model: gpt-4
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
response = client.complete("Hello", model="claude-3")
```

## Configuration

### Models

Define available models with their provider configurations:

```yaml
models:
  model-name:
    provider: openai|anthropic|local
    model_name: actual-model-id
    api_key_env: ENV_VAR_NAME
    max_tokens: 4096
    temperature: 0.7
    timeout: 30
    extra_params:
      custom_param: value
```

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

```python
# Check specific model
health = client.health_check("gpt-4")
print(f"GPT-4 healthy: {health['gpt-4']}")

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
info = client.get_model_info("gpt-4")
print(f"Context length: {info['context_length']}")
```

### Configuration Management

```python
# Reload configuration
client.reload_config()

# List configured tasks
tasks = client.list_tasks()
print(f"Available tasks: {tasks}")
```

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