# Providers Module

## Purpose and Public Surface

The providers module implements a **strategy pattern** for LLM (Large Language Model) integration, allowing the Dataset Generator to work with different model providers through a consistent interface. It provides:

- **Abstract Provider Interface**: Common contract for all LLM providers
- **Factory Pattern**: Dynamic provider selection based on configuration
- **Ollama Implementation**: Production-ready integration with Ollama GPU service
- **Extensible Architecture**: Easy addition of new providers (OpenAI, Anthropic, etc.)

### Public API

| Component | Location | Purpose |
|-----------|----------|---------|
| `ModelProvider` | [`src/core/providers/base.py`](../../src/core/providers/base.py) | Abstract base class defining provider interface |
| `get_provider()` | [`src/core/providers/factory.py`](../../src/core/providers/factory.py) | Factory function for provider instantiation |
| `OllamaProvider` | [`src/core/providers/ollama.py`](../../src/core/providers/ollama.py) | Ollama implementation with GPU support |

## Key Classes and Functions

### ModelProvider (Abstract Base Class)

**Location**: [`src/core/providers/base.py`](../../src/core/providers/base.py)

**Purpose**: Defines the contract that all LLM providers must implement.

```python
class ModelProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod  
    def health_check(self) -> bool:
        """Check if the model provider is available"""
        pass
```

**Used by**:
- [`src/core/data_generator.py`](../../src/core/data_generator.py) - Core generation pipeline
- [`src/core/providers/factory.py`](../../src/core/providers/factory.py) - Return type for factory

### get_provider() Factory Function

**Location**: [`src/core/providers/factory.py`](../../src/core/providers/factory.py)

**Purpose**: Creates provider instances based on configuration or environment variables.

```python
def get_provider(config: Dict[str, Any] = None) -> ModelProvider
```

**Configuration Sources** (in precedence order):
1. Explicit `config` parameter
2. Environment variables (`PROVIDER_NAME`, `PROVIDER_API_URL`, `MODEL_NAME`)
3. Default to "ollama" provider

**Used by**:
- [`src/core/data_generator.py`](../../src/core/data_generator.py) - Initialize model provider
- [`src/api/routes.py`](../../src/api/routes.py) - Dependency injection

### OllamaProvider Implementation

**Location**: [`src/core/providers/ollama.py`](../../src/core/providers/ollama.py)

**Purpose**: Production implementation for Ollama GPU service integration.

**Features**:
- **Retry Logic**: Configurable retries with exponential backoff
- **Chat API**: Uses modern Ollama chat endpoint (`/api/chat`)
- **Error Handling**: Comprehensive timeout and connection error handling
- **Health Monitoring**: Service availability checking

**Configuration Parameters**:
```python
{
    "api_url": "http://ollama:11434",      # Service endpoint
    "model_name": "gemma3:1b-it-qat",      # Model identifier  
    "timeout": 60,                         # Request timeout (seconds)
    "max_retries": 3,                      # Retry attempts
    "retry_delay": 5                       # Delay between retries (seconds)
}
```

## Inputs, Outputs, and Side Effects

### Inputs

**Configuration Sources**:
- **Explicit Config**: Dictionary passed to `get_provider(config)`
- **Environment Variables**: 
  - `PROVIDER_NAME` - Provider type (default: "ollama")
  - `PROVIDER_API_URL` - LLM service endpoint
  - `MODEL_NAME` - Model identifier  
  - `PROVIDER_TIMEOUT` - Request timeout
  - `PROVIDER_MAX_RETRIES` - Maximum retry attempts
  - `PROVIDER_RETRY_DELAY` - Delay between retries
- **Base Configuration**: [`config/config.yaml`](../../config/config.yaml)

**Generation Parameters**:
```python
# Prompt input
prompt = "Generate Estonian FAQ entries about banking services"

# Generation options
options = {
    "temperature": 0.7,        # Randomness (0.0-1.0)
    "num_predict": 4096,       # Maximum tokens to generate
}
```

### Outputs

**Text Generation**:
```python
# Raw text output from LLM
generated_text: str = provider.generate(prompt, options)

# Health check result  
is_healthy: bool = provider.health_check()
```

### Side Effects

#### Network Operations
- **HTTP Requests**: API calls to LLM service (Ollama at `:11434`)
- **Connection Pooling**: Reuses connections for multiple requests
- **Retry Logic**: Multiple attempts on failure with delays

#### Logging
- **Request Logging**: API call details and timing
- **Error Logging**: Failed requests and retry attempts  
- **Health Monitoring**: Service availability status
- **Log Location**: [`logs/synthetic_data_service.log`](../../logs/)

#### Memory Usage
- **Model Loading**: Ollama service loads models on first use
- **Response Caching**: No internal caching (handled by Ollama)
- **Connection Overhead**: HTTP client connection management

## Example Usage

### Basic Provider Creation

```python
from src.core.providers.factory import get_provider

# Create provider with defaults (uses environment variables)
provider = get_provider()

# Create provider with explicit configuration  
config = {
    "name": "ollama",
    "api_url": "http://ollama:11434",
    "model_name": "gemma3:1b-it-qat",
    "timeout": 60
}
provider = get_provider(config)
```

### Text Generation

```python
# Simple generation
prompt = "Generate a question about Estonian banking in Estonian language"
response = provider.generate(prompt)

# Generation with parameters
options = {
    "temperature": 0.8,    # Higher creativity
    "num_predict": 2048    # Shorter responses
}
response = provider.generate(prompt, options)
```

### Health Monitoring

```python
# Check service availability
if provider.health_check():
    print("LLM service is available")
    response = provider.generate(prompt)
else:
    print("LLM service is unavailable")
```

### Integration with Data Generator

**From** [`src/core/data_generator.py`](../../src/core/data_generator.py):

```python
class DataGenerator:
    def __init__(self, config=None):
        # Provider initialization
        provider_config = self.config.get("provider", {})
        self.model_provider = get_provider(provider_config)
        
    def generate(self, structure_name, prompt_template_name, **kwargs):
        # Use provider for text generation
        generated_content = self.model_provider.generate(
            rendered_prompt, 
            generation_options
        )
```

### Configuration Examples

**Environment Variables** (Docker):
```bash
# In docker-compose.yml
environment:
  - PROVIDER_NAME=ollama
  - PROVIDER_API_URL=http://ollama:11434  
  - MODEL_NAME=gemma3:1b-it-qat
  - PROVIDER_TIMEOUT=60
```

**Configuration File** ([`config/config.yaml`](../../config/config.yaml)):
```yaml
provider:
  name: "ollama"
  model_name: "gemma3:1b-it-qat"
  api_url: "http://ollama:11434"
  timeout: 60
  max_retries: 3
  retry_delay: 5
```

## Error Handling

### Network Errors
```python
# Connection timeout
RuntimeError: "Failed to generate text after 3 attempts"

# Service unavailable  
requests.ConnectionError: "Connection refused to http://ollama:11434"
```

### Configuration Errors
```python
# Invalid provider name
# Logs warning and defaults to "ollama"
logger.warning("Unknown provider: invalid_provider. Defaulting to ollama.")
```

### Model Errors
```python
# Model not found
requests.HTTPError: "404 Client Error: model 'invalid_model' not found"

# Generation limits exceeded
requests.HTTPError: "400 Client Error: prompt too long"
```

## Cross-Links and Integration Points

### Core Components
- **Data Generator**: [`src/core/data_generator.py`](../../src/core/data_generator.py) - Primary consumer
- **API Routes**: [`src/api/routes.py`](../../src/api/routes.py) - Health checks and dependency injection
- **Configuration**: [`config/config.yaml`](../../config/config.yaml) - Provider settings

### Docker Integration  
- **Service Container**: [`Dockerfile.service`](../../Dockerfile.service) - API service with provider access
- **Ollama Container**: [`Dockerfile.ollama-gpu`](../../Dockerfile.ollama-gpu) - GPU-enabled LLM service
- **Compose File**: [`docker-compose.yml`](../../docker-compose.yml) - Service orchestration

### Monitoring and Logging
- **Logger**: [`src/utils/logger.py`](../../src/utils/logger.py) - Structured logging
- **Health Endpoint**: `GET /health` - Provider status reporting  
- **MLflow Tracking**: [`src/utils/mlflow_tracking.py`](../../src/utils/mlflow_tracking.py) - Generation metrics

### Configuration Files
- **Base Config**: [`config/config.yaml`](../../config/config.yaml) - Provider configuration
- **Model Config**: [`config/model_config.yaml`](../../config/model_config.yaml) - Model-specific settings

## Extending with New Providers

### Adding a New Provider

1. **Create Provider Class**:
```python
# src/core/providers/openai.py
from src.core.providers.base import ModelProvider

class OpenAIProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize OpenAI client
        pass
        
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        # Implement OpenAI API call
        pass
        
    def health_check(self) -> bool:
        # Check OpenAI API availability
        pass
```

2. **Register in Factory** ([`src/core/providers/factory.py`](../../src/core/providers/factory.py)):
```python
from src.core.providers.openai import OpenAIProvider

providers = {
    "ollama": lambda cfg: OllamaProvider(cfg),
    "openai": lambda cfg: OpenAIProvider(cfg),  # Add new provider
}
```

3. **Update Configuration**:
```yaml
# config/config.yaml
provider:
  name: "openai"
  api_key: "${OPENAI_API_KEY}"
  model_name: "gpt-4"
```

This extensible architecture ensures the Dataset Generator can integrate with any LLM provider while maintaining consistent behavior across the application.
