# Configuration Module

## Purpose and Public Surface

The configuration module provides a **flexible, layered configuration management system** for the Dataset Generator. It enables seamless configuration loading from multiple sources with clear precedence ordering, environment variable integration, and deep merging capabilities. This module serves as the foundation for all configuration management across the application.

### Core Features

- **Layered Configuration**: File-based configs with environment variable overrides
- **Deep Merging**: Hierarchical configuration composition from multiple sources
- **Environment Integration**: Automatic parsing and mapping of environment variables
- **Path Resolution**: Fallback configuration file discovery with container compatibility
- **Type Conversion**: Automatic parsing of strings to appropriate types (int, float, bool)

### Public API

| Component | Location | Purpose |
|-----------|----------|---------|
| `ConfigLoader` | [`src/core/config.py`](../../src/core/config.py) | Main configuration loading class |
| `app_config` | [`src/core/config.py`](../../src/core/config.py) | Global configuration instance |

## Key Classes and Functions

### ConfigLoader Class

**Location**: [`src/core/config.py`](../../src/core/config.py)

**Purpose**: Centralized configuration management with support for multiple sources and environment variable integration.

```python
class ConfigLoader:
    """
    Configuration loader with support for layered configurations and environment variables.
    
    Precedence order (highest to lowest):
    1. Environment variables
    2. Configuration files (YAML)
    3. Default values
    """
    
    DEFAULT_PATHS = [
        "config/config.yaml",
        "src/config/config.yaml", 
        "/app/config/config.yaml"
    ]
```

### Core Methods

#### `load()` - Configuration Loading

```python
@classmethod
def load(
    cls, 
    paths: Optional[List[str]] = None, 
    env_prefix: str = ""
) -> Dict[str, Any]
```

**Purpose**: Load and merge configuration from multiple sources with fallbacks.

**Parameters**:
- `paths`: Custom configuration file paths (defaults to `DEFAULT_PATHS`)
- `env_prefix`: Optional prefix for environment variables

**Returns**: Merged configuration dictionary

**Used by**:
- [`src/main.py`](../../src/main.py) - Application initialization
- [`src/core/metrics.py`](../../src/core/metrics.py) - Metrics configuration loading
- [`src/core/model_client.py`](../../src/core/model_client.py) - Model client configuration
- [`src/core/storage_manager.py`](../../src/core/storage_manager.py) - Storage path configuration
- [`src/utils/validators.py`](../../src/utils/validators.py) - Validation rule configuration

#### `_deep_merge()` - Configuration Merging

```python
@classmethod
def _deep_merge(cls, base: Dict, overlay: Dict) -> Dict
```

**Purpose**: Recursively merge two configuration dictionaries, with overlay taking precedence.

**Use Case**: Combining base configuration files with user-specific overrides.

#### `_get_env_config()` - Environment Variable Processing

```python
@classmethod
def _get_env_config(cls, prefix: str) -> Dict[str, Any]
```

**Purpose**: Extract and parse configuration from environment variables with automatic type conversion.

**Environment Variable Mappings**:
```python
mappings = {
    # Provider settings
    "PROVIDER_NAME": "provider.name",
    "MODEL_NAME": "provider.model_name", 
    "PROVIDER_API_URL": "provider.api_url",
    "PROVIDER_TIMEOUT": "provider.timeout",
    
    # Directories
    "DATA_DIR": "directories.input",
    "OUTPUT_DIR": "directories.output",
    "TEMPLATES_DIR": "directories.templates",
    "USER_CONFIGS_DIR": "directories.user_configs",
    
    # Generation settings
    "DEFAULT_LANGUAGE": "generation.default_language",
    "DEFAULT_NUM_EXAMPLES": "generation.default_num_examples"
}
```

#### `_set_by_path()` - Dotted Path Configuration

```python
@classmethod
def _set_by_path(cls, config: Dict, path: str, value: Any)
```

**Purpose**: Set nested configuration values using dot notation (e.g., `"provider.api_url"`).

### Global Configuration Instance

**Location**: [`src/core/config.py`](../../src/core/config.py) (bottom of file)

```python
app_config = ConfigLoader.load()
```

**Purpose**: Singleton configuration instance used throughout the application.

**Used by**: All modules requiring configuration access.

## Inputs, Outputs, and Side Effects

### Configuration Sources (Inputs)

#### 1. Configuration Files (YAML)

**Primary Configuration** ([`config/config.yaml`](../../config/config.yaml)):
```yaml
# LLM Provider configuration
provider:
  name: "ollama"
  model_name: "gemma3:1b-it-qat"
  api_url: "http://ollama:11434"
  timeout: 60
  max_retries: 3
  retry_delay: 5

# Directory paths
directories:
  input: "data"
  output: "output_datasets" 
  templates: "templates"
  user_configs: "user_configs"

# Generation settings
generation:
  default_num_examples: 10
  default_language: "et"
  parameters:
    temperature: 0.7
    max_tokens: 4096
```

**Model Configuration** ([`config/model_config.yaml`](../../config/model_config.yaml)):
```yaml
model_name: "gemma3:1b-it-qat"
ollama_host: "http://ollama:11434"
generation_defaults:
  temperature: 0.95
  max_tokens_per_response: 5000
```

**Path Resolution Priority**:
1. `config/config.yaml` (development)
2. `src/config/config.yaml` (alternative source location)
3. `/app/config/config.yaml` (container runtime)

#### 2. Environment Variables

**Container Environment** (from [`docker-compose.yml`](../../docker-compose.yml)):
```bash
# Provider configuration
PROVIDER_NAME=ollama
PROVIDER_API_URL=http://ollama:11434
MODEL_NAME=gemma3:1b-it-qat

# Directory overrides
TEMPLATES_DIR=/app/templates
USER_CONFIGS_DIR=/app/user_configs
OUTPUT_DIR=/app/output_datasets

# Generation parameters
DEFAULT_LANGUAGE=et
DEFAULT_TEMPERATURE=0.7
```

**Type Conversion Logic**:
```python
# Automatic parsing
"60" → 60 (int)
"0.7" → 0.7 (float)  
"true" → True (bool)
"false" → False (bool)
"et" → "et" (string)
```

#### 3. User Configuration Overrides

**User Templates** ([`user_configs/`](../../user_configs/)):
- Dataset structures: `user_configs/dataset_structures/single_question.yaml`
- Custom prompts: `user_configs/prompts/institute_topic_question.txt`

### Configuration Output

**Merged Configuration Dictionary**:
```python
{
    "provider": {
        "name": "ollama",
        "api_url": "http://ollama:11434",
        "model_name": "gemma3:1b-it-qat",
        "timeout": 60
    },
    "directories": {
        "input": "data",
        "output": "output_datasets",
        "templates": "templates",
        "user_configs": "user_configs"
    },
    "generation": {
        "default_language": "et",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 4096
        }
    }
}
```

### Side Effects

#### File System Operations
- **Configuration File Reading**: YAML parsing from multiple potential paths
- **Path Validation**: Directory existence checking during configuration loading
- **Error Logging**: File read errors logged to [`logs/`](../../logs/) directory

#### Logging Operations
```python
# Success logging
logger.info(f"Loading configuration from {path}")

# Error logging  
logger.error(f"Error loading config from {path}: {e}")
```

#### Memory Impact
- **Global Instance**: Single `app_config` instance shared across application
- **Deep Copying**: Configuration merging creates new dictionary instances
- **Caching**: No internal caching; configuration loaded once at startup

## Example Usage

### Basic Configuration Loading

**From Application Startup** ([`src/main.py`](../../src/main.py)):
```python
from src.core.config import ConfigLoader

def main():
    # Load with default paths
    config = ConfigLoader.load()
    
    # Load with custom path  
    config = ConfigLoader.load(paths=["custom_config.yaml"])
    
    # Access nested configuration
    provider_config = config.get("provider", {})
    api_url = provider_config.get("api_url", "http://localhost:11434")
```

### Environment Variable Integration

**Docker Environment**:
```yaml
# docker-compose.yml
environment:
  - PROVIDER_API_URL=http://custom-ollama:11434
  - MODEL_NAME=custom-model:latest
  - DEFAULT_LANGUAGE=en
  - DEFAULT_TEMPERATURE=0.9
```

**Application Code**:
```python
# Environment variables automatically override file config
config = ConfigLoader.load()
print(config["provider"]["api_url"])  # → "http://custom-ollama:11434"
print(config["generation"]["parameters"]["temperature"])  # → 0.9
```

### Configuration Access Patterns

**Global Configuration Usage** ([`src/core/metrics.py`](../../src/core/metrics.py)):
```python
from src.core.config import ConfigLoader

class RelevanceCoverageMetric:
    def __init__(self, embedding_model: str):
        self.config = ConfigLoader.load()
        self.segment_weight = self.config.get("relevance_score", {}).get("segment_weight")
        self.query_weight = self.config.get("relevance_score", {}).get("query_weight")
```

**Storage Manager Configuration** ([`src/core/storage_manager.py`](../../src/core/storage_manager.py)):
```python
from src.core.config import app_config

class StorageManager:
    def __init__(self, config=None):
        self.config = config or app_config or {}
        self.dirs = self.config.get("directories", {})
        self.datasets_base_dir = (
            app_config.DATASETS_DIR 
            if hasattr(app_config, "DATASETS_DIR") 
            else "datasets"
        )
```

**Validation Configuration** ([`src/utils/validators.py`](../../src/utils/validators.py)):
```python
from core.config import app_config

def validate_generation_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    # Use configuration for validation rules
    if "language" in parameters:
        if parameters["language"] not in app_config.SUPPORTED_LANGUAGES:
            parameters["language"] = app_config.DEFAULT_LANGUAGE
```

### Custom Configuration Loading

**With Prefix for Multi-Tenant**:
```python
# Load tenant-specific configuration  
tenant_config = ConfigLoader.load(
    paths=["tenants/agency1/config.yaml"],
    env_prefix="AGENCY1_"
)

# Environment: AGENCY1_PROVIDER_API_URL=http://agency1-ollama:11434
# Results in: config["provider"]["api_url"] = "http://agency1-ollama:11434"
```

### Deep Merge Example

**Base Configuration**:
```yaml
# config/config.yaml
provider:
  name: "ollama"
  timeout: 30
  
generation:
  default_language: "et"
  parameters:
    temperature: 0.7
```

**User Override**:
```yaml
# user_configs/override.yaml  
provider:
  timeout: 60  # Override timeout
  
generation:
  parameters:
    max_tokens: 8192  # Add new parameter
```

**Merged Result**:
```python
{
    "provider": {
        "name": "ollama",        # From base
        "timeout": 60            # From override
    },
    "generation": {
        "default_language": "et", # From base
        "parameters": {
            "temperature": 0.7,   # From base
            "max_tokens": 8192    # From override
        }
    }
}
```

## Error Handling

### Configuration File Errors

```python
# File not found - silently continues to next path
if Path(path).exists():
    # Only process existing files
    
# YAML parsing errors
try:
    file_config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Error loading config from {path}: {e}")
    # Continues processing other files
```

### Environment Variable Errors

```python
# Invalid type conversion - falls back to string
if value.isdigit():
    value = int(value)
elif value.replace(".", "", 1).isdigit():
    value = float(value)
# Invalid values remain as strings
```

### Missing Configuration

```python
# Safe access patterns with defaults
provider_config = config.get("provider", {})
api_url = provider_config.get("api_url", "http://localhost:11434")

# Directory configuration with fallbacks
output_dir = config.get("directories", {}).get("output", "output_datasets")
```

## Cross-Links and Integration Points

### Core Components

**Data Generator Integration** ([`src/core/data_generator.py`](../../src/core/data_generator.py)):
- Provider configuration: `config.get("provider", {})`
- Generation parameters: `config.get("generation", {})`
- Directory paths: `config.get("directories", {})`

**API Integration** ([`src/api/routes.py`](../../src/api/routes.py)):
- Configuration dependency injection via `request.app.state.config`
- Background task configuration access
- Callback URL configuration: `config.get("callback", {})`

**Storage Management** ([`src/core/storage_manager.py`](../../src/core/storage_manager.py)):
- Directory path resolution from configuration
- Base directory fallbacks and defaults

### Configuration Files

**Base Configuration** ([`config/config.yaml`](../../config/config.yaml)):
- Primary application configuration
- Provider settings, directories, generation defaults

**Model Configuration** ([`config/model_config.yaml`](../../config/model_config.yaml)):
- Model-specific settings (Gemma 3 12B)
- Generation defaults and processing parameters

**User Configurations** ([`user_configs/`](../../user_configs/)):
- Dataset structures: `user_configs/dataset_structures/`
- Custom prompts: `user_configs/prompts/`

### Template System

**Template Directories** ([`templates/`](../../templates/)):
- Prompt templates: `templates/prompts/default/`
- Dataset structures: `templates/dataset_structures/`
- User override paths from configuration

### Docker Integration

**Container Configuration** ([`docker-compose.yml`](../../docker-compose.yml)):
- Environment variable mapping to configuration paths
- Volume mounts for configuration directories
- Service-specific configuration overrides

**Service Container** ([`Dockerfile.service`](../../Dockerfile.service)):
- Configuration file copying: `COPY config/ /app/config/`
- Environment variable setup for runtime configuration

### Logging and Monitoring

**Logger Integration** ([`src/utils/logger.py`](../../src/utils/logger.py)):
- Configuration loading success/failure logging
- File access error reporting

**MLflow Configuration**:
- Tracking URI from configuration: `config.get("mlflow", {})`
- Experiment name configuration for dataset generation tracking

## Configuration Schema Reference

### Complete Configuration Structure

```yaml
# Provider Configuration
provider:
  name: "ollama"                    # Provider type
  model_name: "gemma3:1b-it-qat"   # Model identifier
  api_url: "http://ollama:11434"   # Service endpoint
  timeout: 60                      # Request timeout (seconds)
  max_retries: 3                   # Retry attempts
  retry_delay: 5                   # Delay between retries

# Directory Paths
directories:
  input: "data"                    # Input data sources
  output: "output_datasets"        # Generated dataset output
  templates: "templates"           # Prompt templates
  user_configs: "user_configs"     # User configuration overrides

# Generation Settings
generation:
  default_num_examples: 10         # Default dataset size
  default_language: "et"           # Default generation language
  max_retries: 3                   # Generation retry attempts
  parameters:
    temperature: 0.7               # Model creativity (0.0-1.0)
    max_tokens: 4096              # Maximum response length

# Dataset Generation
dataset_generation:
  structure_name: "single_question"           # Output structure type
  prompt_template_name: "institute_topic_question"  # Prompt template
  traversal_strategy: "pattern"              # Data source strategy
  output_format: "json"                      # Output file format
  post_processing: "aggregation"             # Post-processing mode

# MLflow Integration
mlflow:
  experiment_name: "synthetic_data_generation"  # Experiment tracking name

# Callback Configuration
callback:
  url: "http://ruuter-public:8086/global-classifier/data/callback"
  max_retries: 3
  timeout: 30
```

### Environment Variable Mappings

| Environment Variable | Configuration Path | Type | Example |
|---------------------|-------------------|------|---------|
| `PROVIDER_NAME` | `provider.name` | string | `ollama` |
| `PROVIDER_API_URL` | `provider.api_url` | string | `http://ollama:11434` |
| `MODEL_NAME` | `provider.model_name` | string | `gemma3:1b-it-qat` |
| `PROVIDER_TIMEOUT` | `provider.timeout` | int | `60` |
| `DATA_DIR` | `directories.input` | string | `data` |
| `OUTPUT_DIR` | `directories.output` | string | `output_datasets` |
| `TEMPLATES_DIR` | `directories.templates` | string | `templates` |
| `USER_CONFIGS_DIR` | `directories.user_configs` | string | `user_configs` |
| `DEFAULT_LANGUAGE` | `generation.default_language` | string | `et` |
| `DEFAULT_TEMPERATURE` | `generation.parameters.temperature` | float | `0.7` |
| `MAX_TOKENS` | `generation.parameters.max_tokens` | int | `4096` |

This configuration module provides the foundational infrastructure for all application settings, enabling flexible deployment across different environments while maintaining consistent behavior and clear configuration precedence.
