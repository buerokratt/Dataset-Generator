# Storage Manager Module

## Purpose and Public Surface

The Storage Manager module provides a **unified interface for file system operations** related to dataset generation, including directory management, path resolution, and dataset discovery. It serves as an abstraction layer between the application and the file system, ensuring consistent directory structures and providing utility methods for common storage operations.

### Key Features

- **Configuration-Based Path Resolution**: Uses [`config/config.yaml`](../../config/config.yaml) with fallbacks to defaults
- **Directory Management**: Automated creation, cleaning, and preparation of directories
- **Dataset Discovery**: Methods to list, check existence, and measure dataset sizes
- **Path Safety**: Ensures consistent directory structures across different environments
- **File System Abstraction**: Centralized storage operations for the entire application

### Public API

| Component | Location | Purpose |
|-----------|----------|---------|
| `StorageManager` | [`src/core/storage_manager.py`](../../src/core/storage_manager.py) | Main storage management class |
| `get_datasets_base_dir()` | [`src/core/storage_manager.py`](../../src/core/storage_manager.py) | Returns configured datasets directory |
| `prepare_directory()` | [`src/core/storage_manager.py`](../../src/core/storage_manager.py) | Creates directories with parents |
| `list_datasets()` | [`src/core/storage_manager.py`](../../src/core/storage_manager.py) | Discovers available datasets |

## Key Classes and Functions

### StorageManager Class

**Location**: [`src/core/storage_manager.py`](../../src/core/storage_manager.py)

**Purpose**: Central storage management with configuration-driven directory resolution.

```python
class StorageManager:
    def __init__(self, config=None):
        # Configuration-based directory setup
        self.config = config or app_config or {}
        self.dirs = self.config.get("directories", {})
        
        # Directory path resolution with fallbacks
        self.datasets_base_dir = app_config.DATASETS_DIR if hasattr(app_config, "DATASETS_DIR") else "datasets"
        self.templates_dir = app_config.TEMPLATES_DIR if hasattr(app_config, "TEMPLATES_DIR") else "templates"
        self.user_configs_dir = app_config.USER_CONFIGS_DIR if hasattr(app_config, "USER_CONFIGS_DIR") else "user_configs"
```

**Configuration Sources**:
1. Explicit `config` parameter
2. Global [`src/core/config.py`](../../src/core/config.py) app_config
3. Hardcoded defaults (`datasets`, `templates`, `user_configs`)

**Used by**:
- [`src/core/data_generator.py`](../../src/core/data_generator.py) - Main dataset generation pipeline
- [`src/api/routes.py`](../../src/api/routes.py) - API endpoints for dataset operations

### Directory Management Methods

#### `prepare_directory(dir_path: str) -> None`

**Purpose**: Creates directory structures with parent directories as needed.

```python
def prepare_directory(self, dir_path: str) -> None:
    """Create a directory if it doesn't exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Prepared directory: {dir_path}")
```

**Used in**: Dataset generation workflow before writing output files

#### `clean_directory(directory: str) -> None`

**Purpose**: Removes all contents from a directory while preserving the directory itself.

```python
def clean_directory(self, directory: str) -> None:
    # Remove all files and subdirectories
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
```

**Side Effects**: **Destructive** - permanently removes files and subdirectories

### Dataset Discovery Methods

#### `list_datasets() -> List[str]`

**Purpose**: Discovers all available datasets in the base datasets directory.

```python
def list_datasets(self) -> List[str]:
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        return []
    
    return [d for d in os.listdir(datasets_dir) 
            if os.path.isdir(os.path.join(datasets_dir, d))]
```

**Returns**: List of dataset directory names (not full paths)

#### `dataset_exists(dataset_name: str) -> bool`

**Purpose**: Checks if a specific dataset directory exists.

#### `get_dataset_size(dataset_name: str) -> int`

**Purpose**: Calculates total size of all files in a dataset directory recursively.

**Performance Note**: Walks entire directory tree - can be slow for large datasets

## Inputs/Outputs and Side Effects

### Inputs

| Method | Input Type | Source | Example |
|--------|------------|--------|---------|
| `__init__` | `config: Dict` | [`config/config.yaml`](../../config/config.yaml) or [`src/core/config.py`](../../src/core/config.py) | `{"directories": {"output": "custom_output"}}` |
| `prepare_directory` | `dir_path: str` | Generated paths from [`src/core/data_generator.py`](../../src/core/data_generator.py) | `"output_datasets/my_dataset_20250809"` |
| `clean_directory` | `directory: str` | User-specified or computed paths | `"output_datasets/temp"` |
| `dataset_exists` | `dataset_name: str` | API requests via [`src/api/routes.py`](../../src/api/routes.py) | `"estonian_qa_dataset"` |

### Outputs

| Method | Output Type | Description | Used By |
|--------|-------------|-------------|---------|
| `get_datasets_base_dir` | `str` | Absolute or relative path to datasets | [`src/core/data_generator.py`](../../src/core/data_generator.py) |
| `list_datasets` | `List[str]` | Dataset directory names | [`src/api/routes.py`](../../src/api/routes.py) API endpoints |
| `get_dataset_size` | `int` | Size in bytes | Monitoring and cleanup operations |

### Side Effects

#### File System Operations
- **Directory Creation**: `prepare_directory()` creates directory trees
- **File Deletion**: `clean_directory()` permanently removes files
- **Directory Scanning**: `list_datasets()` reads directory contents

#### Logging
All operations log to the configured logger:
```python
logger.info(f"Prepared directory: {dir_path}")
logger.info(f"Cleaned directory: {directory}")
```

**Log Output**: Written to [`logs/`](../../logs/) directory

#### Configuration Dependencies
- Reads from [`config/config.yaml`](../../config/config.yaml) `directories` section
- Falls back to environment variables via [`src/core/config.py`](../../src/core/config.py)
- Creates default directories if none configured

## Example Usage

### Basic Dataset Management

```python
from src.core.storage_manager import StorageManager

# Initialize with default configuration
storage = StorageManager()

# Prepare output directory for new dataset
dataset_name = "estonian_finance_qa"
output_path = f"output_datasets/{dataset_name}"
storage.prepare_directory(output_path)

# List all available datasets
available_datasets = storage.list_datasets()
print(f"Found {len(available_datasets)} datasets")

# Check if specific dataset exists
if storage.dataset_exists("previous_dataset"):
    size = storage.get_dataset_size("previous_dataset")
    print(f"Previous dataset size: {size:,} bytes")
```

### Integration with Data Generator

From [`src/core/data_generator.py`](../../src/core/data_generator.py):

```python
class DataGenerator:
    def __init__(self, config=None):
        # Initialize storage manager with same config
        self.storage_manager = StorageManager(config)
        
        # Get configured directories
        self.output_dir = self.storage_manager.get_datasets_base_dir()
        
    def generate(self, structure_name: str, ...):
        # Use storage manager to prepare output directory
        output_dir = Path(self.output_dir) / dataset_name
        self.storage_manager.prepare_directory(str(output_dir))
```

### Configuration-Based Setup

Using [`config/config.yaml`](../../config/config.yaml):

```yaml
directories:
  output: "custom_datasets"
  templates: "custom_templates" 
  user_configs: "agency_configs"
```

```python
# StorageManager automatically uses configured paths
storage = StorageManager()
print(storage.get_datasets_base_dir())  # "custom_datasets"
print(storage.get_templates_dir())      # "custom_templates"
```

### API Integration

From [`src/api/routes.py`](../../src/api/routes.py):

```python
@app.get("/datasets")
async def list_datasets():
    storage = StorageManager()
    datasets = storage.list_datasets()
    
    return {
        "datasets": [
            {
                "name": name,
                "size": storage.get_dataset_size(name),
                "exists": storage.dataset_exists(name)
            }
            for name in datasets
        ]
    }
```

## Cross-Links and Integration Points

### Configuration Files
- **Primary Config**: [`config/config.yaml`](../../config/config.yaml) - Directory path configuration
- **Config Module**: [`src/core/config.py`](../../src/core/config.py) - Configuration loading and environment variables

### Data Flow Integration
- **Data Generator**: [`src/core/data_generator.py`](../../src/core/data_generator.py) - Primary consumer for output directory management
- **API Routes**: [`src/api/routes.py`](../../src/api/routes.py) - Dataset listing and management endpoints
- **Template Registry**: [`src/core/template_registry.py`](../../src/core/template_registry.py) - Template and user config directory access

### Directory Structure Dependencies
- **Output Datasets**: [`output_datasets/`](../../output_datasets/) - Generated dataset storage
- **Templates**: [`templates/`](../../templates/) - Prompt template storage  
- **User Configs**: [`user_configs/`](../../user_configs/) - Agency-specific configurations
- **Logs**: [`logs/`](../../logs/) - Operation logging output

### Docker Integration
- **Volume Mounts**: [`docker-compose.yml`](../../docker-compose.yml) configures persistent storage
- **Container Paths**: Maps host directories to container paths for data persistence

### Error Handling
- **Path Safety**: Validates directory paths to prevent traversal attacks
- **Graceful Fallbacks**: Uses default paths when configuration is missing
- **Logging Integration**: Uses [`src/utils/logger.py`](../../src/utils/logger.py) for consistent logging
