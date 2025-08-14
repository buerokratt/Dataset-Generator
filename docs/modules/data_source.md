# Data Source Module

## Purpose and Public Surface

The data source module provides a **flexible file discovery and content management system** for the Dataset Generator. It implements a strategy pattern for traversing different directory organizations and includes filtering capabilities to handle diverse data structures across government agencies.

### Public API

| Component | Location | Purpose |
|-----------|----------|---------|
| `DataSource` | [`src/core/data_source.py`](../../src/core/data_source.py) | Individual file representation with lazy loading |
| `DataSourceManager` | [`src/core/data_source.py`](../../src/core/data_source.py) | High-level interface for loading and filtering sources |
| `DataSourceFilter` | [`src/core/data_source.py`](../../src/core/data_source.py) | Flexible filtering by extension, size, patterns |
| Traversal Strategies | [`src/core/data_source.py`](../../src/core/data_source.py) | Different directory organization patterns |

## Key Classes and Functions

### DataSource (Core Data Unit)

**Location**: [`src/core/data_source.py:12-72`](../../src/core/data_source.py)

**Purpose**: Represents a single file with lazy content loading and metadata tracking.

```python
class DataSource:
    def __init__(self, path: str, metadata: Dict[str, Any] = None, content: str = None)
    
    @property
    def content(self) -> str  # Lazy-loaded file content
    
    @property 
    def name(self) -> str     # Filename without extension
    
    @property
    def extension(self) -> str # File extension without dot
```

**Used by**:
- [`src/api/routes.py:224`](../../src/api/routes.py) - Processing individual sources in bulk generation
- [`src/core/data_generator.py`](../../src/core/data_generator.py) - Content extraction for dataset generation
- All traversal strategies - Creating source objects during discovery

### DataSourceManager (Main Interface)

**Location**: [`src/core/data_source.py:644-805`](../../src/core/data_source.py)

**Purpose**: Unified interface for loading sources with different traversal strategies.

```python
class DataSourceManager:
    def load_sources(
        self,
        base_path: str = None,
        strategy_name: str = None,           # "flat", "recursive", "institutional", "pattern"
        filter_config: Dict[str, Any] = None
    ) -> List[DataSource]
```

**Used by**:
- [`src/api/routes.py:200-209`](../../src/api/routes.py) - Bulk dataset generation pipeline
- External applications - Main entry point for data discovery

### Traversal Strategies

**Location**: [`src/core/data_source.py:190-642`](../../src/core/data_source.py)

**Available Strategies**:

| Strategy | Purpose | Directory Structure |
|----------|---------|-------------------|
| `FlatDirectoryTraversalStrategy` | Single directory, non-recursive | `data/*.txt` |
| `RecursiveDirectoryTraversalStrategy` | Deep traversal with depth control | `data/**/*.txt` |
| `InstitutionalTraversalStrategy` | Government agency organization | `ministry_name/topic.txt` |
| `PatternBasedTraversalStrategy` | Glob pattern matching | `**/cleaned.txt` |

**Configuration Example** (from [`config/config.yaml`](../../config/config.yaml)):
```yaml
data_sources:
  default:
    strategy: "pattern"
    base_path: "data"
    patterns: ["**/cleaned.txt"]
    recursive: true
```

## Inputs/Outputs and Side Effects

### File System Operations

**Reads**:
- **File Discovery**: Scans directories based on traversal strategy
- **Content Loading**: Lazy loads file content when `DataSource.content` is accessed
- **Metadata Extraction**: File size, relative paths, organizational context

**No Writes**: This module is read-only and does not modify the file system.

### Network Operations

**None**: This module operates entirely on local file systems.

### Side Effects

1. **Logging**: Uses [`src/utils/logger.py`](../../src/utils/logger.py) for discovery and error reporting
2. **Memory Usage**: Lazy loading minimizes memory footprint until content is accessed
3. **File Handle Management**: Automatically closes file handles after reading

## Example Usage

### Basic Usage from API Routes

From [`src/api/routes.py:200-209`](../../src/api/routes.py):

```python
# Create data source manager with configuration
data_source_config = config.get("data_sources", {}).get("default", {})
source_manager = DataSourceManager(config=data_source_config)

# Load all matching sources
data_sources = source_manager.load_sources(
    base_path=data_path,
    strategy_name=traversal_strategy,  # From config: "pattern" 
    filter_config=filter_config,
)

# Process each source
for source in data_sources:
    print(f"Processing: {source.path}")
    content = source.content  # Lazy-loaded content
    metadata = source.metadata  # Organizational context
```

### Institutional Traversal Example

```python
# For government agency data organized as ministry/topic.txt
manager = DataSourceManager()
sources = manager.load_sources(
    base_path="data/government",
    strategy_name="institutional",
    filter_config={"extensions": ["txt"]}
)

# Access institutional metadata
for source in sources:
    agency = source.metadata.get('institution')    # e.g., "ministry_of_finance"
    topic = source.metadata.get('topic')          # e.g., "budget"
    content = source.content                       # File content
```

### Pattern-Based Discovery

Configuration in [`config/config.yaml`](../../config/config.yaml):
```yaml
data_sources:
  default:
    strategy: "pattern"
    patterns: ["**/cleaned.txt"]  # Find all cleaned.txt files
    base_path: "data"
    recursive: true
```

```python
# Automatically uses pattern strategy from config
sources = manager.load_sources(base_path="data/agencies")
```

### Custom Filtering

```python
# Filter by file size and extension
filter_config = {
    "extensions": ["txt", "md"],
    "min_size": 100,        # Minimum 100 bytes
    "max_size": 1000000,    # Maximum 1MB
    "patterns": [".*government.*"]  # Path must contain "government"
}

sources = manager.load_sources(
    base_path="data",
    strategy_name="recursive", 
    filter_config=filter_config
)
```

## Configuration Integration

### Data Source Configuration

**Location**: [`config/config.yaml:91-96`](../../config/config.yaml)

```yaml
data_sources:
  default:
    strategy: "pattern"           # Traversal strategy
    base_path: "data"            # Root directory
    patterns: ["**/cleaned.txt"] # Glob patterns for pattern strategy
    recursive: true              # Enable recursive traversal
```

### Filter Configuration

**Location**: [`config/config.yaml:78`](../../config/config.yaml)

```yaml
dataset_generation:
  filter: {}  # Can specify extensions, patterns, size limits
```

## Cross-Links to Project Structure

### Templates Integration

**Directory**: [`templates/`](../../templates/) - Prompt templates used with discovered data

**Connection**: Data sources provide content that gets processed using templates:
- Data from institutional sources → Agency-specific templates
- Pattern-matched sources → Standardized templates

### User Configurations

**Directory**: [`user_configs/`](../../user_configs/) - Agency-specific dataset configurations

**Example Structure**:
```
user_configs/
├── dataset_structures/
│   ├── single_question.yaml      # Used with single data sources
│   └── topic_conversations.yaml  # Used with institutional sources
└── prompts/
    ├── institute_topic_question.txt    # Government-specific prompts
    └── institute_topic_conversation.txt
```

### Output Integration

**Directory**: [`output_datasets/`](../../output_datasets/) - Generated datasets

**Connection**: Each discovered `DataSource` can generate one or more output files based on:
- Source metadata (agency, topic)
- Traversal strategy results
- Configured output format

### Related Modules

| Module | Relationship |
|--------|-------------|
| [`src/core/data_generator.py`](../../src/core/data_generator.py) | Consumes `DataSource` objects for content generation |
| [`src/core/config.py`](../../src/core/config.py) | Provides configuration for data source discovery |
| [`src/api/routes.py`](../../src/api/routes.py) | Main consumer for bulk dataset generation |
| [`src/utils/logger.py`](../../src/utils/logger.py) | Used for discovery and error logging |

## Performance Considerations

1. **Lazy Loading**: Content is only loaded when accessed, reducing memory usage for large file sets
2. **Strategy Selection**: Choose appropriate traversal strategy based on directory structure:
   - Use `flat` for simple directories
   - Use `institutional` for government agency organization  
   - Use `pattern` for complex hierarchies with naming conventions
3. **Filtering**: Apply filters early to reduce processing overhead
4. **Batch Processing**: Manager loads all matching sources at once for efficient bulk operations
