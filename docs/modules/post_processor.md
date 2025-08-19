# Post Processor Module

## Purpose and Public Surface

The Post Processor module provides **dataset post-processing capabilities** for the synthetic dataset generation pipeline. It transforms generated datasets into final output formats through different processing strategies including ZIP compression and data aggregation. This component serves as the final stage in the dataset generation workflow, preparing generated content for delivery or storage.

### Key Features

- **Multiple Processing Strategies**: ZIP compression and aggregation support via strategy pattern
- **Cross-Dataset Aggregation**: Combines multiple datasets into unified outputs (JSON/CSV)
- **Field Mapping & Transformation**: Configurable data structure transformation for standardized outputs
- **Metadata Preservation**: Maintains dataset metadata through processing operations
- **Flexible Output Formats**: Supports JSON, CSV, and text output formats
- **Data Shuffling**: Optional randomization for training data preparation

### Public API

| Component | Location | Purpose |
|-----------|----------|---------|
| `PostProcessor` | [`src/core/post_processor.py`](../../src/core/post_processor.py) | Abstract base class for post-processing strategies |
| `ZipPostProcessor` | [`src/core/post_processor.py`](../../src/core/post_processor.py) | Creates ZIP archives of generated datasets |
| `AggregationPostProcessor` | [`src/core/post_processor.py`](../../src/core/post_processor.py) | Aggregates multiple datasets into unified files |
| `PostProcessorFactory` | [`src/core/post_processor_factory.py`](../../src/core/post_processor_factory.py) | Factory for creating processor instances |

## Key Classes and Functions

### PostProcessor (Abstract Base Class)

**Location**: [`src/core/post_processor.py`](../../src/core/post_processor.py)

**Purpose**: Defines the contract for all post-processing implementations using the strategy pattern.

```python
class PostProcessor(ABC):
    @abstractmethod
    def process(self, output_paths: List[str], base_output_dir: str) -> Optional[str]:
        """Process generated dataset outputs into final format"""
```

**Used by**:
- [`src/core/post_processor_factory.py`](../../src/core/post_processor_factory.py) - Return type for factory method
- [`src/api/routes.py`](../../src/api/routes.py) - Post-processing in bulk generation pipeline

### ZipPostProcessor Class

**Location**: [`src/core/post_processor.py`](../../src/core/post_processor.py)

**Purpose**: Creates ZIP archives of generated datasets for distribution or storage.

**Key Features**:
- Recursive directory compression
- Preserves original file structure
- Automatic archive naming based on dataset structure

**Input/Output**:
- **Input**: List of dataset output paths, base output directory
- **Output**: Path to created ZIP archive or None if failed
- **Side Effects**: Creates ZIP file in configured output directory

### AggregationPostProcessor Class

**Location**: [`src/core/post_processor.py`](../../src/core/post_processor.py)

**Purpose**: Aggregates multiple dataset files into unified JSON or CSV outputs with configurable field mapping.

**Key Features**:
- **Multi-format Support**: JSON, CSV, and text output formats
- **Field Mapping**: Transforms data structure using configuration-driven mapping
- **Metadata Integration**: Incorporates dataset metadata into aggregated output
- **Data Shuffling**: Optional randomization with sequential ID reassignment
- **Cross-Dataset Processing**: Combines data from multiple sources

**Field Mapping Configuration**:
```yaml
field_mapping:
  enabled: true
  payload_to_output:     # Maps metadata fields to output
    agency_name: agency_name
    agency_id: agency_id
  defaults:              # Default values and auto-generation
    item_id: auto_increment
    dataset_version_id: version_id
  content_fields:        # Maps content fields
    question: data_item
```

### PostProcessorFactory Class

**Location**: [`src/core/post_processor_factory.py`](../../src/core/post_processor_factory.py)

**Purpose**: Creates appropriate post-processor instances based on configuration.

```python
@staticmethod
def create_post_processor(config: dict) -> PostProcessor:
    """Create a post-processor based on configuration."""
```

**Configuration-Driven Selection**:
- `post_processing: "zip"` → `ZipPostProcessor`
- `post_processing: "aggregation"` → `AggregationPostProcessor`

## Inputs/Outputs and Side Effects

### File System Operations

**ZipPostProcessor**:
- **Read**: Generated dataset files and directories
- **Write**: ZIP archives in output directory
- **Structure**: Preserves original directory hierarchy in archives

**AggregationPostProcessor**:
- **Read**: JSON, text files from multiple dataset sources
- **Write**: Aggregated JSON/CSV files with configured naming
- **Structure**: Creates flat aggregated files with optional field transformation

### Network Operations
- **None**: No direct network operations

### Configuration Dependencies
- **Aggregation Settings**: [`config/config.yaml`](../../config/config.yaml) - `dataset_generation.aggregation`
- **Output Directories**: [`config/config.yaml`](../../config/config.yaml) - `directories.output`
- **Field Mapping**: Configuration-driven data transformation rules

### Logging Side Effects
- **Info**: Processing progress, file creation confirmations
- **Warning**: Individual file processing errors (continues processing)
- **Error**: Fatal processing failures

## Example Usage

### Basic ZIP Post-Processing

```python
from src.core.post_processor_factory import PostProcessorFactory

# Configuration for ZIP processing
config = {
    "dataset_generation": {
        "post_processing": "zip"
    },
    "directories": {
        "output": "output_datasets"
    }
}

# Create processor and process datasets
processor = PostProcessorFactory.create_post_processor(config)
zip_path = processor.process(
    output_paths=["/path/to/dataset1", "/path/to/dataset2"],
    base_output_dir="output_datasets/agency_datasets"
)
# Returns: "output_datasets/agency_datasets.zip"
```

### Aggregation with Field Mapping

```python
# Configuration for aggregation with CSV output
config = {
    "dataset_generation": {
        "post_processing": "aggregation",
        "aggregation": {
            "output_filename": "consolidated_data",
            "output_format": "csv",
            "merge_strategy": "combine_arrays",
            "enable_shuffling": True,
            "field_mapping": {
                "enabled": True,
                "payload_to_output": {
                    "agency_name": "agency_name",
                    "agency_id": "agency_id"
                },
                "defaults": {
                    "item_id": "auto_increment",
                    "dataset_version_id": "version_id"
                },
                "content_fields": {
                    "question": "data_item"
                }
            },
            "csv_field_order": ["item_id", "agency_name", "agency_id", "data_item", "dataset_version_id"]
        }
    }
}

# Create processor with metadata
processor = PostProcessorFactory.create_post_processor(config)
dataset_metadata = [
    {"agency_name": "PPA", "agency_id": "ppa", "version_id": "v1.0"},
    {"agency_name": "Id.ee", "agency_id": "idee", "version_id": "v1.0"}
]

csv_path = processor.process(
    output_paths=["dataset1.json", "dataset2.json"],
    base_output_dir="output_datasets",
    dataset_metadata=dataset_metadata
)
# Returns: "output_datasets/consolidated_data.csv"
```

### Real-World Usage in Bulk Generation

From [`src/api/routes.py`](../../src/api/routes.py):

```python
# Individual dataset post-processing (ZIP mode)
if post_processing_type != "aggregation":
    base_output_dir = f"{output_dir}"
    post_processor = PostProcessorFactory.create_post_processor(modified_config)
    final_output_path = post_processor.process(all_output_paths, base_output_dir)

# Cross-dataset aggregation (Aggregation mode)
if post_processing_type == "aggregation":
    final_post_processor = PostProcessorFactory.create_post_processor(
        final_aggregation_config
    )
    # Pass dataset metadata to the processor
    final_aggregated_path = final_post_processor.process(
        all_cross_dataset_output_paths, base_output_dir, all_dataset_metadata
    )
```

### Field Mapping Example

**Input JSON**:
```json
[
  {"question": "Kuidas saan taotleda sotsiaaltoetust?", "category": "social"}
]
```

**Dataset Metadata**:
```json
{"agency_name": "PPA", "agency_id": "ppa", "version_id": "v1.0"}
```

**Output CSV** (after field mapping):
```csv
item_id,agency_name,agency_id,data_item,dataset_version_id
v1.0_1,PPA,ppa,"Kuidas saan taotleda sotsiaaltoetust?",1
```

## Cross-Links and Configuration

### Configuration Sources
- **Post-Processing Type**: [`config/config.yaml`](../../config/config.yaml) - `dataset_generation.post_processing`
- **Aggregation Settings**: [`config/config.yaml`](../../config/config.yaml) - `dataset_generation.aggregation`
- **Output Directory**: [`config/config.yaml`](../../config/config.yaml) - `directories.output`

### Integration Points
- **Bulk Generation**: [`src/api/routes.py`](../../src/api/routes.py) - `background_generate_bulk()` function
- **Single Dataset Processing**: [`src/api/routes.py`](../../src/api/routes.py) - `process_single_dataset()` function
- **Factory Pattern**: [`src/core/post_processor_factory.py`](../../src/core/post_processor_factory.py) - Processor instantiation

### Output Locations
- **Generated Datasets**: [`output_datasets/`](../../output_datasets/) - Source data for post-processing
- **Processed Files**: Configured output directory (default: `output_datasets/`)
- **ZIP Archives**: Named after dataset structure (e.g., `single_question.zip`)
- **Aggregated Files**: Named using `output_filename` configuration

### Post-Processing Modes

**ZIP Mode** (`post_processing: "zip"`):
- Individual dataset compression
- Preserves original file structure
- Suitable for dataset distribution

**Aggregation Mode** (`post_processing: "aggregation"`):
- Cross-dataset data consolidation
- Field mapping and transformation
- CSV/JSON unified outputs
- Supports metadata integration
- Optional data shuffling for training preparation

### Related Components
- **Data Generator**: [`src/core/data_generator.py`](../../src/core/data_generator.py) - Produces input for post-processing
- **Storage Manager**: [`src/core/storage_manager.py`](../../src/core/storage_manager.py) - Output directory management
- **API Routes**: [`src/api/routes.py`](../../src/api/routes.py) - Post-processing orchestration
