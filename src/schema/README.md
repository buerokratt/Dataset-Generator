# Schema Definitions

## Overview

This directory contains Pydantic schema definitions that provide data validation, serialization, and documentation for the synthetic data generation service. While these schemas aren't fully integrated in the current sprint, they provide a foundation for future development.

## Purpose

The schemas in this directory serve several important purposes:

1. **Data Validation**: Define the expected structure and types for various components
2. **Documentation**: Self-document the data models through code
3. **Serialization/Deserialization**: Enable easy conversion between Python objects and JSON/YAML
4. **API Integration**: Provide automatic request/response validation for FastAPI endpoints
5. **IDE Support**: Enable better code completion and type checking

## Available Schemas

### `prompt_schema.py`

Contains schemas for prompt templates and variables:
- `PromptVariable`: Defines a variable that can be substituted in a template
- `PromptTemplate`: Defines a complete prompt template with metadata

### `dataset_schema.py` 

Contains schemas for dataset structures:
- `DatasetFile`: Defines a file in a dataset structure
- `DatasetDirectory`: Defines a directory that can contain files and subdirectories
- `DatasetStructure`: Defines the complete structure of a dataset

## How to Use in Future Sprints

### 1. Schema Validation

Use schemas to validate configuration files when they're loaded:

```python
from schema.dataset_schema import DatasetStructure
import yaml

# Load raw YAML
with open("my_structure.yaml", "r") as f:
    raw_data = yaml.safe_load(f)

# Validate against schema
try:
    structure = DatasetStructure(**raw_data)
    print(f"Valid structure: {structure.name}")
except ValidationError as e:
    print(f"Invalid structure: {e}")

```
### 2. Schema Validation
Use schemas to validate API requests and responses:

```python
from fastapi import APIRouter
from schema.prompt_schema import PromptTemplate

router = APIRouter()

@router.post("/templates/", response_model=PromptTemplate)
async def create_template(template: PromptTemplate):
    """Create a new prompt template"""
    # Template is already validated by FastAPI
    return template_service.save(template)

@router.get("/templates/{name}", response_model=PromptTemplate)
async def get_template(name: str):
    """Get a prompt template by name"""
    return template_service.get(name)
```
### 3. Type hinting
Use schemas as type hints for better IDE support:

```python
from schema.dataset_schema import DatasetStructure

def process_structure(structure: DatasetStructure) -> None:
    """Process a dataset structure"""
    print(f"Processing {structure.name} with {len(structure.root.files)} files")
    # Your IDE will provide autocomplete for structure.root.files, etc.
```
### 4. Extending Schemas
To add new functionality, extend existing schemas:

```python
from schema.prompt_schema import PromptTemplate
from pydantic import Field

class EnhancedPromptTemplate(PromptTemplate):
    """PromptTemplate with additional features"""
    model_name: str = Field(..., description="Name of the model to use")
    temperature: float = Field(0.7, description="Temperature parameter for generation")
```
### Integration Roadmap
In future sprints, consider the following integrations:

1. **Configuration Validation**: Use schemas to validate all YAML configuration files
2. **API Endpoints**: Add CRUD endpoints for managing templates and structures
3. **Database Integration**: Add methods to serialize/deserialize to/from database records
4. **Schema Evolution**: Add versioning to schemas to handle upgrades gracefully
5. **UI Generation**: Use schema definitions to auto-generate form interfaces

### Best Practices
1. Keep schemas focused on data validation, not business logic
2. Leverage Pydantic's validation capabilities for complex rules
3. Use descriptive field names and add descriptions
4. Provide examples where helpful
5. Keep backward compatibility in mind when evolving schemas

### Notes
The current implementation uses Pydantic v1 syntax. If upgrading to Pydantic v2, some syntax changes may be required.