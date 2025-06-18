# Estonian Government AI Dataset Generator

A powerful service for generating synthetic datasets using Large Language Models with configurable structures and templates.

## Overview

The Dataset Generator is a flexible tool that transforms source data into structured synthetic datasets using LLMs. It provides configurable templates, structures, and traversal strategies for versatile dataset generation.

## Features

- **Multiple LLM Provider Support**: Works with Ollama (default) and can be extended to other providers
- **Flexible Data Source Processing**: Multiple traversal strategies for handling different directory structures
- **Configurable Output Formats**: Generates datasets in JSON and other formats
- **Bulk Generation**: Process multiple source files in a single request
- **Docker Ready**: Packaged as Docker images for easy deployment

## Using in External Projects

The Dataset Generator is designed to be used as a service within larger architectures. Here's how to integrate it into your project:

First build docker compose file
```bash
cd Dataset-Generator
docker compose build
```

### Integration via Docker Compose

Add the Dataset Generator services to your project's `docker-compose.yml`:
This is a minimal setup example

```yaml
services:
  # Dataset Generator services
  dataset-gen-ollama:
    image: synthesisai/dataset-generator-ollama:latest
    container_name: dataset-gen-ollama
    ports:
      - "11434:11434"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OLLAMA_USE_GPU=1
      - OLLAMA_HOST=0.0.0.0
    volumes:
      - dataset_gen_ollama_models:/root/.ollama
      - ./src/dataset-generation/ollama-entrypoint.sh:/ollama-entrypoint.sh
    entrypoint: ["bash", "/ollama-entrypoint.sh"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  dataset-gen-service:
    image: synthesisai/dataset-generator:latest
    container_name: dataset-gen-service
    ports:
      - "8000:8000"
    environment:
      - PROVIDER_NAME=ollama
      - MODEL_NAME=gemma3:1b-it-qat
      - PROVIDER_API_URL=http://dataset-gen-ollama:11434
      - SERVICE_DEBUG=false
      - MLFLOW_TRACKING_URI=http://dataset-gen-mlflow:5000
    volumes:
      - ./src/dataset-generation/config:/app/config
      - ./src/dataset-generation/templates:/app/templates
      - ./src/dataset-generation/user_configs:/app/user_configs
      - ./src/dataset-generation/data:/app/data
      - ./src/dataset-generation/output_datasets:/app/output_datasets
      - ./src/dataset-generation/logs:/app/logs
    depends_on:
      - dataset-gen-ollama
      - dataset-gen-mlflow

  dataset-gen-mlflow:
    image: synthesisai/dataset-generator-mlflow:latest
    container_name: dataset-gen-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./src/dataset-generation/mlflow_data:/mlflow/mlflow_data
      - ./src/dataset-generation/mlflow_artifacts:/mlflow/mlflow_artifacts
```

Don't forget to add the necessary volumes:

```yaml
volumes:
  dataset_gen_ollama_models:
```

### Required Directory Structure

Create the following directory structure in your project:

```
src/
└── dataset-generation/
    ├── config/            # Configuration files
    ├── data/              # Input data files
    ├── logs/              # Log files
    ├── mlflow_artifacts/  # MLflow artifact storage
    ├── mlflow_data/       # MLflow metadata storage
    ├── output_datasets/   # Generated datasets
    ├── templates/         # System templates
    ├── user_configs/      # Custom templates and structures
    │   ├── dataset_structures/
    │   └── prompts/
    └── ollama-entrypoint.sh  # Entrypoint script for Ollama
```

### Ollama Entrypoint Script

Create an entrypoint script at `src/dataset-generation/ollama-entrypoint.sh`:
and copy content from 
```bash 
  scripts/ollama-entrypoint.sh
```

Make it executable: `chmod +x src/dataset-generation/ollama-entrypoint.sh`

### Configuration

1. **Create a base config file** in `src/dataset-generation/config/config.yaml`:

```yaml
# LLM Provider configuration
provider:
  name: "ollama"
  model_name: "gemma3:1b-it-qat"
  api_url: "http://dataset-gen-ollama:11434"
  timeout: 60
  max_retries: 3
  retry_delay: 5

# Directory paths
directories:
  input: "data"
  output: "output_datasets"
  templates: "templates"
  user_configs: "user_configs"
  
# Default generation settings
generation:
  default_num_examples: 5
  default_language: "et"
  parameters:
    temperature: 0.7
    max_tokens: 4096
```

### Using the API

You can now call the Dataset Generator API from your services:

```python
import requests

# Generate dataset
response = requests.post("http://dataset-gen-service:8000/generate-bulk", json={
    "dataset_structure_name": "estonian_qa",
    "prompt_template_name": "qa_generator",
    "data_path": "data/government",
    "traversal_strategy": "institutional",
    "no_of_samples": 10,
    "output_format": "json"
})

# Get results
result = response.json()
```

## Customization

### Custom Dataset Structures

Create custom dataset structures in `src/dataset-generation/user_configs/dataset_structures/`:

```yaml
# src/dataset-generation/user_configs/dataset_structures/custom_structure.yaml
name: "custom_structure"
description: "My custom dataset structure"
root:
  files:
    - name: "data.json"
      description: "Main data file"
```

### Custom Prompt Templates

Create custom prompt templates in `src/dataset-generation/user_configs/prompts/`:

```
# src/dataset-generation/user_configs/prompts/custom_prompt.txt
You are generating data about ${topic}.
Please create a detailed response in ${language}.
```

## API Reference

### Generate Bulk Endpoint

`POST /generate-bulk`

Request body:
```json
{
  "dataset_structure_name": "estonian_qa",
  "prompt_template_name": "qa_generator",
  "data_path": "data/government",
  "traversal_strategy": "institutional",
  "no_of_samples": 10,
  "output_format": "json",
  "parameters": {
    "temperature": 0.7
  }
}
```

Response:
```json
{
  "status": "success",
  "message": "Generated datasets for 5 sources",
  "results": [
    {
      "source": "data/government/ministry_of_finance/taxes.txt",
      "output_path": "output_datasets/estonian_qa/ministry_of_finance/taxes_20250527_121901"
    }
  ],
  "zip_path": "output_datasets/estonian_qa.zip"
}
```

### Supported Data Source Patterns

The system supports multiple ways to organize and access your source data through different traversal strategies:

1. ***Flat Directory Structure***

Used with ```traversal_strategy: "flat"``` - processes only files in the top level of a directory.

```bash
data/
├── document1.txt       # ✓ included
├── document2.txt       # ✓ included
├── report.pdf          # ✓ included (if not filtered by extension)
└── subdirectory/
    └── nested.txt      # ✗ excluded (not in top level)
```

2. ***Recursive Directory Structure***

Used with ```traversal_strategy: "recursive"``` - processes files at all levels with optional depth control.

```bash
data/
├── document1.txt          # ✓ included (depth 0)
├── category1/
│   ├── file1.txt          # ✓ included (depth 1)
│   └── subcategory/
│       └── file2.txt      # ✓ included (depth 2)
└── category2/
    └── file3.md           # ✓ included (depth 1)
```

3. ***Institutional Organization***

Used with ```traversal_strategy: "institutional"``` - designed for government/organizational content.

```bash
data/
├── ministry_of_finance/              # Institution
│   ├── taxes.txt                     # ✓ included as topic "taxes"
│   └── budget_planning.md            # ✓ included as topic "budget_planning"
├── ministry_of_education/            # Institution
│   ├── schools.txt                   # ✓ included as topic "schools"
│   └── universities.json             # ✓ included as topic "universities"
└── agency_documents/                 # Institution
    └── regulations.txt               # ✓ included as topic "regulations"
```

Each file is processed with institution and topic metadata automatically extracted.

4. ***Pattern-Based Structure***

Used with ```traversal_strategy: "pattern"``` - finds files using glob patterns.

```bash
data/
├── reports/
│   ├── 2023_report.txt      # ✓ included with "**/*.txt" pattern
│   └── 2023_data.xlsx       # ✗ excluded with "**/*.txt" pattern
├── documents/
│   └── important.txt        # ✓ included with "**/*.txt" pattern
└── reference.pdf            # ✗ excluded with "**/*.txt" pattern
```

### Dataset Structure Examples

The system supports various dataset structure definitions. Here are some examples:

1. Simple Single File Structure
```yml 
name: "simple_qa"
description: "Basic question-answer dataset with a single JSON file"
root:
  files:
    qa_data:
      format: json
```

2. Multi-File Dataset

```yml
name: "comprehensive_dataset"
description: "Multiple files with different formats"
root:
  files:
    - name: "questions.json"
      format: json
      description: "All questions in JSON format"
    - name: "answers.json" 
      format: json
      description: "All answers in JSON format"
    - name: "metadata.yaml"
      format: yaml
      description: "Dataset metadata"
    - name: "README.md"
      format: markdown
      description: "Documentation"
```

3. Hierarchical Structure

```yml
name: "hierarchical_dataset"
description: "Complex directory structure with multiple files"
root:
  files:
    - name: "index.json"
      format: json
  directories:
    categories:
      pattern: "{category_name}"
      files:
        - name: "data.json"
          format: json
      directories:
        subcategories:
          pattern: "{subcategory_name}"
          files:
            - name: "specific_data.json"
              format: json
```

