# Synthetic Data Generation Service

A service for generating synthetic datasets using large language models (LLMs) for various use cases, particularly focused on Estonian government AI applications.

## Overview

This service creates synthetic datasets such as FAQs and conversations based on topic content. It uses template-based prompting and structured output formatting to generate high-quality data that can be used for training and testing AI models.

## Features

- Generate synthetic FAQs and conversations from source text content
- Configurable dataset structures and prompt templates
- Format outputs as JSON or plain text
- Exponential backoff retry mechanism for handling API failures
- MLflow integration for experiment tracking
- Docker support for containerized operation

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) for local LLM serving (or accessible LLM API)
- [MLflow](https://mlflow.org/) (optional, for experiment tracking)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-org/synthetic-data-service.git
cd synthetic-data-service
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Main Configuration

The service is configured via `config.yaml` in the root directory:

```yaml
# API connection settings
api:
  url: "http://localhost:8000"
  timeout_seconds: 30

# Directory paths
directories:
  input: "data"                # Root directory for source data files
  output: "output_datasets"    # Root directory for generated datasets
  
# FAQs generation settings
faqs:
  structure_name: "single_question"
  prompt_template_name: "institute_topic_question"
  num_examples: 5
  parameters:
    redundancy_factor: 0
    difficulty: "intermediate"
    language: "et"  # Estonian

# Conversations generation settings
conversations:
  structure_name: "topic_conversations"
  prompt_template_name: "institute_topic_conversation"
  num_examples: 3
  parameters:
    redundancy_factor: 0
    num_turns: 6
    difficulty: "intermediate"
    language: "et"

# Processing settings
processing:
  wait_between_requests: 1  # Seconds to wait between API requests

mlflow:
  experiment_name: "synthetic_data_generation"
```

### Model Configuration

The LLM settings are configured in `config/model_config.yaml`:

```yaml
model_name: "gemma3:1b-it-qat"
ollama_host: "http://ollama:11434"
ollama_timeout: 60
ollama_max_retries: 3
ollama_retry_delay: 5
...
```

## Project Structure

```
dataset-generator/
│
├── src/                   # Source code
│   ├── api/               # API routes and validation
│   ├── core/              # Core data generation logic
│   ├── schema/            # Pydantic schemas
│   └── utils/             # Utility functions
│
├── config/                # Configuration files
├── templates/             # Default templates
│   ├── dataset_structures/
│   └── prompts/
│
├── user_configs/          # User-defined templates/configs
│   ├── dataset_structures/
│   └── prompts/
│
├── data/                  # Source data directory
├── output_datasets/       # Generated datasets
├── logs/                  # Application logs
└── mlflow/                # MLflow configuration
```

## Usage

### Starting the Service

Run the service with:

By default, the service runs on `localhost:8000`. You can specify different host/port:

```bash
docker compose up --build -d
```

### Generating Datasets

The generation script processes text files and creates synthetic datasets:

```bash
python test_dataset_generation.py
```

This will:
1. Read all topic files from the `data` directory
2. Generate FAQs and conversations for each topic
3. Save datasets to the output directory
4. Log experiments to MLflow (if configured)

### Data Organization

Organize your source data in the `data` directory as follows:

```
data/
└── institution_name/
    ├── topic1.txt
    ├── topic2.txt
    └── ...
```

Each `.txt` file should contain the source content for a specific topic.

## API Endpoints

### Generate Dataset

```
POST /generate

{
  "dataset_name": "optional_name",       // Optional name for dataset
  "output_base_path": "/optional/path",  // Optional explicit output path
  "structure_name": "structure_name",    // Name of dataset structure YAML
  "prompt_template_name": "template_name", // Name of prompt template to use
  "num_examples": 10,                    // Number of examples to generate
  "output_format": "json",               // Output format (json or text)
  "parameters": {                        // Additional parameters
    "language": "et",
    "topic_content": "Content text...",
    ...
  }
}
```

## MLflow Integration

The service can log experiment details to MLflow. Configure MLflow in `mlflow.env`:

```
MLFLOW_TRACKING_USERNAME=mlflowadmin
MLFLOW_TRACKING_PASSWORD=set your password
MLFLOW_FLASK_SERVER_SECRET_KEY=set your secret key
...
```

Access the MLflow UI at http://localhost:5000

## Customizing Templates and Structures

### Dataset Structures

Define dataset structures in YAML files under `user_configs/dataset_structures/`:

```yaml
name: "my_structure"
description: "Custom dataset structure"
root:
  files:
    records: 
      format: "json"
      description: "Main dataset records"
```

### Prompt Templates

Create prompt templates in `user_configs/prompts/`:

```
You are a helpful AI assistant generating synthetic data.

Please generate ${num_examples} examples about ${topic}.

Additional Instructions: ${additional_instructions}
```

## Troubleshooting

### Common Issues

1. **API Connection Error**: Ensure the API service is running at the URL specified in `config.yaml`
2. **LLM Connection Error**: Check that Ollama is running and accessible
3. **File Access Issues**: Ensure proper permissions for reading input and writing output
4. **JSON Parse Errors**: Check output templates to ensure they generate valid JSON

### Logs

Application logs are stored in the `logs/` directory.

## Experiments

The synthetic data generation service is designed to be highly configurable, allowing you to experiment with different settings to optimize output quality. This section explains how to conduct various experiments by modifying configurations and templates.

### Experiment Types

#### 1. Model Configuration Experiments

Modify settings in `config/model_config.yaml` to experiment with:

#### 2. User Configuration Experiments

Modify settings in `config.yaml` to experiment with:

#### 3. Prompt Configuration Experiments

For FAQs: `user_configs/prompts/faqs/your_template_name.txt`
For conversations: `user_configs/prompts/conversations/your_template_name.txt`

Then update the template name in `config.yaml`:
```yaml
faqs:
  prompt_template_name: "your_template_name"
```