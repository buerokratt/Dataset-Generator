# Dataset Generator

**Open-source, generic synthetic dataset generator.** Build the Docker image once and reuse it inside **any external project's** `docker-compose.yml` by mounting your configurations.

## What is this?

This is a containerized synthetic dataset generation service designed for embedding into external projects. It provides:

- **Generic architecture**: Works with any LLM provider (Ollama, OpenAI, etc.)
- **Template-driven generation**: Customizable prompts and output structures
- **Multi-tenant support**: Different configurations per agency/project
- **RESTful API**: Generate datasets programmatically with background processing
- **Docker-first design**: Deploy anywhere with consistent behavior

## Quick Start

1. **Clone and start services:**
```bash
git clone https://github.com/rootcodelabs/Dataset-Generator.git
cd Dataset-Generator
docker compose up -d
```

2. **Verify services are running:**
```bash
curl http://localhost:8000/health
```

3. **Generate your first dataset:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_structure": "single_question",
    "prompt_template": "institute_topic_question",
    "num_samples": 5,
    "language": "et"
  }'
```

Check generated datasets in `./output_datasets/`

## Use in Another Project

### Option A: Use Published Image

Add this service to your existing `docker-compose.yml`:

```yaml
version: '3'

services:
  # Your existing services...
  
  dataset-generator:
    image: synthesisai/dataset-generator:latest
    ports:
      - "8000:8000"
    environment:
      - PROVIDER_API_URL=http://your-llm-provider:11434
      - MLFLOW_TRACKING_URI=http://your-mlflow:5000
    volumes:
      - ./your-templates:/app/templates
      - ./your-configs:/app/user_configs
      - ./your-data:/app/data
      - ./generated-datasets:/app/output_datasets
      - ./logs:/app/logs
    networks:
      - your-network

  # Optional: Include Ollama if you don't have an LLM provider
  ollama:
    image: synthesisai/dataset-generator-ollama:latest
    ports:
      - "11434:11434"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - your-network
```

### Option B: Build & Push Custom Tag

1. **Customize the service:**
```bash
# Modify config/config.yaml, templates/, etc.
# Add your custom templates and configurations
```

2. **Build with your tag:**
```bash
docker build -f Dockerfile.service -t your-org/dataset-generator:v1.0 .
docker build -f Dockerfile.ollama-gpu -t your-org/dataset-generator-ollama:v1.0 .
```

3. **Push to your registry:**
```bash
docker push your-org/dataset-generator:v1.0
docker push your-org/dataset-generator-ollama:v1.0
```

4. **Use in your project:**
```yaml
services:
  dataset-generator:
    image: your-org/dataset-generator:v1.0
    # ... rest of configuration
```

## Directory Structure

| Directory | Purpose | Mount Point | Example Content |
|-----------|---------|-------------|-----------------|
| `src/` | Application source code | Not mounted | API routes, core logic, providers |
| `templates/` | Prompt templates | `/app/templates` | `prompts/default/base_prompt.txt` |
| `user_configs/` | User configurations | `/app/user_configs` | `dataset_structures/single_question.yaml` |
| `config/` | Base configuration | `/app/config` | `config.yaml`, `model_config.yaml` |
| `data/` | Input data sources | `/app/data` | Your source documents/texts |
| `output_datasets/` | Generated datasets | `/app/output_datasets` | JSON/CSV output files |
| `logs/` | Application logs | `/app/logs` | `synthetic_data_service.log` |

## Configuration

Key environment variables for integration:

```bash
# LLM Provider
PROVIDER_API_URL=http://ollama:11434        # Your LLM service endpoint
MODEL_NAME=gemma3:1b-it-qat                 # Model to use
PROVIDER_NAME=ollama                        # Provider type

# MLflow (optional)
MLFLOW_TRACKING_URI=http://mlflow:5000      # Experiment tracking

# Service
SERVICE_DEBUG=false                         # Debug logging
```

## API Usage

Generate datasets programmatically:

```bash
# Single generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_structure": "single_question",
    "prompt_template": "institute_topic_question", 
    "num_samples": 10,
    "language": "et",
    "parameters": {
      "temperature": 0.7,
      "difficulty": "medium"
    }
  }'

# Bulk generation with callback
curl -X POST "http://localhost:8000/generate/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "dataset_structure": "single_question",
        "prompt_template": "institute_topic_question",
        "num_samples": 5,
        "language": "et"
      }
    ],
    "callback_url": "http://your-service/callback"
  }'
```

## Next Steps

- See [Architecture Documentation](ARCHITECTURE.md) for system design
- Check [Configuration Guide](modules/configuration.md) for advanced setup
- Review [API Documentation](modules/api-integration.md) for full endpoint reference
- Explore [Examples](examples/) for common use casesitle: **Open-source, generic synthetic dataset generator.** Build the Docker image once and reuse it inside **any external project’s** `docker-compose.yml` by mounting your /configs.