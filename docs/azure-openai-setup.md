# Azure OpenAI Integration Guide

## Overview

This document provides setup and testing instructions for the Azure OpenAI provider in the Estonian Government AI Dataset Generator.

## Features

- ✅ **Individual Generation**: Single API calls for one-off generations
- ✅ **Batch Generation**: Multiple samples in a single API call for efficiency  
- ✅ **Automatic Fallback**: Falls back to individual calls if batch fails
- ✅ **Rate Limiting**: Respects Azure OpenAI rate limits
- ✅ **Error Handling**: Robust retry logic with exponential backoff

## Setup Instructions

### 1. Azure OpenAI Prerequisites

You need an Azure OpenAI resource with:
- **Azure OpenAI Service** deployed in your Azure subscription
- **GPT model deployment** (e.g., gpt-4, gpt-35-turbo)
- **API credentials** (endpoint, API key, deployment name)

### 2. Environment Variables

Set the following environment variables:

```bash
# Required
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"  # Your deployment name
```

### 3. Install Dependencies

```bash
pip install langchain-openai
```

### 4. Configuration

Update `config/config.yaml`:

```yaml
provider:
  name: "azure-openai"

azure_openai:
  # Optional: Can use environment variables instead
  azure_endpoint: "https://your-resource.openai.azure.com/"
  api_key: "your-api-key" 
  deployment_name: "gpt-4"
  
  # Model parameters
  api_version: "2024-02-15-preview"
  temperature: 0.7
  max_tokens: 4096
  top_p: 1.0
  
  # Rate limiting
  tpm_limit: 200000  # Tokens per minute
  rpm_limit: 6000    # Requests per minute
  
  # Batch generation
  batch_generation:
    enabled: true
    max_batch_size: 10
    max_tokens_per_batch: 20000
```

## Testing

### Quick Test

Run the simple test script:

```bash
python test_azure_simple.py
```

This will:
1. ✅ Check environment variables
2. ✅ Test provider import and initialization  
3. ✅ Test individual generation
4. ✅ Test batch generation

### Comprehensive Test

Run the full test suite:

```bash
python test_azure_openai.py
```

## Usage Examples

### Individual Generation

```python
from core.providers.azure_openai import AzureOpenAIProvider
from core.config import Config

# Load config
config = Config()
config_dict = config._config.copy()
config_dict['provider']['name'] = 'azure-openai'

# Create provider
provider = AzureOpenAIProvider(config_dict)

# Generate single response
prompt = "Generate a question in Estonian: {\"question\": \"...\"}"
response = provider.generate(prompt)
print(response)
```

### Batch Generation

```python
# Generate multiple samples efficiently
prompt = "Generate a question in Estonian: {\"question\": \"...\"}"
responses = provider.generate_batch(prompt, num_samples=5)

for i, response in enumerate(responses, 1):
    print(f"Sample {i}: {response}")
```

### Integration with Data Generator

```python
from core.data_generator import DataGenerator
from core.config import Config

# Configure for Azure OpenAI
config = Config()
config._config['provider']['name'] = 'azure-openai'

# Create generator
generator = DataGenerator(config)

# Generate dataset - will automatically use batch generation if supported
generator.generate_dataset(
    structure_name="single_question",
    prompt_template_name="institute_topic_question", 
    dataset_name="test_azure",
    num_samples=10
)
```

## Performance Comparison

| Method | API Calls | Time (10 samples) | Cost Efficiency |
|--------|-----------|-------------------|-----------------|
| Individual | 10 calls | ~50-80s | Standard |
| Batch | 1 call | ~10-15s | ~80% reduction |

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```
   Error: Invalid API key or endpoint
   ```
   **Solution**: Check `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`

2. **Deployment Not Found**
   ```
   Error: The API deployment for this resource does not exist
   ```
   **Solution**: Verify `AZURE_OPENAI_DEPLOYMENT_NAME` matches your Azure deployment

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   **Solution**: Adjust `tpm_limit` and `rpm_limit` in config to match your Azure quotas

4. **Import Error**
   ```
   ImportError: No module named 'langchain_openai'
   ```
   **Solution**: Install dependency: `pip install langchain-openai`

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
from utils.logger import setup_logger
logger = setup_logger("test", level="DEBUG")
```

## Provider Comparison

| Feature | Ollama | Bedrock Anthropic | Azure OpenAI |
|---------|--------|-------------------|--------------|
| **Batch Support** | ❌ | ✅ | ✅ |
| **Local Hosting** | ✅ | ❌ | ❌ |
| **Enterprise Ready** | ⚠️ | ✅ | ✅ |
| **Cost Efficiency** | ✅ | 💰 | 💰 |
| **Estonian Language** | ⚠️ | ✅ | ✅ |

## Next Steps

1. **Set up Azure OpenAI credentials**
2. **Run test scripts to verify functionality**  
3. **Update your dataset generation configs**
4. **Monitor performance and costs**

For production use, consider:
- Setting up proper Azure authentication (Managed Identity)
- Implementing cost monitoring and alerts
- Fine-tuning rate limits based on your Azure quotas
- Adding custom retry policies for your use case
