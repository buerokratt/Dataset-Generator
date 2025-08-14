# AWS Bedrock Anthropic Provider Guide

This guide covers the complete setup and usage of the AWS Bedrock Anthropic provider for the Estonian Government AI Dataset Generator.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

## Overview

The AWS Bedrock Anthropic provider enables the Dataset Generator to use Anthropic Claude models through AWS Bedrock service. This integration provides:

- **Enterprise-grade reliability** with AWS infrastructure
- **Advanced AI capabilities** through Claude 3 and 3.5 models
- **Cost-effective scaling** with pay-per-use pricing
- **Security compliance** with AWS security standards
- **Rate limiting and retry mechanisms** for production stability

### Supported Models

| Model ID | Model Name | Strengths | TPM Limit | RPM Limit |
|----------|------------|-----------|-----------|-----------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Claude 3.5 Sonnet | Most capable, latest | 200,000 | 1,000 |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Claude 3.5 Haiku | Fast, cost-effective | 400,000 | 2,000 |
| `anthropic.claude-3-opus-20240229-v1:0` | Claude 3 Opus | Most powerful | 80,000 | 400 |
| `anthropic.claude-3-sonnet-20240229-v1:0` | Claude 3 Sonnet | Balanced | 160,000 | 800 |
| `anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku | Fastest | 480,000 | 2,400 |

**Recommendation**: Use Claude 3.5 Sonnet for production workloads and Claude 3.5 Haiku for development/testing.

## Prerequisites

### AWS Account Setup

1. **AWS Account**: Active AWS account with Bedrock access
2. **IAM Permissions**: User/role with the following permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:InvokeModel",
           "bedrock:InvokeModelWithResponseStream",
           "bedrock:GetFoundationModel",
           "bedrock:ListFoundationModels"
         ],
         "Resource": [
           "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
         ]
       }
     ]
   }
   ```

3. **Model Access**: Request access to Anthropic models in AWS Bedrock console
   - Go to AWS Bedrock → Model access
   - Request access for Anthropic Claude models
   - Wait for approval (typically instant for Claude 3/3.5)

4. **AWS Credentials**: One of the following methods:
   - **Environment variables** (recommended for local development)
   - **IAM roles** (recommended for production)
   - **AWS CLI profiles**
   - **Instance profiles** (for EC2 deployment)

### Regional Availability

Bedrock with Anthropic models is available in these regions:

- `us-east-1` (N. Virginia) - **Recommended**
- `us-west-2` (Oregon)
- `eu-west-1` (Ireland)
- `ap-southeast-1` (Singapore)
- `ap-northeast-1` (Tokyo)

## Installation

### 1. Install Dependencies

The Bedrock provider dependencies are already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies added:
- `langchain==0.3.8` - LangChain framework
- `langchain-aws==0.2.4` - AWS Bedrock integration
- `boto3==1.35.87` - AWS SDK
- `tenacity==9.0.0` - Retry mechanisms

### 2. Verify Installation

Run the integration test:

```bash
python test_bedrock_integration.py
```

## Configuration

### Method 1: Environment Variables (Recommended)

Create a `.env` file from the example:

```bash
cp .env.bedrock.example .env
```

Edit `.env` with your AWS credentials:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Provider Selection
PROVIDER_NAME=bedrock-anthropic

# Bedrock Configuration
BEDROCK_MODEL_NAME=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_AWS_REGION=us-east-1
BEDROCK_TEMPERATURE=0.7
BEDROCK_MAX_TOKENS=4096
BEDROCK_TOP_P=1.0
BEDROCK_TPM_LIMIT=200000
```

### Method 2: Configuration File

Update `config/config.yaml`:

```yaml
provider:
  name: "bedrock-anthropic"

bedrock_anthropic:
  model_name: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  aws_region: "us-east-1"
  temperature: 0.7
  max_tokens: 4096
  top_p: 1.0
  tpm_limit: 200000
  stop_sequences: []
```

### Method 3: Docker Compose

Use the Bedrock override file:

```bash
# Copy and configure .env file
cp .env.bedrock.example .env
nano .env

# Run with Bedrock configuration
docker compose -f docker-compose.yml -f docker-compose.bedrock.yml up
```

### Configuration Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `model_name` | Bedrock model identifier | `claude-3-5-sonnet-20241022-v2:0` | `claude-3-5-haiku-20241022-v1:0` |
| `aws_region` | AWS region for Bedrock | `us-east-1` | `eu-west-1` |
| `temperature` | Model creativity (0.0-1.0) | `0.7` | `0.1` (factual), `0.9` (creative) |
| `max_tokens` | Maximum output tokens | `4096` | `1000` |
| `top_p` | Nucleus sampling (0.0-1.0) | `1.0` | `0.9` |
| `tpm_limit` | Tokens per minute limit | `200000` | `50000` (conservative) |
| `stop_sequences` | Stop generation sequences | `[]` | `["\n\n", "END"]` |

## Usage Examples

### Basic Dataset Generation

```python
from src.core.data_generator import DataGenerator

# Initialize with Bedrock provider
config = {
    "provider": {"name": "bedrock-anthropic"},
    "bedrock_anthropic": {
        "model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "aws_region": "us-east-1",
        "temperature": 0.7
    }
}

generator = DataGenerator(config)

# Generate dataset
output_path = generator.generate(
    structure_name="single_question",
    prompt_template_name="institute_topic_question",
    num_examples=10,
    parameters={"topic": "AI governance", "language": "et"}
)
```

### API Request with Provider Override

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "structure_name": "single_question",
    "prompt_template_name": "institute_topic_question",
    "num_examples": 5,
    "provider_override": {
      "name": "bedrock-anthropic",
      "model_name": "anthropic.claude-3-5-haiku-20241022-v1:0",
      "temperature": 0.1,
      "max_tokens": 1000
    },
    "parameters": {
      "topic": "digital transformation",
      "language": "et"
    }
  }'
```

### Bulk Generation

```bash
curl -X POST "http://localhost:8000/generate-bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "structure_name": "single_question",
        "prompt_template_name": "institute_topic_question",
        "num_examples": 10,
        "parameters": {"topic": "cybersecurity", "language": "et"}
      },
      {
        "structure_name": "topic_conversations",
        "prompt_template_name": "institute_topic_conversation",
        "num_examples": 5,
        "parameters": {"topic": "e-governance", "language": "et"}
      }
    ]
  }'
```

## Testing

### 1. Integration Test

Run the comprehensive integration test:

```bash
python test_bedrock_integration.py
```

Expected output:
```
=== Testing BedrockAnthropicProvider Direct Instantiation ===
✓ Provider initialized successfully
  Model: anthropic.claude-3-5-haiku-20241022-v1:0
  Region: us-east-1
  Temperature: 0.1

=== Testing Health Check ===
✓ Health check passed (2.31s)

=== Testing Text Generation ===
Test 1: Simple factual question
✓ Generation successful (1.87s)
Response (43 chars): The capital of Estonia is Tallinn.

=== Testing Factory Integration ===
✓ Factory created provider successfully
✓ Factory provider generation test: Hello! How can I assist you today?

🎉 All tests passed! Bedrock Anthropic provider is ready to use.
```

### 2. Unit Tests

Run provider-specific unit tests:

```bash
python -m pytest tests/test_bedrock_provider.py -v
```

### 3. Load Testing

Test rate limiting and performance:

```bash
python scripts/load_test_bedrock.py --concurrent=5 --requests=50
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

**Error**: `NoCredentialsError: Unable to locate credentials`

**Solutions**:
- Verify AWS credentials in environment variables
- Check IAM user has Bedrock permissions
- Ensure credentials are not expired

```bash
# Test credentials
aws sts get-caller-identity
```

#### 2. Model Access Denied

**Error**: `AccessDeniedException: You don't have access to the model`

**Solutions**:
- Request model access in AWS Bedrock console
- Verify region supports the model
- Check IAM permissions include Bedrock actions

#### 3. Rate Limiting

**Error**: `ThrottlingException: Rate exceeded`

**Solutions**:
- Reduce `tpm_limit` in configuration
- Implement request batching
- Use faster model (e.g., Claude 3.5 Haiku)

#### 4. Network Connectivity

**Error**: `EndpointConnectionError: Could not connect to the endpoint`

**Solutions**:
- Check internet connectivity
- Verify AWS region is correct
- Test with AWS CLI: `aws bedrock list-foundation-models --region us-east-1`

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("src.core.providers.bedrock_anthropic").setLevel(logging.DEBUG)
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

### Health Check Script

Quick connectivity test:

```bash
python -c "
from src.core.providers.bedrock_anthropic import BedrockAnthropicProvider
config = {'model_name': 'anthropic.claude-3-5-haiku-20241022-v1:0', 'aws_region': 'us-east-1'}
provider = BedrockAnthropicProvider(config)
print('Health check:', provider.health_check())
"
```

## Performance Optimization

### 1. Model Selection

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Production datasets** | Claude 3.5 Sonnet | Best quality/speed balance |
| **Development/testing** | Claude 3.5 Haiku | Fastest, most cost-effective |
| **Complex reasoning** | Claude 3 Opus | Highest capability |
| **High-volume generation** | Claude 3.5 Haiku | Highest rate limits |

### 2. Rate Limiting Strategy

```python
# Conservative settings for production
config = {
    "tpm_limit": 50000,  # 25% of max rate
    "temperature": 0.1,  # Consistent outputs
    "max_tokens": 1000   # Shorter responses
}

# Aggressive settings for development
config = {
    "tpm_limit": 200000,  # Near max rate
    "temperature": 0.7,   # More creative
    "max_tokens": 4096    # Full responses
}
```

### 3. Cost Optimization

**Input Tokens** (per 1K tokens):
- Claude 3.5 Sonnet: $3.00
- Claude 3.5 Haiku: $1.00
- Claude 3 Opus: $15.00

**Output Tokens** (per 1K tokens):
- Claude 3.5 Sonnet: $15.00
- Claude 3.5 Haiku: $5.00
- Claude 3 Opus: $75.00

**Strategies**:
1. Use shorter prompts when possible
2. Set appropriate `max_tokens` limits
3. Use Claude 3.5 Haiku for simple tasks
4. Monitor costs with AWS Cost Explorer

### 4. Monitoring and Metrics

The provider automatically logs metrics for MLflow tracking:

```python
# Metrics logged per request
{
    "provider": "bedrock-anthropic",
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350,
    "duration": 2.4,
    "temperature": 0.7,
    "max_tokens": 4096
}
```

View metrics in MLflow UI:
```bash
docker compose up mlflow
# Open http://localhost:5000
```

## Security Considerations

1. **Credential Management**:
   - Use IAM roles in production
   - Rotate access keys regularly
   - Never commit credentials to code

2. **Network Security**:
   - Use VPC endpoints for private connectivity
   - Enable CloudTrail for API logging
   - Implement network access controls

3. **Data Privacy**:
   - Bedrock doesn't use data for training
   - Consider data residency requirements
   - Implement data encryption in transit/at rest

4. **Access Control**:
   - Use least-privilege IAM policies
   - Implement resource-based policies
   - Monitor usage with CloudWatch

---

For additional support, please refer to:
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude)
- [Project Issues](https://github.com/rootcodelabs/Dataset-Generator/issues)
