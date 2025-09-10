# Batch Generation Implementation

## Overview

This document describes the batch generation feature implemented for the Estonian Government AI Dataset Generator. The feature allows cloud-based LLM providers (like AWS Bedrock Anthropic) to generate multiple samples in a single API call, significantly improving cost-efficiency and performance.

## Architecture

### Provider Interface Updates

The `ModelProvider` abstract base class now includes:

```python
def generate_batch(self, prompt: str, num_samples: int, options: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate multiple samples from the same prompt"""
    
def supports_batch_generation(self) -> bool:
    """Return True if provider supports efficient batch generation"""
```

### Provider Implementations

#### 1. Ollama Provider (Default Behavior)
- **Batch Support**: `False`
- **Behavior**: Uses default implementation (multiple individual API calls)
- **No Changes**: Maintains existing behavior for backward compatibility

#### 2. Bedrock Anthropic Provider (New Batch Support)
- **Batch Support**: `True`
- **Behavior**: Single API call with batch prompt engineering
- **Fallback**: Automatically falls back to individual calls if batch fails

## Implementation Details

### Batch Generation Flow

```
1. DataGenerator checks provider.supports_batch_generation()
2. If True: Use generate_batch() for multiple samples
3. If False or batch fails: Fall back to individual generate() calls
4. Process all results consistently regardless of generation method
```

### Bedrock Batch Implementation

The Bedrock provider uses prompt engineering to request multiple samples:

```python
batch_prompt = f"""Generate {num_samples} diverse variations of the following request. 
Each variation should be unique and creative while following the same requirements.

Format your response as a JSON array with exactly {num_samples} elements...

Original prompt: {prompt}
"""
```

### Configuration

#### Global Batch Settings
```yaml
# config/config.yaml
batch_generation:
  enabled: true
  max_batch_size: 10
  fallback_on_failure: true

processing:
  wait_between_requests: 1  # Only for individual calls
```

#### Provider-Specific Settings
```yaml
bedrock_anthropic:
  batch_generation:
    enabled: true
    max_batch_size: 5
    max_tokens_per_batch: 20000
```

## Performance Improvements

### Cost Reduction
- **Before**: 10 samples = 10 API calls = 10 × (input_tokens + output_tokens) × price
- **After**: 10 samples = 1 API call = 1 × (input_tokens + 10×output_tokens) × price
- **Savings**: ~90% reduction in input token costs

### Latency Reduction
- **Before**: 10 × (network_latency + generation_time)
- **After**: 1 × (network_latency + generation_time)
- **Improvement**: ~90% reduction in total latency

### Rate Limiting Benefits
- **Before**: 10 separate requests against rate limits
- **After**: 1 request against rate limits
- **Result**: Allows higher throughput within provider limits

## Error Handling

### Robust Fallback Strategy
1. **Batch Generation Attempted**: Provider tries batch generation first
2. **JSON Parsing**: Attempts to extract structured results from batch response
3. **Validation**: Checks if correct number of samples returned
4. **Automatic Fallback**: Falls back to individual calls if any step fails
5. **Logging**: Comprehensive logging at each step for debugging

### Example Error Scenarios
- **Invalid JSON Response**: Falls back to individual calls
- **Wrong Number of Samples**: Falls back to individual calls  
- **Token Limit Exceeded**: Falls back with smaller batch size
- **Network Errors**: Retry with exponential backoff

## Usage Examples

### Basic Usage (Transparent)
```python
# DataGenerator automatically chooses best method
generator = DataGenerator(config)
results = generator.generate(
    structure_name="qa_dataset",
    prompt_template_name="question_generator",
    num_examples=10  # Will use batch if provider supports it
)
```

### Direct Provider Usage
```python
# Direct batch generation
provider = get_provider({"name": "bedrock-anthropic"})
samples = provider.generate_batch(
    prompt="Generate a question about AI",
    num_samples=5
)
```

### Configuration Override
```python
# Force individual generation even for batch-capable providers
config["batch_generation"]["enabled"] = False
generator = DataGenerator(config)
```

## Testing

### Test Coverage
- ✅ Provider batch support detection
- ✅ Batch generation success path
- ✅ Fallback to individual generation
- ✅ Configuration loading
- ✅ Integration with DataGenerator
- ✅ Error handling scenarios

### Running Tests
```bash
# Test batch generation functionality
python test_batch_generation.py

# Full integration test
python test_bedrock_integration.py
```

## Monitoring and Logging

### Batch Generation Logs
```
🚀 USING BATCH GENERATION for 10 samples
📦 Generating batch of 5 samples...
✅ BATCH SUCCESS: Generated 5 samples in 3.24s
📦 Generating batch of 5 samples...  
✅ BATCH SUCCESS: Generated 5 samples in 2.87s
TOTAL EXAMPLES GENERATED: 10/10
```

### Fallback Logs
```
❌ BATCH GENERATION FAILED: Could not parse JSON array
🔄 FALLING BACK to individual generation
🤖 GENERATING INDIVIDUAL EXAMPLE 1/10
✅ INDIVIDUAL EXAMPLE 1 completed
```

## Configuration Reference

### Environment Variables
```bash
# Provider selection
PROVIDER_NAME=bedrock-anthropic

# Bedrock configuration
BEDROCK_MODEL_NAME=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_AWS_REGION=us-east-1
BEDROCK_TPM_LIMIT=200000

# AWS credentials
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### YAML Configuration
```yaml
provider:
  name: "bedrock-anthropic"

bedrock_anthropic:
  model_name: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  aws_region: "us-east-1"
  temperature: 0.7
  max_tokens: 4096
  batch_generation:
    enabled: true
    max_batch_size: 5

batch_generation:
  enabled: true
  max_batch_size: 10
  fallback_on_failure: true
```

## Future Enhancements

### Planned Improvements
1. **Azure OpenAI Provider**: Add batch support for OpenAI GPT models
2. **Dynamic Batch Sizing**: Automatically adjust batch size based on token limits
3. **Parallel Batching**: Split large requests across multiple parallel batch calls
4. **Cost Tracking**: Add detailed cost estimation and tracking per batch
5. **Quality Metrics**: Compare batch vs individual generation quality

### Provider Roadmap
- ✅ AWS Bedrock Anthropic (Claude models)
- 🔄 Azure OpenAI (GPT models) - Planned
- 🔄 Google Vertex AI (Gemini models) - Future
- 🔄 Anthropic Direct API - Future

## Troubleshooting

### Common Issues

#### 1. Batch Generation Not Working
```
Check: provider.supports_batch_generation() returns True
Check: batch_generation.enabled = true in config
Check: AWS credentials are configured correctly
```

#### 2. Frequent Fallbacks to Individual Generation
```
Issue: Batch responses not parseable as JSON
Solution: Check model prompt engineering, adjust max_tokens
```

#### 3. Token Limit Errors
```
Issue: Batch prompt + responses exceed model limits
Solution: Reduce max_batch_size in configuration
```

#### 4. High Costs Despite Batch Generation
```
Check: Logs show "USING BATCH GENERATION" messages
Check: Not falling back to individual generation frequently
```

### Debug Commands
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Test provider capabilities
python -c "
from src.core.providers.factory import get_provider
p = get_provider({'name': 'bedrock-anthropic'})
print(f'Batch support: {p.supports_batch_generation()}')
"

# Test configuration loading
python -c "
from src.core.config import ConfigLoader
c = ConfigLoader.load()
print(f'Batch config: {c.get(\"batch_generation\", {})}')
"
```

## Migration Guide

### From Individual to Batch Generation
1. **No Code Changes Required**: Existing code automatically benefits
2. **Configuration Updates**: Add batch_generation section to config
3. **Provider Updates**: Ensure using batch-capable provider
4. **Testing**: Run test_batch_generation.py to verify
5. **Monitoring**: Check logs for batch generation usage

### Backward Compatibility
- ✅ All existing code continues to work unchanged
- ✅ Ollama provider maintains individual generation behavior
- ✅ Configuration defaults preserve existing behavior
- ✅ API contracts remain identical
- ✅ Output formats and structures unchanged
