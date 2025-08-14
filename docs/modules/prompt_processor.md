# Prompt Processor Module

## Purpose and Public Surface

The Prompt Processor module provides **template processing and content extraction** capabilities for the synthetic dataset generation pipeline. It handles variable substitution in prompt templates and robust extraction of structured data (particularly JSON) from LLM responses. This component acts as a bridge between template-based prompts and the model responses, ensuring clean data extraction even from imperfectly formatted outputs.

### Key Features

- **Template Variable Substitution**: Uses Python's `string.Template` for `${variable}` syntax processing
- **Robust JSON Extraction**: Multiple extraction patterns to handle markdown-wrapped, malformed, or text-embedded JSON
- **Error Resilience**: Graceful handling of invalid templates and malformed model responses
- **Content Validation**: JSON validation before extraction to ensure data integrity

### Public API

| Component | Location | Purpose |
|-----------|----------|---------|
| `PromptProcessor` | [`src/core/prompt_processor.py`](../../src/core/prompt_processor.py) | Main processing class for templates and content extraction |
| `process()` | [`src/core/prompt_processor.py`](../../src/core/prompt_processor.py) | Template variable substitution with parameters |
| `extract_json()` | [`src/core/prompt_processor.py`](../../src/core/prompt_processor.py) | JSON extraction from mixed-content responses |

## Key Classes and Functions

### PromptProcessor Class

**Location**: [`src/core/prompt_processor.py`](../../src/core/prompt_processor.py)

**Purpose**: Processes prompt templates with variable substitution and extracts structured content from model responses.

```python
class PromptProcessor:
    def process(self, template: str, params: Dict[str, Any]) -> str:
        """Process a prompt template with parameter substitution"""
        
    def extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain non-JSON content"""
```

**Used by**:
- [`src/core/data_generator.py`](../../src/core/data_generator.py) - Template processing and response extraction in generation pipeline

### process() Method

**Location**: [`src/core/prompt_processor.py`](../../src/core/prompt_processor.py)

**Purpose**: Substitutes variables in prompt templates using `${variable}` syntax.

**Input/Output**:
- **Input**: Template string with `${variable}` placeholders, parameter dictionary
- **Output**: Processed template with variables substituted
- **Side Effects**: Logs warnings for unresolved template variables

**Template Variables** (commonly used):
- `${system_prompt}` - System instructions for the model
- `${language_name}` - Target language name (e.g., "Estonian")
- `${language_code}` - Target language code (e.g., "et")
- `${difficulty}` - Generation difficulty level
- `${style}` - Output style preferences
- `${file_name}` - Source file name
- `${file_content}` - Source file content
- `${topic}` - Topic information
- `${index}` - Generation item index
- `${path}` - Output path identifier

### extract_json() Method

**Location**: [`src/core/prompt_processor.py`](../../src/core/prompt_processor.py)

**Purpose**: Extracts valid JSON from model responses that may contain markdown formatting, explanations, or other text.

**Extraction Patterns**:
1. **Markdown Code Blocks**: ````json { "data": "value" } ```
2. **Plain JSON Objects**: `{ "data": "value" }`
3. **Fallback**: Returns original text if no valid JSON found

**Input/Output**:
- **Input**: Raw model response text (potentially mixed content)
- **Output**: Extracted JSON string or original text
- **Side Effects**: Validates JSON structure before returning

## Inputs/Outputs and Side Effects

### File System Operations
- **Read**: Template files from [`templates/prompts/`](../../templates/prompts/) and [`user_configs/prompts/`](../../user_configs/prompts/)
- **No Writes**: Pure processing component, no file system modifications

### Network Operations
- **None**: No direct network operations

### Logging Side Effects
- **Warning**: Unresolved template variables during substitution
- **Error**: Template processing failures
- **Debug**: Template parameter details

## Example Usage

### Basic Template Processing

```python
from src.core.prompt_processor import PromptProcessor

processor = PromptProcessor()

# Load template from user_configs/prompts/institute_topic_question.txt
template = """${system_prompt}

Generate a realistic user question in ${language_name} about:
Topic: ${file_name}
Content: "${file_content}"

Difficulty: ${difficulty}
Style: ${style}
"""

params = {
    "system_prompt": "You are a helpful assistant...",
    "language_name": "Estonian", 
    "file_name": "social_services",
    "file_content": "Information about social benefits...",
    "difficulty": "medium",
    "style": "conversational"
}

processed_prompt = processor.process(template, params)
```

### JSON Extraction from Model Response

```python
# Model response with markdown formatting
response = '''Here's the generated question:

```json
{
    "question": "Kuidas saan taotleda sotsiaaltoetust?",
    "category": "social_services",
    "difficulty": "medium"
}
```

This question asks about applying for social benefits.'''

# Extract clean JSON
json_content = processor.extract_json(response)
# Returns: {"question": "Kuidas saan taotleda sotsiaaltoetust?", ...}
```

### Real-World Usage in Data Generation

From [`src/core/data_generator.py`](../../src/core/data_generator.py):

```python
# Template processing during dataset generation
prompt_params = {
    "index": i,
    "path": relative_file_key,
    "format": file_format,
    "language_name": language_name,
    "language_code": current_language_code,
    "system_prompt": current_system_prompt,
    **parameters  # User-provided parameters
}

prompt = self.prompt_processor.process(prompt_template, prompt_params)
content = self.model_client.generate(prompt)

# JSON extraction from model response
if file_format == "json":
    try:
        parsed = json.loads(content)
        generated_items.append(parsed)
    except json.JSONDecodeError:
        # Fallback to extraction
        extracted = self.prompt_processor.extract_json(content)
        if extracted:
            parsed = json.loads(extracted)
            generated_items.append(parsed)
```

## Cross-Links and Configuration

### Template Sources
- **User Templates**: [`user_configs/prompts/`](../../user_configs/prompts/) - Agency-specific templates (highest priority)
- **Default Templates**: [`templates/prompts/default/`](../../templates/prompts/default/) - System default templates
- **Example Templates**: [`templates/prompts/examples/`](../../templates/prompts/examples/) - Template examples

### Template Resolution Order
1. `user_configs/prompts/faqs/{template_name}.txt`
2. `user_configs/prompts/conversations/{template_name}.txt` 
3. `user_configs/prompts/{template_name}.txt`
4. `templates/prompts/examples/faqs/{template_name}.txt`
5. `templates/prompts/examples/conversations/{template_name}.txt`
6. `templates/prompts/default/{template_name}.txt`

### Configuration Integration
- **Parameter Sources**: [`config/config.yaml`](../../config/config.yaml) - Default generation parameters
- **Language Settings**: Configured via `generation.default_language` and `generation.parameters`
- **Template Variables**: Dynamically populated from data sources and configuration

### Related Components
- **Data Generator**: [`src/core/data_generator.py`](../../src/core/data_generator.py) - Primary consumer of template processing
- **Storage Manager**: [`src/core/storage_manager.py`](../../src/core/storage_manager.py) - Template file location resolution
- **Configuration**: [`src/core/config.py`](../../src/core/config.py) - Parameter defaults and overrides
