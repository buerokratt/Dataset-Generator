"""
API routes for the Synthetic Dataset Generation Service
"""
from fastapi import FastAPI, HTTPException
from typing import  Dict, Any, Optional
from pydantic import BaseModel

from core.data_generator import DataGenerator
from core.prompt_processor import PromptProcessor
from core.model_client import OllamaClient
from utils.logger import setup_logger
from utils.validators import validate_structure_exists, validate_prompt_template_exists, validate_output_path, validate_generation_parameters, validate_format, ValidationError
from core.config import app_config
from src.utils.logger import logger

# logger.add(sink="synthetic_data_service.log")
setup_logger("synthetic-data-service", "INFO")

# Initialize FastAPI app
app = FastAPI(title="Synthetic Dataset Generation Service")

# Initialize core components with configuration from app_config
model_client = OllamaClient(
    api_url=app_config.OLLAMA_HOST,        # Host URL for LLM API
    model_name=app_config.MODEL_NAME,      # Name of the language model to use
    timeout=app_config.OLLAMA_TIMEOUT,     # Request timeout in seconds
    max_retries=app_config.OLLAMA_MAX_RETRIES,  # Number of retry attempts
    retry_delay=app_config.OLLAMA_RETRY_DELAY   # Delay between retries in seconds
)
prompt_processor = PromptProcessor()  # Initialize prompt template processor
data_generator = DataGenerator(
    model_client=model_client,            # LLM client for text generation
    prompt_processor=prompt_processor,    # Processor for handling templates
    config=app_config                     # Application configuration
)

# Define Pydantic model for API request validation
class GenerationRequest(BaseModel):
    """Request model for dataset generation"""
    dataset_name: Optional[str] = None       # Optional name for the dataset (used for output path)
    output_base_path: Optional[str] = None   # Optional explicit output path (overrides dataset_name)
    structure_name: str                      # Name of dataset structure YAML to use
    prompt_template_name: str                # Name of prompt template to use
    num_examples: int = 10                   # Number of examples to generate (default: 10)
    output_format: str = "json"              # Output format, either "json" or "text" (default: json)
    parameters: Dict[str, Any] = {}          # Additional parameters for template substitution

@app.get("/")
async def root():
    """Root endpoint for service health check"""
    return {"message": "Synthetic Dataset Generation Service"}

@app.post("/generate")
async def generate_dataset(request: GenerationRequest):
    """
    Generate a synthetic dataset based on the provided request
    
    Steps:
    1. Validate inputs (structure, template, output path)
    2. Process and validate parameters
    3. Generate dataset using data_generator
    4. Return success message with output path
    """
    try:
        # Validate inputs using validator functions
        validate_structure_exists(request.structure_name)      # Ensure structure YAML exists
        validate_prompt_template_exists(request.prompt_template_name)  # Ensure template exists
        validate_output_path(request.dataset_name, request.output_base_path)  # Ensure valid output path
        
        # Process and enhance parameters with defaults as needed
        validated_parameters = validate_generation_parameters(request.parameters)
        
        # Ensure output format is supported
        validated_format = validate_format(request.output_format)

        # Call data generator to create the dataset
        output_path = data_generator.generate(
            # Pass both path options, generator will determine which to use
            dataset_name=request.dataset_name,
            output_base_path=request.output_base_path,
            structure_name=request.structure_name,
            prompt_template_name=request.prompt_template_name,
            num_examples=request.num_examples,
            output_format=validated_format,
            parameters=validated_parameters
        )

        # Return success response with path to generated dataset
        return {"message": "Dataset generated successfully", "path": output_path}

    except ValidationError as ve:
        # Handle validation errors with appropriate status code
        logger.error(f"Validation error: {ve.message}")
        raise HTTPException(status_code=ve.status_code, detail=ve.message)
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        # Catch and log any other exceptions
        logger.exception("Error generating dataset")
        raise HTTPException(status_code=500, detail=str(e))