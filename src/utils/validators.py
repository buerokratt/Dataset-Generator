"""
Validators for the Synthetic Dataset Generation Service
"""
import os
from typing import Dict, Any, Optional

from core.config import app_config
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

def validate_structure_exists(structure_name: str) -> bool:
    """
    Validate that a dataset structure exists
    
    Args:
        structure_name: Name of the structure to validate
        
    Returns:
        True if the structure exists
        
    Raises:
        ValidationError: If the structure does not exist
    """
    templates_dir = app_config.TEMPLATES_DIR
    user_configs_dir = app_config.USER_CONFIGS_DIR
    
    structure_exists = os.path.exists(f"{templates_dir}/dataset_structures/{structure_name}.yaml") or \
                      os.path.exists(f"{user_configs_dir}/dataset_structures/{structure_name}.yaml")
    
    if not structure_exists:
        raise ValidationError(f"Dataset structure '{structure_name}' not found", 404)
    
    return True

def validate_prompt_template_exists(prompt_template_name: str) -> bool:
    """
    Validate that a prompt template exists
    
    Args:
        prompt_template_name: Name of the prompt template to validate
        
    Returns:
        True if the prompt template exists
        
    Raises:
        ValidationError: If the prompt template does not exist
    """
    templates_dir = app_config.TEMPLATES_DIR
    user_configs_dir = app_config.USER_CONFIGS_DIR
    
    prompt_exists = (
        # User configs (highest priority)
        os.path.exists(f"{user_configs_dir}/prompts/faqs/{prompt_template_name}.txt") or
        os.path.exists(f"{user_configs_dir}/prompts/conversations/{prompt_template_name}.txt") or
        os.path.exists(f"{user_configs_dir}/prompts/{prompt_template_name}.txt") or
        
        # Default templates
        os.path.exists(f"{templates_dir}/prompts/examples/faqs/{prompt_template_name}.txt") or
        os.path.exists(f"{templates_dir}/prompts/examples/conversations/{prompt_template_name}.txt") or
        os.path.exists(f"{templates_dir}/prompts/default/{prompt_template_name}.txt") or
        os.path.exists(f"{templates_dir}/prompts/examples/{prompt_template_name}.txt")
    )
    
    if not prompt_exists:
        raise ValidationError(f"Prompt template '{prompt_template_name}' not found", 404)
    
    return True

def validate_output_path(dataset_name: Optional[str], output_base_path: Optional[str]) -> bool:
    """
    Validate that an output path can be determined
    
    Args:
        dataset_name: Name of the dataset
        output_base_path: Base path for output
        
    Returns:
        True if an output path can be determined
        
    Raises:
        ValidationError: If an output path cannot be determined
    """
    if not dataset_name and not output_base_path:
        raise ValidationError("Either 'dataset_name' or 'output_base_path' must be provided")
    
    return True

def validate_generation_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and possibly enhance generation parameters
    
    Args:
        parameters: Parameters for generation
        
    Returns:
        Validated and possibly enhanced parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate language
    if "language" in parameters:
        if parameters["language"] not in app_config.SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{parameters['language']}' not in supported languages. Using default.")
            parameters["language"] = app_config.DEFAULT_LANGUAGE
    else:
        parameters["language"] = app_config.DEFAULT_LANGUAGE
        
    # Ensure other required parameters have reasonable defaults
    if "difficulty" not in parameters:
        parameters["difficulty"] = "intermediate"
    
    return parameters

def validate_format(output_format: str) -> str:
    """
    Validate output format
    
    Args:
        output_format: Format to validate
        
    Returns:
        Validated format
        
    Raises:
        ValidationError: If format is invalid
    """
    if output_format not in app_config.SUPPORTED_FORMATS:
        logger.warning(f"Format '{output_format}' not supported. Using default.")
        return app_config.DEFAULT_SAVE_FORMAT
    
    return output_format