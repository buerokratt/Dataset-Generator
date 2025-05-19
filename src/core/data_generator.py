"""
Core data generation logic for the Synthetic Dataset Generation Service
"""
import os
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.model_client import OllamaClient
from core.prompt_processor import PromptProcessor
from core.storage_manager import StorageManager
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")

class DataGenerator:
    """Core component for generating synthetic datasets"""
    
    def __init__(self, model_client: OllamaClient, prompt_processor: PromptProcessor, config=None):
        self.model_client = model_client
        self.prompt_processor = prompt_processor
        self.storage_manager = StorageManager()
        self.config = config

        
    def generate(
        self,
        structure_name: str,
        prompt_template_name: str,
        dataset_name: Optional[str] = None, # Keep for backward compatibility or alternative use
        output_base_path: Optional[str] = None, # Added parameter
        num_examples: int = 100,
        output_format: str = "json", # Default format if not specified elsewhere
        parameters: Dict[str, Any] = {}
    ) -> str:
        """
        Generate a synthetic dataset based on the specified parameters.
        Prioritizes output_base_path if provided.
        """
        # --- Determine Output Directory ---
        if output_base_path:
            # Use the explicitly provided path
            output_dir = Path(output_base_path)
            # Derive a name for logging/metadata if dataset_name wasn't provided
            effective_dataset_name = dataset_name or output_dir.name
        elif dataset_name:
            # Fallback to using dataset_name under the configured base datasets directory
            # Assumes storage_manager provides the base path
            base_datasets_dir = self.storage_manager.get_datasets_base_dir() # Requires this method in StorageManager
            output_dir = Path(base_datasets_dir) / dataset_name
            effective_dataset_name = dataset_name
        else:
            # Should be caught by API validation, but raise error defensively
            raise ValueError("Cannot determine output directory: provide 'dataset_name' or 'output_base_path'")

        logger.info(f"Generating dataset '{effective_dataset_name}' at '{output_dir}'")

        # --- Load Configs ---
        structure_config = self._load_structure(structure_name) # Load the full structure dict
        structure_root = structure_config.get('root', {}) # Get the 'root' node for processing
        prompt_template = self._load_prompt_template(prompt_template_name)

        # --- Prepare Directory and Metadata ---
        self.storage_manager.prepare_directory(str(output_dir)) # Create directory if needed

        metadata = {
            "name": effective_dataset_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "structure_name": structure_name,
            "prompt_template_name": prompt_template_name,
            "num_examples_requested": num_examples, # Clarify this is requested total
            "output_format": output_format, # Record the final format used
            "parameters": parameters,
            "output_path": str(output_dir.resolve()) # Store absolute path
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # --- Generate Data ---
        # Pass the 'root' of the structure to the generation logic
        self._generate_data(
            output_dir=str(output_dir),
            structure_root=structure_root, # Pass the relevant part of the structure
            prompt_template=prompt_template,
            num_examples=num_examples, # This is the total requested
            output_format=output_format, # Pass the determined format
            parameters=parameters
        )

        return str(output_dir)
    
    def _load_structure(self, structure_name: str) -> Dict[str, Any]:
        """
        Load a dataset structure from file.
        
        Args:
            structure_name: Name of the structure file (without extension)
        
        Returns:
            The dataset structure as a dictionary
        """
        templates_dir = self.storage_manager.get_templates_dir()
        user_configs_dir = self.storage_manager.get_user_configs_dir()
        
        # Try user configs first
        user_path = f"{user_configs_dir}/dataset_structures/{structure_name}.yaml"
        if os.path.exists(user_path):
            logger.info(f"Loading user structure '{structure_name}' from {user_path}")
            with open(user_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Try default templates
        default_path = f"{templates_dir}/dataset_structures/{structure_name}.yaml"
        if os.path.exists(default_path):
            logger.info(f"Loading structure '{structure_name}' from {default_path}")
            with open(default_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        raise FileNotFoundError(f"Dataset structure '{structure_name}' not found")
    
    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from file"""
        
        # Get base directories from config
        templates_dir = self.storage_manager.get_templates_dir()
        user_configs_dir = self.storage_manager.get_user_configs_dir()
        
        # Define all possible template locations in order of precedence
        template_locations = [
            # User configs (highest priority)
            f"{user_configs_dir}/prompts/faqs/{template_name}.txt",
            f"{user_configs_dir}/prompts/conversations/{template_name}.txt",
            f"{user_configs_dir}/prompts/{template_name}.txt",
            
            # Default templates
            f"{templates_dir}/prompts/examples/faqs/{template_name}.txt",
            f"{templates_dir}/prompts/examples/conversations/{template_name}.txt",
            f"{templates_dir}/prompts/default/{template_name}.txt",
            f"{templates_dir}/prompts/examples/{template_name}.txt"
        ]
        
        # Try each location
        for location in template_locations:
            if os.path.exists(location):
                logger.info(f"Loading prompt template from {location}")
                with open(location, "r", encoding="utf-8") as f:
                    return f.read()
        
        # Not found
        raise FileNotFoundError(f"Prompt template '{template_name}' not found in any search location")
    

    def _create_directory_structure(self, base_dir: str, structure_node: Dict[str, Any]) -> None:
         """Create directory structure based on 'subdirectories' in a node."""
         if 'subdirectories' in structure_node and structure_node['subdirectories']:
             for name, content in structure_node['subdirectories'].items():
                 dir_path = Path(base_dir) / name
                 dir_path.mkdir(parents=True, exist_ok=True)
                 self._create_directory_structure(str(dir_path), content) # Recurse
    
    def _generate_data(
        self,
        output_dir: str,
        structure_root: Dict[str, Any], # Expecting the 'root' node
        prompt_template: str,
        num_examples: int, # Total examples requested for the whole call
        output_format: str, # The final output format (e.g., 'json')
        parameters: Dict[str, Any]
    ) -> None:
        """Generate data for files defined in the structure_root."""
         # Convert Windows backslashes to forward slashes for Docker compatibility
        base_output_path = Path(output_dir.replace('\\', '/'))

        # Flatten the structure to get defined files relative to the root
        for relative_file_key, file_info in self._flatten_structure(structure_root):
            # file_info contains format specified in YAML, etc.
            # Use the output_format passed to the function as the definitive format
            file_format = output_format # Override YAML format if needed, or use file_info['format']
            file_extension = f".{file_format}"

            # Construct the full path for the output file with forward slashes
            # relative_file_key is like 'faqs' from the YAML
            output_file_path = base_output_path / f"{relative_file_key}{file_extension}"

            # Ensure parent directory exists with proper paths
            os.makedirs(os.path.dirname(str(output_file_path)), exist_ok=True)

            # Log the file path being written to
            logger.info(f"Writing to file: {output_file_path}")

            # Ensure the directory for this file exists (important for nested structures if any)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine number of examples for *this specific file*
            # Use the relative_file_key for parameter lookup
            path_examples = self._get_path_examples(relative_file_key, num_examples, parameters)
            logger.info(f"Generating {path_examples} examples for file: {output_file_path}")

            generated_items = []
            for i in range(path_examples):
                if self.config:
                    default_language = getattr(self.config, 'DEFAULT_LANGUAGE', 'et')
                    supported_languages = getattr(self.config, 'SUPPORTED_LANGUAGES', {'en': 'English', 'et': 'Estonian', 'fi': 'Finnish'})
                    default_system_prompt = getattr(self.config, 'DEFAULT_SYSTEM_PROMPT', 'You are a helpful assistant providing accurate information based on topic content.')
                else:
                    default_language = 'et'
                    supported_languages = {'en': 'English', 'et': 'Estonian', 'fi': 'Finnish'}
                    default_system_prompt = 'You are a helpful assistant providing accurate information based on topic content.'
                current_language_code = parameters.get("language", default_language)
                language_name = supported_languages.get(current_language_code, current_language_code)
                current_system_prompt = parameters.get("system_prompt", default_system_prompt)
                prompt_params = { 
                    "index": i, 
                    "path": relative_file_key, 
                    "format": file_format,
                    "language_name": language_name,
                    "language_code": current_language_code,
                    "system_prompt": current_system_prompt,
                    **parameters 
                }
                
                logger.debug(f"Using language: {language_name} ({current_language_code})")
                logger.debug(f"Prompt params: {prompt_params}")
                prompt = self.prompt_processor.process(prompt_template, prompt_params)
                content = self.model_client.generate(prompt)

                # Process content (especially for JSON aggregation)
                if file_format == "json":
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list): # If model returns a list for one call
                            generated_items.extend(parsed)
                        else: # Assume model returns one item per call
                            generated_items.append(parsed)
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON response for item {i}: {content[:100]}...")
                        # Optionally try to extract or add raw content
                        extracted = self.prompt_processor.extract_json(content)
                        if extracted:
                             try:
                                 parsed = json.loads(extracted)
                                 if isinstance(parsed, list): generated_items.extend(parsed)
                                 else: generated_items.append(parsed)
                             except json.JSONDecodeError:
                                 logger.warning(f"Extracted content still not JSON: {extracted[:100]}...")
                                 # generated_items.append({"error": "invalid_json", "raw": content})
                        else:
                             # generated_items.append({"error": "invalid_json", "raw": content})
                             pass # Skip invalid items silently or log

                else: # For text or other formats, append raw content
                    generated_items.append(content)

            # Save all generated items to the single file
            logger.info(f"Writing {len(generated_items)} items to {output_file_path}")
            with open(output_file_path, "w", encoding='utf-8') as f:
                if file_format == "json":
                    json.dump(generated_items, f, indent=2)
                else: # Assume text, join with newlines
                    f.write("\n".join(generated_items))

            try:
                with open(output_file_path, "w", encoding='utf-8') as f:
                    if file_format == "json":
                        json.dump(generated_items, f, indent=2, ensure_ascii=False)
                    else:  # Assume text, join with newlines
                        f.write("\n".join(generated_items))
    
                # Verify the file exists after writing
                if os.path.exists(output_file_path):
                    logger.info(f"Successfully wrote file: {output_file_path}")
                else:
                    logger.error(f"Failed to create file: {output_file_path}")
            except Exception as e:
                logger.error(f"Error writing to {output_file_path}: {e}")
    
    def _flatten_structure(self, structure_node: Dict[str, Any], current_path: str = "") -> List[tuple]:
        """ Flattens structure node, returning list of (relative_key, file_info)."""
        items = []
        base_path = Path(current_path)
        # Files at current level
        if 'files' in structure_node and structure_node['files']:
            for file_key, file_info in structure_node['files'].items():
                 # The key itself (e.g., 'faqs') is the identifier relative to current path
                 items.append((str(base_path / file_key), file_info))
        # Recurse into subdirectories
        if 'subdirectories' in structure_node and structure_node['subdirectories']:
            for dir_key, dir_content in structure_node['subdirectories'].items():
                 items.extend(self._flatten_structure(dir_content, str(base_path / dir_key)))
        return items
    
    def _get_path_examples(self, relative_file_key: str, total_examples: int, parameters: Dict[str, Any]) -> int:
         """Determine examples for a specific file key (e.g., 'faqs')."""
         # Normalize key: replace path separators if any, though likely none for simple structures
         normalized_key = relative_file_key.replace(os.sep, "_")
         count_param = f"{normalized_key}_count"
         if count_param in parameters:
             try: return int(parameters[count_param])
             except (ValueError, TypeError): logger.warning(f"Invalid count for {count_param}")
         return total_examples