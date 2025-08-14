from typing import Dict, Any, Optional, List
import os
import json
import time
from pathlib import Path

from src.core.providers.factory import get_provider
from src.core.template_registry import TemplateRegistry
from src.core.prompt_processor import PromptProcessor
from src.core.storage_manager import StorageManager
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class DataGenerator:
    """
    Core component for generating synthetic datasets using configurable templates and LLM providers.

    This class orchestrates the entire dataset generation pipeline, from loading dataset structure
    definitions and prompt templates to generating content via LLM providers and saving the output
    in the specified structure.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the generator
        model_provider (ModelProvider): Provider for LLM access
        model_client (ModelProvider): Client for interacting with the LLM
        template_registry (TemplateRegistry): Registry for dataset structures and prompt templates
        prompt_processor (PromptProcessor): Processor for template variable substitution
        storage_manager (StorageManager): Manager for file operations
        generation_defaults (Dict[str, Any]): Default generation parameters
        output_dir (str): Directory for generated datasets
        templates_dir (str): Directory for template files
        user_configs_dir (str): Directory for user configuration files
        data_dir (str): Directory for input data files

    Example:
        ```python
        # Initialize with configuration
        config = {"provider": {"name": "ollama", "model": "gemma3:1b-it-qat"}}
        generator = DataGenerator(config)

        # Generate a dataset
        output_path = generator.generate(
            structure_name="estonian_qa",
            prompt_template_name="qa_generator",
            num_examples=10,
            parameters={"language": "et", "topic": "finance"}
        )
        ```
    """

    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or {}

        # Initialize provider
        provider_config = self.config.get("provider", {})
        
        # Merge provider-specific configuration if applicable
        provider_name = provider_config.get("name", "ollama")
        if provider_name == "bedrock-anthropic":
            # Merge bedrock_anthropic config into provider config
            bedrock_config = self.config.get("bedrock_anthropic", {})
            provider_config = {**provider_config, **bedrock_config}
            logger.debug(f"Merged Bedrock configuration: {list(bedrock_config.keys())}")
        elif provider_name == "azure-openai":
            # Merge azure_openai config into provider config
            azure_config = self.config.get("azure_openai", {})
            provider_config = {**provider_config, **azure_config}
            logger.debug(f"Merged Azure OpenAI configuration: {list(azure_config.keys())}")
        
        self.model_provider = get_provider(provider_config)

        # Create model client from provider
        self.model_client = self.model_provider if self.model_provider else None

        # Initialize template registry
        self.template_registry = TemplateRegistry(self.config)

        # Initialize prompt processor
        self.prompt_processor = PromptProcessor()

        # Initialize storage manager
        self.storage_manager = StorageManager()

        # Get generation defaults
        self.generation_defaults = self.config.get("generation", {})

        # Get directories config
        directories = self.config.get("directories", {})
        self.output_dir = directories.get("output", "output_datasets")
        self.templates_dir = directories.get("templates", "templates")
        self.user_configs_dir = directories.get("user_configs", "user_configs")
        self.data_dir = directories.get("input", "data")

        # Create necessary directories
        for dir_path in [self.output_dir, self.templates_dir, self.user_configs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def generate(
        self,
        structure_name: str,
        prompt_template_name: str,
        dataset_name: Optional[str] = None,
        output_base_path: Optional[str] = None,
        num_examples: Optional[int] = None,
        output_format: str = "json",
        parameters: Dict[str, Any] = None,
    ) -> str:
        """
        Generate a dataset

        Args:
            structure_name: Name of the dataset structure
            prompt_template_name: Name of the prompt template
            dataset_name: Name for the dataset (used if output_base_path not provided)
            output_base_path: Path where to save the dataset
            num_examples: Number of examples to generate (defaults to config)
            output_format: Format for output files
            parameters: Additional parameters for generation

        Returns:
            Path to the generated dataset
        """
        parameters = parameters or {}
        if "language" in parameters:
            if (
                not isinstance(parameters["language"], str)
                or len(parameters["language"]) > 10
            ):
                logger.warning("Invalid language parameter detected, using default")
                parameters["language"] = "et"

        if output_base_path and ".." in output_base_path:
            logger.warning(
                f"Path traversal attempt detected in output_base_path: {output_base_path}"
            )
            output_base_path = output_base_path.replace("..", "")

        num_examples = num_examples or self.generation_defaults.get(
            "default_num_examples", 5
        )

        # Determine output directory
        if output_base_path:
            output_dir = Path(output_base_path)
            effective_dataset_name = os.path.basename(output_base_path)
        elif dataset_name:
            output_dir = Path(self.output_dir) / dataset_name
            effective_dataset_name = dataset_name
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            effective_dataset_name = f"{structure_name}_{timestamp}"
            output_dir = Path(self.output_dir) / effective_dataset_name

        logger.info(f"Generating dataset '{effective_dataset_name}' at '{output_dir}'")

        # Load structure and template
        try:
            structure = self.template_registry.get_structure(structure_name)
            structure_root = structure.get("root", {})
        except KeyError as e:
            logger.error(f"Failed to load structure: {e}")
            raise ValueError(f"Dataset structure '{structure_name}' not found")

        try:
            template_path = self.template_registry.get_template_path(
                prompt_template_name
            )
            with open(template_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except (KeyError, FileNotFoundError) as e:
            logger.error(f"Failed to load prompt template: {e}")
            raise ValueError(f"Prompt template '{prompt_template_name}' not found")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "name": effective_dataset_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "structure_name": structure_name,
            "prompt_template_name": prompt_template_name,
            "num_examples_requested": num_examples,
            "output_format": output_format,
            "parameters": parameters,
            "output_path": str(output_dir.resolve()),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Generate data
        self._generate_data(
            output_dir=str(output_dir),
            structure_root=structure_root,
            prompt_template=prompt_template,
            num_examples=num_examples,
            output_format=output_format,
            parameters=parameters,
        )

        logger.info(f"Dataset generation completed: {output_dir}")
        return str(output_dir)

    def _generate_data(
        self,
        output_dir: str,
        structure_root: Dict[str, Any],  # Expecting the 'root' node
        prompt_template: str,
        num_examples: int,  # Total examples requested for the whole call
        output_format: str,  # The final output format (e.g., 'json')
        parameters: Dict[str, Any],
    ) -> None:
        """Generate data for files defined in the structure_root."""
        base_output_path = Path(output_dir.replace("\\", "/"))

        # Flatten the structure to get defined files relative to the root
        for relative_file_key, file_info in self._flatten_structure(structure_root):
            # file_info contains format specified in YAML, etc.
            # Use the output_format passed to the function as the definitive format
            file_format = output_format  # Override YAML format if needed, or use file_info['format']
            file_extension = f".{file_format}"

            # Construct the full path for the output file with forward slashes
            # relative_file_key is like 'faqs' from the YAML
            output_file_path = base_output_path / f"{relative_file_key}{file_extension}"

            # Log the file path being written to
            logger.info(f"Writing to file: {output_file_path}")

            # Ensure the directory for this file exists (important for nested structures if any)
            os.makedirs(output_file_path.parent, exist_ok=True)

            # Determine number of examples for *this specific file*
            # Use the relative_file_key for parameter lookup
            path_examples = self._get_path_examples(
                relative_file_key, num_examples, parameters
            )
            logger.info(
                f"Generating {path_examples} examples for file: {output_file_path}"
            )

            # Log provider information and batch support
            logger.info("=" * 80)
            logger.info(f"STARTING GENERATION FOR FILE: {relative_file_key}")
            logger.info(f"REQUESTED EXAMPLES: {path_examples}")
            logger.info(f"PROVIDER: {self.model_provider.__class__.__name__}")
            logger.info(f"BATCH SUPPORT: {self.model_provider.supports_batch_generation()}")
            logger.info("=" * 80)

            generated_items = []
            
            # Check if provider supports efficient batch generation
            if self.model_provider.supports_batch_generation() and path_examples > 1:
                logger.info(f"🚀 USING BATCH GENERATION for {path_examples} examples")
                
                try:
                    # Prepare common parameters for all samples
                    if self.config:
                        default_language = getattr(self.config, "DEFAULT_LANGUAGE", "et")
                        supported_languages = getattr(
                            self.config,
                            "SUPPORTED_LANGUAGES",
                            {"en": "English", "et": "Estonian", "fi": "Finnish"},
                        )
                        default_system_prompt = getattr(
                            self.config,
                            "DEFAULT_SYSTEM_PROMPT",
                            "You are a helpful assistant providing accurate information based on topic content.",
                        )
                    else:
                        default_language = "et"
                        supported_languages = {
                            "en": "English",
                            "et": "Estonian",
                            "fi": "Finnish",
                        }
                        default_system_prompt = "You are a helpful assistant providing accurate information based on topic content."
                    
                    current_language_code = parameters.get("language", default_language)
                    language_name = supported_languages.get(
                        current_language_code, current_language_code
                    )
                    current_system_prompt = parameters.get(
                        "system_prompt", default_system_prompt
                    )
                    
                    # Create base prompt parameters (same for all samples)
                    base_prompt_params = {
                        "path": relative_file_key,
                        "format": file_format,
                        "language_name": language_name,
                        "language_code": current_language_code,
                        "system_prompt": current_system_prompt,
                        **parameters,
                    }
                    
                    # Process template once for batch generation
                    prompt = self.prompt_processor.process(prompt_template, base_prompt_params)
                    
                    # Generate all samples in batches
                    batch_size = min(path_examples, 10)  # Limit batch size to avoid token limits
                    remaining_examples = path_examples
                    
                    while remaining_examples > 0:
                        current_batch_size = min(remaining_examples, batch_size)
                        
                        logger.info(f"📦 Generating batch of {current_batch_size} examples...")
                        
                        # Single batch call
                        batch_responses = self.model_provider.generate_batch(
                            prompt, 
                            current_batch_size
                        )
                        
                        # Process each response in the batch
                        for i, content in enumerate(batch_responses):
                            try:
                                # Batch responses from providers are already properly formatted
                                if file_format == "json":
                                    try:
                                        # Try to parse as JSON first
                                        parsed = json.loads(content)
                                        if isinstance(parsed, list):  # If model returns a list for one call
                                            generated_items.extend(parsed)
                                        else:  # Assume model returns one item per call
                                            generated_items.append(parsed)
                                        logger.debug(f"BATCH SAMPLE {i+1}: parsed JSON successfully")
                                    except json.JSONDecodeError:
                                        # If it's not JSON, try to extract JSON
                                        extracted = self.prompt_processor.extract_json(content)
                                        if extracted:
                                            try:
                                                parsed = json.loads(extracted)
                                                if isinstance(parsed, list):
                                                    generated_items.extend(parsed)
                                                else:
                                                    generated_items.append(parsed)
                                                logger.debug(f"BATCH SAMPLE {i+1}: extracted and parsed JSON")
                                            except json.JSONDecodeError:
                                                # If extraction also fails, skip this sample
                                                logger.debug(f"BATCH SAMPLE {i+1}: could not parse as JSON, skipping")
                                                continue
                                        else:
                                            logger.debug(f"BATCH SAMPLE {i+1}: no JSON found, skipping")
                                            continue
                                else:  # For text or other formats, append raw content
                                    generated_items.append(content)
                                    logger.debug(f"BATCH SAMPLE {i+1}: using raw content")
                                
                            except Exception as e:
                                logger.error(f"Failed to process batch sample {i+1}: {str(e)}")
                        
                        remaining_examples -= current_batch_size
                        
                except Exception as e:
                    logger.error(f"BATCH GENERATION FAILED: {str(e)}")
                    logger.info("FALLING BACK to individual generation")
                    # Clear any partial results and fall through to individual generation
                    generated_items = []
            
            # Individual generation (for Ollama or batch fallback)
            if len(generated_items) < path_examples:
                remaining = path_examples - len(generated_items)
                logger.info(f"USING INDIVIDUAL GENERATION for {remaining} examples")
                
                for i in range(remaining):
                    current_example_num = len(generated_items) + 1
                    logger.info(f"GENERATING INDIVIDUAL EXAMPLE {current_example_num}/{path_examples}")
                    
                    try:
                        if self.config:
                            default_language = getattr(self.config, "DEFAULT_LANGUAGE", "et")
                            supported_languages = getattr(
                                self.config,
                                "SUPPORTED_LANGUAGES",
                                {"en": "English", "et": "Estonian", "fi": "Finnish"},
                            )
                            default_system_prompt = getattr(
                                self.config,
                                "DEFAULT_SYSTEM_PROMPT",
                                "You are a helpful assistant providing accurate information based on topic content.",
                            )
                        else:
                            default_language = "et"
                            supported_languages = {
                                "en": "English",
                                "et": "Estonian",
                                "fi": "Finnish",
                            }
                            default_system_prompt = "You are a helpful assistant providing accurate information based on topic content."
                        
                        current_language_code = parameters.get("language", default_language)
                        language_name = supported_languages.get(
                            current_language_code, current_language_code
                        )
                        current_system_prompt = parameters.get(
                            "system_prompt", default_system_prompt
                        )
                        
                        prompt_params = {
                            "index": current_example_num - 1,
                            "path": relative_file_key,
                            "format": file_format,
                            "language_name": language_name,
                            "language_code": current_language_code,
                            "system_prompt": current_system_prompt,
                            **parameters,
                        }

                        logger.debug(
                            f"Using language: {language_name} ({current_language_code})"
                        )
                        logger.debug(f"Prompt params: {prompt_params}")
                        prompt = self.prompt_processor.process(prompt_template, prompt_params)
                        content = self.model_client.generate(prompt)

                        # Process content (especially for JSON aggregation)
                        if file_format == "json":
                            try:
                                parsed = json.loads(content)
                                if isinstance(
                                    parsed, list
                                ):  # If model returns a list for one call
                                    generated_items.extend(parsed)
                                else:  # Assume model returns one item per call
                                    generated_items.append(parsed)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Non-JSON response for item {current_example_num}: {content[:100]}..."
                                )
                                # Optionally try to extract or add raw content
                                extracted = self.prompt_processor.extract_json(content)
                                if extracted:
                                    try:
                                        parsed = json.loads(extracted)
                                        if isinstance(parsed, list):
                                            generated_items.extend(parsed)
                                        else:
                                            generated_items.append(parsed)
                                    except json.JSONDecodeError:
                                        logger.debug(
                                            f"Extracted content still not JSON: {extracted[:100]}..."
                                        )
                                else:
                                    continue
                        else:  # For text or other formats, append raw content
                            generated_items.append(content)
                            
                        logger.info(f"INDIVIDUAL EXAMPLE {current_example_num} completed")
                        
                        # Wait between individual requests
                        if i < remaining - 1:
                            wait_time = self.config.get("processing", {}).get("wait_between_requests", 1)
                            if wait_time > 0:
                                logger.info(f"Waiting {wait_time}s before next example...")
                                time.sleep(wait_time)
                                
                    except Exception as e:
                        logger.error(f"INDIVIDUAL EXAMPLE {current_example_num} FAILED: {str(e)}")

            logger.info("=" * 80)
            logger.info(f"GENERATION COMPLETED FOR FILE: {relative_file_key}")
            logger.info(f"TOTAL EXAMPLES GENERATED: {len(generated_items)}/{path_examples}")
            logger.info("=" * 80)

            # Save all generated items to the single file
            logger.info(f"Writing {len(generated_items)} items to {output_file_path}")

            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
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

    def _flatten_structure(
        self, structure_node: Dict[str, Any], current_path: str = ""
    ) -> List[tuple]:
        """
        Flatten a hierarchical structure definition into a list of file paths with their metadata.

        This method recursively traverses the nested directory structure defined in the dataset
        structure YAML, converting it to a flat list of files that need to be generated. It handles
        both files at the current level and files within subdirectories.

        Args:
            structure_node (Dict[str, Any]): The current node in the structure tree, containing
                'files' and 'subdirectories' keys
            current_path (str, optional): The relative path to the current node. Defaults to ""

        Returns:
            List[tuple]: A list of tuples, each containing:
                - relative_file_key (str): The path to the file relative to the output directory
                - file_info (Dict): File metadata from the structure definition

        Example:
            For a structure like:
            ```
            root:
            files:
                faqs: {}
            subdirectories:
                examples:
                files:
                    sample1: {}
            ```

            Returns:
            [('faqs', {}), ('examples/sample1', {})]
        """
        items = []
        if ".." in current_path:
            logger.warning(f"Path traversal attempt detected in: {current_path}")
            current_path = current_path.replace("..", "")
        base_path = Path(current_path)
        # Files at current level
        if "files" in structure_node and structure_node["files"]:
            for file_key, file_info in structure_node["files"].items():
                # The key itself (e.g., 'faqs') is the identifier relative to current path
                items.append((str(base_path / file_key), file_info))
        # Recurse into subdirectories
        if "subdirectories" in structure_node and structure_node["subdirectories"]:
            for dir_key, dir_content in structure_node["subdirectories"].items():
                items.extend(
                    self._flatten_structure(dir_content, str(base_path / dir_key))
                )
        return items

    def _get_path_examples(
        self, relative_file_key: str, total_examples: int, parameters: Dict[str, Any]
    ) -> int:
        """
        Determine how many examples to generate for a specific file path.

        This method allows for file-specific control over the number of examples to generate.
        It checks if a specific count parameter exists for this file (by normalizing the path
        and looking for a corresponding parameter), and if found, uses that value instead of
        the global total_examples value.

        Args:
            relative_file_key (str): The relative path to the file (e.g., 'examples/sample1')
            total_examples (int): The default number of examples to generate if no specific count found
            parameters (Dict[str, Any]): Generation parameters that may contain file-specific counts

        Returns:
            int: Number of examples to generate for this specific file

        """
        normalized_key = relative_file_key.replace(os.sep, "_")
        count_param = f"{normalized_key}_count"
        if count_param in parameters:
            try:
                return int(parameters[count_param])
            except (ValueError, TypeError):
                logger.warning(f"Invalid count for {count_param}")
        return total_examples
