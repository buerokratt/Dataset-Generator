import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.utils.logger import logger


class ConfigLoader:
    """
    Configuration loader with support for layered configurations and environment variables.

    This class provides a flexible configuration management system that loads settings
    from multiple sources with a clear precedence order:
    1. Environment variables (highest priority)
    2. Configuration files (YAML)
    3. Default values (lowest priority)

    It supports deep merging of nested configurations, automatic type conversion for
    environment variables, and dot-notation for accessing nested configuration paths.

    Attributes:
        DEFAULT_PATHS (List[str]): Default configuration file paths to try in order:
            - "config/config.yaml"
            - "src/config/config.yaml"
            - "/app/config/config.yaml"

    Class Methods:
        load(paths: Optional[List[str]] = None, env_prefix: str = "") -> Dict[str, Any]:
            Load and merge configuration from files and environment variables

        _deep_merge(base: Dict, overlay: Dict) -> Dict:
            Recursively merge two dictionaries

        _get_env_config(prefix: str) -> Dict[str, Any]:
            Extract configuration from environment variables

        _set_by_path(config: Dict, path: str, value: Any):
            Set a value in a dictionary using a dotted path notation

    Example:
        ```python
        # Load using default paths
        config = ConfigLoader.load()

        # Access configuration values
        model_name = config.get("provider", {}).get("model_name", "default_model")

        # Load with custom path and environment prefix
        custom_config = ConfigLoader.load(
            paths=["my_config.yaml"],
            env_prefix="APP_"
        )
        ```
    """

    DEFAULT_PATHS = [
        "config/config.yaml",
        "src/config/config.yaml",
        "/app/config/config.yaml"
    ]

    @classmethod
    def load(
        cls, paths: Optional[List[str]] = None, env_prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Load configuration from multiple sources with fallbacks

        Args:
            paths: List of configuration file paths to try
            env_prefix: Prefix for environment variables

        Returns:
            Merged configuration dictionary
        """
        paths = paths or cls.DEFAULT_PATHS
        config = {}

        # Try each path in order
        for path in paths:
            if Path(path).exists():
                logger.info(f"Loading configuration from {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            # Deep merge with existing config
                            config = cls._deep_merge(config, file_config)
                except Exception as e:
                    logger.error(f"Error loading config from {path}: {e}")

        # Override with environment variables
        env_config = cls._get_env_config(env_prefix)
        if env_config:
            config = cls._deep_merge(config, env_config)

        return config

    @classmethod
    def _deep_merge(cls, base: Dict, overlay: Dict) -> Dict:
        """
        Recursively merge two dictionaries

        Args:
            base: Base dictionary
            overlay: Dictionary to overlay (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @classmethod
    def _get_env_config(cls, prefix: str) -> Dict[str, Any]:
        """
        Get configuration from environment variables

        Args:
            prefix: Prefix for environment variables

        Returns:
            Dictionary of configuration from environment
        """
        config = {}

        # Common environment variable mappings
        mappings = {
            # Provider settings
            "PROVIDER_NAME": "provider.name",
            "MODEL_NAME": "provider.model_name",
            "PROVIDER_API_URL": "provider.api_url",
            "PROVIDER_TIMEOUT": "provider.timeout",
            "PROVIDER_MAX_RETRIES": "provider.max_retries",
            "PROVIDER_RETRY_DELAY": "provider.retry_delay",
            # Azure OpenAI provider settings
            "AZURE_OPENAI_ENDPOINT": "azure_openai.azure_endpoint",
            "AZURE_OPENAI_API_KEY": "azure_openai.api_key",
            "AZURE_OPENAI_API_VERSION": "azure_openai.api_version",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "azure_openai.deployment_name",
            "AZURE_OPENAI_MODEL_NAME": "azure_openai.model_name",
            "AZURE_OPENAI_TEMPERATURE": "azure_openai.temperature",
            "AZURE_OPENAI_MAX_TOKENS": "azure_openai.max_tokens",
            "AZURE_OPENAI_TOP_P": "azure_openai.top_p",
            "AZURE_OPENAI_TPM_LIMIT": "azure_openai.tpm_limit",
            "AZURE_OPENAI_RPM_LIMIT": "azure_openai.rpm_limit",
            "AZURE_OPENAI_BATCH_ENABLED": "azure_openai.batch_generation.enabled",
            "AZURE_OPENAI_MAX_BATCH_SIZE": "azure_openai.batch_generation.max_batch_size",
            "AZURE_OPENAI_MAX_TOKENS_PER_BATCH": "azure_openai.batch_generation.max_tokens_per_batch",
            # Ollama provider settings
            "OLLAMA_HOST": "ollama.host",
            "OLLAMA_TIMEOUT": "ollama.timeout",
            "OLLAMA_MAX_RETRIES": "ollama.max_retries",
            "OLLAMA_RETRY_DELAY": "ollama.retry_delay",
            "OLLAMA_MODEL_NAME": "ollama.model_name",
            # Bedrock Anthropic provider settings
            "BEDROCK_MODEL_NAME": "bedrock_anthropic.model_name",
            "BEDROCK_AWS_REGION": "bedrock_anthropic.aws_region",
            "BEDROCK_TEMPERATURE": "bedrock_anthropic.temperature",
            "BEDROCK_MAX_TOKENS": "bedrock_anthropic.max_tokens",
            "BEDROCK_TOP_P": "bedrock_anthropic.top_p",
            "BEDROCK_TPM_LIMIT": "bedrock_anthropic.tpm_limit",
            # Directories
            "DATA_DIR": "directories.input",
            "OUTPUT_DIR": "directories.output",
            "TEMPLATES_DIR": "directories.templates",
            "USER_CONFIGS_DIR": "directories.user_configs",
            # Generation settings
            "DEFAULT_LANGUAGE": "generation.default_language",
            "DEFAULT_NUM_EXAMPLES": "generation.default_num_examples",
            "DEFAULT_TEMPERATURE": "generation.parameters.temperature",
            "MAX_TOKENS": "generation.parameters.max_tokens",
            # Content processing
            "MAX_CONTENT_LENGTH": "content_processing.max_content_length",
            "CONTENT_OVERLAP": "content_processing.content_overlap",
            # Language settings
            "DEFAULT_SYSTEM_PROMPT": "language_settings.default_system_prompt",
            # Storage settings
            "DATASETS_DIR": "storage.datasets_dir",
            "TEMPLATES_DIR_STORAGE": "storage.templates_dir",
            "USER_CONFIGS_DIR_STORAGE": "storage.user_configs_dir",
        }

        for env_var, config_path in mappings.items():
            env_name = f"{prefix}{env_var}" if prefix else env_var
            if env_name in os.environ:
                value = os.environ[env_name]

                # Try to parse numbers and booleans
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
                    value = float(value)
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"

                # Set in config using dotted path
                cls._set_by_path(config, config_path, value)

        return config

    @classmethod
    def _set_by_path(cls, config: Dict, path: str, value: Any):
        """
        Set a value in a dictionary using a dotted path

        Args:
            config: Dictionary to modify
            path: Dotted path (e.g. "provider.name")
            value: Value to set
        """
        parts = path.split(".")
        current = config

        # Navigate to the correct location
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        current[parts[-1]] = value


app_config = ConfigLoader.load()
