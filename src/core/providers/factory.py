from typing import Dict, Any
from src.core.providers.base import ModelProvider
from src.core.providers.ollama import OllamaProvider
from src.core.providers.bedrock_anthropic import BedrockAnthropicProvider
from src.core.providers.azure_openai import AzureOpenAIProvider
from src.utils.logger import logger
import os


def get_provider(config: Dict[str, Any] = None) -> ModelProvider:
    """
    Factory function that creates and returns a ModelProvider instance based on configuration.

    This function implements the Factory Pattern to dynamically select and instantiate
    the appropriate ModelProvider implementation based on the provided configuration.
    If no configuration is provided, it falls back to environment variables or defaults.

    Args:
        config (Dict[str, Any], optional): Configuration dictionary containing provider
            settings. At minimum, should contain a "name" key specifying the provider type.
            If None, will use environment variables or default to "ollama".

    Returns:
        ModelProvider: An initialized provider instance that implements the ModelProvider
            interface, configured according to the provided settings

    Raises:
        No exceptions - falls back to "ollama" provider if the requested provider is unknown

    Example:
        ```python
        # Create provider from explicit configuration
        provider_config = {"name": "ollama", "model_name": "gemma3:1b-it-qat"}
        provider = get_provider(provider_config)

        # Create provider using environment variables
        provider = get_provider()  # Uses PROVIDER_NAME from environment or defaults to "ollama"
        ```

    Note:
        To add support for a new provider type, add an entry to the `providers` dictionary
        with a lambda function that instantiates the provider with the given configuration.
    """
    config = config or {}
    provider_name = config.get("name") or os.getenv("PROVIDER_NAME", "ollama")

    providers = {
        "ollama": lambda cfg: OllamaProvider(cfg),
        "bedrock-anthropic": lambda cfg: BedrockAnthropicProvider(cfg),
        "azure-openai": lambda cfg: AzureOpenAIProvider(cfg)
        # Add more providers here as needed
    }

    if provider_name.lower() not in providers:
        logger.warning(f"Unknown provider: {provider_name}. Defaulting to ollama.")
        provider_name = "ollama"

    provider_factory = providers[provider_name.lower()]

    return provider_factory(config)
