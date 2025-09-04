from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class ModelProvider(ABC):
    """
    Abstract base class defining the interface for all model providers.

    This class serves as a contract that all concrete model provider implementations
    must follow, enabling the application to interact with different LLM backends
    (such as Ollama, VLLM, OpenAI, etc.) through a consistent interface.

    The ModelProvider abstraction supports the strategy pattern, allowing the system
    to switch between different LLM implementations at runtime based on configuration.

    Implementers must override:
    - generate(): To produce text from a prompt
    - health_check(): To verify connectivity with the model service

    Example:
        ```python
        class OllamaProvider(ModelProvider):
            def __init__(self, config: Dict[str, Any]):
                self.client = OllamaClient(
                    api_url=config.get("api_url"),
                    model_name=config.get("model_name")
                )

            def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
                return self.client.generate(prompt, options)

            def health_check(self) -> bool:
                return self.client.health_check()
        ```
    """

    @abstractmethod
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from a prompt"""
        pass

    def generate_batch(
        self, prompt: str, num_samples: int, options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate multiple samples from the same prompt.

        Default implementation: call generate() multiple times (for Ollama)
        Cloud providers can override for efficient batch generation.

        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            options: Generation options

        Returns:
            List of generated responses
        """
        from src.utils.logger import logger

        logger.info(
            f"Generating {num_samples} samples using default batch method (multiple API calls)"
        )
        results = []
        for i in range(num_samples):
            logger.info(f"Generating sample {i + 1}/{num_samples}")
            response = self.generate(prompt, options)
            results.append(response)
        return results

    def supports_batch_generation(self) -> bool:
        """Return True if provider supports efficient batch generation"""
        return False  # Default: False for Ollama-like providers

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the model provider is available"""
        pass
