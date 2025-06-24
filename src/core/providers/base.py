from abc import abstractmethod
from typing import Dict, Any, Optional
from dspy import dspy


class ModelProvider(dspy.LM):
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
    def predict(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the model provider is available"""
        pass
