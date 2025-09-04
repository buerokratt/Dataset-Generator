import requests
import time
import os
from typing import Dict, Any, Optional
from src.core.providers.base import ModelProvider
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class OllamaProvider(ModelProvider):
    """Provider for Ollama models"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with config dict or environment variables"""
        config = config or {}
        self.api_url = config.get("api_url") or os.getenv(
            "PROVIDER_API_URL", "http://ollama:11434"
        )
        self.model_name = config.get("model_name") or os.getenv(
            "MODEL_NAME", "gemma3:1b-it-qat"
        )
        self.timeout = int(config.get("timeout") or os.getenv("PROVIDER_TIMEOUT", "60"))
        self.max_retries = int(
            config.get("max_retries") or os.getenv("PROVIDER_MAX_RETRIES", "3")
        )
        self.retry_delay = int(
            config.get("retry_delay") or os.getenv("PROVIDER_RETRY_DELAY", "5")
        )

    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate text using Ollama API"""
        options = options or {}

        # Use chat API endpoint for newer Ollama versions (0.7.0+)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": options.get("temperature", 0.7),
            "num_predict": options.get("num_predict", 4096),
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/chat", json=payload, timeout=self.timeout
                )

                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")

    def health_check(self) -> bool:
        """Check if Ollama API is responsive"""
        try:
            response = requests.get(f"{self.api_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
