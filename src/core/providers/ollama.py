import requests
import time
import os
from typing import Dict, Any
from src.core.providers.base import ModelProvider
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class OllamaLM(ModelProvider):
    def __init__(self, config: Dict[str, Any] = None):
        self.model_name = config.get("model_name") or os.getenv(
            "MODEL_NAME", "gemma3:1b-it-qat"
        )
        self.api_url = config.get("api_url") or os.getenv(
            "PROVIDER_API_URL", "http://localhost:11434"
        )
        self.max_retries = int(
            config.get("max_retries") or os.getenv("PROVIDER_MAX_RETRIES", "3")
        )
        self.timeout = int(config.get("timeout") or os.getenv("PROVIDER_TIMEOUT", "60"))
        super().__init__(self.model_name)

    def predict(self, prompt: str, **kwargs) -> str:
        options = kwargs.get("options", {})
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
                    f"Ollama attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")

    def health_check(self) -> bool:
        """Check if Ollama API is responsive"""
        try:
            response = requests.get(f"{self.api_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
