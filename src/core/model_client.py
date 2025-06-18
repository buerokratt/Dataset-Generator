"""
Client for interacting with the Gemma model via Ollama API
"""

import requests
import json
from typing import Dict, Any, Optional, List

from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class OllamaClient:
    """Client for interacting with the Ollama API to access the Gemma model"""

    def __init__(
        self,
        api_url=None,
        model_name=None,
        timeout=None,
        max_retries=None,
        retry_delay=None,
    ):
        """
        Initialize the Ollama client

        Args:
            api_url: URL of the Ollama API
            model_name: Name of the model to use
            timeout: Timeout for requests
            max_retries: Maximum number of retries
            retry_delay: Delay between retries
        """
        from core.config import app_config

        self.api_url = api_url or app_config.OLLAMA_HOST
        self.model_name = model_name or app_config.MODEL_NAME
        self.timeout = timeout or app_config.OLLAMA_TIMEOUT
        self.max_retries = max_retries or app_config.OLLAMA_MAX_RETRIES
        self.retry_delay = retry_delay or app_config.OLLAMA_RETRY_DELAY
        self.default_temperature = app_config.DEFAULT_TEMPERATURE
        self.default_num_predict = app_config.DEFAULT_NUM_PREDICT

    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from the model

        Args:
            prompt: The input prompt to send to the model
            options: Additional options for generation

        Returns:
            The generated text response
        """
        if options is None:
            options = {}

        # Default options
        default_options = {
            "temperature": self.default_temperature,
            "max_tokens": self.default_num_predict,
            "top_p": 0.9,
        }

        # Merge with provided options
        merged_options = {**default_options, **options}

        # Prepare request
        data = {"model": self.model_name, "prompt": prompt, **merged_options}

        # Send request
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=data,
                timeout=30,  # 2-minute timeout
            )

            if response.status_code != 200:
                logger.error(
                    f"Error generating from model: {response.status_code}, {response.text}"
                )
                raise requests.HTTPError(
                    f"Error generating from model: {response.status_code}, {response.text}"
                )

            # Parse response
            full_response = ""
            for line in response.text.strip().split("\n"):
                if not line:
                    continue

                try:
                    resp_json = json.loads(line)
                    if "response" in resp_json:
                        full_response += resp_json["response"]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response line: {line}")

            return full_response

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise requests.RequestException(f"Request error: {e}")

    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=10)

            if response.status_code != 200:
                logger.error(
                    f"Error listing models: {response.status_code}, {response.text}"
                )
                return []

            # Parse response
            resp_json = response.json()
            models = [model["name"] for model in resp_json.get("models", [])]

            return models

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return []

    def health_check(self) -> bool:
        """Check if the Ollama server is responsive"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)

            return response.status_code == 200

        except requests.RequestException:
            return False
