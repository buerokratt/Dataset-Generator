import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")

# Determine the base directory of the application
# Assuming this file is in src/core, config is at src/../config
BASE_DIR = Path(__file__).resolve().parent.parent 
CONFIG_FILE_PATH = BASE_DIR / "config" / "model_config.yaml"

class AppConfig:
    def __init__(self, config_data: Dict[str, Any]):
        # Ollama Client settings
        self.MODEL_NAME: str = os.getenv("MODEL_NAME", config_data.get("model_name", "gemma3:1b-it-qat"))
        self.OLLAMA_HOST: str = os.getenv("MODEL_API_URL", config_data.get("ollama_host", "http://ollama:11434"))
        self.OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", config_data.get("ollama_timeout", 60)))
        self.OLLAMA_MAX_RETRIES: int = int(os.getenv("OLLAMA_MAX_RETRIES", config_data.get("ollama_max_retries", 3)))
        self.OLLAMA_RETRY_DELAY: int = int(os.getenv("OLLAMA_RETRY_DELAY", config_data.get("ollama_retry_delay", 5)))

        # Storage settings
        storage_settings = config_data.get("storage", {})
        self.DATASETS_DIR = os.getenv("DATASETS_DIR", storage_settings.get("datasets_dir", "datasets"))
        self.TEMPLATES_DIR = os.getenv("TEMPLATES_DIR", storage_settings.get("templates_dir", "templates"))
        self.USER_CONFIGS_DIR = os.getenv("USER_CONFIGS_DIR", storage_settings.get("user_configs_dir", "user_configs"))

        # Generation defaults
        gen_defaults = config_data.get("generation_defaults", {})
        self.DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", gen_defaults.get("temperature", 0.7)))
        # Ollama uses 'num_predict' for max tokens.
        self.DEFAULT_NUM_PREDICT: int = int(os.getenv("DEFAULT_NUM_PREDICT", gen_defaults.get("num_predict", 4096)))


        # Language settings
        lang_settings = config_data.get("language_settings", {})
        self.DEFAULT_SYSTEM_PROMPT: str = os.getenv("DEFAULT_SYSTEM_PROMPT", lang_settings.get("default_system_prompt", "You are a helpful assistant."))
        self.DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", lang_settings.get("default_language", "en"))
        self.SUPPORTED_LANGUAGES: Dict[str, str] = lang_settings.get("supported_languages", {"en": "English"})

        # Output defaults
        output_defaults = config_data.get("output_defaults", {})
        self.DEFAULT_SAVE_FORMAT: str = os.getenv("DEFAULT_SAVE_FORMAT", output_defaults.get("save_format", "json"))
        self.SUPPORTED_FORMATS = output_defaults.get("supported_formats", ["json", "text"])
        
        # Content processing (for application logic, not directly for OllamaClient)
        content_proc = config_data.get("content_processing", {})
        self.MAX_CONTENT_LENGTH: int = int(content_proc.get("max_content_length", 15000))
        self.CONTENT_OVERLAP: int = int(content_proc.get("content_overlap", 500))


_app_config_instance: Optional[AppConfig] = None

def load_config() -> AppConfig:
    global _app_config_instance
    if _app_config_instance is None:
        if not CONFIG_FILE_PATH.exists():
            logger.error(f"Configuration file not found: {CONFIG_FILE_PATH}")
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE_PATH}")
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            if not config_data:
                logger.error(f"Configuration file is empty or invalid: {CONFIG_FILE_PATH}")
                raise ValueError(f"Configuration file is empty or invalid: {CONFIG_FILE_PATH}")
            _app_config_instance = AppConfig(config_data)
            logger.info(f"Configuration loaded successfully from {CONFIG_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error loading configuration from {CONFIG_FILE_PATH}: {e}")
            raise
    return _app_config_instance

# Load config once when module is imported
app_config = load_config()