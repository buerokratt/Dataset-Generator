from loguru import logger
import yaml
import sys
import re
from typing import List, Tuple
import numpy as np 
import json
import glob 
import os
from constants import PREPROCESS_TEXT_PATTERN

logger.remove()
# add stdout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

def read_file(file_path):
    """
    Reads a file and returns its content as a list of lines.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""
    


class Config:
    _instance = None
    
    def __new__(cls, config_path='eval_config.yaml'):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            with open(config_path, 'r') as file:
                cls._instance.data = yaml.safe_load(file)
        return cls._instance
    
    def get(self, key, default=None):
        """Access nested keys with dot notation: config.get('database.url')"""
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


def load_stopwords() -> List[str]:
    config = Config()
    try:
        stopwords_file = config.get("general.stopwords_file")
        with open(stopwords_file, "r", encoding="utf-8") as f:
            est_stopwords = [line.strip() for line in f if line.strip()]
        return est_stopwords
    except:
        logger.warning("Estonian stopwords file not found. Using default stopwords.")
        return []
    
    
    
def preprocess_text(text: str) -> str:
    """Preprocess Estonian text."""
    text = text.lower()
        
    text = re.sub(PREPROCESS_TEXT_PATTERN, '', text)
        
    text = re.sub(r'\s+', ' ', text).strip()
        
    estonian_chars = 'äöüõÄÖÜÕ'
    text = re.sub(f'[^a-zA-Z0-9{estonian_chars} ]', ' ', text)
        
    return text

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def read_conversations_from_directory(directory: str) -> Tuple[List[str], List[str]]:
    """
    Read all conversation files from a directory matching a pattern.

    """
    config = Config()
    files = sorted(glob.glob(os.path.join(directory, config.get("general.conversation_file_pattern"))))
    contents = []
    filenames = []
    
    for f in files:
        content = read_file(f)
        if content.strip():  
            contents.append(content)
            filenames.append(os.path.basename(f))
    
    return contents, filenames