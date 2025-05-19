"""
Manage storage for synthetic datasets
"""
import os
import shutil
from pathlib import Path
from typing import List

from src.utils.logger import logger, setup_logger
from core.config import app_config

setup_logger("synthetic-data-service", "INFO")

class StorageManager:
    """Manage storage for synthetic datasets"""

    def __init__(self):
        self.datasets_base_dir = app_config.DATASETS_DIR if hasattr(app_config, 'DATASETS_DIR') else 'datasets'
        self.templates_dir = app_config.TEMPLATES_DIR if hasattr(app_config, 'TEMPLATES_DIR') else 'templates'
        self.user_configs_dir = app_config.USER_CONFIGS_DIR if hasattr(app_config, 'USER_CONFIGS_DIR') else 'user_configs'
        
        # Ensure base directories exist
        for dir_path in [self.datasets_base_dir, self.templates_dir, self.user_configs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_datasets_base_dir(self) -> str:
         """Returns the configured base directory for datasets."""
         return self.datasets_base_dir
    
    def get_templates_dir(self) -> str:
         """Returns the configured directory for templates."""
         return self.templates_dir
    
    def get_user_configs_dir(self) -> str:
         """Returns the configured directory for user configs."""
         return self.user_configs_dir
    
    def prepare_directory(self, dir_path: str) -> None:
        """Create a directory if it doesn't exist."""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Prepared directory: {dir_path}")
    
    def clean_directory(self, directory: str) -> None:
        """
        Clean a directory by removing all files and subdirectories
        
        Args:
            directory: Path to the directory
        """
        # Check if directory exists
        if not os.path.exists(directory):
            return
        
        # Remove all files and subdirectories
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        logger.info(f"Cleaned directory: {directory}")
    
    def list_datasets(self) -> List[str]:
        """
        List all datasets
        
        Returns:
            List of dataset names
        """
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            return []
        
        # List all directories in the datasets directory
        return [
            d for d in os.listdir(datasets_dir) 
            if os.path.isdir(os.path.join(datasets_dir, d))
        ]
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if the dataset exists, False otherwise
        """
        dataset_path = os.path.join("datasets", dataset_name)
        return os.path.exists(dataset_path)
    
    def get_dataset_size(self, dataset_name: str) -> int:
        """
        Get the size of a dataset in bytes
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Size of the dataset in bytes
        """
        dataset_path = os.path.join("datasets", dataset_name)
        if not os.path.exists(dataset_path):
            return 0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return total_size