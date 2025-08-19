import os
import glob
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class TemplateRegistry:
    """Registry for managing dataset structures and prompt templates"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config
        self.structures = {}
        self.templates = {}

        # Get directory paths from config with fallbacks
        dirs = config.get("directories", {})
        self.templates_dir = dirs.get("templates", "templates")
        self.user_configs_dir = dirs.get("user_configs", "user_configs")

        # Scan for available templates and structures
        self.refresh()

    def refresh(self):
        """Refresh the registry by scanning directories"""
        self._scan_structures()
        self._scan_templates()

    def _scan_structures(self):
        """Scan for dataset structures"""
        self.structures = {}

        # Search paths in order of precedence
        search_paths = [
            os.path.join(self.user_configs_dir, "dataset_structures"),
            os.path.join(self.templates_dir, "dataset_structures"),
        ]

        for path in search_paths:
            if not os.path.exists(path):
                continue

            for file_path in glob.glob(os.path.join(path, "*.yaml")):
                structure_name = os.path.splitext(os.path.basename(file_path))[0]

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        structure = yaml.safe_load(f)

                    # Only add if not already in registry (precedence)
                    if structure_name not in self.structures:
                        self.structures[structure_name] = {
                            "path": file_path,
                            "data": structure,
                        }
                except Exception as e:
                    logger.error(f"Error loading structure from {file_path}: {e}")

        logger.info(f"Found {len(self.structures)} dataset structures")

    def _scan_templates(self):
        """Scan for prompt templates"""
        self.templates = {}

        # Search paths in order of precedence
        search_paths = [
            # User templates by category
            os.path.join(self.user_configs_dir, "prompts", "**", "*.txt"),
            # User templates in root
            os.path.join(self.user_configs_dir, "prompts", "*.txt"),
            # Default templates by category
            os.path.join(self.templates_dir, "prompts", "**", "*.txt"),
            # Default templates
            os.path.join(self.templates_dir, "prompts", "*.txt"),
        ]

        for pattern in search_paths:
            for file_path in glob.glob(pattern, recursive=True):
                template_name = os.path.splitext(os.path.basename(file_path))[0]

                # Only add if not already in registry (precedence)
                if template_name not in self.templates:
                    self.templates[template_name] = {
                        "path": file_path,
                        "category": self._get_template_category(file_path),
                    }

        logger.info(f"Found {len(self.templates)} prompt templates")

    def _get_template_category(self, template_path: str) -> Optional[str]:
        """Extract category from template path if available"""
        parts = Path(template_path).parts

        # Check for category in path
        for i, part in enumerate(parts):
            if (
                part == "prompts"
                and i + 1 < len(parts)
                and parts[i + 1] not in ("default", "examples")
            ):
                return parts[i + 1]

        return None

    def get_structure(self, name: str) -> Dict[str, Any]:
        """
        Get a dataset structure by name

        Args:
            name: Structure name

        Returns:
            Structure data

        Raises:
            KeyError: If structure not found
        """
        if name not in self.structures:
            raise KeyError(f"Dataset structure '{name}' not found")

        return self.structures[name]["data"]

    def get_template_path(self, name: str) -> str:
        """
        Get a template path by name

        Args:
            name: Template name

        Returns:
            Path to template file

        Raises:
            KeyError: If template not found
        """
        if name not in self.templates:
            raise KeyError(f"Prompt template '{name}' not found")

        return self.templates[name]["path"]

    def list_structures(self) -> List[str]:
        """List all available structure names"""
        return list(self.structures.keys())

    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())

    def list_templates_by_category(self) -> Dict[str, List[str]]:
        """Group templates by category"""
        result = {"default": []}

        for name, info in self.templates.items():
            category = info.get("category") or "default"

            if category not in result:
                result[category] = []

            result[category].append(name)

        return result
