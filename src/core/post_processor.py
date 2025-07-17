from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os
from src.utils.logger import logger, setup_logger
import time
import random

setup_logger("synthetic-data-service", "INFO")


class PostProcessor(ABC):
    """Abstract base class for post-processing generated datasets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def process(self, output_paths: List[str], base_output_dir: str) -> Optional[str]:
        """
        Process the generated dataset outputs.

        Args:
            output_paths: List of paths to generated datasets
            base_output_dir: Base directory for output

        Returns:
            Path to the final processed output or None if failed
        """
        pass


class ZipPostProcessor(PostProcessor):
    """Post-processor that creates ZIP archives of datasets."""

    def process(self, output_paths: List[str], base_output_dir: str) -> Optional[str]:
        """Create ZIP archive of the datasets."""
        try:
            import zipfile
            from pathlib import Path

            base_path = Path(base_output_dir)
            if not base_path.exists():
                logger.error(f"Dataset path not found: {base_path}")
                return None

            # Extract the correct structure name (last part of the path)
            structure_name = base_path.name
            zip_filename = f"{structure_name}.zip"

            # Get output directory from config
            output_dir = self.config.get("directories", {}).get(
                "output", "output_datasets"
            )
            zip_path = os.path.join(output_dir, zip_filename)

            # Create the ZIP file
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=base_path)
                        zipf.write(file_path, arcname)

            logger.info(f"Created ZIP archive: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            return None


class AggregationPostProcessor(PostProcessor):
    """Post-processor that aggregates datasets into a single file."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.field_mapping_config = (
            config.get("dataset_generation", {})
            .get("aggregation", {})
            .get("field_mapping", {})
        )

    def _aggregate_json_files(
        self,
        file_paths: List[str],
        merge_strategy: str,
        include_metadata: bool,
        dataset_metadata: List[dict] = None,
    ) -> Dict[str, Any]:
        """Aggregate JSON files with dynamic field mapping."""
        aggregated = {"aggregated_data": [], "total_items": 0}

        if include_metadata:
            aggregated["metadata"] = {
                "aggregation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "merge_strategy": merge_strategy,
                "source_count": len(file_paths),
            }

        # Collect all items first
        all_items = []
        id_counter = 1

        for i, file_path in enumerate(file_paths):
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                dataset_dir = Path(file_path)
                json_files = list(dataset_dir.glob("*.json"))

                current_dataset_metadata = (
                    dataset_metadata[i]
                    if dataset_metadata and i < len(dataset_metadata)
                    else {}
                )

                for json_file in json_files:
                    if json_file.name == "metadata.json":
                        continue

                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    items_to_add = data if isinstance(data, list) else [data]

                    for item in items_to_add:
                        mapped_item = self._apply_field_mapping(
                            item, current_dataset_metadata, id_counter
                        )
                        all_items.append(mapped_item)
                        id_counter += 1

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        # Shuffle all items and replace id with sequential values
        if all_items:
            logger.info(f"Shuffling {len(all_items)} items")
            random.shuffle(all_items)

            # Replace id with sequential values after shuffling
            aggregated["aggregated_data"] = [
                {**item, "id": str(idx + 1)} for idx, item in enumerate(all_items)
            ]
            aggregated["total_items"] = len(all_items)

            logger.info("Added sequential id values after shuffling")

        return aggregated

    def _apply_field_mapping(
        self, item: dict, dataset_metadata: dict, id_counter: int
    ) -> dict:
        """Apply field mapping configuration to transform the item."""
        if not self.field_mapping_config.get("enabled", False):
            return item

        mapped_item = {}

        # Apply payload to output mapping
        payload_mapping = self.field_mapping_config.get("payload_to_output", {})
        for payload_field, output_field in payload_mapping.items():
            if payload_field in dataset_metadata:
                mapped_item[output_field] = dataset_metadata[payload_field]

        # Apply defaults
        defaults = self.field_mapping_config.get("defaults", {})
        for field, default_value in defaults.items():
            if field not in mapped_item:
                if default_value == "auto_increment":
                    mapped_item[field] = str(id_counter)
                else:
                    mapped_item[field] = default_value

        # Copy content fields from generated data
        content_fields = self.field_mapping_config.get("content_fields", [])
        for field in content_fields:
            if field in item:
                mapped_item[field] = item[field]

        # Copy any remaining fields from original item that aren't mapped
        for key, value in item.items():
            if key not in mapped_item:
                mapped_item[key] = value

        return mapped_item

    def process(
        self,
        output_paths: List[str],
        base_output_dir: str,
        dataset_metadata: List[dict] = None,
    ) -> Optional[str]:
        """Aggregate datasets into a single file with metadata support."""
        try:
            aggregation_config = self.config.get("dataset_generation", {}).get(
                "aggregation", {}
            )

            filename = aggregation_config.get("output_filename", "aggregated_data")
            output_format = self.config.get("dataset_generation", {}).get(
                "output_format", "json"
            )
            merge_strategy = aggregation_config.get("merge_strategy", "combine_arrays")
            include_metadata = aggregation_config.get("include_metadata", True)

            # Create aggregated file path
            aggregated_path = Path(base_output_dir) / f"{filename}.{output_format}"
            os.makedirs(base_output_dir, exist_ok=True)

            logger.info(
                f"Aggregating {len(output_paths)} datasets into {aggregated_path}"
            )

            if output_format == "json":
                aggregated_data = self._aggregate_json_files(
                    output_paths, merge_strategy, include_metadata, dataset_metadata
                )
                with open(aggregated_path, "w", encoding="utf-8") as f:
                    json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
            else:
                aggregated_content = self._aggregate_text_files(
                    output_paths, include_metadata
                )
                with open(aggregated_path, "w", encoding="utf-8") as f:
                    f.write(aggregated_content)

            logger.info(f"Successfully created aggregated file: {aggregated_path}")
            return str(aggregated_path)

        except Exception as e:
            logger.error(f"Error aggregating datasets: {e}")
            return None

    def _aggregate_text_files(
        self, file_paths: List[str], include_metadata: bool
    ) -> str:
        """Aggregate text files by concatenating content."""
        aggregated_content = []

        if include_metadata:
            aggregated_content.append("# Aggregated Dataset")
            aggregated_content.append(
                f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            aggregated_content.append(f"# Source files: {len(file_paths)}")
            aggregated_content.append("")

        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    continue

                dataset_dir = Path(file_path)
                text_files = list(dataset_dir.glob("*.txt"))

                for text_file in text_files:
                    aggregated_content.append(f"## Source: {text_file.name}")
                    aggregated_content.append("")

                    with open(text_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        aggregated_content.append(content)

                    aggregated_content.append("")
                    aggregated_content.append("---")
                    aggregated_content.append("")

            except Exception as e:
                logger.error(f"Error processing text file {file_path}: {e}")
                continue

        return "\n".join(aggregated_content)
