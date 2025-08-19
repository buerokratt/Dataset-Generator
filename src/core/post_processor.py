from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os
from src.utils.logger import logger, setup_logger
import time
import random
import csv

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
        """Aggregate multiple JSON files into a single structure."""
        all_items = []
        id_counter = 1

        for i, file_path in enumerate(file_paths):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Get corresponding metadata for this file
                metadata = (
                    dataset_metadata[i]
                    if dataset_metadata and i < len(dataset_metadata)
                    else {}
                )

                # Extract items from the JSON structure
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # Look for common array fields
                    items = data.get(
                        "questions", data.get("data", data.get("items", [data]))
                    )
                else:
                    items = [data]

                # Apply field mapping if enabled
                if self.field_mapping_config.get("enabled", False):
                    mapped_items = []
                    for item in items:
                        mapped_item = self._apply_field_mapping(
                            item, metadata, id_counter
                        )
                        mapped_items.append(mapped_item)
                        id_counter += 1
                    items = mapped_items

                all_items.extend(items)

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue

        # Get aggregation config
        aggregation_config = self.config.get("dataset_generation", {}).get(
            "aggregation", {}
        )

        # Check if shuffling is enabled
        enable_shuffling = aggregation_config.get("enable_shuffling", False)

        # Handle shuffling
        if all_items:
            if enable_shuffling:
                logger.info(f"Shuffling {len(all_items)} items")
                random.shuffle(all_items)

                # Re-assign sequential IDs after shuffling (preserving version_id format if used)
                if self.field_mapping_config.get("enabled", False):
                    # Get version_id from first item's metadata or config
                    sample_metadata = dataset_metadata[0] if dataset_metadata else {}
                    version_id = sample_metadata.get("version_id")
                    if not version_id:
                        aggregation_config = self.config.get(
                            "dataset_generation", {}
                        ).get("aggregation", {})
                        version_id = aggregation_config.get("version_id")

                    # Re-assign IDs with proper formatting
                    for idx, item in enumerate(all_items):
                        if isinstance(item, dict):
                            if version_id:
                                item["id"] = f"{version_id}_{idx + 1}"
                            else:
                                item["id"] = str(idx + 1)

                logger.info("Added sequential id values after shuffling")
            else:
                logger.info(f"Aggregating {len(all_items)} items without shuffling")

                logger.info("Added sequential id values without shuffling")

            # REMOVE THIS BLOCK - it was overriding the versioned IDs:
            # # If field mapping wasn't applied earlier, assign sequential IDs now
            # if not self.field_mapping_config.get("enabled", False):
            #     for idx, item in enumerate(all_items):
            #         if isinstance(item, dict):
            #             item["id"] = str(idx + 1)

        # Create aggregated structure
        aggregated = {"aggregated_data": all_items, "total_items": len(all_items)}

        # # Add metadata if requested
        # if include_metadata:
        #     aggregated["metadata"] = {
        #         "aggregation_timestamp": datetime.now().isoformat(),
        #         "source_count": len(file_paths),
        #         "merge_strategy": merge_strategy
        #     }

        return aggregated

    def _apply_field_mapping(
        self, item: Dict[str, Any], metadata: Dict[str, Any], id_counter: int
    ) -> Dict[str, Any]:
        """Apply field mapping to transform item structure."""
        mapped_item = {}

        # Get version_id from metadata with fallback handling
        version_id = metadata.get("version_id")
        if not version_id:
            aggregation_config = self.config.get("dataset_generation", {}).get(
                "aggregation", {}
            )
            version_id = aggregation_config.get("version_id")
        if not version_id:
            version_id = None

        # Process payload_to_output mappings
        for source_field, target_field in self.field_mapping_config.get(
            "payload_to_output", {}
        ).items():
            if source_field in metadata and source_field != "version_id":
                mapped_item[target_field] = metadata[source_field]

        # Process defaults
        for target_field, default_value in self.field_mapping_config.get(
            "defaults", {}
        ).items():
            if default_value == "auto_increment":
                if version_id:
                    mapped_item[target_field] = f"{version_id}_{id_counter}"
                else:
                    mapped_item[target_field] = str(id_counter)
            elif target_field == "dataset_version_id":
                # Set from version_id if available, else use default
                try:
                    mapped_item[target_field] = (
                        int(version_id) if version_id is not None else default_value
                    )
                except Exception:
                    mapped_item[target_field] = default_value
            else:
                mapped_item[target_field] = default_value

        # Process content fields with proper renaming
        content_fields = self.field_mapping_config.get("content_fields", {})
        if isinstance(content_fields, dict):
            for source_field, target_field in content_fields.items():
                if source_field in item:
                    mapped_item[target_field] = item[source_field]
        elif isinstance(content_fields, list):
            for field in content_fields:
                if isinstance(field, dict):
                    for source_field, target_field in field.items():
                        if source_field in item:
                            mapped_item[target_field] = item[source_field]
                elif isinstance(field, str) and field in item:
                    mapped_item[field] = item[field]

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
            output_format = aggregation_config.get(
                "output_format", "json"
            )  # Add this line
            merge_strategy = aggregation_config.get("merge_strategy", "combine_arrays")
            include_metadata = aggregation_config.get("include_metadata", True)

            # Create aggregated file path
            aggregated_path = Path(base_output_dir) / f"{filename}.{output_format}"
            os.makedirs(base_output_dir, exist_ok=True)

            logger.info(
                f"Aggregating {len(output_paths)} datasets into {aggregated_path} as {output_format.upper()}"
            )

            if output_format == "csv":
                # Generate CSV output without metadata
                json_data = self._aggregate_json_files(
                    output_paths, merge_strategy, include_metadata, dataset_metadata
                )

                csv_data = json_data.get("aggregated_data", [])

                if csv_data:
                    with open(aggregated_path, "w", newline="", encoding="utf-8") as f:
                        csv_field_order = aggregation_config.get("csv_field_order")
                        if csv_field_order:
                            fieldnames = csv_field_order
                        else:
                            fieldnames = list(csv_data[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)

                    logger.info(
                        f"Successfully created CSV file with {len(csv_data)} rows (metadata excluded)"
                    )
                else:
                    logger.warning("No data to write to CSV file")

            elif output_format == "json":
                # Generate JSON output (existing logic with metadata)
                aggregated_data = self._aggregate_json_files(
                    output_paths, merge_strategy, include_metadata, dataset_metadata
                )
                with open(aggregated_path, "w", encoding="utf-8") as f:
                    json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

            else:
                # Text format (existing logic)
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
