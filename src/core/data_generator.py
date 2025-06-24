# data_generator.py
from typing import Dict, Any, Optional, List
import os
import json
import re
import time
from pathlib import Path

from src.core.providers.factory import get_provider
from src.core.template_registry import TemplateRegistry
from src.core.prompt_processor import PromptProcessor
from src.core.storage_manager import StorageManager
from src.utils.logger import logger, setup_logger

from src.core.dspy_generation import PromptPipeline, optimize_prompt_with_dspy

from src.core.metrics import (
    InformationCoverageMetric,
    PerSampleQualityEvaluator,
    RelevanceCoverageMetric,
)

setup_logger("synthetic-data-service", "INFO")


class DataGenerator:
    """
    A class to generate synthetic datasets using language models (LLMs) and DSPy prompt optimization.

    This generator loads a specified prompt template, fills it with context parameters,
    invokes the model to produce structured data, and evaluates the output using various
    metrics (information coverage, relevance, and per-sample quality). It also optionally
    optimizes prompts using DSPy if the output quality is below threshold.

    Attributes:
        config (dict): Configuration dictionary for model, directories, and generation behavior.
        model_provider: The LLM backend instance (e.g., Ollama, OpenAI).
        template_registry: Manages prompt templates and structure definitions.
        prompt_processor: Replaces placeholders in templates with actual values.
        storage_manager: Manages file storage interactions.
    """
    def __init__(self, config=None):
        """
        Initializes the DataGenerator with optional configuration.

        Args:
            config (dict, optional): Configuration settings including provider, directories, and generation defaults.
        """
        self.config = config or {}
        self.model_provider = get_provider(self.config.get("provider", {}))
        self.template_registry = TemplateRegistry(self.config)
        self.prompt_processor = PromptProcessor()
        self.storage_manager = StorageManager()
        self.generation_defaults = self.config.get("generation", {})
        self.evaluation_models_defaults = self.config.get("models", {})
        self.embedding_model = self.evaluation_models_defaults.get("embedding_model")

        directories = self.config.get("directories", {})
        self.output_dir = directories.get("output", "output_datasets")
        self.templates_dir = directories.get("templates", "templates")
        self.user_configs_dir = directories.get("user_configs", "user_configs")
        self.data_dir = directories.get("input", "data")
        self.info_metric = None
        self.relevance_metric = None
        self.per_sample_evaluator = None

        for dir_path in [self.output_dir, self.templates_dir, self.user_configs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_info_metric(self):
        """
        Lazily initializes and returns the InformationCoverageMetric instance.

        Returns:
            InformationCoverageMetric: Metric to evaluate semantic content completeness.
        """
        
        if self.info_metric is None:
            self.info_metric = InformationCoverageMetric(
                embedding_model=self.embedding_model
            )
        return self.info_metric

    def get_relevance_metric(self):
        """
        Lazily initializes and returns the RelevanceCoverageMetric instance.

        Returns:
            RelevanceCoverageMetric: Metric to evaluate how relevant the generated output is to the prompt.
        """
        if self.relevance_metric is None:
            self.relevance_metric = RelevanceCoverageMetric(
                embedding_model=self.embedding_model
            )
        return self.relevance_metric

    def get_per_sample_evaluator(self):
        """
        Lazily initializes and returns the PerSampleQualityEvaluator instance.

        Returns:
            PerSampleQualityEvaluator: Evaluates each generated sample for structure and quality.
        """
        if self.per_sample_evaluator is None:
            self.per_sample_evaluator = PerSampleQualityEvaluator(
                embedding_model=self.embedding_model
            )
        return self.per_sample_evaluator

    def generate(
        self,
        structure_name: str,
        prompt_template_name: str,
        dataset_name: Optional[str] = None,
        output_base_path: Optional[str] = None,
        num_examples: Optional[int] = None,
        output_format: str = "json",
        parameters: Dict[str, Any] = None,
        optimize_prompts: bool = False,
    ) -> str:
        """
        Entry point for generating a synthetic dataset based on a structure and template.

        Args:
            structure_name (str): Name of the registered data structure to use.
            prompt_template_name (str): Filename of the prompt template to use.
            dataset_name (str, optional): Output folder name. If not provided, timestamp will be used.
            output_base_path (str, optional): Full output path to use instead of computed name.
            num_examples (int, optional): Number of examples to generate per file.
            output_format (str): File format, defaults to "json".
            parameters (dict): Runtime variables for prompt substitution.
            optimize_prompts (bool): Whether to apply DSPy prompt optimization if output is subpar.

        Returns:
            str: Path to the generated dataset directory.
        """
        parameters = parameters or {}

        num_examples = num_examples or self.generation_defaults.get(
            "default_num_examples", 5
        )

        if output_base_path:
            output_dir = Path(output_base_path)
        elif dataset_name:
            output_dir = Path(self.output_dir) / dataset_name
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.output_dir) / f"{structure_name}_{timestamp}"

        structure = self.template_registry.get_structure(structure_name)["root"]
        template_path = self.template_registry.get_template_path(prompt_template_name)
        with open(template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        if optimize_prompts:
            prompt_template = optimize_prompt_with_dspy(
                prompt_template, self.model_provider
            )

        os.makedirs(output_dir, exist_ok=True)
        self._generate_data_with_dspy(
            str(output_dir),
            structure,
            prompt_template,
            num_examples,
            output_format,
            parameters,
        )
        return str(output_dir)

    def _generate_data_with_dspy(
        self,
        output_dir: str,
        structure_root: dict,
        prompt_template: str,
        num_examples: int,
        output_format: str,
        parameters: dict,
    ):
        """
        Internal method to drive the actual generation logic using LLMs and evaluate results.

        Args:
            output_dir (str): Directory where outputs are saved.
            structure_root (dict): Structure dictionary describing subdirectories and files.
            prompt_template (str): Template text with placeholders for generation.
            num_examples (int): Number of samples to generate for each file.
            output_format (str): Output format (e.g., json, txt).
            parameters (dict): Contextual values to fill the prompt.
        """
        pipeline = PromptPipeline(self.model_provider)
        info_metric = self.get_info_metric()
        relevance_metric = self.get_relevance_metric()
        per_sample_evaluator = self.get_per_sample_evaluator()

        for relative_file_key, file_info in self._flatten_structure(structure_root):
            file_format = file_info.get("format", output_format)
            output_path = Path(output_dir) / f"{relative_file_key}.{file_format}"
            os.makedirs(output_path.parent, exist_ok=True)
            examples_count = self._get_path_examples(
                relative_file_key, num_examples, parameters
            )

            regenerate = True
            attempts = 0
            best_items = []

            while regenerate and attempts < 3:
                attempts += 1
                items = []
                high_quality_samples = []

                for i in range(examples_count):
                    context = dict(parameters)
                    context["index"] = i + 1
                    processed = self.prompt_processor.process(prompt_template, context)
                    output = pipeline.generate_raw_output(
                        requirements=processed, context="{}"
                    )
                    parsed_items = self.extract_json_from_llm_output([output])

                    for item in parsed_items:
                        score = per_sample_evaluator(item, context)
                        if score >= 0.2:
                            high_quality_samples.append(item)

                    items.extend(parsed_items)

                batch_text = " ".join(
                    json.dumps(item, ensure_ascii=False) for item in items
                )
                info_score, _ = info_metric(batch_text, [prompt_template])
                relevance_score, _ = relevance_metric(batch_text, [prompt_template])
                avg = (info_score + relevance_score) / 2
                logger.info(
                    f"[{relative_file_key}] Attempt {attempts}: Info={info_score:.3f}, Relevance={relevance_score:.3f}, Avg={avg:.3f}"
                )

                if avg >= 0.4:
                    regenerate = False
                    best_items = items
                elif attempts == 2 and high_quality_samples:
                    # Optional: Use good samples for prompt tuning here before final retry
                    logger.info(
                        f"[{relative_file_key}] Avg Score ({avg:.2f}) still below threshold. Found {len(high_quality_samples)} good samples. Applying DSPy prompt optimization."
                    )
                    context = dict(parameters)
                    proceesed_prompt_templete_to_optimmize = (
                        self.prompt_processor.process(prompt_template, context)
                    )
                    prompt_template = optimize_prompt_with_dspy(
                        proceesed_prompt_templete_to_optimmize, self.model_provider
                    )

            with open(output_path, "w", encoding="utf-8") as f:
                if file_format == "json":
                    json.dump(best_items, f, indent=2, ensure_ascii=False)
                else:
                    for x in best_items:
                        f.write(str(x) + "\n")

    def _flatten_structure(
        self, structure_node: Dict[str, Any], current_path: str = ""
    ) -> List[tuple]:
        """
        Recursively flattens a nested directory structure into a list of file paths and metadata.

        Args:
            structure_node (dict): The nested structure definition.
            current_path (str): Current path in recursion.

        Returns:
            List[tuple]: List of (relative_file_path, file_info) pairs.
        """
        items = []
        base_path = Path(current_path)
        for k, v in structure_node.get("files", {}).items():
            items.append((str(base_path / k), v))
        for d, sd in structure_node.get("subdirectories", {}).items():
            items.extend(self._flatten_structure(sd, str(base_path / d)))
        return items

    def _get_path_examples(
        self, relative_file_key: str, total_examples: int, parameters: Dict[str, Any]
    ) -> int:
        """
        Gets the number of examples to generate for a specific file, allowing per-file overrides.

        Args:
            relative_file_key (str): Normalized path key for the file.
            total_examples (int): Default number of examples.
            parameters (dict): Parameters that might contain file-specific counts.

        Returns:
            int: Number of examples to generate.
        """
        normalized = relative_file_key.replace(os.sep, "_")
        key = f"{normalized}_count"
        return int(parameters.get(key, total_examples))

    def extract_json_from_llm_output(self, llm_output):
        """
        Attempts to parse JSON-structured output from LLM responses.

        Args:
            llm_output (list): List of raw string outputs from the model.

        Returns:
            list: Parsed JSON objects (dicts or lists) extracted from model output.
        """
        all_questions = []
        for item in llm_output:
            cleaned = re.sub(r"```json|```", "", item, flags=re.IGNORECASE).strip()
            try:
                if cleaned.startswith("[") or cleaned.startswith("{"):
                    parsed = json.loads(cleaned)
                else:
                    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                    else:
                        continue
                if isinstance(parsed, list):
                    all_questions.extend(parsed)
                elif isinstance(parsed, dict):
                    all_questions.append(parsed)
            except Exception as e:
                print(f"Failed to parse item: {e}\nContent: {cleaned[:100]}...")
                continue
        return all_questions
