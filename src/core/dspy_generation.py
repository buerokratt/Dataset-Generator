import json
import dspy
from typing import Any
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")




"""
dspy_generation.py

This module defines the core classes and utilities for generating synthetic data using DSPy.
It includes prompt pipelines, quality assessors, and prompt optimizers that leverage DSPy's
modular framework for signature-based generation, quality assurance, and prompt enhancement.
"""

class JSONQualityAssessor:
    """
    Assesses whether the given model output is valid JSON.

    Returns:
        float: 1.0 if output can be parsed as JSON, otherwise 0.0.
    """
    def __call__(self, output: str) -> float:
        try:
            json.loads(output)
            return 1.0
        except Exception:
            return 0.0

class DataGenerationSignature(dspy.Signature):
    """
    Defines the input/output schema for DSPy-based data generation.

    Fields:
        requirements (str): Prompt describing dataset generation constraints.
        context (str): Optional additional context (e.g., examples, metadata).
        generated_data (str): Output field for generated sample.
    """
    requirements = dspy.InputField(
        desc="Dataset requirements and format specifications"
    )
    context = dspy.InputField(desc="Additional context or examples")
    generated_data = dspy.OutputField(desc="Generated dataset entry")


class DsPyGenerateModule(dspy.Module):
    """
    A DSPy module implementing the data generation process using Chain-of-Thought prompting.

    Methods:
        forward(requirements: str, context: str) -> dict:
            Executes the generation pipeline and returns structured output.
    """
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(DataGenerationSignature)

    def forward(self, requirements: str, context: str = ""):
        """
        Performs forward generation using DSPy's Chain-of-Thought logic.

        Args:
            requirements (str): Dataset specification or generation instructions.
            context (str): Optional examples or metadata for in-context learning.

        Returns:
            dict: Output containing 'generated_data' as key.
        """
        return self.generate(requirements=requirements, context=context)


class PromptPipeline:
    """
    A simple pipeline to format and invoke LLM prediction using DSPy's configured provider.

    Args:
        provider (Any): A language model backend implementing a `predict()` method.

    Methods:
        generate_raw_output(requirements: str, context: str) -> str:
            Returns generated output from the model, substituting context if available.
    """
    def __init__(self, provider: Any):
        self.provider = provider
        dspy.configure(lm=self.provider)

    def generate_raw_output(self, requirements: str, context: str = "") -> str:
        """
        Generates raw text output by filling the prompt with context and sending to the model.

        Args:
            requirements (str): Template prompt with placeholders.
            context (str): JSON-formatted context values for substitution.

        Returns:
            str: Model's generated response.
        """
        try:
            prompt = (
                requirements.format(**json.loads(context))
                if context.strip()
                else requirements
            )
        except Exception as e:
            logger.warning(f"Context formatting failed: {e}. Using raw requirements.")
            prompt = requirements
        return self.provider.predict(prompt)


def optimize_prompt_with_dspy(prompt_template: str, provider: Any) -> str:
    """
    Attempts to optimize a prompt template by evaluating multiple prompt variants.

    The prompt with the longest model output is heuristically selected as the best variant.
    Each variant is logged along with its score.

    Args:
        prompt_template (str): The base prompt to optimize.
        provider (Any): LLM backend that supports a `predict()` method.

    Returns:
        str: The best-performing prompt variant selected from the candidates.
    """
    logger.info("Starting DSPy prompt optimization...")
    dspy.configure(lm=provider)

    prompt_variants = [
        prompt_template,
        prompt_template + "\n\nEnsure output is JSON and follows instructions.",
        "Please follow the requirements and output in JSON:\n" + prompt_template,
    ]

    def simulate_response(prompt):
        return provider.predict(prompt)

    scored_variants = []
    for variant in prompt_variants:
        output = simulate_response(variant)
        score = len(output)
        scored_variants.append((score, variant))
        logger.info(f"Prompt variant scored: {score}")

    best_variant = max(scored_variants, key=lambda x: x[0])[1]
    logger.info("[DSPy] Selected optimized prompt variant:")
    logger.info(best_variant[:300])
    return best_variant
