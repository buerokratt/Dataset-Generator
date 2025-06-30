import re
import json
from typing import Dict, Any
from string import Template

from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class PromptProcessor:
    """
    Process prompt templates for synthetic data generation.

    This class handles the processing of text templates with variable substitution
    and extraction of structured data (particularly JSON) from model responses.
    It serves as a core utility in the dataset generation pipeline, enabling:
    1. Variable interpolation in prompt templates using ${variable} syntax
    2. Robust JSON extraction from potentially malformed or text-wrapped responses

    The processor uses Python's string.Template for parameter substitution
    and regular expressions for JSON extraction, with multiple fallback patterns
    to handle common response formatting issues from LLMs.

    Methods:
        process(template: str, params: Dict[str, Any]) -> str:
            Processes a template string by substituting variables with values from params

        extract_json(text: str) -> str:
            Extracts valid JSON from text that might contain non-JSON content or formatting

    Example:
        ```python
        processor = PromptProcessor()

        # Process a template with variables
        template = "Generate data about ${topic} in ${language}."
        params = {"topic": "astronomy", "language": "Estonian"}
        prompt = processor.process(template, params)

        # Extract JSON from model response
        response = "Here's the data: ```json { \"name\": \"value\" } ```"
        json_str = processor.extract_json(response)  # Returns: { "name": "value" }
        ```
    """

    def process(self, template: str, params: Dict[str, Any]) -> str:
        """
        Process a prompt template with parameters

        Args:
            template: The prompt template string
            params: Parameters to inject into the template

        Returns:
            The processed prompt
        """
        # Use string.Template for parameter substitution
        try:
            template_obj = Template(template)
            processed = template_obj.safe_substitute(params)

            # Check for unresolved variables
            unresolved_vars = re.findall(r"\$\{([^}]+)\}", processed)
            if unresolved_vars:
                logger.warning(f"Unresolved template variables: {unresolved_vars}")
                logger.debug(f"Available parameters: {list(params.keys())}")

            return processed
        except Exception as e:
            logger.error(f"Error processing template: {e}")
            return template

    def extract_json(self, text: str) -> str:
        """
        Extract JSON from text that might contain non-JSON content

        Args:
            text: Text that might contain JSON

        Returns:
            The extracted JSON string, or original text if no JSON is found
        """
        # First, check for Markdown-style code blocks with JSON
        markdown_pattern = r"```(?:json)?\s*(.*?)```"
        markdown_match = re.search(markdown_pattern, text, re.DOTALL)

        if markdown_match:
            json_str = markdown_match.group(1).strip()
            # Add additional cleanup for common issues
            json_str = json_str.replace("\n", " ").replace("  ", " ")

            # Try to find a valid JSON object
            json_object_pattern = r"(\{.*\})"
            object_match = re.search(json_object_pattern, json_str, re.DOTALL)

            if object_match:
                potential_json = object_match.group(1)
                try:
                    # Validate it's actually JSON
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    pass

        # Then try to find JSON content between braces (your existing code)
        json_pattern = r"(\{[^\}]*\})"
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
            try:
                # Validate it's actually JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        # Fall back to original text
        return text
