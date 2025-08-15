import os
import json
from loguru import logger
import sys

import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from utils import Config

logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class TopicAgencyFitEvaluator:
    """
    A qualitative evaluator that assesses how well conversations fit their intended topics
    and agencies, using a lightweight model that can run on a GPU with 9GB VRAM.
    """

    def __init__(
        self,
        cpu_only: bool = False,
        device: str = None,
    ):
        """
        Initialize the evaluator with a lightweight model.
        """
        if device is None:
            if cpu_only:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = Config()
        model_name = self.config.get("models.qualitative_model")
        logger.info(f"Initializing evaluator with model: {model_name} on {device}")

        # Configure quantization if using 4-bit precision
        if self.config.get("models.use_4bit_quantization") and device == "cuda":
            logger.info("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure model
        model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Set generation parameters
        self.max_new_tokens = self.config.get("qualitative_evaluation.max_new_tokens")
        self.temperature = self.config.get("qualitative_evaluation.temperature")
        self.device = device

        # Define evaluation criteria templates
        self.criteria_templates = self._define_evaluation_criteria()

    def _define_evaluation_criteria(self) -> Dict[str, str]:
        """Define the evaluation criteria templates."""
        prompt_file = self.config.get("qualitative_evaluation.prompt_file_topic")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data

    def _extract_score(self, response: str) -> Tuple[Optional[float], str]:
        """
        Extract numerical score from a generated response.


        """
        try:
            if "Score:" in response and "\n" in response:
                score_part, reasoning = response.split("\n", 1)
                score_str = score_part.replace("Score:", "").strip()

                score = float(score_str)

                if 1 <= score <= 5:
                    return score, reasoning.strip()

            import re

            number_matches = re.findall(r"\b([1-5](?:\.\d)?)\b", response[:50])
            if number_matches:
                return float(number_matches[0]), response

            return None, response
        except:
            return None, response

    def _clean_conversation(self, conversation: str) -> str:
        """
        Clean and format the conversation text for evaluation.

        Args:
            conversation: The  conversation text

        Returns:
            Cleaned conversation text
        """
        # Basic cleaning
        conversation = conversation.strip()

        # Make sure user/assistant markers are clear
        conversation = conversation.replace("**Kasutaja**:", "User:")
        conversation = conversation.replace("**Robot**:", "Assistant:")
        conversation = conversation.replace("Kasutaja:", "User:")
        conversation = conversation.replace("Robot:", "Assistant:")

        # Ensure there are line breaks between turns
        conversation = conversation.replace("User:", "\nUser:")
        conversation = conversation.replace("Assistant:", "\nAssistant:")

        return conversation

    def read_topic_description(self, topic_file: str) -> str:
        """
        Read and process a topic description file.

        Args:
            topic_file: Path to the topic description file

        Returns:
            Processed topic description
        """
        try:
            with open(topic_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Basic processing to extract key information
            # Extract the first 1000 characters if very long
            if len(content) > 2000:
                content = content[:2000] + "... [content truncated]"

            return content
        except Exception as e:
            logger.error(f"Error reading topic file {topic_file}: {e}")
            return "No topic description available."

    def read_agency_description(self, agency_name: str) -> str:
        """
        Get a description of the agency based on its name.

        Args:
            agency_name: Name of the agency

        Returns:
            Agency description
        """
        # This could be expanded to read from actual files describing agencies
        agency_descriptions = {
            "ID.ee": (
                "ID.ee provides information and services related to Estonian digital identity, "
                "including ID cards, Mobile-ID, Smart-ID, and digital signatures. They handle "
                "questions about applying for, using, and troubleshooting Estonian digital "
                "identity services."
            ),
            "Politsei-_ja_Piirivalveamet": (
                "The Police and Border Guard Board (Politsei- ja Piirivalveamet) handles law enforcement, "
                "border control, citizenship and migration issues. They provide services related to "
                "passports, identity documents, residence permits, citizenship applications, and "
                "reporting crimes or incidents."
            ),
        }

        # Extract agency name from path if needed
        if "/" in agency_name or "\\" in agency_name:
            agency_name = os.path.basename(agency_name)

        # Try to match with known descriptions
        for known_agency, description in agency_descriptions.items():
            if (
                known_agency.lower() in agency_name.lower()
                or agency_name.lower() in known_agency.lower()
            ):
                return description

        # If no match, provide a generic description
        return f"This is an Estonian government agency named '{agency_name}'."

    def evaluate_topic_agency_fit(
        self,
        conversation: str,
        topic_name: str,
        topic_description: str,
        agency_name: str,
        agency_description: str,
        criteria: str = "topic_fit",
    ) -> Dict[str, Any]:
        """
        Evaluate how well a conversation fits a topic and agency.

        Args:
            conversation: The conversation text
            topic_name: Name of the topic
            topic_description: Description of the topic
            agency_name: Name of the agency
            agency_description: Description of the agency
            criteria: The evaluation criteria to use

        Returns:
            Dictionary with evaluation results
        """
        # Clean the conversation
        clean_conv = self._clean_conversation(conversation)

        # Get the evaluation prompt
        prompt_template = self.criteria_templates.get(criteria)
        if not prompt_template:
            logger.error(f"Unknown criteria: {criteria}")
            return {"error": f"Unknown criteria: {criteria}"}

        # Construct the full prompt
        prompt = (
            f"You are an expert conversation evaluator for Estonian government agencies.\n\n"
            f"AGENCY: {agency_name}\n"
            f"AGENCY DESCRIPTION: {agency_description}\n\n"
            f"TOPIC: {topic_name}\n"
            f"TOPIC DESCRIPTION: {topic_description}\n\n"
            f"CONVERSATION TO EVALUATE:\n{clean_conv}\n\n"
            f"EVALUATION TASK:\n{prompt_template}\n\n"
            f"YOUR EVALUATION:"
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate evaluation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
            )

        # Decode the response and extract only the generated part
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt) :].strip()

        # Extract score if applicable
        score, reasoning = self._extract_score(response)

        return {
            "criteria": criteria,
            "score": score,
            "response": response,
            "reasoning": reasoning if score is not None else response,
        }

    def evaluate_conversation_topic_agency_fit(
        self,
        conversation: str,
        topic_name: str,
        topic_description: str,
        agency_name: str,
        agency_description: str,
    ) -> Dict[str, Any]:
        """
        Perform a full evaluation of a conversation's fit to topic and agency.

        Args:
            conversation: The conversation text
            topic_name: Name of the topic
            topic_description: Description of the topic
            agency_name: Name of the agency
            agency_description: Description of the agency

        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {}
        numerical_scores = {}

        # Evaluate score-based criteria
        score_criteria = ["topic_fit", "agency_fit", "classification_confidence"]

        for criteria in tqdm(score_criteria, desc="Evaluating fit criteria"):
            eval_result = self.evaluate_topic_agency_fit(
                conversation,
                topic_name,
                topic_description,
                agency_name,
                agency_description,
                criteria,
            )
            results[criteria] = eval_result

            if eval_result["score"] is not None:
                numerical_scores[criteria] = eval_result["score"]

        # Evaluate non-score criteria
        results["potential_confusion"] = self.evaluate_topic_agency_fit(
            conversation,
            topic_name,
            topic_description,
            agency_name,
            agency_description,
            "potential_confusion",
        )

        results["key_indicators"] = self.evaluate_topic_agency_fit(
            conversation,
            topic_name,
            topic_description,
            agency_name,
            agency_description,
            "key_indicators",
        )

        # Calculate aggregate score
        if numerical_scores:
            # Weight classification confidence slightly higher
            weights = {
                "topic_fit": 0.35,
                "agency_fit": 0.35,
                "classification_confidence": 0.3,
            }

            weighted_sum = sum(
                numerical_scores.get(k, 0) * v
                for k, v in weights.items()
                if k in numerical_scores
            )
            total_weight = sum(v for k, v in weights.items() if k in numerical_scores)

            aggregate_score = weighted_sum / total_weight if total_weight > 0 else None
        else:
            aggregate_score = None

        return {
            "topic_name": topic_name,
            "agency_name": agency_name,
            "detailed_scores": results,
            "numerical_scores": numerical_scores,
            "aggregate_score": aggregate_score,
        }


def path_to_str(obj):
    """
    Convert any Path objects to strings for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with all Path objects converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [path_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(path_to_str(item) for item in obj)
    else:
        return obj


def find_topic_description_file(topic_dir: str) -> Optional[str]:
    """
    Find a suitable topic description file in the specified directory or its parent.

    Args:
        topic_dir: Path to the topic directory

    Returns:
        Path to the topic description file, or None if not found
    """
    # Check for common topic description file patterns in the topic directory
    topic_name = os.path.basename(topic_dir)
    parent_dir = os.path.dirname(topic_dir)

    # Possible patterns for topic description files
    potential_files = [
        # In the topic directory
        os.path.join(topic_dir, "description.txt"),
        os.path.join(topic_dir, f"{topic_name}.txt"),
        os.path.join(topic_dir, "topic.txt"),
        # In the parent directory
        os.path.join(parent_dir, f"{topic_name}.txt"),
        os.path.join(parent_dir, f"{topic_name}_description.txt"),
        os.path.join(
            parent_dir,
            "output_" + os.path.basename(parent_dir),
            f"{topic_name}_-_{os.path.basename(parent_dir)}.txt",
        ),
        os.path.join(
            parent_dir,
            "..",
            "data",
            "output_" + os.path.basename(parent_dir),
            f"{topic_name}_-_{os.path.basename(parent_dir)}.txt",
        ),
    ]

    # Try each potential file
    for file_path in potential_files:
        if os.path.isfile(file_path):
            logger.info(f"Found topic description file: {file_path}")
            return file_path

    logger.warning(f"No topic description file found for {topic_name}")
    return None


# Modify these functions in your code to handle explicit conversation paths


def evaluate_topic_directory(
    evaluator: TopicAgencyFitEvaluator,
    topic_dir: str,
    agency_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
    conversation_path: Optional[str] = None,  # Add this parameter
) -> Dict[str, Any]:
    """
    Evaluate all conversations in a topic directory for topic and agency fit.

    Args:
        evaluator: The evaluator instance
        topic_dir: Path to the topic directory
        agency_dir: Path to the agency directory
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files
        conversation_path: Optional explicit path to conversation files (overrides topic_dir)

    Returns:
        Dictionary with evaluation summary
    """
    # Find conversation files - use explicit path if provided
    if conversation_path:
        # Use the explicit conversation path
        conversation_files = sorted(list(Path(conversation_path).glob(pattern)))
        logger.info(f"Looking for conversations in explicit path: {conversation_path}")
    else:
        # Look in the topic directory (original behavior)
        conversation_files = sorted(list(Path(topic_dir).glob(pattern)))
        logger.info(f"Looking for conversations in topic directory: {topic_dir}")

    if not conversation_files:
        logger.warning(f"No conversation files matching pattern '{pattern}' found")
        return {"error": f"No conversation files found"}

    # Create output directory with topic name
    topic_name = os.path.basename(topic_dir)
    topic_output_dir = os.path.join(output_dir, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    # Evaluate each conversation
    all_results = []
    for file_path in conversation_files:
        try:
            result = evaluate_conversation_file(
                evaluator, str(file_path), topic_dir, agency_dir, topic_output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")

    # Calculate summary statistics
    topic_summary = {
        "topic_name": topic_name,
        "agency_name": os.path.basename(agency_dir),
        "total_conversations": len(all_results),
        "conversation_files": [r.get("file_name") for r in all_results],
    }

    # Calculate average scores
    scores = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        valid_scores = [
            r.get("numerical_scores", {}).get(criteria)
            for r in all_results
            if r.get("numerical_scores", {}).get(criteria) is not None
        ]
        if valid_scores:
            scores[criteria] = {
                "average": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
            }

    # Calculate aggregate score
    aggregate_scores = [
        r.get("aggregate_score")
        for r in all_results
        if r.get("aggregate_score") is not None
    ]
    if aggregate_scores:
        topic_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )
        topic_summary["min_aggregate_score"] = min(aggregate_scores)
        topic_summary["max_aggregate_score"] = max(aggregate_scores)

    topic_summary["scores"] = scores

    # Save topic summary
    with open(
        os.path.join(topic_output_dir, "topic_fit_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(path_to_str(topic_summary), f, ensure_ascii=False, indent=2)

    # Generate a readable report
    _generate_topic_fit_report(
        topic_summary,
        all_results,
        os.path.join(topic_output_dir, "topic_fit_report.md"),
    )

    return topic_summary


# Modified functions to work with flat directory structure


def find_topic_files(agency_dir: str) -> List[Tuple[str, str]]:
    """
    Find topic files in an agency directory with flat structure.

    Args:
        agency_dir: Path to the agency directory

    Returns:
        List of tuples: (topic_filename, full_path_to_topic_file)
    """
    # Look for .txt files directly in the agency directory
    topic_files = []

    # Check if directory exists
    if not os.path.isdir(agency_dir):
        logger.warning(f"Agency directory not found: {agency_dir}")
        return []

    # Find all .txt files in the agency directory
    for file_path in Path(agency_dir).glob("*.txt"):
        topic_name = file_path.stem  # Get filename without extension
        topic_files.append((topic_name, str(file_path)))

    logger.info(f"Found {len(topic_files)} topic files in {agency_dir}")
    return topic_files


def evaluate_topic_file(
    evaluator: TopicAgencyFitEvaluator,
    topic_name: str,
    topic_file_path: str,
    agency_dir: str,
    conversation_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate conversations for a topic based on a topic file.

    Args:
        evaluator: The evaluator instance
        topic_name: Name of the topic (filename without extension)
        topic_file_path: Path to the topic file
        agency_dir: Path to the agency directory
        conversation_dir: Path to the conversation directory
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files

    Returns:
        Dictionary with evaluation summary
    """
    # Construct path to conversation directory for this topic
    topic_conversation_dir = os.path.join(
        conversation_dir, os.path.basename(agency_dir), topic_name
    )

    # Check if the conversation directory exists
    if not os.path.isdir(topic_conversation_dir):
        logger.warning(f"Conversation directory not found: {topic_conversation_dir}")
        return {
            "topic_name": topic_name,
            "error": f"Conversation directory not found: {topic_conversation_dir}",
        }

    # Find conversation files
    config = Config()
    conversation_files = sorted(
        list(
            Path(topic_conversation_dir).glob(
                config.get("general.conversation_pattern")
            )
        )
    )

    if not conversation_files:
        # logger.warning(
        #    f"No conversation files matching pattern '{config.get("general.conversation_pattern")}' found in {topic_conversation_dir}"
        # )
        return {
            "topic_name": topic_name,
            "error": f"No conversation files found in {topic_conversation_dir}",
        }

    # Create output directory with topic name
    agency_name = os.path.basename(agency_dir)
    topic_output_dir = os.path.join(output_dir, agency_name, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    # Read topic description
    topic_description = evaluator.read_topic_description(topic_file_path)
    agency_description = evaluator.read_agency_description(agency_name)

    # Evaluate each conversation
    all_results = []
    for file_path in conversation_files:
        try:
            # Read conversation
            with open(file_path, "r", encoding="utf-8") as f:
                conversation = f.read()

            # Evaluate conversation
            logger.info(f"Evaluating conversation: {file_path.name}")
            result = evaluator.evaluate_conversation_topic_agency_fit(
                conversation,
                topic_name,
                topic_description,
                agency_name,
                agency_description,
            )

            # Add file information
            result["file_name"] = file_path.name
            result["file_path"] = str(file_path)

            # Save individual result
            file_stem = file_path.stem
            output_file = os.path.join(
                topic_output_dir, f"{file_stem}_topic_agency_fit.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(path_to_str(result), f, ensure_ascii=False, indent=2)

            all_results.append(result)

        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")

    # Calculate summary statistics
    topic_summary = {
        "topic_name": topic_name,
        "agency_name": agency_name,
        "total_conversations": len(all_results),
        "conversation_files": [r.get("file_name") for r in all_results],
    }

    # Calculate average scores
    scores = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        valid_scores = [
            r.get("numerical_scores", {}).get(criteria)
            for r in all_results
            if r.get("numerical_scores", {}).get(criteria) is not None
        ]
        if valid_scores:
            scores[criteria] = {
                "average": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
            }

    # Calculate aggregate score
    aggregate_scores = [
        r.get("aggregate_score")
        for r in all_results
        if r.get("aggregate_score") is not None
    ]
    if aggregate_scores:
        topic_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )
        topic_summary["min_aggregate_score"] = min(aggregate_scores)
        topic_summary["max_aggregate_score"] = max(aggregate_scores)

    topic_summary["scores"] = scores

    # Save topic summary
    with open(
        os.path.join(topic_output_dir, "topic_fit_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(path_to_str(topic_summary), f, ensure_ascii=False, indent=2)

    # Generate a readable report
    _generate_topic_fit_report(
        topic_summary,
        all_results,
        os.path.join(topic_output_dir, "topic_fit_report.md"),
    )

    return topic_summary


def evaluate_agency_directory_flat(
    evaluator: TopicAgencyFitEvaluator,
    agency_dir: str,
    conversation_base_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
) -> Dict[str, Any]:
    """
    Evaluate all topics in an agency directory with flat structure.

    Args:
        evaluator: The evaluator instance
        agency_dir: Path to the agency directory with topic files
        conversation_base_dir: Base directory containing conversation files
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files

    Returns:
        Dictionary with agency evaluation summary
    """
    # Find topic files in the agency directory
    topic_files = find_topic_files(agency_dir)

    if not topic_files:
        logger.warning(f"No topic files found in {agency_dir}")
        return {"error": f"No topic files found in {agency_dir}"}

    # Create output directory with agency name
    agency_name = os.path.basename(agency_dir)
    agency_output_dir = os.path.join(output_dir, agency_name)
    os.makedirs(agency_output_dir, exist_ok=True)

    # Evaluate each topic
    topic_results = []
    for topic_name, topic_file_path in topic_files:
        logger.info(f"Evaluating topic: {topic_name}")
        result = evaluate_topic_file(
            evaluator,
            topic_name,
            topic_file_path,
            agency_dir,
            conversation_base_dir,
            output_dir,
            pattern,
        )
        topic_results.append(result)

    # Calculate agency summary
    agency_summary = {
        "agency_name": agency_name,
        "topics_evaluated": len(topic_results),
        "topic_results": {
            r.get("topic_name", ""): r for r in topic_results if "topic_name" in r
        },
    }

    # Calculate overall agency scores
    scores_by_criteria = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        all_scores = []
        for topic in topic_results:
            if "scores" in topic and criteria in topic["scores"]:
                avg_score = topic["scores"][criteria]["average"]
                all_scores.append(avg_score)

        if all_scores:
            scores_by_criteria[criteria] = sum(all_scores) / len(all_scores)

    agency_summary["average_scores"] = scores_by_criteria

    # Calculate overall aggregate score
    aggregate_scores = [
        topic.get("average_aggregate_score")
        for topic in topic_results
        if "average_aggregate_score" in topic
    ]
    if aggregate_scores:
        agency_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )

    # Save agency summary
    with open(
        os.path.join(agency_output_dir, "agency_fit_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(path_to_str(agency_summary), f, ensure_ascii=False, indent=2)

    # Generate agency report
    _generate_agency_fit_report(
        agency_summary, os.path.join(agency_output_dir, "agency_fit_report.json")
    )

    return agency_summary


def evaluate_conversation_file(
    evaluator: TopicAgencyFitEvaluator,
    conversation_file: str,
    topic_dir: str,
    agency_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate a single conversation file for topic and agency fit.

    Args:
        evaluator: The evaluator instance
        conversation_file: Path to the conversation file
        topic_dir: Path to the topic directory
        agency_dir: Path to the agency directory
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation results
    """
    # Read conversation
    try:
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation = f.read()
    except Exception as e:
        logger.error(f"Error reading conversation file {conversation_file}: {e}")
        return {"error": f"Error reading file: {e}"}

    # Get topic and agency names
    topic_name = os.path.basename(topic_dir)
    agency_name = os.path.basename(agency_dir)

    # Find topic description file
    topic_description_file = find_topic_description_file(topic_dir)
    if topic_description_file:
        topic_description = evaluator.read_topic_description(topic_description_file)
    else:
        topic_description = f"This topic is about {topic_name}."

    # Get agency description
    agency_description = evaluator.read_agency_description(agency_name)

    # Evaluate the conversation
    logger.info(f"Evaluating conversation: {os.path.basename(conversation_file)}")
    result = evaluator.evaluate_conversation_topic_agency_fit(
        conversation, topic_name, topic_description, agency_name, agency_description
    )

    # Add file information
    result["file_name"] = os.path.basename(conversation_file)
    result["file_path"] = conversation_file

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save evaluation result
    file_stem = os.path.splitext(os.path.basename(conversation_file))[0]
    output_file = os.path.join(output_dir, f"{file_stem}_topic_agency_fit.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(path_to_str(result), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation result to {output_file}")

    return result


def evaluate_topic_directory(
    evaluator: TopicAgencyFitEvaluator,
    topic_dir: str,
    agency_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
) -> Dict[str, Any]:
    """
    Evaluate all conversations in a topic directory for topic and agency fit.


    Returns:
        Dictionary with evaluation summary
    """
    # Find conversation files
    conversation_files = sorted(list(Path(topic_dir).glob(pattern)))

    if not conversation_files:
        logger.warning(
            f"No conversation files matching pattern '{pattern}' found in {topic_dir}"
        )
        return {"error": f"No conversation files found in {topic_dir}"}

    # Create output directory with topic name
    topic_name = os.path.basename(topic_dir)
    topic_output_dir = os.path.join(output_dir, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    # Evaluate each conversation
    all_results = []
    for file_path in conversation_files:
        try:
            result = evaluate_conversation_file(
                evaluator, str(file_path), topic_dir, agency_dir, topic_output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")

    # Calculate summary statistics
    topic_summary = {
        "topic_name": topic_name,
        "agency_name": os.path.basename(agency_dir),
        "total_conversations": len(all_results),
        "conversation_files": [r.get("file_name") for r in all_results],
    }

    # Calculate average scores
    scores = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        valid_scores = [
            r.get("numerical_scores", {}).get(criteria)
            for r in all_results
            if r.get("numerical_scores", {}).get(criteria) is not None
        ]
        if valid_scores:
            scores[criteria] = {
                "average": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
            }

    # Calculate aggregate score
    aggregate_scores = [
        r.get("aggregate_score")
        for r in all_results
        if r.get("aggregate_score") is not None
    ]
    if aggregate_scores:
        topic_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )
        topic_summary["min_aggregate_score"] = min(aggregate_scores)
        topic_summary["max_aggregate_score"] = max(aggregate_scores)

    topic_summary["scores"] = scores

    # Save topic summary
    with open(
        os.path.join(topic_output_dir, "topic_fit_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(path_to_str(topic_summary), f, ensure_ascii=False, indent=2)

    # Generate a readable report
    _generate_topic_fit_report(
        topic_summary,
        all_results,
        os.path.join(topic_output_dir, "topic_fit_report.json"),
    )

    return topic_summary


def _generate_topic_fit_report(
    topic_summary: Dict[str, Any], all_results: List[Dict[str, Any]], output_file: str
):
    """
    Generate a readable Markdown report for topic fit evaluation.


    """
    # write to json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def _generate_agency_fit_report(agency_summary: Dict[str, Any], output_file: str):
    """
    Generate a readable Markdown report for agency fit evaluation.


    """
    # write to json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(agency_summary, f, ensure_ascii=False, indent=2)
