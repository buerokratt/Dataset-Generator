import os
import json
import logging
import argparse
import numpy as np
import torch
import sys
from loguru import logger
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from utils import Config

# Configure logging
logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class LightweightQualitativeEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the evaluator with a lightweight model.

        """

        self.config = Config()
        self.model_name = self.config.get("models.qualitative_model")
        logger.info(f"Initializing evaluator with model: {self.model_name}")

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Configure model
        model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        self.max_new_tokens = self.config.get("qualitative_evaluation.max_new_tokens")
        self.temperature = self.config.get("qualitative_evaluation.temperature")
        self.device = device

        self.criteria_templates = self._define_evaluation_criteria()

    def _define_evaluation_criteria(self) -> Dict[str, str]:
        """Define the evaluation criteria templates."""
        # read prompt in json
        prompt_file = self.config.get("qualitative_evaluation.prompt_file")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data

    def _extract_score(self, response: str) -> Tuple[Optional[float], str]:
        """
        Extract numerical score from a generated response.

        Args:
            response: The generated response text

        Returns:
            Tuple of (score, reasoning) or (None, original response)
        """
        try:
            # Look for score pattern like "Score: 4" or "Score: 4.5"
            if "Score:" in response and "\n" in response:
                score_part, reasoning = response.split("\n", 1)
                score_str = score_part.replace("Score:", "").strip()

                # Convert to float
                score = float(score_str)

                # Validate score is in range 1-5
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


        """
        # Basic cleaning
        conversation = conversation.strip()

        conversation = conversation.replace("**Kasutaja**:", "User:")
        conversation = conversation.replace("**Robot**:", "Assistant:")
        conversation = conversation.replace("Kasutaja:", "User:")
        conversation = conversation.replace("Robot:", "Assistant:")

        conversation = conversation.replace("User:", "\nUser:")
        conversation = conversation.replace("Assistant:", "\nAssistant:")

        return conversation

    def evaluate_conversation(
        self, conversation: str, criteria: str = "overall_quality"
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation using the specified criteria.

        """
        clean_conv = self._clean_conversation(conversation)

        prompt_template = self.criteria_templates.get(
            criteria, self.criteria_templates["overall_quality"]
        )

        prompt = (
            f"You are an expert conversation evaluator.\n\n"
            f"CONVERSATION TO EVALUATE:\n{clean_conv}\n\n"
            f"EVALUATION TASK:\n{prompt_template}\n\n"
            f"YOUR EVALUATION:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt) :].strip()

        score, reasoning = self._extract_score(response)

        return {
            "criteria": criteria,
            "score": score,
            "response": response,
            "reasoning": reasoning if score is not None else response,
        }

    def evaluate_conversation_full(self, conversation: str) -> Dict[str, Any]:
        """
        Perform a full evaluation of a conversation using all criteria.

        Args:
            conversation: The conversation text to evaluate

        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {}
        numerical_scores = {}

        score_criteria = [
            "overall_quality",
            "coherence",
            "relevance",
            "factual_accuracy",
            "helpfulness",
            "natural_language",
            "completeness",
        ]

        for criteria in tqdm(score_criteria, desc="Evaluating criteria"):
            eval_result = self.evaluate_conversation(conversation, criteria)
            results[criteria] = eval_result

            if eval_result["score"] is not None:
                numerical_scores[criteria] = eval_result["score"]

        results["strengths_weaknesses"] = self.evaluate_conversation(
            conversation, "strengths_weaknesses"
        )
        results["improvement_suggestions"] = self.evaluate_conversation(
            conversation, "improvement_suggestions"
        )

        if numerical_scores:
            aggregate_score = sum(numerical_scores.values()) / len(numerical_scores)
        else:
            aggregate_score = None

        return {
            "detailed_scores": results,
            "numerical_scores": numerical_scores,
            "aggregate_score": aggregate_score,
        }

    def aggregate_evaluation(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation results into a summary.

        Args:
            evaluation_results: The full evaluation results dictionary

        Returns:
            Dictionary with aggregated evaluation summary
        """
        numerical_scores = evaluation_results.get("numerical_scores", {})

        if not numerical_scores:
            return {
                "status": "error",
                "message": "No valid numerical scores in evaluation results",
            }

        # Calculate statistics
        scores_list = list(numerical_scores.values())

        summary = {
            "aggregate_score": evaluation_results.get("aggregate_score"),
            "min_score": min(scores_list) if scores_list else None,
            "max_score": max(scores_list) if scores_list else None,
            "score_by_criteria": numerical_scores,
        }

        # Extract strengths and weaknesses
        strengths_weaknesses = evaluation_results.get("detailed_scores", {}).get(
            "strengths_weaknesses", {}
        )
        if strengths_weaknesses:
            summary["strengths_weaknesses"] = strengths_weaknesses.get("response", "")

        # Extract improvement suggestions
        improvement_suggestions = evaluation_results.get("detailed_scores", {}).get(
            "improvement_suggestions", {}
        )
        if improvement_suggestions:
            summary["improvement_suggestions"] = improvement_suggestions.get(
                "response", ""
            )

        return summary

    def batch_evaluate_conversations(
        self, conversations: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of conversations.

        Args:
            conversations: List of conversation texts to evaluate

        Returns:
            List of evaluation results for each conversation
        """
        results = []

        for i, conversation in enumerate(
            tqdm(conversations, desc="Evaluating conversations")
        ):
            logger.info(f"Evaluating conversation {i + 1}/{len(conversations)}")

            if self.config.get("qualitative_evaluation.full_evaluation"):
                eval_result = self.evaluate_conversation_full(conversation)
            else:
                eval_result = self.evaluate_conversation(
                    conversation, "overall_quality"
                )

            results.append(eval_result)

        return results

    def evaluate_and_save(
        self, conversation_files: List[str], output_dir: str
    ) -> Dict[str, Any]:
        """
        Evaluate multiple conversation files and save results.

        Args:
            conversation_files: List of file paths containing conversations
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary with overall evaluation summary
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        all_conversations = []

        # Load and evaluate each conversation
        for file_path in conversation_files:
            try:
                # Read conversation from file
                with open(file_path, "r", encoding="utf-8") as f:
                    conversation = f.read()

                # Skip empty files
                if not conversation.strip():
                    logger.warning(f"Skipping empty file: {file_path}")
                    continue

                all_conversations.append(conversation)

                file_name = os.path.basename(file_path)
                file_stem = os.path.splitext(file_name)[0]

                logger.info(f"Evaluating: {file_name}")

                if self.config.get("qualitative_evaluation.full_evaluation"):
                    eval_result = self.evaluate_conversation_full(conversation)
                else:
                    eval_result = self.evaluate_conversation(
                        conversation, "overall_quality"
                    )

                eval_result["file_name"] = file_name
                eval_result["file_path"] = file_path

                output_file = os.path.join(output_dir, f"{file_stem}_evaluation.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(eval_result, f, ensure_ascii=False, indent=2)

                all_results.append(eval_result)

            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")

        summary = {
            "total_conversations": len(all_results),
            "average_aggregate_score": np.mean(
                [
                    r.get("aggregate_score", 0)
                    for r in all_results
                    if r.get("aggregate_score") is not None
                ]
            ),
            "conversation_files": [r.get("file_name") for r in all_results],
        }

        with open(
            os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._generate_evaluation_report(
            all_results, os.path.join(output_dir, "evaluation_report.md")
        )

        return summary

    def _generate_evaluation_report(
        self, all_results: List[Dict[str, Any]], output_file: str
    ):
        """
        Generate a readable Markdown report from evaluation results.

        Args:
            all_results: List of evaluation results
            output_file: Path to save the report
        """
        # write to json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)


def evaluate_topic_directory(
    evaluator: LightweightQualitativeEvaluator,
    topic_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
) -> Dict[str, Any]:
    """
    Evaluate all conversations in a topic directory.

    Args:
        evaluator: The evaluator instance
        topic_dir: Path to the topic directory
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files

    Returns:
        Dictionary with evaluation summary
    """
    conversation_files = sorted(list(Path(topic_dir).glob(pattern)))

    if not conversation_files:
        logger.warning(
            f"No conversation files matching pattern '{pattern}' found in {topic_dir}"
        )
        return {"error": f"No conversation files found in {topic_dir}"}

    topic_name = os.path.basename(topic_dir)
    topic_output_dir = os.path.join(output_dir, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    logger.info(f"Evaluating {len(conversation_files)} conversations in {topic_dir}")
    return evaluator.evaluate_and_save(conversation_files, topic_output_dir)


def evaluate_agency_directory(
    evaluator: LightweightQualitativeEvaluator,
    agency_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate all topics in an agency directory.

    """
    # Find topic directories
    topic_dirs = [d for d in Path(agency_dir).iterdir() if d.is_dir()]

    if not topic_dirs:
        logger.warning(f"No topic directories found in {agency_dir}")

        conversation_files = sorted(list(Path(agency_dir).glob("conversation_*.txt")))
        if conversation_files:
            logger.info(
                f"Found {len(conversation_files)} conversations directly in agency directory"
            )
            agency_name = os.path.basename(agency_dir)
            agency_output_dir = os.path.join(output_dir, agency_name)
            return evaluator.evaluate_and_save(conversation_files, agency_output_dir)

        return {"error": f"No topic directories or conversations found in {agency_dir}"}

    agency_name = os.path.basename(agency_dir)
    agency_output_dir = os.path.join(output_dir, agency_name)

    topic_results = []
    for topic_dir in topic_dirs:
        logger.info(f"Evaluating topic: {topic_dir.name}")
        topic_result = evaluate_topic_directory(
            evaluator, str(topic_dir), agency_output_dir
        )
        topic_results.append({"topic": topic_dir.name, "result": topic_result})

    valid_scores = []
    for topic in topic_results:
        avg_score = topic["result"].get("average_aggregate_score")
        if avg_score is not None:
            valid_scores.append(avg_score)

    agency_summary = {
        "agency_name": agency_name,
        "topics_evaluated": len(topic_results),
        "average_agency_score": sum(valid_scores) / len(valid_scores)
        if valid_scores
        else None,
        "topic_results": {t["topic"]: t["result"] for t in topic_results},
    }

    agency_summary_file = os.path.join(
        agency_output_dir, "agency_evaluation_summary.json"
    )
    with open(agency_summary_file, "w", encoding="utf-8") as f:
        json.dump(agency_summary, f, ensure_ascii=False, indent=2)

    agency_report_file = os.path.join(agency_output_dir, "agency_evaluation_report.md")
    with open(agency_report_file, "w", encoding="utf-8") as f:
        f.write(f"# {agency_name} Conversation Evaluation Report\n\n")

        if valid_scores:
            avg_agency_score = sum(valid_scores) / len(valid_scores)
            f.write(f"## Agency Overview\n\n")
            f.write(f"- **Topics Evaluated**: {len(topic_results)}\n")
            f.write(f"- **Average Agency Score**: {avg_agency_score:.2f}/5.0\n\n")

            # Topic scores table
            f.write("## Topic Scores\n\n")
            f.write("| Topic | Average Score | Conversations Evaluated |\n")
            f.write("|-------|--------------|-------------------------|\n")

            # Sort topics by score
            sorted_topics = sorted(
                topic_results,
                key=lambda t: t["result"].get("average_aggregate_score", 0)
                if t["result"].get("average_aggregate_score") is not None
                else 0,
                reverse=True,
            )

            for topic in sorted_topics:
                topic_name = topic["topic"]
                topic_score = topic["result"].get("average_aggregate_score")
                conv_count = topic["result"].get("total_conversations", 0)

                if topic_score is not None:
                    f.write(f"| {topic_name} | {topic_score:.2f} | {conv_count} |\n")
                else:
                    f.write(f"| {topic_name} | N/A | {conv_count} |\n")

    return agency_summary
