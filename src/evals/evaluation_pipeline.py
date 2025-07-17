from typing import List, Dict, Tuple, Set, Optional, Any
import os
import re
import sys
import glob
import json
import numpy as np
from pathlib import Path
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import all evaluation modules
from information_coverage import compute_information_coverage
from length_appropriateness import compute_length_appropriateness_score
from relevance_score import compute_weighted_segment_relevance
from query_diversity_score import compute_query_diversity_for_topic
from redundancy_penalty import (
    compute_intra_conversation_redundancy,
    compute_inter_conversation_redundancy,
)
from topic_coverage_score import TopicCoverageAnalyzer
from topic_modeling_based_score import TopicConsistencyEvaluator
from agency_confusion_analysis import AgencyConfusionAnalyzer
from llm_based_quality_scoring import LightweightQualitativeEvaluator
from llm_topic_agency_analysis import TopicAgencyFitEvaluator

from utils import Config, read_file

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class EvaluationPipeline:
    """
    Main evaluation pipeline that orchestrates the execution of all evaluation metrics
    based on configuration settings.
    """

    def __init__(self, config_path: str = "eval_config.yaml"):
        """
        Initialize the evaluation pipeline with the given configuration.

        Args:
            config_path: Path to the evaluation configuration file
        """
        self.config = Config(config_path)
        self.output_dir = self.config.get("general.output_dir")

        # Configure file structure settings
        self.file_structure = {
            # Directory structure configuration
            "root_dir": self.config.get("file_structure.root_dir", "data"),
            "output_dir": self.config.get("file_structure.output_dir", "output"),
            "agency_subdir_pattern": self.config.get(
                "file_structure.agency_subdir_pattern", ""
            ),
            "topic_subdir_pattern": self.config.get(
                "file_structure.topic_subdir_pattern", ""
            ),
            # Conversation files configuration
            "conversation_dir": self.config.get(
                "file_structure.conversation_dir", "conversations"
            ),
            "conversation_file_pattern": self.config.get(
                "file_structure.conversation_file_pattern", "conversation_*.txt"
            ),
            "conversation_json_path": self.config.get(
                "file_structure.conversation_json_path", ""
            ),
            "conversation_json_key": self.config.get(
                "file_structure.conversation_json_key", "conversations"
            ),
            # JSON conversation format configuration
            "conversation_json_format": self.config.get(
                "file_structure.conversation_json_format", "simple_array"
            ),
            "conversation_question_key": self.config.get(
                "file_structure.conversation_question_key", "question"
            ),
            "conversation_speaker_key": self.config.get(
                "file_structure.conversation_speaker_key", "role"
            ),
            "conversation_content_key": self.config.get(
                "file_structure.conversation_content_key", "content"
            ),
            # Topic document files configuration
            "topic_dir": self.config.get("file_structure.topic_dir", "topics"),
            "topic_file_pattern": self.config.get(
                "file_structure.topic_file_pattern", "*.txt"
            ),
            "topic_json_path": self.config.get("file_structure.topic_json_path", ""),
            "topic_json_key": self.config.get(
                "file_structure.topic_json_key", "content"
            ),
        }

        # For backward compatibility
        self.topic_dir = self.config.get(
            "general.topic_directory", self.file_structure["topic_dir"]
        )
        self.conversation_dir = self.config.get(
            "general.conversation_directory", self.file_structure["conversation_dir"]
        )
        self.conversation_pattern = self.config.get(
            "general.conversation_pattern",
            self.file_structure["conversation_file_pattern"],
        )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize results dictionary
        self.results = {
            "by_agency": {},
            "by_topic": {},
            "by_conversation": {},
            "overall": {},
        }

        # Initialize modules that require initialization
        if self.config.get("topic_coverage.enabled"):
            self.topic_coverage_analyzer = TopicCoverageAnalyzer()

        if self.config.get("topic_consistency.enabled"):
            self.topic_consistency_evaluator = TopicConsistencyEvaluator()

        if self.config.get("agency_confusion.enabled"):
            self.agency_confusion_analyzer = AgencyConfusionAnalyzer()

        if self.config.get("qualitative_evaluation.enabled"):
            self.qualitative_evaluator = LightweightQualitativeEvaluator()

        if self.config.get("topic_agency_fit.enabled", False):
            self.topic_agency_fit_evaluator = TopicAgencyFitEvaluator()

    def gather_agencies_topics_conversations(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Gather all agencies, topics, and conversations based on the configured directory structure.

        Returns:
            Dictionary mapping agency names to topics and their conversations
        """
        agency_topics_conversations = {}

        # Handle different file structure configurations
        if self.file_structure["conversation_json_path"]:
            # JSON-based structure
            return self._gather_from_json()
        else:
            # Directory-based structure
            return self._gather_from_directories()

    def _gather_from_directories(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Gather conversations from a directory-based structure.

        Returns:
            Dictionary mapping agency names to topics and their conversations
        """
        agency_topics_conversations = {}

        # Check if the conversation directory exists
        if not os.path.exists(self.conversation_dir):
            logger.error(f"Conversation directory not found: {self.conversation_dir}")
            return {}

        # If agency pattern is specified, use it to identify agency directories
        if self.file_structure["agency_subdir_pattern"]:
            agency_dir_pattern = os.path.join(
                self.conversation_dir, self.file_structure["agency_subdir_pattern"]
            )
            agency_dirs = glob.glob(agency_dir_pattern)

            for agency_path in agency_dirs:
                agency_name = os.path.basename(agency_path)
                agency_topics_conversations[agency_name] = {}

                # If topic pattern is specified, use it to identify topic directories
                if self.file_structure["topic_subdir_pattern"]:
                    topic_dir_pattern = os.path.join(
                        agency_path, self.file_structure["topic_subdir_pattern"]
                    )
                    topic_dirs = glob.glob(topic_dir_pattern)

                    for topic_path in topic_dirs:
                        topic_name = os.path.basename(topic_path)
                        conversations = self._read_conversations_from_directory(
                            topic_path
                        )

                        if conversations:
                            agency_topics_conversations[agency_name][topic_name] = (
                                conversations
                            )
                else:
                    # Assume topics are direct subdirectories
                    topic_dirs = [
                        d
                        for d in os.listdir(agency_path)
                        if os.path.isdir(os.path.join(agency_path, d))
                    ]

                    for topic in topic_dirs:
                        topic_path = os.path.join(agency_path, topic)
                        conversations = self._read_conversations_from_directory(
                            topic_path
                        )

                        if conversations:
                            agency_topics_conversations[agency_name][topic] = (
                                conversations
                            )
        else:
            # Assume direct agency directories
            agency_dirs = [
                d
                for d in os.listdir(self.conversation_dir)
                if os.path.isdir(os.path.join(self.conversation_dir, d))
            ]

            for agency in agency_dirs:
                agency_path = os.path.join(self.conversation_dir, agency)
                agency_topics_conversations[agency] = {}

                # Get all topic directories for this agency
                topic_dirs = [
                    d
                    for d in os.listdir(agency_path)
                    if os.path.isdir(os.path.join(agency_path, d))
                ]

                for topic in topic_dirs:
                    topic_path = os.path.join(agency_path, topic)
                    conversations = self._read_conversations_from_directory(topic_path)

                    if conversations:
                        agency_topics_conversations[agency][topic] = conversations

        return agency_topics_conversations

    def _gather_from_json(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Gather conversations from a JSON-based structure.

        Returns:
            Dictionary mapping agency names to topics and their conversations
        """
        agency_topics_conversations = {}

        # Load conversations from JSON
        json_path = self.file_structure["conversation_json_path"]
        if not os.path.exists(json_path):
            logger.error(f"Conversation JSON file not found: {json_path}")
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract using the specified key path
            json_key = self.file_structure["conversation_json_key"]
            if json_key:
                keys = json_key.split(".")
                content = data
                for key in keys:
                    if key in content:
                        content = content[key]
                    else:
                        logger.error(f"Key '{key}' not found in JSON structure")
                        return {}
            else:
                content = data

            # Process based on the content structure
            if isinstance(content, dict):
                # If content is a dict, keys might be agencies or topics
                if (
                    "agency_key" in self.file_structure
                    and self.file_structure["agency_key"]
                ):
                    # Structure has explicit agency keys
                    agency_key = self.file_structure["agency_key"]
                    topic_key = self.file_structure["topic_key"]

                    for item_id, item in content.items():
                        agency = item.get(agency_key, "default_agency")
                        topic = item.get(topic_key, "default_topic")

                        if agency not in agency_topics_conversations:
                            agency_topics_conversations[agency] = {}

                        if topic not in agency_topics_conversations[agency]:
                            agency_topics_conversations[agency][topic] = []

                        # Extract conversation text
                        conversation = self._extract_conversation_from_json(item)
                        if conversation:
                            agency_topics_conversations[agency][topic].append(
                                conversation
                            )
                else:
                    # Assume first level keys are agencies
                    for agency, topics in content.items():
                        agency_topics_conversations[agency] = {}

                        if isinstance(topics, dict):
                            # Assume second level keys are topics
                            for topic, conversations in topics.items():
                                if isinstance(conversations, list):
                                    agency_topics_conversations[agency][topic] = [
                                        self._extract_conversation_from_json(conv)
                                        for conv in conversations
                                    ]
                                else:
                                    # Single conversation
                                    conversation = self._extract_conversation_from_json(
                                        conversations
                                    )
                                    if conversation:
                                        agency_topics_conversations[agency][topic] = [
                                            conversation
                                        ]
            elif isinstance(content, list):
                # If content is a list, might be a flat list of conversations
                default_agency = "default_agency"
                default_topic = "default_topic"

                agency_topics_conversations[default_agency] = {default_topic: []}

                for item in content:
                    conversation = self._extract_conversation_from_json(item)
                    if conversation:
                        agency_topics_conversations[default_agency][
                            default_topic
                        ].append(conversation)

        except Exception as e:
            logger.error(f"Error reading JSON conversations: {e}")
            return {}

        return agency_topics_conversations

    def _extract_conversation_from_json(self, item) -> str:
        """
        Extract conversation text from a JSON item.

        Args:
            item: JSON item containing conversation data

        Returns:
            Extracted conversation text
        """
        if isinstance(item, str):
            return item

        # Get the conversation format configuration
        json_format = self.file_structure["conversation_json_format"]
        question_key = self.file_structure["conversation_question_key"]

        # Handle simple array format: [{"question": "text"}, {"question": "text"}...]
        if json_format == "simple_array" and isinstance(item, list):
            questions = []
            for entry in item:
                if isinstance(entry, dict) and question_key in entry:
                    questions.append(entry[question_key])
                elif isinstance(entry, str):
                    questions.append(entry)

            if questions:
                # Format as a conversation with numbered questions
                formatted_questions = []
                for i, question in enumerate(questions, 1):
                    formatted_questions.append(f"Küsimus {i}: {question}")
                return "\n".join(formatted_questions)

        # Handle single object format: {"question": "text"}
        if (
            json_format == "simple_array"
            and isinstance(item, dict)
            and question_key in item
        ):
            return f"Küsimus: {item[question_key]}"

        # Handle message array format (common in chat applications)
        if (
            isinstance(item, dict)
            and "messages" in item
            and isinstance(item["messages"], list)
        ):
            messages = item["messages"]
            formatted_conversation = []

            for message in messages:
                if (
                    isinstance(message, dict)
                    and self.file_structure["conversation_speaker_key"] in message
                    and self.file_structure["conversation_content_key"] in message
                ):
                    role = message[self.file_structure["conversation_speaker_key"]]
                    content = message[self.file_structure["conversation_content_key"]]

                    # Map common roles to standard format
                    if role.lower() == "user":
                        formatted_role = "Kasutaja"
                    elif role.lower() in ["assistant", "bot", "ai"]:
                        formatted_role = "Robot"
                    else:
                        formatted_role = role

                    formatted_conversation.append(f"**{formatted_role}**: {content}")

            return "\n\n".join(formatted_conversation)

        # Try various common keys that might contain the conversation
        if isinstance(item, dict):
            # First try the configured question key
            if question_key in item:
                if isinstance(item[question_key], str):
                    return f"Küsimus: {item[question_key]}"
                elif isinstance(item[question_key], list):
                    # Handle list of questions
                    questions = []
                    for i, q in enumerate(item[question_key], 1):
                        if isinstance(q, str):
                            questions.append(f"Küsimus {i}: {q}")
                        elif isinstance(q, dict) and question_key in q:
                            questions.append(f"Küsimus {i}: {q[question_key]}")
                    return "\n".join(questions)

            # Fallback to other common keys
            for key in ["text", "conversation", "content", "message"]:
                if key in item:
                    if isinstance(item[key], str):
                        return item[key]
                    elif isinstance(item[key], list):
                        # Handle list of messages/turns
                        return self._format_conversation_turns(item[key])

            # If we have specific configuration for turns
            turns_key = self.file_structure.get("conversation_turns_key", "")
            if turns_key and turns_key in item:
                return self._format_conversation_turns(item[turns_key])

        elif isinstance(item, list):
            # Might be a list of questions or turns
            if json_format == "simple_array":
                # Handle list of question objects
                questions = []
                for i, entry in enumerate(item, 1):
                    if isinstance(entry, dict) and question_key in entry:
                        questions.append(f"Küsimus {i}: {entry[question_key]}")
                    elif isinstance(entry, str):
                        questions.append(f"Küsimus {i}: {entry}")

                if questions:
                    return "\n".join(questions)

            # Fallback to general turn formatting
            return self._format_conversation_turns(item)

        return ""

    def _format_conversation_turns(self, turns) -> str:
        """
        Format a list of conversation turns into a single string.

        Args:
            turns: List of conversation turns

        Returns:
            Formatted conversation text
        """
        if not turns:
            return ""

        # Try to detect the structure of turns
        if all(isinstance(t, str) for t in turns):
            # Simple list of strings
            return "\n".join(turns)

        if all(isinstance(t, dict) for t in turns):
            # List of dictionaries with turn information
            formatted_turns = []

            # Try to find common speaker/content keys
            speaker_keys = ["speaker", "role", "user", "agent", "sender"]
            content_keys = ["text", "content", "message", "utterance"]
            question_key = self.file_structure["conversation_question_key"]

            for turn in turns:
                speaker = None
                content = None

                # First check for the configured question key
                if question_key in turn:
                    content = turn[question_key]
                    speaker = "Küsimus"
                else:
                    # Try to find speaker
                    for key in speaker_keys:
                        if key in turn:
                            speaker = turn[key]
                            break

                    # Try to find content
                    for key in content_keys:
                        if key in turn:
                            content = turn[key]
                            break

                if speaker and content:
                    formatted_turns.append(f"{speaker}: {content}")
                elif content:
                    formatted_turns.append(content)

            return "\n".join(formatted_turns)

        return ""

    def _read_conversations_from_directory(self, directory: str) -> List[str]:
        """
        Read all conversation files from a directory.

        Args:
            directory: Path to the directory

        Returns:
            List of conversation texts
        """
        # First check for conversation text files (if not preferring JSON)
        conversations = []

        if not self.file_structure.get("prefer_json", False):
            conversation_files = glob.glob(
                os.path.join(directory, self.conversation_pattern)
            )

            for conv_file in conversation_files:
                content = read_file(conv_file)
                if content.strip():
                    conversations.append(content)

        # Check for JSON files (primary method when prefer_json is True)
        if (
            not conversations
            or self.file_structure.get("check_json_fallback", True)
            or self.file_structure.get("prefer_json", False)
        ):
            # Check for conversations.json file specifically
            conversations_json = os.path.join(directory, "conversations.json")
            if os.path.exists(conversations_json):
                try:
                    with open(conversations_json, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Treat the entire JSON file as one conversation
                    if isinstance(data, list):
                        # Extract conversation from the entire list
                        conversation = self._extract_conversation_from_json(data)
                        if conversation:
                            conversations.append(conversation)
                            logger.info(
                                f"Loaded conversation from {conversations_json} with {len(data)} questions"
                            )
                    elif isinstance(data, dict):
                        # Handle single conversation object
                        conversation = self._extract_conversation_from_json(data)
                        if conversation:
                            conversations.append(conversation)
                    else:
                        logger.warning(
                            f"Unexpected JSON structure in {conversations_json}"
                        )

                except Exception as e:
                    logger.warning(f"Error reading conversations.json: {e}")

            # Check other JSON files as fallback
            if not conversations:
                json_files = glob.glob(os.path.join(directory, "*.json"))

                for json_file in json_files:
                    if json_file == conversations_json:  # Skip if already processed
                        continue

                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Treat each JSON file as one conversation
                        if isinstance(data, list):
                            conversation = self._extract_conversation_from_json(data)
                            if conversation:
                                conversations.append(conversation)
                        elif isinstance(data, dict):
                            if "conversations" in data and isinstance(
                                data["conversations"], list
                            ):
                                # Handle wrapped conversation lists
                                conversation = self._extract_conversation_from_json(
                                    data["conversations"]
                                )
                                if conversation:
                                    conversations.append(conversation)
                            else:
                                # Handle single conversation object
                                conversation = self._extract_conversation_from_json(
                                    data
                                )
                                if conversation:
                                    conversations.append(conversation)

                    except Exception as e:
                        logger.warning(f"Error reading JSON file {json_file}: {e}")

        return conversations

    def gather_topic_documents(self) -> Dict[str, Dict[str, str]]:
        """
        Gather all topic documents based on the configured file structure.

        Returns:
            Dictionary mapping agency names to topics and their documents
        """
        agency_topic_documents = {}

        # Handle different file structure configurations
        if self.file_structure["topic_json_path"]:
            # JSON-based structure
            return self._gather_topic_documents_from_json()
        else:
            # Directory-based structure
            return self._gather_topic_documents_from_directories()

    def _gather_topic_documents_from_directories(self) -> Dict[str, Dict[str, str]]:
        """
        Gather topic documents from a directory-based structure.

        Returns:
            Dictionary mapping agency names to topics and their documents
        """
        agency_topic_documents = {}

        # Check if the topic directory exists
        if not os.path.exists(self.topic_dir):
            logger.error(f"Topic directory not found: {self.topic_dir}")
            return {}

        # Handle simplified directory structure: topics/agency_name/topic_*.txt
        if self.file_structure.get("simplified_topic_structure", False):
            agency_dirs = [
                d
                for d in os.listdir(self.topic_dir)
                if os.path.isdir(os.path.join(self.topic_dir, d))
            ]

            for agency in agency_dirs:
                agency_path = os.path.join(self.topic_dir, agency)
                agency_topic_documents[agency] = {}

                # Get all topic files for this agency
                topic_files = glob.glob(
                    os.path.join(agency_path, self.file_structure["topic_file_pattern"])
                )

                for topic_file in topic_files:
                    # Extract topic name from filename (removing .txt extension)
                    filename = os.path.basename(topic_file)
                    topic_name, ext = os.path.splitext(filename)

                    # Read the content
                    topic_content = read_file(topic_file)
                    if topic_content.strip():
                        agency_topic_documents[agency][topic_name] = topic_content
                        logger.info(f"Found topic document for {agency}/{topic_name}")

            return agency_topic_documents

        # Look for agency directories using patterns
        if self.file_structure["agency_subdir_pattern"]:
            agency_pattern = self.file_structure["agency_subdir_pattern"]
            if "output_" in agency_pattern:
                # Handle special case for output_* directories
                directories = [
                    d
                    for d in os.listdir(self.topic_dir)
                    if os.path.isdir(os.path.join(self.topic_dir, d))
                    and d.startswith("output_")
                ]

                for output_dir in directories:
                    agency_match = re.search(r"output_(.+)", output_dir)
                    if agency_match:
                        agency = agency_match.group(1)
                        agency_topic_documents[agency] = {}

                        # Get all topic documents for this agency
                        topic_files = glob.glob(
                            os.path.join(
                                self.topic_dir,
                                output_dir,
                                self.file_structure["topic_file_pattern"],
                            )
                        )

                        for topic_file in topic_files:
                            # Try to extract topic name from filename using patterns
                            topic_name = self._extract_topic_name_from_filename(
                                topic_file, agency
                            )

                            topic_content = read_file(topic_file)
                            if topic_content.strip():
                                agency_topic_documents[agency][topic_name] = (
                                    topic_content
                                )
                                logger.info(
                                    f"Found topic document for {agency}/{topic_name}"
                                )
            else:
                # Use the provided pattern directly
                agency_dir_pattern = os.path.join(self.topic_dir, agency_pattern)
                agency_dirs = glob.glob(agency_dir_pattern)

                for agency_path in agency_dirs:
                    agency_name = os.path.basename(agency_path)
                    agency_topic_documents[agency_name] = {}

                    # If topic pattern is specified, use it to identify topic files
                    topic_files = glob.glob(
                        os.path.join(
                            agency_path, self.file_structure["topic_file_pattern"]
                        )
                    )

                    for topic_file in topic_files:
                        topic_name = self._extract_topic_name_from_filename(
                            topic_file, agency_name
                        )
                        topic_content = read_file(topic_file)

                        if topic_content.strip():
                            agency_topic_documents[agency_name][topic_name] = (
                                topic_content
                            )
        else:
            # Look for output_* directories as a fallback for backward compatibility
            output_dirs = [
                d
                for d in os.listdir(self.topic_dir)
                if os.path.isdir(os.path.join(self.topic_dir, d))
                and d.startswith("output_")
            ]

            for output_dir in output_dirs:
                agency_match = re.search(r"output_(.+)", output_dir)
                if agency_match:
                    agency = agency_match.group(1)
                    agency_topic_documents[agency] = {}

                    # Get all topic documents for this agency
                    topic_files = glob.glob(
                        os.path.join(self.topic_dir, output_dir, "*.txt")
                    )

                    for topic_file in topic_files:
                        topic_name = self._extract_topic_name_from_filename(
                            topic_file, agency
                        )
                        topic_content = read_file(topic_file)

                        if topic_content.strip():
                            agency_topic_documents[agency][topic_name] = topic_content
                            logger.info(
                                f"Found topic document for {agency}/{topic_name}"
                            )

        return agency_topic_documents

    def _gather_topic_documents_from_json(self) -> Dict[str, Dict[str, str]]:
        """
        Gather topic documents from a JSON-based structure.

        Returns:
            Dictionary mapping agency names to topics and their documents
        """
        agency_topic_documents = {}

        # Load topic documents from JSON
        json_path = self.file_structure["topic_json_path"]
        if not os.path.exists(json_path):
            logger.error(f"Topic JSON file not found: {json_path}")
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract using the specified key path
            json_key = self.file_structure["topic_json_key"]
            if json_key:
                keys = json_key.split(".")
                content = data
                for key in keys:
                    if key in content:
                        content = content[key]
                    else:
                        logger.error(f"Key '{key}' not found in JSON structure")
                        return {}
            else:
                content = data

            # Process based on the content structure
            if isinstance(content, dict):
                # If content is a dict, keys might be agencies or topics
                if (
                    "agency_key" in self.file_structure
                    and self.file_structure["agency_key"]
                ):
                    # Structure has explicit agency keys
                    agency_key = self.file_structure["agency_key"]
                    topic_key = self.file_structure["topic_key"]
                    content_key = self.file_structure.get(
                        "topic_content_key", "content"
                    )

                    for item_id, item in content.items():
                        agency = item.get(agency_key, "default_agency")
                        topic = item.get(topic_key, "default_topic")
                        topic_content = item.get(content_key, "")

                        if agency not in agency_topic_documents:
                            agency_topic_documents[agency] = {}

                        if topic_content:
                            agency_topic_documents[agency][topic] = topic_content
                else:
                    # Assume first level keys are agencies
                    for agency, topics in content.items():
                        agency_topic_documents[agency] = {}

                        if isinstance(topics, dict):
                            # Assume second level keys are topics
                            for topic, topic_content in topics.items():
                                if isinstance(topic_content, str):
                                    agency_topic_documents[agency][topic] = (
                                        topic_content
                                    )
                                elif (
                                    isinstance(topic_content, dict)
                                    and "content" in topic_content
                                ):
                                    agency_topic_documents[agency][topic] = (
                                        topic_content["content"]
                                    )
            elif isinstance(content, list):
                # If content is a list, might be a flat list of topic documents
                default_agency = "default_agency"
                agency_topic_documents[default_agency] = {}

                for i, item in enumerate(content):
                    if isinstance(item, str):
                        topic_name = f"topic_{i + 1}"
                        agency_topic_documents[default_agency][topic_name] = item
                    elif isinstance(item, dict):
                        topic_name = item.get("topic", f"topic_{i + 1}")
                        topic_content = item.get("content", "")

                        if topic_content:
                            agency_topic_documents[default_agency][topic_name] = (
                                topic_content
                            )

        except Exception as e:
            logger.error(f"Error reading JSON topic documents: {e}")
            return {}

        return agency_topic_documents

    def _extract_topic_name_from_filename(self, filename: str, agency: str) -> str:
        """
        Extract topic name from a filename.

        Args:
            filename: Path to the topic file
            agency: Agency name for fallback patterns

        Returns:
            Extracted topic name
        """
        basename = os.path.basename(filename)

        # Try different patterns to extract topic name
        # Pattern 1: "TopicName_-_Agency.txt"
        match = re.match(r"(.+)_-_" + re.escape(agency) + r"\.txt", basename)
        if match:
            return match.group(1)

        # Pattern 2: "TopicName.txt"
        name, ext = os.path.splitext(basename)
        return name

    def evaluate_information_coverage(self, conversation: str, topic_doc: str) -> Dict:
        """
        Evaluate information coverage for a single conversation.

        Args:
            conversation: The conversation text
            topic_doc: The topic document text

        Returns:
            Dictionary with coverage score and details
        """
        topic_docs = [topic_doc]  # The function expects a list
        score, matched_chunks = compute_information_coverage(conversation, topic_docs)

        return {
            "information_coverage_score": float(score),
            "matched_chunks_count": len(matched_chunks),
        }

    def evaluate_length_appropriateness(
        self, conversation: str, topic_doc: str
    ) -> Dict:
        """
        Evaluate length appropriateness for a single conversation.

        Args:
            conversation: The conversation text
            topic_doc: The topic document text

        Returns:
            Dictionary with appropriateness score and details
        """
        topic_docs = [topic_doc]  # The function expects a list
        results = compute_length_appropriateness_score(conversation, topic_docs)

        return {
            "length_appropriateness_score": float(results["appropriateness_score"]),
            "actual_turns": results["actual_turns"],
            "recommended_turns": f"{results['recommended_min_turns']} - {results['recommended_max_turns']}",
            "topic_complexity": results["topic_complexity"]["complexity_category"],
        }

    def evaluate_relevance(self, conversation: str, topic_doc: str) -> Dict:
        """
        Evaluate relevance for a single conversation.

        Args:
            conversation: The conversation text
            topic_doc: The topic document text

        Returns:
            Dictionary with relevance score and details
        """
        topic_docs = [topic_doc]  # The function expects a list
        results = compute_weighted_segment_relevance(conversation, topic_docs)

        return {
            "relevance_score": float(results["relevance_score"]),
            "segment_score": float(results["segment_score"]),
            "query_score": float(results["query_score"]),
            "term_score": float(results["term_score"]),
        }

    def evaluate_topic_consistency(self, conversation: str, topic_doc: str) -> Dict:
        """
        Evaluate topic consistency for a single conversation.

        Args:
            conversation: The conversation text
            topic_doc: The topic document text

        Returns:
            Dictionary with consistency score and details
        """
        self.topic_consistency_evaluator.set_topic_documents([topic_doc])
        results = self.topic_consistency_evaluator.evaluate_conversation_topic_quality(
            conversation
        )

        return {
            "topic_consistency_score": float(results["topic_quality_score"]),
            "coherence_score": float(results["coherence_score"]),
            "alignment_score": float(results["alignment_score"]),
            "quality_assessment": results["quality_assessment"],
        }

    def evaluate_intra_redundancy(self, conversation: str) -> Dict:
        """
        Evaluate intra-conversation redundancy.

        Args:
            conversation: The conversation text

        Returns:
            Dictionary with redundancy score
        """
        score = compute_intra_conversation_redundancy(conversation)

        return {"intra_redundancy_score": float(score)}

    def evaluate_query_diversity(
        self, conversations: List[str], topic_name: str
    ) -> Dict:
        """
        Evaluate query diversity for a set of conversations.

        Args:
            conversations: List of conversation texts
            topic_name: Name of the topic

        Returns:
            Dictionary with diversity scores and details
        """
        score = compute_query_diversity_for_topic(conversations, topic_name)

        return {"query_diversity_score": float(score)}

    def evaluate_inter_redundancy(self, conversations: List[str]) -> Dict:
        """
        Evaluate inter-conversation redundancy.

        Args:
            conversations: List of conversation texts

        Returns:
            Dictionary with redundancy scores and details
        """
        results = compute_inter_conversation_redundancy(conversations)

        return {
            "inter_redundancy_score": float(results["redundancy_score"]),
            "redundant_pairs_count": len(results["redundant_pairs"]),
        }

    def evaluate_topic_coverage(self, conversations: List[str], topic_doc: str) -> Dict:
        """
        Evaluate topic coverage for a set of conversations.

        Args:
            conversations: List of conversation texts
            topic_doc: The topic document text

        Returns:
            Dictionary with coverage scores and details
        """
        results = self.topic_coverage_analyzer.analyze_topic_coverage(
            topic_doc, conversations
        )

        if "error" in results:
            return {"topic_coverage_score": 0.0, "error": results["error"]}

        coverage_percentage = float(results["coverage_percentage"]) / 100

        return {
            "topic_coverage_score": coverage_percentage,
            "total_topics": results["total_topics"],
            "covered_topics": results["covered_topics"],
            "uncovered_topics_count": len(results["uncovered_topics"]),
        }

    def evaluate_qualitative(self, conversation: str) -> Dict:
        """
        Perform qualitative evaluation of a conversation using LLM.

        Args:
            conversation: The conversation text

        Returns:
            Dictionary with qualitative scores and details
        """
        result = self.qualitative_evaluator.evaluate_conversation(
            conversation, "overall_quality"
        )

        score = result.get("score")
        if score is not None:
            normalized_score = score / 5.0  # Normalize to 0-1 range
        else:
            normalized_score = 0.0

        return {
            "qualitative_score": float(normalized_score),
            "raw_score": score,
            "reasoning": result.get("reasoning", "No reasoning provided"),
        }

    def evaluate_topic_agency_fit(
        self, conversation: str, topic_name: str, topic_doc: str, agency_name: str
    ) -> Dict:
        """
        Evaluate how well a conversation fits its intended topic and agency.

        Args:
            conversation: The conversation text
            topic_name: Name of the topic
            topic_doc: The topic document text
            agency_name: Name of the agency

        Returns:
            Dictionary with fit scores and details
        """
        agency_description = self.topic_agency_fit_evaluator.read_agency_description(
            agency_name
        )

        result = self.topic_agency_fit_evaluator.evaluate_conversation_topic_agency_fit(
            conversation, topic_name, topic_doc, agency_name, agency_description
        )

        score = result.get("aggregate_score")
        if score is not None:
            normalized_score = score / 5.0  # Normalize to 0-1 range
        else:
            normalized_score = 0.0

        return {
            "topic_agency_fit_score": float(normalized_score),
            "topic_fit_score": float(
                result.get("numerical_scores", {}).get("topic_fit", 0) / 5.0
            ),
            "agency_fit_score": float(
                result.get("numerical_scores", {}).get("agency_fit", 0) / 5.0
            ),
        }

    def evaluate_agency_confusion(
        self, agency_topics_conversations: Dict[str, Dict[str, List[str]]]
    ) -> Dict:
        """
        Analyze potential confusion between different agencies' conversations.

        Args:
            agency_topics_conversations: Dictionary mapping agencies to topics and conversations

        Returns:
            Dictionary with confusion analysis results
        """
        results = self.agency_confusion_analyzer.analyze_cross_agency_confusion(
            agency_topics_conversations
        )

        if "error" in results:
            return {"agency_confusion_score": 1.0, "error": results["error"]}

        confusion_rate = float(results["overall_confusion_rate"])
        confusion_score = 1.0 - confusion_rate  # Invert for scoring (higher is better)

        return {
            "agency_confusion_score": confusion_score,
            "overall_similarity": float(results["overall_similarity"]),
            "total_confusion_pairs": results["total_confusion_pairs"],
        }

    def evaluate_conversation(
        self, conversation: str, topic_doc: str, agency_name: str, topic_name: str
    ) -> Dict:
        """
        Run all applicable single-conversation evaluations.

        Args:
            conversation: The conversation text
            topic_doc: The topic document text
            agency_name: Name of the agency
            topic_name: Name of the topic

        Returns:
            Dictionary with all evaluation scores
        """
        results = {}

        # Run all enabled single-conversation evaluations
        if self.config.get("information_coverage.enabled"):
            results.update(self.evaluate_information_coverage(conversation, topic_doc))

        if self.config.get("length_appropriateness.enabled"):
            results.update(
                self.evaluate_length_appropriateness(conversation, topic_doc)
            )

        if self.config.get("relevance_score.enabled"):
            results.update(self.evaluate_relevance(conversation, topic_doc))

        if self.config.get("topic_consistency.enabled"):
            results.update(self.evaluate_topic_consistency(conversation, topic_doc))

        if self.config.get("redundancy_penalty.enabled"):
            results.update(self.evaluate_intra_redundancy(conversation))

        if self.config.get("qualitative_evaluation.enabled"):
            results.update(self.evaluate_qualitative(conversation))

        if self.config.get("topic_agency_fit.enabled", False):
            results.update(
                self.evaluate_topic_agency_fit(
                    conversation, topic_name, topic_doc, agency_name
                )
            )

        return results

    def evaluate_topic(
        self,
        conversations: List[str],
        topic_doc: str,
        agency_name: str,
        topic_name: str,
    ) -> Dict:
        """
        Run all applicable topic-level evaluations.

        Args:
            conversations: List of conversation texts
            topic_doc: The topic document text
            agency_name: Name of the agency
            topic_name: Name of the topic

        Returns:
            Dictionary with all evaluation scores
        """
        results = {}

        # Run all enabled topic-level evaluations
        if self.config.get("query_diversity.enabled"):
            results.update(self.evaluate_query_diversity(conversations, topic_name))

        if self.config.get("redundancy_penalty.enabled"):
            results.update(self.evaluate_inter_redundancy(conversations))

        if self.config.get("topic_coverage.enabled"):
            results.update(self.evaluate_topic_coverage(conversations, topic_doc))

        return results

    def run_evaluation(self) -> Dict:
        """
        Run the complete evaluation pipeline.

        Returns:
            Dictionary with all evaluation results
        """
        # Gather all data
        agency_topics_conversations = self.gather_agencies_topics_conversations()
        print(agency_topics_conversations.keys())
        agency_topic_documents = self.gather_topic_documents()

        if not agency_topics_conversations or not agency_topic_documents:
            logger.error("No data found for evaluation")
            return {"error": "No data found for evaluation"}

        # Evaluate agency confusion if enabled
        if (
            self.config.get("agency_confusion.enabled")
            and len(agency_topics_conversations) > 1
        ):
            agency_confusion_results = self.evaluate_agency_confusion(
                agency_topics_conversations
            )
            self.results["overall"]["agency_confusion"] = agency_confusion_results

        # Process each agency
        for agency_name, topics_conversations in agency_topics_conversations.items():
            self.results["by_agency"][agency_name] = {"by_topic": {}, "overall": {}}

            agency_conversation_results = []

            # Process each topic
            for topic_name, conversations in topics_conversations.items():
                if (
                    agency_name not in agency_topic_documents
                    or topic_name not in agency_topic_documents[agency_name]
                ):
                    logger.warning(
                        f"No topic document found for {agency_name}/{topic_name}"
                    )
                    continue

                topic_doc = agency_topic_documents[agency_name][topic_name]
                topic_results = {"by_conversation": {}, "overall": {}}

                # Evaluate each conversation
                conversation_results = []
                for i, conversation in enumerate(conversations):
                    conv_id = f"conversation_{i + 1}"
                    conv_results = self.evaluate_conversation(
                        conversation, topic_doc, agency_name, topic_name
                    )

                    topic_results["by_conversation"][conv_id] = conv_results
                    conversation_results.append(conv_results)
                    agency_conversation_results.append(conv_results)

                    # Add to global conversation results
                    self.results["by_conversation"][
                        f"{agency_name}/{topic_name}/{conv_id}"
                    ] = conv_results

                # Evaluate topic level metrics
                topic_level_results = self.evaluate_topic(
                    conversations, topic_doc, agency_name, topic_name
                )

                # Compute average of conversation-level metrics for the topic
                avg_results = self.average_results(conversation_results)

                # Combine topic-level and averaged conversation-level results
                topic_results["overall"] = {**avg_results, **topic_level_results}

                # Save topic results
                self.results["by_agency"][agency_name]["by_topic"][topic_name] = (
                    topic_results
                )
                self.results["by_topic"][f"{agency_name}/{topic_name}"] = topic_results[
                    "overall"
                ]

            # Compute average metrics for the agency
            self.results["by_agency"][agency_name]["overall"] = self.average_results(
                agency_conversation_results
            )

        # Compute overall averages across all agencies and topics
        all_topic_results = list(self.results["by_topic"].values())
        self.results["overall"]["average_scores"] = self.average_results(
            all_topic_results
        )

        # Generate summary report
        self.generate_summary_report()

        return self.results

    def average_results(self, results_list: List[Dict]) -> Dict:
        """
        Compute the average of numeric values across a list of result dictionaries.

        Args:
            results_list: List of result dictionaries

        Returns:
            Dictionary with averaged results
        """
        if not results_list:
            return {}

        averaged_results = {}
        score_counts = {}

        for results in results_list:
            for key, value in results.items():
                if isinstance(value, (int, float)) and key.endswith("_score"):
                    averaged_results[key] = averaged_results.get(key, 0.0) + value
                    score_counts[key] = score_counts.get(key, 0) + 1

        # Calculate averages
        for key in averaged_results:
            if score_counts[key] > 0:
                averaged_results[key] = averaged_results[key] / score_counts[key]

        return averaged_results

    def generate_summary_report(self) -> None:
        """
        Generate a summary report of all evaluation results.
        """
        # Create summary dataframe for topics
        topic_data = []
        for topic_id, results in self.results["by_topic"].items():
            row = {"topic": topic_id}
            for key, value in results.items():
                if isinstance(value, (int, float)) and key.endswith("_score"):
                    row[key] = value
            topic_data.append(row)

        if topic_data:
            topic_df = pd.DataFrame(topic_data)

            # Save topic summary
            topic_summary_file = os.path.join(self.output_dir, "topic_summary.csv")
            topic_df.to_csv(topic_summary_file, index=False)

            # Generate topic scores heatmap
            self.generate_score_heatmap(
                topic_df,
                "topic",
                os.path.join(self.output_dir, "topic_scores_heatmap.png"),
            )

        # Create summary dataframe for agencies
        agency_data = []
        for agency_name, agency_results in self.results["by_agency"].items():
            row = {"agency": agency_name}
            for key, value in agency_results["overall"].items():
                if isinstance(value, (int, float)) and key.endswith("_score"):
                    row[key] = value
            agency_data.append(row)

        if agency_data:
            agency_df = pd.DataFrame(agency_data)

            # Save agency summary
            agency_summary_file = os.path.join(self.output_dir, "agency_summary.csv")
            agency_df.to_csv(agency_summary_file, index=False)

            # Generate agency scores heatmap
            self.generate_score_heatmap(
                agency_df,
                "agency",
                os.path.join(self.output_dir, "agency_scores_heatmap.png"),
            )

        # Save complete results to JSON
        results_file = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

        logger.info(f"Evaluation results saved to {results_file}")

        # Generate overall scores summary
        overall_scores = self.results["overall"].get("average_scores", {})
        if overall_scores:
            overall_file = os.path.join(self.output_dir, "overall_scores.csv")
            overall_df = pd.DataFrame(
                [{"metric": k, "score": v} for k, v in overall_scores.items()]
            )
            overall_df.to_csv(overall_file, index=False)

            # Generate overall scores bar chart
            self.generate_overall_scores_chart(
                overall_scores, os.path.join(self.output_dir, "overall_scores.png")
            )

    def generate_score_heatmap(
        self, df: pd.DataFrame, id_column: str, output_file: str
    ) -> None:
        """
        Generate a heatmap visualization of scores.

        Args:
            df: DataFrame containing score data
            id_column: Column containing identifiers (e.g., 'topic', 'agency')
            output_file: Path to save the visualization
        """
        # Get score columns
        score_columns = [col for col in df.columns if col.endswith("_score")]

        if not score_columns:
            return

        # Prepare data for heatmap
        pivot_df = df.set_index(id_column)[score_columns]

        # Create a more readable column names
        column_labels = [
            col.replace("_score", "").replace("_", " ").title()
            for col in pivot_df.columns
        ]

        plt.figure(figsize=(12, max(8, len(pivot_df) * 0.4)))

        # Create heatmap
        ax = sns.heatmap(
            pivot_df,
            annot=True,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            xticklabels=column_labels,
            linewidths=0.5,
            fmt=".2f",
        )

        plt.title(f"{id_column.title()} Evaluation Scores")
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Score heatmap saved to {output_file}")

    def generate_overall_scores_chart(
        self, scores: Dict[str, float], output_file: str
    ) -> None:
        """
        Generate a bar chart of overall scores.

        Args:
            scores: Dictionary of score names and values
            output_file: Path to save the visualization
        """
        # Filter out non-score items
        score_items = [
            (k.replace("_score", "").replace("_", " ").title(), v)
            for k, v in scores.items()
            if k.endswith("_score")
        ]

        if not score_items:
            return

        # Sort by score value
        score_items.sort(key=lambda x: x[1], reverse=True)

        labels, values = zip(*score_items)

        plt.figure(figsize=(12, 8))

        bars = plt.barh(labels, values, color=plt.cm.YlGnBu(np.array(values) * 0.8))

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                max(width + 0.01, 0.05),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
            )

        plt.xlim(0, 1.05)
        plt.title("Overall Evaluation Scores")
        plt.xlabel("Score (0-1)")
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Overall scores chart saved to {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the conversation evaluation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="Path to the evaluation configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory (overrides config)",
    )

    args = parser.parse_args()

    # Initialize and run the pipeline
    pipeline = EvaluationPipeline(args.config)

    # Override output directory if specified
    if args.output:
        pipeline.output_dir = args.output
        os.makedirs(pipeline.output_dir, exist_ok=True)

    # Run the evaluation
    results = pipeline.run_evaluation()

    if "error" in results:
        logger.error(f"Evaluation failed: {results['error']}")
        sys.exit(1)

    logger.info(
        f"Evaluation completed successfully. Results saved to {pipeline.output_dir}"
    )

    # Print summary scores
    if "average_scores" in results["overall"]:
        logger.info("Overall scores:")
        for metric, score in results["overall"]["average_scores"].items():
            if metric.endswith("_score"):
                logger.info(f"  {metric}: {score:.4f}")
