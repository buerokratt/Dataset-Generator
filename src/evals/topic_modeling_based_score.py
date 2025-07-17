from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import warnings
import os
import glob
import json
from pathlib import Path
from utils import read_file
from utils import Config
from utils import load_stopwords
from utils import preprocess_text
from constants import USER_QUERY_PATTERNS
from loguru import logger
import sys

logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class TopicConsistencyEvaluator:
    """
    A class for evaluating how well generated Estonian conversations align with their source topic documents.
    Uses direct comparison between conversations and topic documents instead of requiring reference conversations.
    """

    def __init__(self, topic_documents: List[str] = None):
        """
        Initialize the topic consistency evaluator.

        Args:
            topic_documents: List of topic documents that were used to generate conversations
            embedding_model: Name of the sentence-transformer model to use
        """
        self.config = Config()
        self.embedding_model_name = self.config.get("topic_consistency.embedding_model")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.topic_documents = topic_documents or []
        self.topic_keywords = {}
        self.topic_embeddings = {}
        self.stopwords = load_stopwords()

        if self.topic_documents:
            self._process_topic_documents()

    def _extract_segments(self, text: str, max_length: int = 200) -> List[str]:
        """Split text into meaningful segments for analysis."""
        sentences = re.split(r"[.!?]+", text)
        segments = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) <= max_length:
                current += " " + sentence if current else sentence
            else:
                if current:
                    segments.append(preprocess_text(current))
                current = sentence

        if current:
            segments.append(preprocess_text(current))

        if not segments:
            text = preprocess_text(text)
            words = text.split()
            for i in range(0, len(words), max_length // 5):  # Approx 5 chars per word
                segment = " ".join(words[i : i + max_length // 5])
                if segment:
                    segments.append(segment)

        return segments

    def _extract_keywords(self, text: str, n: int = 20) -> List[Tuple[str, float]]:
        """Extract important keywords from text using TF-IDF."""
        preprocessed = preprocess_text(text)

        try:
            vectorizer = TfidfVectorizer(
                stop_words=self.stopwords, ngram_range=(1, 2), min_df=1, max_df=0.9
            )

            tfidf_matrix = vectorizer.fit_transform([preprocessed, "dummy text"])
            feature_names = vectorizer.get_feature_names_out()

            scores = zip(feature_names, tfidf_matrix[0].toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            return sorted_scores[:n]

        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            words = preprocessed.split()
            word_counts = Counter(words)
            return [
                (w, c / max(1, sum(word_counts.values())))
                for w, c in word_counts.most_common(n)
            ]

    def _process_topic_documents(self):
        """Process topic documents to extract embeddings and keywords."""
        if not self.topic_documents:
            logger.warning("No topic documents provided for processing.")
            return

        for i, doc in enumerate(self.topic_documents):
            topic_id = i  # Use index as topic ID

            preprocessed = preprocess_text(doc)

            if not preprocessed:
                continue

            keywords = self._extract_keywords(doc)
            self.topic_keywords[topic_id] = keywords

            embedding = self.embedding_model.encode(preprocessed)
            self.topic_embeddings[topic_id] = embedding

            segments = self._extract_segments(doc)
            if segments:
                segment_embeddings = self.embedding_model.encode(segments)
                self.topic_embeddings[f"{topic_id}_segments"] = segment_embeddings

    def set_topic_documents(self, topic_documents: List[str]):
        """
        Set or update the topic documents.

        Args:
            topic_documents: List of topic documents
        """
        self.topic_documents = topic_documents or []
        self.topic_keywords = {}
        self.topic_embeddings = {}

        if self.topic_documents:
            self._process_topic_documents()

    def get_conversation_segments(self, conversation: str) -> Dict:
        """
        Extract different types of segments from a conversation for analysis.

        Args:
            conversation: The conversation text

        Returns:
            Dictionary with various segment types
        """
        preprocessed = preprocess_text(conversation)

        user_queries = []
        for pattern in USER_QUERY_PATTERNS:
            matches = re.findall(pattern, conversation, re.DOTALL | re.IGNORECASE)
            user_queries.extend([preprocess_text(m) for m in matches if m.strip()])

        turns = []
        if re.search(r"\*\*(?:Kasutaja|Robot)\*\*:", conversation, re.IGNORECASE):
            turn_matches = re.findall(
                r"\*\*(?:Kasutaja|Robot)\*\*:(.*?)(?=\*\*(?:Kasutaja|Robot)\*\*:|$)",
                conversation,
                re.DOTALL | re.IGNORECASE,
            )
            turns = [preprocess_text(t) for t in turn_matches if t.strip()]
        elif re.search(r"(?:Kasutaja|Robot):", conversation, re.IGNORECASE):
            turn_matches = re.findall(
                r"(?:Kasutaja|Robot):(.*?)(?=(?:Kasutaja|Robot):|$)",
                conversation,
                re.DOTALL | re.IGNORECASE,
            )
            turns = [preprocess_text(t) for t in turn_matches if t.strip()]
        else:
            segments = self._extract_segments(conversation)
            turns = segments

        if not turns and not user_queries:
            segments = self._extract_segments(conversation)
            if segments:
                turns = segments
            else:
                turns = [preprocessed] if preprocessed else []

        return {
            "full_text": preprocessed,
            "user_queries": user_queries,
            "turns": turns,
            "word_count": len(preprocessed.split()) if preprocessed else 0,
        }

    def compute_topic_coherence(self, conversation: str) -> Dict:
        """
        Compute internal coherence of a conversation (how well it maintains a consistent topic).

        Args:
            conversation: The conversation text

        Returns:
            Dictionary with coherence metrics
        """
        segments = self.get_conversation_segments(conversation)
        turns = segments["turns"]

        if len(turns) < 2:
            return {
                "coherence_score": 0.0,
                "reason": "Not enough turns to compute coherence",
            }

        try:
            turn_embeddings = self.embedding_model.encode(turns)

            similarities = []
            for i in range(len(turns) - 1):
                sim = cosine_similarity([turn_embeddings[i]], [turn_embeddings[i + 1]])[
                    0
                ][0]
                similarities.append(sim)

            avg_similarity = sum(similarities) / len(similarities)
            min_similarity = min(similarities)
            variance = np.var(similarities)

            coherence_score = (
                0.5 * avg_similarity + 0.3 * min_similarity + 0.2 * (1 - variance)
            )

            return {
                "coherence_score": float(coherence_score),
                "average_turn_similarity": float(avg_similarity),
                "minimum_turn_similarity": float(min_similarity),
                "similarity_variance": float(variance),
                "number_of_turns": len(turns),
            }

        except Exception as e:
            logger.error(f"Error computing topic coherence: {e}")
            return {"error": str(e), "coherence_score": 0.0}

    def compute_topic_alignment(self, conversation: str, topic_id: int = 0) -> Dict:
        """
        Compute how well a conversation aligns with a specific topic document.

        Args:
            conversation: The conversation text
            topic_id: The ID of the topic document to compare with

        Returns:
            Dictionary with alignment metrics
        """
        if not self.topic_embeddings or topic_id not in self.topic_embeddings:
            return {"alignment_score": 0.0, "error": "Topic document not available"}

        segments = self.get_conversation_segments(conversation)
        full_text = segments["full_text"]
        user_queries = segments["user_queries"]
        turns = segments["turns"]

        if not full_text:
            return {"alignment_score": 0.0, "reason": "Empty conversation"}

        try:
            topic_embedding = self.topic_embeddings[topic_id]

            scores = {}

            full_text_embedding = self.embedding_model.encode(full_text)
            full_text_sim = cosine_similarity([full_text_embedding], [topic_embedding])[
                0
            ][0]
            scores["full_text_similarity"] = float(full_text_sim)

            if user_queries:
                query_embeddings = self.embedding_model.encode(user_queries)
                query_sims = [
                    cosine_similarity([q_emb], [topic_embedding])[0][0]
                    for q_emb in query_embeddings
                ]
                weights = [1.0] + [0.7] * (len(query_sims) - 1)
                weighted_sims = [s * w for s, w in zip(query_sims, weights)]
                query_sim = sum(weighted_sims) / sum(weights)
                scores["query_similarity"] = float(query_sim)
            else:
                scores["query_similarity"] = scores["full_text_similarity"]

            if f"{topic_id}_segments" in self.topic_embeddings and turns:
                turn_embeddings = self.embedding_model.encode(turns)
                topic_segment_embeddings = self.topic_embeddings[f"{topic_id}_segments"]

                turn_max_sims = []
                for turn_emb in turn_embeddings:
                    sims = [
                        cosine_similarity([turn_emb], [seg_emb])[0][0]
                        for seg_emb in topic_segment_embeddings
                    ]
                    turn_max_sims.append(max(sims))

                segment_sim = sum(turn_max_sims) / len(turn_max_sims)
                scores["segment_similarity"] = float(segment_sim)
            else:
                scores["segment_similarity"] = scores["full_text_similarity"]

            if topic_id in self.topic_keywords:
                topic_words = [word for word, _ in self.topic_keywords[topic_id]]
                conversation_words = full_text.split()

                matching_words = [
                    word for word in conversation_words if word in topic_words
                ]
                matching_ratio = (
                    len(matching_words) / len(conversation_words)
                    if conversation_words
                    else 0
                )
                scores["keyword_overlap"] = float(matching_ratio)

                matched_keywords = []
                for word, score in self.topic_keywords[topic_id]:
                    if word in conversation_words:
                        matched_keywords.append((word, score))
            else:
                scores["keyword_overlap"] = 0.0
                matched_keywords = []

            alignment_score = (
                0.3 * scores["full_text_similarity"]
                + 0.3 * scores["query_similarity"]
                + 0.3 * scores["segment_similarity"]
                + 0.1 * scores["keyword_overlap"]
            )

            return {
                "alignment_score": float(alignment_score),
                "component_scores": scores,
                "matching_keywords": matched_keywords[:10],  # Top 10 matched keywords
                "matching_keyword_count": len(matched_keywords),
            }

        except Exception as e:
            logger.error(f"Error computing topic alignment: {e}")
            return {"error": str(e), "alignment_score": 0.0}

    def evaluate_conversation_topic_quality(
        self, conversation: str, topic_id: int = 0
    ) -> Dict:
        """
        Perform  topic quality evaluation for a conversation.

        """
        coherence = self.compute_topic_coherence(conversation)

        alignment = self.compute_topic_alignment(conversation, topic_id)

        segments = self.get_conversation_segments(conversation)

        if "error" in coherence or "error" in alignment:
            errors = []
            if "error" in coherence:
                errors.append(f"Coherence error: {coherence['error']}")
            if "error" in alignment:
                errors.append(f"Alignment error: {alignment['error']}")

            return {"topic_quality_score": 0.0, "errors": errors}

        topic_quality_score = (
            0.5 * coherence["coherence_score"] + 0.5 * alignment["alignment_score"]
        )

        result = {
            "topic_quality_score": float(topic_quality_score),
            "coherence_score": coherence["coherence_score"],
            "alignment_score": alignment["alignment_score"],
            "turn_count": coherence["number_of_turns"],
            "word_count": segments["word_count"],
            "component_scores": alignment.get("component_scores", {}),
            "matching_keywords": alignment.get("matching_keywords", []),
            "matching_keyword_count": alignment.get("matching_keyword_count", 0),
        }

        if topic_quality_score >= 0.8:
            result["quality_assessment"] = "excellent"
        elif topic_quality_score >= 0.6:
            result["quality_assessment"] = "good"
        elif topic_quality_score >= 0.4:
            result["quality_assessment"] = "acceptable"
        elif topic_quality_score >= 0.2:
            result["quality_assessment"] = "poor"
        else:
            result["quality_assessment"] = "inadequate"

        return result

    def evaluate_conversation_set(
        self, conversations: List[str], topic_id: int = 0
    ) -> Dict:
        """
        Evaluate topic quality for a set of conversations.


        """
        results = []

        for i, conv in enumerate(conversations):
            result = self.evaluate_conversation_topic_quality(conv, topic_id)
            result["conversation_index"] = i
            results.append(result)

        quality_scores = [
            r["topic_quality_score"] for r in results if "topic_quality_score" in r
        ]
        coherence_scores = [
            r["coherence_score"] for r in results if "coherence_score" in r
        ]
        alignment_scores = [
            r["alignment_score"] for r in results if "alignment_score" in r
        ]

        quality_counts = Counter(
            [r.get("quality_assessment", "unknown") for r in results]
        )

        return {
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "average_coherence": np.mean(coherence_scores) if coherence_scores else 0.0,
            "average_alignment": np.mean(alignment_scores) if alignment_scores else 0.0,
            "quality_distribution": dict(quality_counts),
            "excellent_percent": quality_counts.get("excellent", 0) / len(results) * 100
            if results
            else 0,
            "good_or_better_percent": (
                quality_counts.get("excellent", 0) + quality_counts.get("good", 0)
            )
            / len(results)
            * 100
            if results
            else 0,
            "evaluated_count": len(results),
            "detailed_results": results,
        }


def read_topic_documents(topic_docs_path: str) -> List[str]:
    """Read topic documents from files."""
    if os.path.isfile(topic_docs_path):
        return [read_file(topic_docs_path)]
    elif os.path.isdir(topic_docs_path):
        files = glob.glob(os.path.join(topic_docs_path, "*.txt"))
        return [read_file(f) for f in files]
    else:
        if topic_docs_path.startswith("[") and topic_docs_path.endswith("]"):
            try:
                file_list = json.loads(topic_docs_path)
                return [read_file(f) for f in file_list if isinstance(f, str)]
            except:
                pass
        files = re.split(r",|\n", topic_docs_path)
        return [read_file(f.strip()) for f in files if f.strip()]


def read_conversations_from_directory(
    directory: str, pattern: str = "conversation_*.txt"
) -> List[str]:
    """Read all conversation files from a directory matching a pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    return [read_file(f) for f in files]


def compute_topic_quality_score(conversation: str, topic_document: str) -> float:
    """
    Simple interface for computing topic quality score.

    Args:
        conversation: The conversation to evaluate
        topic_document: The topic document used to generate the conversation

    Returns:
        Float score between 0 and 1
    """
    evaluator = TopicConsistencyEvaluator([topic_document])
    result = evaluator.evaluate_conversation_topic_quality(conversation)

    if "topic_quality_score" in result:
        return result["topic_quality_score"]
    else:
        return 0.0


def evaluate_topic_quality_from_files(
    conversation_path: str, topic_docs_path: str, output_file: str = None
) -> Dict:
    """
    Evaluate topic quality for conversations from files.

    Args:
        conversation_path: Path to conversation file or directory
        topic_docs_path: Path to topic documents file or directory
        output_file: Optional path to write results to

    Returns:
        Dictionary with evaluation results
    """
    topic_docs = read_topic_documents(topic_docs_path)

    if not topic_docs:
        logger.error(f"No topic documents found at {topic_docs_path}")
        return {"error": "No topic documents found"}

    evaluator = TopicConsistencyEvaluator(topic_docs)

    conversations = []
    filenames = []

    if os.path.isdir(conversation_path):
        files = glob.glob(os.path.join(conversation_path, "conversation_*.txt"))
        for file_path in sorted(files):
            filename = os.path.basename(file_path)
            content = read_file(file_path)
            if content.strip():
                filenames.append(filename)
                conversations.append(content)
    else:
        filename = os.path.basename(conversation_path)
        content = read_file(conversation_path)
        if content.strip():
            filenames.append(filename)
            conversations.append(content)

    if not conversations:
        logger.error(f"No conversations found at {conversation_path}")
        return {"error": "No conversations found"}

    results = evaluator.evaluate_conversation_set(conversations)

    for i, detailed in enumerate(results["detailed_results"]):
        if i < len(filenames):
            detailed["filename"] = filenames[i]

    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # write to json file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            logger.info(f"Results written to {output_file}")
        except Exception as e:
            logger.error(f"Error writing results to {output_file}: {e}")

    return results
