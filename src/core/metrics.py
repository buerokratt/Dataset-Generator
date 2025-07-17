"""
metrics.py

Evaluation metrics for assessing the quality of generated text data.
Includes metrics for semantic diversity, keyword coverage, information coverage,
and relevance coverage. These are used to provide both single-sample and batch-level
feedback for data generation and prompt optimization workflows.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import re
from loguru import logger
import sys
import json
import ast
from typing import Union, Dict
from src.core.config import ConfigLoader

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class SemanticDiversityMetric:
    """Measures the semantic diversity across generated outputs using TF-IDF cosine similarity."""

    def __call__(self, outputs: List[str]) -> float:
        """Compute semantic diversity score for a list of outputs.

        Args:
            outputs (List[str]): List of generated text outputs.

        Returns:
            float: Diversity score between 0 and 1 (higher means more diverse).
        """
        if len(outputs) < 2:
            return 0.5
        try:
            vectorizer = TfidfVectorizer().fit_transform(outputs)
            similarity_matrix = cosine_similarity(vectorizer)
            upper_triangular = np.triu(similarity_matrix, k=1)
            avg_similarity = upper_triangular.sum() / (
                len(outputs) * (len(outputs) - 1) / 2
            )
            diversity_score = 1.0 - avg_similarity
            return round(float(diversity_score), 3)
        except Exception:
            return 0.0


class KeywordCoverageMetric:
    """Calculates how well required keywords are covered in generated outputs."""

    def __init__(self, required_keywords: List[str]):
        self.required_keywords = [kw.lower() for kw in required_keywords]

    def __call__(self, outputs: List[str]) -> float:
        """Evaluate the proportion of required keywords present in outputs.

        Args:
            outputs (List[str]): List of text outputs.

        Returns:
            float: Coverage ratio of keywords found.
        """
        matched = 0
        total = len(self.required_keywords) * len(outputs)
        if total == 0:
            return 1.0

        for output in outputs:
            lower_output = output.lower()
            matched += sum(1 for kw in self.required_keywords if kw in lower_output)

        return round(matched / total, 3)


class InformationCoverageMetric:
    """Measures how much of the topic content is semantically covered in the conversation."""

    def __init__(self, embedding_model: str):
        self.model_name = embedding_model
        self.model = SentenceTransformer(self.model_name)

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def split_to_chunks(self, text: str) -> List[str]:
        return [
            self.clean_text(s) for s in re.split(r"[.!?]", text) if len(s.strip()) > 10
        ]

    def extract_key_chunks(self, topics: List[str]) -> List[str]:
        chunks = []
        for topic in topics:
            chunks.extend(self.split_to_chunks(topic))
        return list(set(chunks))

    def __call__(
        self, conversation: str, topic_docs: List[str], threshold: float = 0.5
    ) -> Tuple[float, List[str]]:
        """Compute information coverage score and matched topic chunks.

        Args:
            conversation (str): Generated conversation string.
            topic_docs (List[str]): List of reference topic documents.
            threshold (float): Similarity threshold.

        Returns:
            Tuple[float, List[str]]: Coverage score and list of matched topic chunks.
        """
        logger.info("Computing information coverage...")
        key_chunks = self.extract_key_chunks(topic_docs)
        conv_chunks = self.split_to_chunks(conversation)

        if not key_chunks or not conv_chunks:
            return 0.0, []

        topic_embeddings = self.model.encode(key_chunks, convert_to_tensor=True)
        conv_embeddings = self.model.encode(conv_chunks, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(topic_embeddings, conv_embeddings)
        max_similarities = scores.max(dim=1).values

        matched = max_similarities >= threshold
        score = matched.sum().item() / len(key_chunks)
        logger.info(f"Information coverage score: {score:.2f}")
        matched_chunks = [key_chunks[i] for i, m in enumerate(matched) if m]

        return score, matched_chunks


class RelevanceCoverageMetric:
    """Computes relevance score between conversation and topic documents based on segment, query, and term alignment."""

    def __init__(self, embedding_model: str):
        self.config = ConfigLoader.load()
        self.model = SentenceTransformer(embedding_model)
        self.segment_weight = self.config.get("relevance_score", {}).get(
            "segment_weight"
        )
        self.query_weight = self.config.get("relevance_score", {}).get("query_weight")
        self.term_weight = self.config.get("relevance_score", {}).get("term_weight")

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def split_into_segments(self, text: str, max_length: int = 200) -> List[str]:
        try:
            sentences = sent_tokenize(text)
            segments, current = [], ""
            for sentence in sentences:
                if len(current) + len(sentence) <= max_length:
                    current += " " + sentence if current else sentence
                else:
                    if current:
                        segments.append(self.clean_text(current))
                    current = sentence
            if current:
                segments.append(self.clean_text(current))
            return segments
        except Exception:
            return [
                self.clean_text(s)
                for s in re.split(r"[.!?]", text)
                if len(s.strip()) > 10
            ]

    def extract_user_queries(self, batch: Union[List[Dict], str]) -> List[str]:
        """
        Extracts question-like sentences from a batch of dicts or raw string input.
        This version is key-agnostic and works with any schema.
        """
        queries = []

        try:
            # If input is a list of dicts, flatten all string values
            if isinstance(batch, list) and all(
                isinstance(item, dict) for item in batch
            ):
                text_values = []
                for item in batch:
                    for value in item.values():
                        if isinstance(value, str):
                            text_values.append(value)
                text_blob = " ".join(text_values)
            else:
                # Fallback: treat as raw text
                text_blob = batch if isinstance(batch, str) else str(batch)

            # Tokenize into sentences and filter questions
            sentences = sent_tokenize(text_blob)
            queries = [s.strip() for s in sentences if s.strip().endswith("?")]

        except Exception:
            return []

        return queries

    def extract_key_terms(self, text: str, n: int = 10) -> List[str]:
        if not text.strip():
            return []
        try:
            vectorizer = TfidfVectorizer(
                min_df=self.config.get("relevance_score.min_df"),
                max_df=self.config.get("relevance_score.max_df"),
                ngram_range=ast.literal_eval(
                    self.config.get("relevance_score.ngram_range")
                ),
            )
            tfidf_matrix = vectorizer.fit_transform([text, "dummy text"])
            feature_names = vectorizer.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix[0].toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            return [term for term, _ in sorted_scores[:n]]
        except Exception:
            words = re.findall(r"\b\w{3,}\b", text)
            return list(set(words))[:n]

    def __call__(
        self, conversation: str, topic_docs: List[str], threshold: float = 0.0
    ) -> Tuple[float, dict]:
        """Compute relevance score and diagnostic breakdown.

        Args:
            conversation (str): Generated conversation.
            topic_docs (List[str]): Reference documents.
            threshold (float): Similarity threshold (unused).

        Returns:
            Tuple[float, dict]: Composite score and detail dictionary.
        """
        conversation = self.clean_text(conversation)
        topic_combined = self.clean_text(" ".join(topic_docs))

        if not conversation or not topic_combined:
            return 0.0, {}

        # Segment Relevance
        topic_segments = self.split_into_segments(topic_combined)
        conv_segments = self.split_into_segments(conversation)

        segment_score = 0.0
        if topic_segments and conv_segments:
            topic_embeddings = self.model.encode(topic_segments, convert_to_tensor=True)
            conv_embeddings = self.model.encode(conv_segments, convert_to_tensor=True)
            similarity_matrix = util.pytorch_cos_sim(conv_embeddings, topic_embeddings)
            max_similarities = torch.max(similarity_matrix, dim=1).values
            segment_score = torch.mean(max_similarities).item()

        # Query Relevance
        queries = self.extract_user_queries(conversation)
        query_score = 0.0
        if queries:
            query_embeddings = self.model.encode(queries, convert_to_tensor=True)
            topic_embedding = self.model.encode(topic_combined, convert_to_tensor=True)
            query_similarities = [
                util.pytorch_cos_sim(q_emb, topic_embedding).item()
                for q_emb in query_embeddings
            ]
            if len(query_similarities) > 1:
                weights = [1.0] + [0.7] * (len(query_similarities) - 1)
                weighted_sims = [s * w for s, w in zip(query_similarities, weights)]
                query_score = sum(weighted_sims) / sum(weights)
            else:
                query_score = query_similarities[0] if query_similarities else 0.0

        # Term Overlap
        topic_terms = self.extract_key_terms(topic_combined)
        conv_terms = self.extract_key_terms(conversation)
        if topic_terms and conv_terms:
            topic_set, conv_set = set(topic_terms), set(conv_terms)
            intersection = len(topic_set.intersection(conv_set))
            union = len(topic_set.union(conv_set))
            term_score = intersection / union if union > 0 else 0.0
        else:
            term_score = 0.0

        final_score = (
            self.segment_weight * segment_score
            + self.query_weight * query_score
            + self.term_weight * term_score
        )
        final_score = max(0.0, min(final_score, 1.0))

        return final_score, {
            "segment_score": segment_score,
            "query_score": query_score,
            "term_score": term_score,
            "segments": conv_segments,
            "queries": queries,
            "terms_overlap": list(set(topic_terms).intersection(set(conv_terms))),
        }


class PerSampleQualityEvaluator:
    """Evaluates a single sample using semantic similarity to the original context."""

    def __init__(self, embedding_model: str):
        self.semantic_metric = InformationCoverageMetric(embedding_model)

    def __call__(self, output_obj: dict, context_str: str) -> float:
        """Computes quality score for one output sample given context.

        Args:
            output_obj (dict): Generated structured output.
            context_str (str): Prompt input or user-provided context.

        Returns:
            float: Semantic similarity-based quality score.
        """
        if not isinstance(output_obj, dict):
            return 0.0

        output_text = " ".join(str(v) for v in output_obj.values())

        # Convert context_str to a flattened string representation
        try:
            if isinstance(context_str, dict):
                context_text = " ".join(f"{k}: {v}" for k, v in context_str.items())
            else:
                context_dict = json.loads(context_str)
                context_text = " ".join(f"{k}: {v}" for k, v in context_dict.items())
        except Exception:
            context_text = ""

        info_score, _ = self.semantic_metric(output_text, [context_text])
        sample_query_score = info_score
        return round(sample_query_score, 3)
