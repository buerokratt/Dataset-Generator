from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from loguru import logger
import sys

# remove the default stderr handler
logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

from utils import Config


def clean_text(text: str) -> str:
    """
    Clean text by removing extra spaces and newlines.
    """
    return re.sub(r"\s+", " ", text.strip())


def split_to_chunks(text: str) -> List[str]:
    """
    Split text into sentences or chunks for comparison.
    """
    return [clean_text(s) for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]


def extract_key_chunks(topics: List[str]) -> List[str]:
    """
    Extract chunks from all topic documents.
    """
    chunks = []
    for topic in topics:
        chunks.extend(split_to_chunks(topic))
    return list(set(chunks))


def compute_information_coverage(
    conversation: str, topic_docs: List[str], threshold: float = 0.5
) -> Tuple[float, List[str]]:
    """
    Compute coverage of topic-doc content in the Estonian conversation.

    Returns:
        score: float between 0 and 1
        matched_chunks: List of matched chunks
    """
    logger.info("Computing information coverage...")
    key_chunks = extract_key_chunks(topic_docs)
    conv_chunks = split_to_chunks(conversation)

    if not key_chunks or not conv_chunks:
        return 0.0, []
    config = Config()
    model = SentenceTransformer(config.get("models.embedding_model"))
    topic_embeddings = model.encode(key_chunks, convert_to_tensor=True)
    conv_embeddings = model.encode(conv_chunks, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(topic_embeddings, conv_embeddings)
    max_similarities = scores.max(dim=1).values

    matched = max_similarities >= threshold
    score = matched.sum().item() / len(key_chunks)
    logger.info(f"Information coverage score: {score:.2f}")
    matched_chunks = [key_chunks[i] for i, m in enumerate(matched) if m]

    return score, matched_chunks
