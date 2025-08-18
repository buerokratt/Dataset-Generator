import json
import hashlib
import logging
import os
import glob
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict
import redis
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error(
        "SentenceTransformers not available. Please install: pip install sentence-transformers"
    )


@dataclass
class EmbeddingRecord:
    """Data class for embedding records."""

    id: int
    text_content: str
    embedding_vector: List[float]  # Always required now
    agency: str
    topic: str
    model_name: str
    created_at: str
    content_hash: str

    def to_dict(self):
        return {
            "id": self.id,
            "text_content": self.text_content,
            "embedding_vector": self.embedding_vector,
            "agency": self.agency,
            "topic": self.topic,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class UnifiedEmbeddingManager:
    """
    Unified embedding management system with Redis caching.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        redis_url: str = "redis://localhost:6379",
        redis_db: int = 0,
        topic_documents_path: str = "topic_documents",
    ):
        """
        Initialize unified embedding manager.

        Args:
            model_name: SentenceTransformer model name
            redis_url: Redis connection URL
            redis_db: Redis database number
            topic_documents_path: Path to local topic documents (legacy, now optional)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers required but not available")

        self.model_name = model_name
        self.topic_documents_path = topic_documents_path

        # Initialize SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")

        # Initialize Redis client
        self.redis_client = redis.from_url(
            redis_url, db=redis_db, decode_responses=False
        )

        # Test Redis connection
        if not self.health_check():
            raise ConnectionError(f"Cannot connect to Redis at {redis_url}")

        logger.info(f"Initialized UnifiedEmbeddingManager with Redis: {redis_url}")

    # === REDIS CONNECTION METHODS ===

    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    # === KEY GENERATION ===

    def _get_persistent_key(self, agency: str, topic: str) -> str:
        """Generate Redis key for persistent topic embeddings."""
        return f"embeddings:persistent:{agency}:{topic}"

    def _get_temporary_key(self, session_id: str) -> str:
        """Generate Redis key for temporary question embeddings."""
        return f"embeddings:temp:{session_id}"

    # === SERIALIZATION ===

    def _serialize_embeddings(self, embeddings: List[EmbeddingRecord]) -> bytes:
        """Serialize list of embedding records."""
        data = [emb.to_dict() for emb in embeddings]
        return json.dumps(data).encode("utf-8")

    def _deserialize_embeddings(self, data: bytes) -> List[EmbeddingRecord]:
        """Deserialize list of embedding records."""
        parsed_data = json.loads(data.decode("utf-8"))
        return [EmbeddingRecord.from_dict(item) for item in parsed_data]

    def _serialize_question_embeddings(
        self, embeddings: Dict[str, np.ndarray]
    ) -> bytes:
        """Serialize question embeddings dict."""
        serializable = {key: emb.tolist() for key, emb in embeddings.items()}
        return json.dumps(serializable).encode("utf-8")

    def _deserialize_question_embeddings(self, data: bytes) -> Dict[str, np.ndarray]:
        """Deserialize question embeddings dict."""
        parsed_data = json.loads(data.decode("utf-8"))
        return {key: np.array(emb) for key, emb in parsed_data.items()}

    # === EXTRACT TOPIC CONTENT FROM RESULTS ===

    def _extract_topic_content_from_results(self, results: List[Dict]) -> List[Dict]:
        """
        Extract topic content from actual result files (metadata.json).

        Args:
            results: List of result dictionaries with 'source' and 'output_path' keys

        Returns:
            List of documents with content and metadata
        """
        documents = []
        doc_id = 1

        for result in results:
            try:
                output_path = result.get("output_path", "")
                source_path = result.get("source", "")

                if not output_path or not os.path.exists(output_path):
                    logger.warning(f"Output path doesn't exist: {output_path}")
                    continue

                # Check if output_path is a file or directory
                if os.path.isfile(output_path):
                    # If it's a file, get the directory
                    output_dir = os.path.dirname(output_path)
                else:
                    # If it's already a directory, use it directly
                    output_dir = output_path

                metadata_path = os.path.join(output_dir, "metadata.json")

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Extract content from metadata
                    file_content = metadata.get("parameters", {}).get(
                        "file_content", ""
                    )
                    if not file_content:
                        logger.warning(f"No file_content in metadata: {metadata_path}")
                        continue

                    # Extract agency and topic from source path
                    # Source: /app/data/agencies/sm_someuuid/d934abece3ce5ea3ceaa55e41f3cfe0eb7ea6f97/cleaned.txt
                    source_parts = source_path.strip("/").split("/")
                    if len(source_parts) >= 4:
                        agency = source_parts[-3]  # sm_someuuid
                        topic = source_parts[
                            -2
                        ]  # d934abece3ce5ea3ceaa55e41f3cfe0eb7ea6f97
                    else:
                        # Fallback to extracting from output path
                        output_parts = output_path.strip("/").split("/")
                        if len(output_parts) >= 3:
                            agency = (
                                output_parts[-3]
                                if len(output_parts) >= 3
                                else "unknown"
                            )
                            topic = (
                                output_parts[-2]
                                if len(output_parts) >= 2
                                else "unknown"
                            )
                        else:
                            agency = "unknown"
                            topic = f"topic_{doc_id}"

                    content_hash = hashlib.sha256(
                        file_content.encode("utf-8")
                    ).hexdigest()

                    documents.append(
                        {
                            "id": doc_id,
                            "text_content": file_content.strip(),
                            "agency": agency,
                            "topic": topic,
                            "content_hash": content_hash,
                            "metadata_path": metadata_path,
                            "source_path": source_path,
                            "output_path": output_path,
                        }
                    )
                    doc_id += 1
                    logger.info(
                        f"Extracted content for {agency}-{topic} from {metadata_path}"
                    )

                else:
                    logger.warning(f"Metadata file not found: {metadata_path}")

            except Exception as e:
                logger.error(f"Error extracting content from result: {e}")
                continue

        logger.info(f"Extracted content from {len(documents)} result files")
        return documents

    def _load_topic_documents_for_pairs(
        self, pairs: List[Tuple[str, str]], results: List[Dict]
    ) -> List[Dict]:
        """
        Load topic documents from actual result files, filtered by specific pairs.

        Args:
            pairs: List of (agency, topic) pairs to load
            results: List of result dictionaries

        Returns:
            List of filtered documents matching the specific pairs
        """
        all_documents = self._extract_topic_content_from_results(results)

        # Convert pairs to set for faster lookup
        target_pairs = set(pairs)

        # Filter documents to only include the specific pairs requested
        filtered_documents = []
        for doc in all_documents:
            doc_pair = (doc["agency"], doc["topic"])
            if doc_pair in target_pairs:
                filtered_documents.append(doc)

        logger.info(
            f"Filtered to {len(filtered_documents)} documents matching {len(pairs)} requested pairs"
        )
        return filtered_documents

    # === LEGACY DOCUMENT LOADING (fallback) ===

    def _load_topic_documents_from_source_paths(
        self, results: List[Dict], agencies: List[str], topics: List[str]
    ) -> List[Dict]:
        """
        Load topic documents directly from source paths in results.
        Used as fallback when metadata.json is not available.
        """
        documents = []
        doc_id = 1

        logger.info(
            f"Loading from source paths - looking for agencies: {agencies}, topics: {topics}"
        )

        for result in results:
            try:
                source_path = result.get("source", "")
                logger.debug(f"Processing source path: {source_path}")

                if not source_path or not os.path.exists(source_path):
                    logger.warning(
                        f"Source path doesn't exist or is empty: {source_path}"
                    )
                    continue

                # Extract agency and topic from source path
                source_parts = source_path.strip("/").split("/")
                logger.debug(f"Source path parts: {source_parts}")

                if len(source_parts) >= 4:
                    agency = source_parts[-3]  # sm_someuuid
                    topic = source_parts[-2]  # d934abece3ce5ea3ceaa55e41f3cfe0eb7ea6f97
                    logger.debug(f"Extracted agency: {agency}, topic: {topic}")
                else:
                    logger.warning(
                        f"Source path doesn't have expected structure: {source_path}"
                    )
                    continue

                # Filter by agencies and topics if specified
                agency_match = not agencies or agency in agencies
                topic_match = not topics or topic in topics
                logger.debug(
                    f"Filter match for {agency}-{topic}: agency_match={agency_match}, topic_match={topic_match}"
                )

                if not (agency_match and topic_match):
                    continue

                # Read content from source file
                try:
                    with open(source_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if content:
                        documents.append(
                            {
                                "id": doc_id,
                                "text_content": content,
                                "agency": agency,
                                "topic": topic,
                                "content_hash": hashlib.sha256(
                                    content.encode("utf-8")
                                ).hexdigest(),
                                "source_path": source_path,
                                "file_name": os.path.basename(source_path),
                            }
                        )
                        doc_id += 1
                        logger.info(
                            f"Loaded content for {agency}-{topic} from {source_path}"
                        )

                except Exception as e:
                    logger.warning(f"Error reading source file {source_path}: {e}")

            except Exception as e:
                logger.warning(f"Error processing source path for result: {e}")

        logger.info(f"Loaded {len(documents)} topic documents from source paths")
        return documents

    def _load_topic_documents_legacy(
        self, agencies: List[str], topics: List[str]
    ) -> List[Dict]:
        """
        Legacy method: Load topic documents from local file system.
        Used as last resort fallback.
        """
        documents = []
        doc_id = 1

        for agency in agencies:
            agency_dir = os.path.join(self.topic_documents_path, agency)
            if not os.path.exists(agency_dir):
                logger.warning(f"Agency directory not found: {agency_dir}")
                continue

            for topic in topics:
                topic_dir = os.path.join(agency_dir, topic)
                if not os.path.exists(topic_dir):
                    logger.warning(f"Topic directory not found: {topic_dir}")
                    continue

                # Load all .txt files in topic directory
                try:
                    txt_files = glob.glob(os.path.join(topic_dir, "*.txt"))

                    for txt_file in txt_files:
                        try:
                            with open(txt_file, "r", encoding="utf-8") as f:
                                content = f.read().strip()

                            if content:
                                documents.append(
                                    {
                                        "id": doc_id,
                                        "text_content": content,
                                        "agency": agency,
                                        "topic": topic,
                                        "content_hash": hashlib.sha256(
                                            content.encode("utf-8")
                                        ).hexdigest(),
                                        "file_path": txt_file,
                                        "filename": os.path.basename(txt_file),
                                    }
                                )
                                doc_id += 1

                        except Exception as e:
                            logger.warning(f"Error reading {txt_file}: {e}")

                except Exception as e:
                    logger.warning(f"Error processing topic directory {topic_dir}: {e}")

        logger.info(f"Loaded {len(documents)} topic documents from legacy file system")
        return documents

    def _generate_embeddings_for_pairs(
        self, pairs: List[Tuple[str, str]], results: List[Dict] = None
    ) -> List[EmbeddingRecord]:
        """
        Generate embeddings for specific agency-topic pairs.

        Args:
            pairs: List of (agency, topic) pairs to generate embeddings for
            results: Optional list of result dictionaries to extract content from

        Returns:
            List of EmbeddingRecord objects with computed embeddings
        """
        logger.info(f"Generating embeddings for {len(pairs)} agency-topic pairs")

        if results:
            # Load documents for the specific pairs
            documents = self._load_topic_documents_for_pairs(pairs, results)
        else:
            # Fallback to legacy method
            agencies = list(set(pair[0] for pair in pairs))
            topics = list(set(pair[1] for pair in pairs))
            documents = self._load_topic_documents_legacy(agencies, topics)

            # Filter to only the requested pairs
            target_pairs = set(pairs)
            documents = [
                doc
                for doc in documents
                if (doc["agency"], doc["topic"]) in target_pairs
            ]

        if not documents:
            logger.warning("No documents found to generate embeddings")
            return []

        # Extract texts for batch embedding
        texts = [doc["text_content"] for doc in documents]

        # Generate embeddings in batches
        logger.info(f"Computing embeddings for {len(texts)} documents...")
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=len(texts) > 10,
            )
            all_embeddings.extend(batch_embeddings)

        # Create EmbeddingRecord objects
        embedding_records = []
        current_time = datetime.now().isoformat()

        for doc, embedding in zip(documents, all_embeddings):
            record = EmbeddingRecord(
                id=doc["id"],
                text_content=doc["text_content"],
                embedding_vector=embedding.tolist(),
                agency=doc["agency"],
                topic=doc["topic"],
                model_name=self.model_name,
                created_at=current_time,
                content_hash=doc["content_hash"],
            )
            embedding_records.append(record)

        logger.info(f"Generated {len(embedding_records)} embeddings for pairs")
        return embedding_records

    def get_embeddings_for_pairs(
        self, pairs: List[Tuple[str, str]], results: List[Dict] = None
    ) -> Dict[Tuple[str, str], List[EmbeddingRecord]]:
        """
        Get topic embeddings for specific agency-topic pairs.

        Args:
            pairs: List of (agency, topic) pairs
            results: Optional list of result dictionaries for content extraction

        Returns:
            Dict mapping (agency, topic) to list of EmbeddingRecord objects
        """
        logger.info(f"Getting embeddings for {len(pairs)} specific agency-topic pairs")

        embeddings_by_context = {}
        missing_pairs = []

        # Check Redis cache for each pair
        for agency, topic in pairs:
            redis_key = self._get_persistent_key(agency, topic)
            try:
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    cached_embeddings = self._deserialize_embeddings(cached_data)
                    embeddings_by_context[(agency, topic)] = cached_embeddings
                    logger.info(
                        f"Retrieved {len(cached_embeddings)} cached embeddings for {agency}-{topic}"
                    )
                else:
                    missing_pairs.append((agency, topic))
            except Exception as e:
                logger.warning(
                    f"Error reading cached embeddings for {agency}-{topic}: {e}"
                )
                missing_pairs.append((agency, topic))

        # Generate missing embeddings if any
        if missing_pairs:
            logger.info(f"Generating embeddings for {len(missing_pairs)} missing pairs")

            all_embeddings = self._generate_embeddings_for_pairs(missing_pairs, results)

            # Group by pair and cache
            embeddings_by_pair = defaultdict(list)
            for embedding in all_embeddings:
                key = (embedding.agency, embedding.topic)
                embeddings_by_pair[key].append(embedding)

            # Cache each pair
            for (agency, topic), pair_embeddings in embeddings_by_pair.items():
                if pair_embeddings:
                    # Cache in Redis
                    redis_key = self._get_persistent_key(agency, topic)
                    serialized_data = self._serialize_embeddings(pair_embeddings)
                    self.redis_client.set(redis_key, serialized_data)

                    # Add to results
                    embeddings_by_context[(agency, topic)] = pair_embeddings
                    logger.info(
                        f"Generated and cached {len(pair_embeddings)} embeddings for {agency}-{topic}"
                    )

        logger.info(
            f"Retrieved embeddings for {len(embeddings_by_context)} agency-topic pairs"
        )
        return embeddings_by_context

    # === LEGACY API (DEPRECATED) ===

    def get_topic_embeddings(
        self,
        agency: str,
        topic: str,
        force_refresh: bool = False,
        results: List[Dict] = None,
    ) -> Optional[List[EmbeddingRecord]]:
        """
        Get topic embeddings for agency-topic combination.
        """
        pairs = [(agency, topic)]
        results_dict = self.get_embeddings_for_pairs(pairs, results)
        return results_dict.get((agency, topic))

    def get_bulk_topic_embeddings(
        self, agencies: List[str], topics: List[str], results: List[Dict] = None
    ) -> Dict[Tuple[str, str], List[EmbeddingRecord]]:
        """
        Get topic embeddings for multiple agency-topic combinations.

        Consider using get_embeddings_for_pairs() instead.
        """
        logger.warning(
            f"Using legacy get_bulk_topic_embeddings - creates cross-product of {len(agencies)}×{len(topics)}={len(agencies) * len(topics)} combinations"
        )

        # Create cross-product (legacy behavior)
        pairs = [(agency, topic) for agency in agencies for topic in topics]
        return self.get_embeddings_for_pairs(pairs, results)

    def has_topic_embeddings(self, agency: str, topic: str) -> bool:
        """Check if topic embeddings exist in cache."""
        redis_key = self._get_persistent_key(agency, topic)
        return self.redis_client.exists(redis_key) > 0

    def delete_topic_embeddings(self, agency: str, topic: Optional[str] = None) -> bool:
        """
        Delete topic embeddings.

        Args:
            agency: Agency name
            topic: Topic name (if None, deletes all topics for agency)
        """
        try:
            if topic:
                redis_key = self._get_persistent_key(agency, topic)
                deleted = self.redis_client.delete(redis_key)
                logger.info(
                    f"Deleted topic embeddings for {agency}-{topic}: {deleted} keys"
                )
            else:
                pattern = f"embeddings:persistent:{agency}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(
                        f"Deleted all topic embeddings for {agency}: {deleted} keys"
                    )

            return True

        except Exception as e:
            logger.error(f"Error deleting topic embeddings for {agency}-{topic}: {e}")
            return False

    # === TEMPORARY QUESTION EMBEDDINGS ===

    def start_question_session(self) -> str:
        """Start a new question embedding session."""
        session_id = str(uuid.uuid4())
        logger.info(f"Started question embedding session: {session_id}")
        return session_id

    def generate_and_cache_question_embeddings(
        self,
        session_id: str,
        questions_by_context: Dict[Tuple[str, str], List[str]],
        ttl_seconds: int = 3600,
    ) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
        """
        Generate and cache question embeddings for evaluation session.

        Args:
            session_id: Unique session identifier
            questions_by_context: Dict mapping (agency, topic) to question lists
            ttl_seconds: Time to live for cached embeddings

        Returns:
            Dict mapping (agency, topic) to question embeddings
        """
        logger.info(f"Generating question embeddings for session {session_id}")

        # Collect all questions
        all_questions = []
        question_contexts = []

        for (agency, topic), questions in questions_by_context.items():
            for question in questions:
                all_questions.append(question)
                question_contexts.append((agency, topic))

        if not all_questions:
            logger.warning("No questions to generate embeddings for")
            return {}

        # Generate embeddings in batches
        logger.info(f"Computing embeddings for {len(all_questions)} questions...")
        batch_size = 32
        all_question_embeddings = []

        for i in range(0, len(all_questions), batch_size):
            batch_questions = all_questions[i : i + batch_size]
            batch_embeddings = self.model.encode(
                batch_questions,
                batch_size=len(batch_questions),
                show_progress_bar=len(all_questions) > 10,
            )
            all_question_embeddings.extend(batch_embeddings)

        # Organize embeddings by context
        embeddings_by_context = defaultdict(dict)
        flattened_for_redis = {}

        for question, embedding, (agency, topic) in zip(
            all_questions, all_question_embeddings, question_contexts
        ):
            # Store by context for return value
            embeddings_by_context[(agency, topic)][question] = embedding

            # Store flattened for Redis
            unique_key = f"{agency}|{topic}|{question}"
            flattened_for_redis[unique_key] = embedding

        # Cache in Redis with TTL
        redis_key = self._get_temporary_key(session_id)
        serialized_data = self._serialize_question_embeddings(flattened_for_redis)
        self.redis_client.setex(redis_key, ttl_seconds, serialized_data)

        logger.info(
            f"Generated and cached {len(all_questions)} question embeddings for session {session_id}"
        )
        return dict(embeddings_by_context)

    def get_question_embeddings(
        self, session_id: str
    ) -> Optional[Dict[Tuple[str, str], Dict[str, np.ndarray]]]:
        """
        Get question embeddings for session, organized by context.

        Args:
            session_id: Session identifier

        Returns:
            Dict mapping (agency, topic) to question embeddings, or None if not found
        """
        redis_key = self._get_temporary_key(session_id)

        try:
            cached_data = self.redis_client.get(redis_key)
            if not cached_data:
                logger.warning(f"No question embeddings found for session {session_id}")
                return None

            # Deserialize flattened data
            flattened_embeddings = self._deserialize_question_embeddings(cached_data)

            # Organize by context
            embeddings_by_context = defaultdict(dict)
            for unique_key, embedding in flattened_embeddings.items():
                # Parse key format: "agency|topic|question"
                parts = unique_key.split("|", 2)
                if len(parts) == 3:
                    agency, topic, question = parts
                    embeddings_by_context[(agency, topic)][question] = embedding

            logger.info(
                f"Retrieved question embeddings for {len(embeddings_by_context)} contexts from session {session_id}"
            )
            return dict(embeddings_by_context)

        except Exception as e:
            logger.error(
                f"Error retrieving question embeddings for session {session_id}: {e}"
            )
            return None

    def cleanup_question_session(self, session_id: str) -> bool:
        """
        Clean up question embeddings session.

        Args:
            session_id: Session identifier to clean up
        """
        try:
            redis_key = self._get_temporary_key(session_id)
            deleted = self.redis_client.delete(redis_key)
            logger.info(
                f"Cleaned up question session {session_id}: {deleted} keys deleted"
            )
            return True

        except Exception as e:
            logger.error(f"Error cleaning up question session {session_id}: {e}")
            return False

    # === CACHE MANAGEMENT ===

    def clear_all_topic_embeddings(self) -> bool:
        """Clear all persistent topic embeddings."""
        try:
            keys = self.redis_client.keys("embeddings:persistent:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared all topic embeddings: {deleted} keys")
            return True
        except Exception as e:
            logger.error(f"Error clearing topic embeddings: {e}")
            return False

    def clear_all_question_sessions(self) -> bool:
        """Clear all temporary question embeddings."""
        try:
            keys = self.redis_client.keys("embeddings:temp:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared all question sessions: {deleted} keys")
            return True
        except Exception as e:
            logger.error(f"Error clearing question sessions: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            persistent_keys = self.redis_client.keys("embeddings:persistent:*")
            temp_keys = self.redis_client.keys("embeddings:temp:*")

            # Analyze persistent cache
            persistent_stats = {
                "total_keys": len(persistent_keys),
                "agency_topic_combinations": [],
            }

            total_persistent_embeddings = 0
            for key in persistent_keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                # Extract agency-topic from key
                parts = key_str.split(":")
                if len(parts) >= 4:
                    agency, topic = parts[2], parts[3]
                    persistent_stats["agency_topic_combinations"].append(
                        f"{agency}-{topic}"
                    )

                # Count embeddings in this key
                try:
                    data = self.redis_client.get(key)
                    if data:
                        embeddings = self._deserialize_embeddings(data)
                        total_persistent_embeddings += len(embeddings)
                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
                    pass

            persistent_stats["total_embeddings"] = total_persistent_embeddings
            persistent_stats["combinations_count"] = len(
                persistent_stats["agency_topic_combinations"]
            )

            # Analyze temporary cache
            temp_stats = {"total_keys": len(temp_keys), "active_sessions": []}

            total_temp_embeddings = 0
            for key in temp_keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= 3:
                    session_id = parts[2]
                    temp_stats["active_sessions"].append(session_id)

                # Count embeddings
                try:
                    data = self.redis_client.get(key)
                    if data:
                        embeddings = self._deserialize_question_embeddings(data)
                        total_temp_embeddings += len(embeddings)
                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")

            temp_stats["total_embeddings"] = total_temp_embeddings
            temp_stats["sessions_count"] = len(temp_stats["active_sessions"])

            # Redis info
            memory_info = self.redis_client.info("memory")

            return {
                "model_name": self.model_name,
                "persistent_cache": persistent_stats,
                "temporary_cache": temp_stats,
                "redis_info": {
                    "used_memory": memory_info.get("used_memory_human", "N/A"),
                    "total_keys": len(persistent_keys) + len(temp_keys),
                    "connected_clients": self.redis_client.info().get(
                        "connected_clients", 0
                    ),
                    "health": self.health_check(),
                },
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


# Convenience functions for backward compatibility
def create_embedding_manager(
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    redis_url: str = None,
    topic_documents_path: str = "topic_documents",
) -> UnifiedEmbeddingManager:
    """Create a unified embedding manager with sensible defaults."""
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    return UnifiedEmbeddingManager(
        model_name=model_name,
        redis_url=redis_url,
        topic_documents_path=topic_documents_path,
    )
