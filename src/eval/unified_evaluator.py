import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


@dataclass
class EvaluationThresholds:
    """Thresholds for each evaluation metric."""

    topic_coverage_min: float = 0.7
    information_coverage_min: float = 0.6
    similarity_coverage_min: float = 0.6
    overall_min: float = 0.65
    context_coverage_min: float = 0.5


@dataclass
class EvaluationResult:
    """Evaluation result with all metrics."""

    topic_coverage_score: float
    information_coverage_score: float
    similarity_coverage_score: float
    overall_score: float
    should_regenerate: bool
    failed_metrics: List[str]
    context_coverage: Dict[str, float]
    contexts_analyzed: int
    contexts_passed: int


class UnifiedEvaluator:
    """
    Unified evaluator that uses the UnifiedEmbeddingManager for all evaluation tasks.
    """

    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        logger.info("Initialized UnifiedEvaluator")

    def evaluate_with_session(
        self,
        session_id: str,
        agencies: List[str],
        topics: List[str],
        thresholds: Optional[EvaluationThresholds] = None,
        results: List[Dict] = None,
    ) -> EvaluationResult:
        """
        Evaluate using cached embeddings from a session.

        Args:
            session_id: Session ID with cached question embeddings
            agencies: List of agencies being evaluated
            topics: List of topics being evaluated
            thresholds: Optional evaluation thresholds
            results: Results list to extract correct agency-topic pairs

        Returns:
            EvaluationResult with all metrics and decision
        """
        if thresholds is None:
            thresholds = EvaluationThresholds()

        logger.info(
            f"Evaluating with session {session_id} for {len(agencies)} agencies, {len(topics)} topics"
        )

        try:
            # FIXED: Extract correct agency-topic pairs from results instead of cross product
            if results:
                valid_pairs = self._extract_valid_agency_topic_pairs(results)
                logger.info(
                    f"Using {len(valid_pairs)} valid agency-topic pairs from results"
                )
            else:
                # Fallback: create cross product (legacy behavior)
                valid_pairs = [
                    (agency, topic) for agency in agencies for topic in topics
                ]
                logger.info(
                    f"Using cross-product approach: {len(valid_pairs)} combinations"
                )

            # FIXED: Use new pair-based API instead of cross-product
            topic_embeddings_by_context = (
                self.embedding_manager.get_embeddings_for_pairs(
                    valid_pairs, results=results
                )
            )

            # No filtering needed - we already got exactly what we asked for
            filtered_topic_embeddings = topic_embeddings_by_context

            if not filtered_topic_embeddings:
                logger.error("No topic embeddings available for valid pairs")
                return self._create_failed_result("No topic embeddings available")

            # Get cached question embeddings
            question_embeddings_by_context = (
                self.embedding_manager.get_question_embeddings(session_id)
            )

            if not question_embeddings_by_context:
                logger.error(f"No question embeddings found for session {session_id}")
                return self._create_failed_result("No question embeddings found")

            logger.info(
                f"Using {len(filtered_topic_embeddings)} topic contexts and {len(question_embeddings_by_context)} question contexts"
            )

            # Compute evaluation metrics for each valid context
            context_scores = {}
            context_metrics = {}

            for agency, topic in valid_pairs:
                context_name = f"{agency}-{topic}"

                # Get embeddings for this context
                topic_records = filtered_topic_embeddings.get((agency, topic), [])
                question_embeddings = question_embeddings_by_context.get(
                    (agency, topic), {}
                )

                if not topic_records:
                    logger.warning(f"No topic embeddings for context {context_name}")
                    context_scores[context_name] = 0.0
                    continue

                if not question_embeddings:
                    logger.warning(f"No questions for context {context_name}")
                    context_scores[context_name] = 0.0
                    continue

                # Convert to numpy arrays
                topic_embeddings = np.array(
                    [record.embedding_vector for record in topic_records]
                )
                questions_embeddings = np.array(list(question_embeddings.values()))

                # Compute context metrics
                metrics = self._compute_context_metrics(
                    topic_embeddings, questions_embeddings, context_name
                )
                context_metrics[(agency, topic)] = metrics
                context_scores[context_name] = metrics["overall_context_score"]

                logger.info(
                    f"{context_name}: Score={metrics['overall_context_score']:.3f}"
                )

            # Aggregate metrics
            overall_metrics = self._aggregate_metrics(context_metrics, thresholds)

            # Determine failed contexts
            failed_contexts = [
                name
                for name, score in context_scores.items()
                if score < thresholds.context_coverage_min
            ]

            # Check failed metrics
            failed_metrics = []
            if overall_metrics["topic_coverage_score"] < thresholds.topic_coverage_min:
                failed_metrics.append("topic_coverage")
            if (
                overall_metrics["information_coverage_score"]
                < thresholds.information_coverage_min
            ):
                failed_metrics.append("information_coverage")
            if (
                overall_metrics["similarity_coverage_score"]
                < thresholds.similarity_coverage_min
            ):
                failed_metrics.append("similarity_coverage")
            if overall_metrics["overall_score"] < thresholds.overall_min:
                failed_metrics.append("overall_score")
            if len(failed_contexts) > 0:
                failed_metrics.append("context_coverage")

            should_regenerate = len(failed_metrics) > 0

            # Create result
            result = EvaluationResult(
                topic_coverage_score=round(overall_metrics["topic_coverage_score"], 3),
                information_coverage_score=round(
                    overall_metrics["information_coverage_score"], 3
                ),
                similarity_coverage_score=round(
                    overall_metrics["similarity_coverage_score"], 3
                ),
                overall_score=round(overall_metrics["overall_score"], 3),
                should_regenerate=should_regenerate,
                failed_metrics=failed_metrics,
                context_coverage=context_scores,
                contexts_analyzed=len(context_scores),
                contexts_passed=len(context_scores) - len(failed_contexts),
            )

            # Log results
            self._log_results(result, thresholds, failed_contexts)

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return self._create_failed_result(f"Evaluation error: {str(e)}")

    def _extract_valid_agency_topic_pairs(
        self, results: List[Dict]
    ) -> List[Tuple[str, str]]:
        """
        Extract valid agency-topic pairs from results, maintaining correct relationships.

        Args:
            results: List of result dictionaries

        Returns:
            List of (agency, topic) tuples with correct pairings
        """
        valid_pairs = []

        for i, result in enumerate(results):
            # Extract agency and topic using the same logic as the embedding manager
            source_path = result.get("source", "")
            output_path = result.get("output_path", "")

            agency = f"agency{i + 1}"  # fallback
            topic = f"topic{i + 1}"  # fallback

            # Try to extract from source path first (more reliable)
            if source_path:
                source_parts = source_path.strip("/").split("/")
                if len(source_parts) >= 4:
                    agency = source_parts[-3]  # sm_someuuid
                    topic = source_parts[-2]  # d934abece3ce5ea3ceaa55e41f3cfe0eb7ea6f97
            else:
                # Fallback to output path
                if output_path:
                    output_parts = output_path.strip("/").split("/")
                    if len(output_parts) >= 5:
                        agency = output_parts[-3]
                        topic = output_parts[-2]

            pair = (agency, topic)
            if pair not in valid_pairs:
                valid_pairs.append(pair)

        logger.info(
            f"Extracted {len(valid_pairs)} valid agency-topic pairs: {valid_pairs}"
        )
        return valid_pairs

    def _compute_context_metrics(
        self,
        topic_embeddings: np.ndarray,
        question_embeddings: np.ndarray,
        context_name: str,
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a specific context."""
        if topic_embeddings.size == 0 or question_embeddings.size == 0:
            logger.warning(f"Empty embeddings for context {context_name}")
            return {
                "similarity_score": 0.0,
                "coverage_score": 0.0,
                "information_score": 0.0,
                "overall_context_score": 0.0,
            }

        try:
            # Normalize embeddings
            topic_norm = topic_embeddings / np.linalg.norm(
                topic_embeddings, axis=1, keepdims=True
            )
            question_norm = question_embeddings / np.linalg.norm(
                question_embeddings, axis=1, keepdims=True
            )

            # Compute similarity matrix
            similarity_matrix = np.dot(question_norm, topic_norm.T)

            # Similarity score: how well questions match topics
            max_similarities = np.max(similarity_matrix, axis=1)
            similarity_score = np.mean(max_similarities)

            # Coverage score: how many topics are covered by questions
            topic_max_similarities = np.max(similarity_matrix, axis=0)
            coverage_threshold = 0.5
            covered_topics = np.sum(topic_max_similarities >= coverage_threshold)
            coverage_score = covered_topics / len(topic_embeddings)

            # Information score: overall alignment
            information_score = np.mean(similarity_matrix)

            # Overall context score
            overall_context_score = (
                similarity_score + coverage_score + information_score
            ) / 3.0

            return {
                "similarity_score": float(similarity_score),
                "coverage_score": float(coverage_score),
                "information_score": float(information_score),
                "overall_context_score": float(overall_context_score),
            }

        except Exception as e:
            logger.error(f"Error computing metrics for {context_name}: {e}")
            return {
                "similarity_score": 0.0,
                "coverage_score": 0.0,
                "information_score": 0.0,
                "overall_context_score": 0.0,
            }

    def _aggregate_metrics(
        self,
        context_metrics: Dict[Tuple[str, str], Dict[str, float]],
        thresholds: EvaluationThresholds,
    ) -> Dict[str, float]:
        """Aggregate metrics across all contexts."""
        if not context_metrics:
            return {
                "topic_coverage_score": 0.0,
                "information_coverage_score": 0.0,
                "similarity_coverage_score": 0.0,
                "overall_score": 0.0,
            }

        similarity_scores = [
            metrics["similarity_score"] for metrics in context_metrics.values()
        ]
        coverage_scores = [
            metrics["coverage_score"] for metrics in context_metrics.values()
        ]
        information_scores = [
            metrics["information_score"] for metrics in context_metrics.values()
        ]

        topic_coverage_score = np.mean(coverage_scores)
        information_coverage_score = np.mean(information_scores)
        similarity_coverage_score = np.mean(similarity_scores)

        overall_score = (
            topic_coverage_score
            + information_coverage_score
            + similarity_coverage_score
        ) / 3.0

        return {
            "topic_coverage_score": float(topic_coverage_score),
            "information_coverage_score": float(information_coverage_score),
            "similarity_coverage_score": float(similarity_coverage_score),
            "overall_score": float(overall_score),
        }

    def _create_failed_result(self, error_message: str) -> EvaluationResult:
        """Create a failed evaluation result."""
        logger.error(f"Creating failed result: {error_message}")
        return EvaluationResult(
            topic_coverage_score=0.0,
            information_coverage_score=0.0,
            similarity_coverage_score=0.0,
            overall_score=0.0,
            should_regenerate=True,
            failed_metrics=["evaluation_error"],
            context_coverage={},
            contexts_analyzed=0,
            contexts_passed=0,
        )

    def _log_results(
        self,
        result: EvaluationResult,
        thresholds: EvaluationThresholds,
        failed_contexts: List[str],
    ):
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("UNIFIED EMBEDDING EVALUATION RESULTS")
        logger.info("=" * 60)

        decision_icon = "✅" if not result.should_regenerate else "❌"
        logger.info(
            f"Decision: {decision_icon} {'ACCEPT' if not result.should_regenerate else 'REGENERATE'}"
        )
        logger.info(f"Overall Score: {result.overall_score:.1%}")
        logger.info(
            f"Topic Coverage: {result.topic_coverage_score:.1%} (min: {thresholds.topic_coverage_min:.1%})"
        )
        logger.info(
            f"Information Coverage: {result.information_coverage_score:.1%} (min: {thresholds.information_coverage_min:.1%})"
        )
        logger.info(
            f"Similarity Coverage: {result.similarity_coverage_score:.1%} (min: {thresholds.similarity_coverage_min:.1%})"
        )
        logger.info(
            f"Contexts: {result.contexts_passed}/{result.contexts_analyzed} passed"
        )

        if result.context_coverage:
            logger.info("Context Breakdown:")
            for context_name, score in result.context_coverage.items():
                status = "✅" if score >= thresholds.context_coverage_min else "❌"
                logger.info(f"   {status} {context_name}: {score:.1%}")

        if failed_contexts:
            logger.warning(f"Failed contexts: {failed_contexts}")

        if result.should_regenerate:
            logger.warning(
                f"REGENERATE RECOMMENDED - Failed: {', '.join(result.failed_metrics)}"
            )
        else:
            logger.info("DATASET QUALITY ACCEPTABLE")


# Main evaluation function that uses the unified system
def eval_single_agency_level(
    results: List[Dict],
    custom_thresholds: Optional[Dict[str, float]] = None,
    embedding_manager=None,
    session_id: Optional[str] = None,
) -> Dict:
    """
    Single agency evaluation using unified embedding management.

    Args:
        results: List of result dictionaries with output paths
        custom_thresholds: Optional custom evaluation thresholds
        embedding_manager: UnifiedEmbeddingManager instance
        session_id: Session ID for cached embeddings

    Returns:
        Dictionary with evaluation results and decision
    """
    start_time = time.time()

    # Validate inputs
    if not results:
        return {
            "error": "No results provided for evaluation",
            "should_regenerate": True,
            "decision": "REGENERATE",
            "evaluation_time_ms": (time.time() - start_time) * 1000,
        }

    # FIXED: Extract valid agency-topic pairs instead of cross product
    valid_pairs = _extract_valid_agency_topic_pairs_from_results(results)
    agencies = list(set(pair[0] for pair in valid_pairs))
    topics = list(set(pair[1] for pair in valid_pairs))

    if not agencies or not topics:
        return {
            "error": "No agencies or topics found in results",
            "should_regenerate": True,
            "decision": "REGENERATE",
            "evaluation_time_ms": (time.time() - start_time) * 1000,
        }

    # If no embedding manager provided, we can't proceed
    if embedding_manager is None:
        logger.error("No embedding manager provided - cannot proceed with evaluation")
        return {
            "error": "No embedding manager provided",
            "should_regenerate": True,
            "decision": "REGENERATE",
            "evaluation_time_ms": (time.time() - start_time) * 1000,
        }

    # Create evaluator
    evaluator = UnifiedEvaluator(embedding_manager)

    # Set up thresholds
    thresholds = EvaluationThresholds()
    if custom_thresholds:
        for key, value in custom_thresholds.items():
            if hasattr(thresholds, key):
                setattr(thresholds, key, value)

    # If no session_id provided, we need to generate embeddings
    if session_id is None:
        logger.info("No session_id provided, generating embeddings on-demand")

        try:
            # Extract questions from results
            questions_by_context = _extract_questions_from_results(results)

            if not questions_by_context:
                return {
                    "error": "No questions found in result files",
                    "should_regenerate": True,
                    "decision": "REGENERATE",
                    "evaluation_time_ms": (time.time() - start_time) * 1000,
                }

            # Generate session with embeddings
            session_id = embedding_manager.start_question_session()
            embedding_manager.generate_and_cache_question_embeddings(
                session_id, questions_by_context
            )

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {
                "error": f"Failed to generate embeddings: {str(e)}",
                "should_regenerate": True,
                "decision": "REGENERATE",
                "evaluation_time_ms": (time.time() - start_time) * 1000,
            }

    # Run evaluation with results parameter to preserve agency-topic relationships
    try:
        result = evaluator.evaluate_with_session(
            session_id, agencies, topics, thresholds, results
        )

        eval_time = (time.time() - start_time) * 1000

        # Convert to expected format
        evaluation_result = {
            "topic_coverage_score": result.topic_coverage_score,
            "information_coverage_score": result.information_coverage_score,
            "similarity_coverage_score": result.similarity_coverage_score,
            "overall_score": result.overall_score,
            "should_regenerate": result.should_regenerate,
            "failed_metrics": result.failed_metrics,
            "decision": "REGENERATE" if result.should_regenerate else "ACCEPT",
            "context_coverage": result.context_coverage,
            "contexts_analyzed": result.contexts_analyzed,
            "contexts_passed": result.contexts_passed,
            "evaluation_metadata": {
                "agencies_evaluated": agencies,
                "topics_evaluated": topics,
                "valid_pairs": len(valid_pairs),
                "evaluation_mode": "unified_redis",
                "session_id": session_id,
            },
            "performance": {
                "evaluation_time_ms": round(eval_time, 2),
                "used_unified_cache": True,
            },
            "summary": f"Unified Evaluation | Overall: {result.overall_score:.1%} | "
            + f"Contexts: {result.contexts_passed}/{result.contexts_analyzed} | "
            + f"Time: {eval_time:.0f}ms",
        }

        return evaluation_result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "error": f"Evaluation failed: {str(e)}",
            "should_regenerate": True,
            "decision": "REGENERATE",
            "evaluation_time_ms": (time.time() - start_time) * 1000,
            "session_id": session_id,
        }


def _extract_valid_agency_topic_pairs_from_results(
    results: List[Dict],
) -> List[Tuple[str, str]]:
    """
    Extract valid agency-topic pairs from results, maintaining correct relationships.
    This ensures we don't create cross-products of agencies and topics.
    """
    valid_pairs = []

    for i, result in enumerate(results):
        # Extract agency and topic using the same logic as the embedding manager
        source_path = result.get("source", "")
        output_path = result.get("output_path", "")

        agency = f"agency{i + 1}"  # fallback
        topic = f"topic{i + 1}"  # fallback

        # Try to extract from source path first (more reliable)
        if source_path:
            source_parts = source_path.strip("/").split("/")
            if len(source_parts) >= 4:
                agency = source_parts[-3]
                topic = source_parts[-2]
        else:
            # Fallback to output path
            if output_path:
                output_parts = output_path.strip("/").split("/")
                if len(output_parts) >= 5:
                    agency = output_parts[-3]
                    topic = output_parts[-2]

        pair = (agency, topic)
        if pair not in valid_pairs:
            valid_pairs.append(pair)

    logger.info(f"Extracted {len(valid_pairs)} valid agency-topic pairs from results")
    return valid_pairs


def _extract_agencies_topics_from_results(
    results: List[Dict],
) -> Tuple[List[str], List[str]]:
    """Extract agencies and topics from result dictionaries - DEPRECATED."""
    logger.warning(
        "Using deprecated _extract_agencies_topics_from_results - should use _extract_valid_agency_topic_pairs_from_results"
    )
    agencies = set()
    topics = set()

    for i, result in enumerate(results):
        # Use generic names as fallback
        agency = f"agency{i + 1}"
        topic = f"topic{i + 1}"

        # Try to extract from source path first (more reliable)
        source_path = result.get("source", "")
        if source_path:
            source_parts = source_path.strip("/").split("/")
            if len(source_parts) >= 4:
                agency = source_parts[-3]
                topic = source_parts[-2]
        else:
            # Fallback to output path
            output_path = result.get("output_path", "")
            if output_path:
                output_parts = output_path.strip("/").split("/")
                if len(output_parts) >= 5:
                    agency = output_parts[-3]
                    topic = output_parts[-2]

        agencies.add(agency)
        topics.add(topic)

    logger.info(f"Extracted agencies: {list(agencies)}, topics: {list(topics)}")
    return list(agencies), list(topics)


def _extract_questions_from_results(
    results: List[Dict],
) -> Dict[Tuple[str, str], List[str]]:
    """Extract questions from result files organized by context."""
    questions_by_context = defaultdict(list)

    for i, result in enumerate(results):
        output_path = result.get("output_path", "")
        source_path = result.get("source", "")

        # Extract agency and topic using the same logic as _extract_valid_agency_topic_pairs_from_results
        agency = f"agency{i + 1}"
        topic = f"topic{i + 1}"

        # Try to extract from source path first (more reliable)
        if source_path:
            source_parts = source_path.strip("/").split("/")
            if len(source_parts) >= 4:
                agency = source_parts[-3]
                topic = source_parts[-2]
        else:
            # Fallback to output path
            if output_path:
                output_parts = output_path.strip("/").split("/")
                if len(output_parts) >= 5:
                    agency = output_parts[-3]
                    topic = output_parts[-2]

        # Find the directory containing faqs.json
        faqs_dir = output_path
        if os.path.isfile(output_path):
            faqs_dir = os.path.dirname(output_path)

        faqs_path = os.path.join(faqs_dir, "faqs.json")

        if os.path.exists(faqs_path):
            try:
                with open(faqs_path, "r", encoding="utf-8") as f:
                    faqs = json.load(f)

                if isinstance(faqs, list):
                    questions = []
                    for faq in faqs:
                        if isinstance(faq, dict) and "question" in faq:
                            question_text = faq["question"].strip()
                            if question_text:
                                questions.append(question_text)

                    if questions:
                        questions_by_context[(agency, topic)].extend(questions)
                        logger.info(
                            f"Extracted {len(questions)} questions for {agency}-{topic}"
                        )

            except Exception as e:
                logger.warning(f"Failed to read questions from {faqs_path}: {e}")
        else:
            logger.warning(f"FAQ file not found: {faqs_path}")

    return dict(questions_by_context)


# Compatibility functions
def eval_with_context(
    results: List[Dict],
    custom_thresholds: Optional[Dict[str, float]] = None,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    session_id: Optional[str] = None,
) -> Dict:
    """Contextual evaluation using unified system."""
    return eval_single_agency_level(results, custom_thresholds, None, session_id)


def clear_evaluation_cache():
    """Clear evaluation cache (placeholder for compatibility)."""
    logger.info("Cache clearing handled by unified embedding manager")
