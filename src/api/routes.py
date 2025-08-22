from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
import httpx
from src.core.data_generator import DataGenerator
from src.core.data_source import DataSourceManager
from src.core.post_processor_factory import PostProcessorFactory

# Import the corrected unified evaluator
from src.eval.unified_evaluator import eval_single_agency_level

# Import the unified Redis embedding manager
from src.eval.redis_embedding_manager import UnifiedEmbeddingManager

from pydantic import BaseModel, Field, field_validator
from src.utils.logger import logger
import time
import uuid
import asyncio
import re
import os
import json

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error(
        "SentenceTransformers not available. Please install: pip install sentence-transformers"
    )


router = APIRouter()

# Store for tracking background task status
task_status_store = {}


# Global unified embedding manager instance
unified_embedding_manager = None
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
logger.info("Initialized SentenceTransformer model for unified embedding manager")


def get_embedding_manager(request: Request) -> UnifiedEmbeddingManager:
    """Get or create unified embedding manager instance."""
    global unified_embedding_manager
    if unified_embedding_manager is None:
        # Get configuration from request
        config = request.app.state.config

        # Extract configuration
        model_name = config.get("models", {}).get(
            "embedding_model", "paraphrase-multilingual-mpnet-base-v2"
        )
        redis_url = config.get("redis", {}).get(
            "url", os.environ.get("REDIS_URL", "redis://localhost:6379")
        )
        topic_documents_path = config.get("embedding", {}).get(
            "topic_documents_path", "topic_documents"
        )

        unified_embedding_manager = UnifiedEmbeddingManager(
            model=model,
            model_name=model_name,
            redis_url=redis_url,
            topic_documents_path=topic_documents_path,
        )
        logger.info("Initialized unified embedding manager")

    return unified_embedding_manager


def get_embedding_manager_from_config(config: dict) -> UnifiedEmbeddingManager:
    """Get or create unified embedding manager from config dict."""
    global unified_embedding_manager
    if unified_embedding_manager is None:
        # Extract configuration
        model_name = config.get("models", {}).get(
            "embedding_model", "paraphrase-multilingual-mpnet-base-v2"
        )
        redis_url = config.get("redis", {}).get(
            "url", os.environ.get("REDIS_URL", "redis://localhost:6379")
        )
        topic_documents_path = config.get("embedding", {}).get(
            "topic_documents_path", "topic_documents"
        )

        unified_embedding_manager = UnifiedEmbeddingManager(
            model_name=model_name,
            redis_url=redis_url,
            topic_documents_path=topic_documents_path,
        )
        logger.info("Initialized unified embedding manager from config")

    return unified_embedding_manager


class DatasetRequest(BaseModel):
    """
    This schema defines the parameters required to generate a synthetic dataset from a specific data source.
    It is used as part of bulk dataset generation requests and supports validation of input fields.
    """

    data_path: str = Field(..., description="Path to the data source")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    version_id: Optional[str] = Field(
        "v1", description="Version identifier for the dataset"
    )
    session_id: Optional[int] = Field(
        None, description="Session ID for tracking dataset generation sessions"
    )

    class Config:
        extra = "allow"  # Allow additional fields like agency_id, etc.

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v):
        if not v or not v.strip():
            raise ValueError("data_path cannot be empty")
        if ".." in v:
            raise ValueError("data_path contains invalid characters")
        return v.strip()

    @field_validator("output_filename")
    @classmethod
    def validate_output_filename(cls, v):
        if v is not None:
            if not v or not v.strip():
                raise ValueError("output_filename cannot be empty if provided")

            if not re.match(r"^[a-zA-Z0-9_-]+$", v.strip()):
                raise ValueError("output_filename contains invalid characters")
        return v.strip() if v else None

    @field_validator("version_id")
    @classmethod
    def validate_version_id(cls, v):
        if v is not None:
            if not v or not v.strip():
                raise ValueError("version_id cannot be empty if provided")
            if not re.match(r"^[a-zA-Z0-9._-]+$", v.strip()):
                raise ValueError("version_id contains invalid characters")
        return v.strip() if v else "v1"


class BulkGenerateRequest(BaseModel):
    datasets: List[DatasetRequest] = Field(
        ..., description="List of datasets to generate"
    )


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    message: str


def get_data_generator(request: Request) -> DataGenerator:
    """
    Get a DataGenerator instance initialized with the application configuration.

    This function serves as a FastAPI dependency that creates and returns a properly
    configured DataGenerator instance using the application's configuration stored
    in request.app.state.config.

    Args:
        request (Request): The FastAPI request object containing application state

    Returns:
        DataGenerator: An initialized data generator instance

    Raises:
        HTTPException: 500 error if application configuration is not found

    Example:
        @router.post("/generate")
        async def generate_dataset(
            request_data: SomeModel,
            data_generator: DataGenerator = Depends(get_data_generator)
        ):
            result = data_generator.generate(...)
            return {"status": "success", "data": result}
    """
    if not hasattr(request.app.state, "config"):
        raise HTTPException(
            status_code=500, detail="Application configuration not found"
        )
    return DataGenerator(request.app.state.config)


async def send_callback(
    callback_url: str, payload: dict, max_retries: int = 3, timeout: float = 30.0
):
    """Send callback notification to external system with enhanced error handling"""
    if not callback_url:
        logger.info("No callback URL configured, skipping callback")
        return {"success": False, "reason": "no_url_configured"}

    callback_start_time = time.time()
    last_error = None

    for attempt in range(max_retries):
        try:
            # Calculate remaining timeout
            elapsed = time.time() - callback_start_time
            remaining_timeout = max(5.0, timeout - elapsed)  # Minimum 5 seconds

            if remaining_timeout <= 0:
                logger.error(f"Callback timeout exceeded before attempt {attempt + 1}")
                return {
                    "success": False,
                    "reason": "timeout_exceeded",
                    "attempts": attempt,
                    "last_error": "Timeout exceeded before all retry attempts",
                }

            async with httpx.AsyncClient(timeout=remaining_timeout) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()

                logger.info(
                    f"Callback sent successfully to {callback_url} on attempt {attempt + 1}"
                )
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "response_status": response.status_code,
                }

        except httpx.TimeoutException as e:
            last_error = f"Timeout error: {str(e)}"
            logger.warning(f"Callback attempt {attempt + 1} timed out: {str(e)}")
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP error {e.response.status_code}: {str(e)}"
            logger.warning(
                f"Callback attempt {attempt + 1} failed with HTTP error: {str(e)}"
            )
        except httpx.RequestError as e:
            last_error = f"Request error: {str(e)}"
            logger.warning(
                f"Callback attempt {attempt + 1} failed with request error: {str(e)}"
            )
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            logger.warning(
                f"Callback attempt {attempt + 1} failed with unexpected error: {str(e)}"
            )

        # Exponential backoff with jitter, but don't wait on the last attempt
        if attempt < max_retries - 1:
            backoff_time = min(
                30.0, (2**attempt) + (time.time() % 1)
            )  # Max 30 seconds with jitter
            logger.info(
                f"Waiting {backoff_time:.1f} seconds before retry {attempt + 2}"
            )
            await asyncio.sleep(backoff_time)

    logger.error(
        f"All {max_retries} callback attempts failed for {callback_url}. Last error: {last_error}"
    )
    return {
        "success": False,
        "reason": "all_attempts_failed",
        "attempts": max_retries,
        "last_error": last_error,
    }


async def process_single_dataset(
    dataset_request: DatasetRequest,
    config: dict,
    data_generator: DataGenerator,
    task_id: str,
    agency_name: str,
) -> dict:
    """
    Process a single dataset generation request with Redis-based embedding caching.

    This function handles the end-to-end logic for generating a synthetic dataset based on the parameters
    provided in a DatasetRequest. It now includes Redis-based embedding management for better performance
    and proper cleanup of temporary embeddings based on evaluation results.

    Args:
        dataset_request (DatasetRequest): The dataset generation request containing source path, output filename, and other parameters.
        config (dict): The full configuration dictionary, including dataset generation and directory settings.
        data_generator (DataGenerator): The main generator instance used to create synthetic datasets.
        task_id (str): The unique identifier for the current bulk generation task.
        agency_name (str): Name of the agency being processed.

    Returns:
        dict: A dictionary containing the result of the dataset generation, including success status, output paths,
              any errors encountered, and metadata about the dataset and configuration used.

    Notes:
        - Now integrates with Redis for embedding management
        - Automatically cleans up temporary embeddings after successful evaluation
        - Keeps topic embeddings cached for future use
        - Supports both individual and bulk post-processing modes (e.g., zip, aggregation).
        - Handles validation and error reporting for missing data sources or configuration issues.
        - Used internally by the bulk dataset generation API endpoint.
    """
    error_details = {
        "category": None,
        "stage": None,
        "source_path": dataset_request.data_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recoverable": False,
    }

    evaluation_session_id = None
    max_regeneration_attempts = config.get("evaluation", {}).get(
        "max_regeneration_attempts", 3
    )
    logger.info(dataset_request)
    try:
        start_time = time.time()
        timeout_seconds = config.get("processing", {}).get(
            "timeout_seconds", 3600
        )  # 1 hour default

        error_details["stage"] = "configuration_loading"
        dataset_config = config.get("dataset_generation", {})

        # Extract all parameters from config
        structure_name = dataset_config.get("structure_name")
        prompt_template_name = dataset_config.get("prompt_template_name")
        traversal_strategy = dataset_config.get("traversal_strategy")
        data_source_config = config.get("data_sources", {}).get("default", {})
        num_examples = dataset_config.get("num_samples")
        output_format = dataset_config.get("output_format")
        parameters = dataset_config.get("parameters", {})
        filter_config = dataset_config.get("filter", {})
        post_processing_type = dataset_config.get("post_processing", "zip")

        # Validate required parameters
        if not structure_name or not prompt_template_name:
            error_details.update(
                {
                    "category": "configuration_error",
                    "stage": "validation",
                    "recoverable": True,
                    "details": f"Missing required configuration: structure_name={structure_name}, prompt_template_name={prompt_template_name}",
                }
            )
            return {
                "success": False,
                "error": "Missing required configuration parameters",
                "error_details": error_details,
                "dataset_metadata": dataset_request.dict(),
            }

        error_details["stage"] = "data_source_loading"

        # Override output_filename if provided in request
        modified_config = config.copy()
        if dataset_request.output_filename:
            modified_config["dataset_generation"] = dataset_config.copy()

            if "aggregation" not in modified_config["dataset_generation"]:
                modified_config["dataset_generation"]["aggregation"] = {}
            else:
                modified_config["dataset_generation"]["aggregation"] = (
                    dataset_config.get("aggregation", {}).copy()
                )

            modified_config["dataset_generation"]["aggregation"]["output_filename"] = (
                dataset_request.output_filename
            )

        data_path = dataset_request.data_path
        output_dir = config.get("directories", {}).get("output")

        # Create data source manager
        source_manager = DataSourceManager(config=data_source_config)

        # Load all matching sources with timeout protection
        if time.time() - start_time > timeout_seconds:
            error_details.update(
                {
                    "category": "timeout_error",
                    "stage": "data_source_loading",
                    "recoverable": False,
                    "details": f"Operation timed out after {timeout_seconds} seconds",
                }
            )
            return {
                "success": False,
                "error": f"Processing timeout after {timeout_seconds} seconds",
                "error_details": error_details,
                "dataset_metadata": dataset_request.dict(),
            }

        data_sources = source_manager.load_sources(
            base_path=data_path,
            strategy_name=traversal_strategy,
            filter_config=filter_config,
        )

        if not data_sources:
            error_details.update(
                {
                    "category": "data_error",
                    "stage": "data_source_loading",
                    "recoverable": True,
                    "details": f"No matching data sources found in {data_path} with strategy {traversal_strategy}",
                }
            )
            return {
                "success": False,
                "error": f"No data sources found matching the criteria in {data_path}",
                "error_details": error_details,
                "dataset_metadata": dataset_request.dict(),
            }

        logger.info(
            f"Found {len(data_sources)} data sources to process for {data_path}"
        )

        # Regeneration loop - generate and evaluate until thresholds are met or max attempts reached
        attempt = 0
        final_results = None
        final_evaluation_results = None
        best_evaluation = None
        best_results = None
        while attempt < max_regeneration_attempts:
            attempt += 1
            logger.info(
                f"Generation attempt {attempt}/{max_regeneration_attempts} for {data_path}"
            )

            error_details["stage"] = "dataset_generation"
            # Track all output paths for post-processing
            all_output_paths = []
            results = []
            generation_errors = []
            for source in data_sources:
                # Check timeout before processing each source
                if time.time() - start_time > timeout_seconds:
                    error_details.update(
                        {
                            "category": "timeout_error",
                            "stage": "dataset_generation",
                            "recoverable": False,
                            "details": f"Operation timed out after {timeout_seconds} seconds during source processing",
                        }
                    )
                    return {
                        "success": False,
                        "error": f"Processing timeout after {timeout_seconds} seconds",
                        "error_details": error_details,
                        "dataset_metadata": dataset_request.dict(),
                        "partial_results": results,
                    }

                try:
                    # Extract metadata for parameters
                    source_params = parameters.copy()

                    # Add source metadata to parameters
                    if traversal_strategy == "institutional":
                        source_params["institution"] = source.metadata.get(
                            "institution", "unknown"
                        )
                        source_params["topic"] = source.metadata.get("topic", "unknown")
                        source_params["topic_content"] = source.content
                    else:
                        source_params["file_path"] = source.path
                        source_params["file_content"] = source.content
                        source_params["file_name"] = source.name

                        for key, value in source.metadata.items():
                            if key not in source_params:
                                source_params[key] = value

                    # Determine output path based on metadata
                    if traversal_strategy == "institutional":
                        topic = source.metadata.get("topic", "unknown")
                        output_base_path = (
                            f"{output_dir}/{structure_name}/{agency_name}/{topic}"
                        )
                    else:
                        rel_path = source.metadata.get("relative_path", source.name)
                        output_base_path = f"{output_dir}/{agency_name}/{rel_path}"

                    logger.info(
                        f"Generating dataset for source: {source.path} -> {output_base_path}"
                    )

                    # Generate dataset
                    result_path = data_generator.generate(
                        structure_name=structure_name,
                        prompt_template_name=prompt_template_name,
                        output_base_path=output_base_path,
                        num_examples=num_examples,
                        output_format=output_format,
                        parameters=source_params,
                    )

                    all_output_paths.append(result_path)
                    results.append({"source": source.path, "output_path": result_path})

                except Exception as source_error:
                    error_msg = f"Failed to generate dataset for source {source.path}: {str(source_error)}"
                    logger.error(error_msg)
                    generation_errors.append(
                        {
                            "source": source.path,
                            "error": str(source_error),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    # Continue processing other sources instead of failing completely

            # Check if we have any successful results
            if not all_output_paths and generation_errors:
                error_details.update(
                    {
                        "category": "generation_error",
                        "stage": "dataset_generation",
                        "recoverable": False,
                        "details": f"All {len(generation_errors)} sources failed to generate",
                        "source_errors": generation_errors,
                    }
                )
                return {
                    "success": False,
                    "error": f"All {len(generation_errors)} data sources failed to generate datasets",
                    "error_details": error_details,
                    "dataset_metadata": dataset_request.dict(),
                    "generation_errors": generation_errors,
                }
            logger.info(results)

            # === UNIFIED EMBEDDING MANAGEMENT FOR EVALUATION ===
            error_details["stage"] = "embedding_preloading"

            try:
                # Extract unique agencies and topics from results for embedding preloading
                agencies_for_evaluation = []
                topics_for_evaluation = []
                valid_pairs = []
                for result in results:
                    source_path = result["source"]
                    source_parts = source_path.strip("/").split("/")
                    if len(source_parts) >= 4:
                        agency = source_parts[-3]  # sm_someuuid
                        topic = source_parts[-2]    # d934abece3ce5ea3ceaa55e41f3cfe0eb7ea6f97
                        pair = (agency, topic)
                        if pair not in valid_pairs:
                            valid_pairs.append(pair)
                    
                    

                
                logger.info(
                    f"Preloading embeddings for evaluation: {agencies_for_evaluation} agencies, {topics_for_evaluation} topics"
                )

                # Get unified embedding manager
                embedding_manager = get_embedding_manager_from_config(config)

                topic_embeddings_by_context = embedding_manager.get_embeddings_for_pairs(
                valid_pairs, 
                results=results  # Pass results for content extraction
            )


                preload_success = len(topic_embeddings_by_context) > 0

                if not preload_success:
                    logger.warning(
                        "Failed to preload topic embeddings, evaluation may fail"
                    )
                else:
                    logger.info(
                        f"Successfully preloaded topic embeddings for {len(topic_embeddings_by_context)} contexts"
                    )

                # Pre-generate and cache question embeddings
                logger.info("Generating question embeddings for evaluation")
                questions_by_context = {}

                for result in results:
                    source_path = result["source"]
                    output_path = result["output_path"]
                    # Try to read the generated FAQs to get questions
                    source_parts = source_path.strip("/").split("/")
                    if len(source_parts) >=4: 
                        result_agency = source_parts[-3]  # sm_someuuid
                        result_topic = source_parts[-2]    # d934abece3ceaa55e41f3cfe0eb7ea6f97 
                        
                        faqs_path = os.path.join(output_path, "faqs.json")

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
                                        context_key = (result_agency, result_topic)
                                        if context_key not in questions_by_context:
                                            questions_by_context[context_key] = []
                                        questions_by_context[context_key].extend(
                                            questions
                                        )
                                    
                                        logger.info(
                                        f"Extracted {len(questions)} questions for context {context_key}"
                                        )

                                
                            except Exception as faq_error:
                                logger.warning(
                                    f"Failed to read FAQs from {faqs_path}: {faq_error}"
                                )
                    else:
                        logger.warning(f"No FAQs found at {faqs_path}")
                # Generate and cache question embeddings
                if questions_by_context:
                    try:
                        evaluation_session_id = (
                            embedding_manager.start_question_session()
                        )
                        question_embeddings_by_context = (
                            embedding_manager.generate_and_cache_question_embeddings(
                                evaluation_session_id, questions_by_context
                            )
                        )
                        logger.info(
                            f"Generated and cached question embeddings for {len(question_embeddings_by_context)} contexts"
                        )
                    except Exception as q_error:
                        logger.warning(
                            f"Failed to generate question embeddings: {q_error}"
                        )
                        evaluation_session_id = None
                else:
                    logger.warning("No questions found to generate embeddings")

            except Exception as embedding_error:
                logger.warning(f"Error during embedding management: {embedding_error}")
                preload_success = False
                evaluation_session_id = None

            # === EVALUATION WITH SMART REDIS INTEGRATION ===
            error_details["stage"] = "evaluation"

            try:
                # Use smart evaluation with the embedding manager and session ID
                embedding_manager = get_embedding_manager_from_config(config)
                evaluation_results = eval_single_agency_level(
                    results=results,
                    embedding_manager=embedding_manager,
                    session_id=evaluation_session_id,
                )
                logger.info(f"Smart evaluation results: {evaluation_results}")

                # Check if evaluation passed
                should_regenerate = evaluation_results.get("should_regenerate", True)

                # Keep track of the best evaluation so far
                if best_evaluation is None or evaluation_results.get(
                    "overall_score", 0
                ) > best_evaluation.get("overall_score", 0):
                    best_evaluation = evaluation_results
                    best_results = results

                if not should_regenerate:
                    logger.info(
                        f"Evaluation passed on attempt {attempt}, stopping regeneration"
                    )
                    final_results = results
                    final_evaluation_results = evaluation_results
                    break
                else:
                    logger.info(
                        f"Evaluation failed on attempt {attempt}, overall_score: {evaluation_results.get('overall_score', 0)}"
                    )

            except Exception as eval_error:
                logger.error(f"Smart evaluation failed: {eval_error}")
                # This should not happen as smart eval has built-in fallback
                evaluation_results = {
                    "decision": "REGENERATE",
                    "should_regenerate": True,
                    "error": str(eval_error),
                    "overall_score": 0.0,
                }

                # Keep track of the best evaluation so far (even failed ones)
                if best_evaluation is None or evaluation_results.get(
                    "overall_score", 0
                ) > best_evaluation.get("overall_score", 0):
                    best_evaluation = evaluation_results
                    best_results = results

            # Clean up temporary embeddings if we're going to regenerate
            if (
                evaluation_session_id
                and should_regenerate
                and attempt < max_regeneration_attempts
            ):
                try:
                    embedding_manager.cleanup_question_session(evaluation_session_id)
                    logger.info(
                        f"Cleaned up temporary embeddings after attempt {attempt}"
                    )
                except Exception as cleanup_error:
                    logger.warning(f"Error during embedding cleanup: {cleanup_error}")

        # If we didn't find a passing evaluation, use the best one we found
        if final_results is None:
            logger.info(
                f"Max regeneration attempts ({max_regeneration_attempts}) reached, using best result"
            )
            final_results = best_results
            final_evaluation_results = best_evaluation

        final_output_paths = []
        if final_results:
            for result in final_results:
                final_output_paths.append(result["output_path"])
        

        error_details["stage"] = "post_processing"
        # MODIFIED: Post-processing logic to handle cross-dataset aggregation
        final_output_path = None

        # For aggregation mode, skip individual post-processing - will be handled at bulk level
        if post_processing_type == "aggregation":
            logger.info(
                f"Skipping individual aggregation for {dataset_request.data_path} - will be handled at bulk level for cross-dataset aggregation"
            )
            # Individual files will be collected for final cross-dataset aggregation

            result = {
                "success": True,
                "dataset_metadata": dataset_request.model_dump(),
                "metrics": final_evaluation_results,
                "post_processing_type": post_processing_type,
                "final_output_path": None,  # Will be set after cross-dataset aggregation
                "_internal_results": final_results,  # Keep for internal aggregation logic (NOT exposed in callback)
                "configuration_used": {
                    "structure_name": structure_name,
                    "prompt_template_name": prompt_template_name,
                    "traversal_strategy": traversal_strategy,
                    "num_examples": num_examples,
                    "output_format": output_format,
                    "post_processing": post_processing_type,
                },
                "embedding_session_id": evaluation_session_id,  # Track for potential cleanup
                "embedding_preload_success": preload_success
                if "preload_success" in locals()
                else False,
                "regeneration_attempts": attempt,
                "max_regeneration_attempts": max_regeneration_attempts,
            }

            # Add error information if there were partial failures
            if generation_errors:
                result.update(
                    {
                        "generation_errors": generation_errors,
                        "sources_processed": len(data_sources),
                        "sources_successful": len(final_results),
                        "sources_failed": len(generation_errors),
                    }
                )

            return result
        else:
            # For non-aggregation modes (like zip), perform individual post-processing
            logger.info(
                f"Performing individual {post_processing_type} post-processing for {dataset_request.data_path}"
            )

            try:
                base_output_dir = f"{output_dir}"
                post_processor = PostProcessorFactory.create_post_processor(
                    modified_config
                )
                final_output_path = post_processor.process(
                    all_output_paths, base_output_dir
                )
            except Exception as post_error:
                error_details.update(
                    {
                        "category": "post_processing_error",
                        "stage": "post_processing",
                        "recoverable": True,
                        "details": f"Post-processing failed: {str(post_error)}",
                    }
                )
                result = {
                    "success": False,
                    "error": f"Post-processing failed: {str(post_error)}",
                    "error_details": error_details,
                    "dataset_metadata": dataset_request.dict(),
                    "_internal_results": final_results,
                    "embedding_session_id": evaluation_session_id,
                    "regeneration_attempts": attempt,
                    "max_regeneration_attempts": max_regeneration_attempts,
                }
                if generation_errors:
                    result["generation_errors"] = generation_errors
                return result

            if final_output_path:
                result = {
                    "success": True,
                    "dataset_metadata": dataset_request.dict(),
                    "metrics": final_evaluation_results,
                    "post_processing_type": post_processing_type,
                    "final_output_path": final_output_path,
                    "_internal_results": final_results,
                    "configuration_used": {
                        "structure_name": structure_name,
                        "prompt_template_name": prompt_template_name,
                        "traversal_strategy": traversal_strategy,
                        "num_examples": num_examples,
                        "output_format": output_format,
                        "post_processing": post_processing_type,
                    },
                    "embedding_session_id": evaluation_session_id,
                    "embedding_preload_success": preload_success
                    if "preload_success" in locals()
                    else False,
                    "regeneration_attempts": attempt,
                    "max_regeneration_attempts": max_regeneration_attempts,
                }

                # Add error information if there were partial failures
                if generation_errors:
                    result.update(
                        {
                            "generation_errors": generation_errors,
                            "sources_processed": len(data_sources),
                            "sources_successful": len(final_results),
                            "sources_failed": len(generation_errors),
                        }
                    )

                return result
            else:
                error_details.update(
                    {
                        "category": "post_processing_error",
                        "stage": "post_processing",
                        "recoverable": True,
                        "details": "Post-processing completed but no output path was returned",
                    }
                )
                result = {
                    "success": False,
                    "error": "Post-processing failed",
                    "error_details": error_details,
                    "dataset_metadata": dataset_request.dict(),
                    "_internal_results": final_results,
                    "embedding_session_id": evaluation_session_id,
                    "regeneration_attempts": attempt,
                    "max_regeneration_attempts": max_regeneration_attempts,
                }
                if generation_errors:
                    result["generation_errors"] = generation_errors
                return result

    except Exception as e:
        # Categorize the error based on its type and stage
        if "timeout" in str(e).lower():
            error_details.update(
                {
                    "category": "timeout_error",
                    "recoverable": False,
                    "details": f"Operation timed out: {str(e)}",
                }
            )
        elif "permission" in str(e).lower() or "access" in str(e).lower():
            error_details.update(
                {
                    "category": "permission_error",
                    "recoverable": True,
                    "details": f"Access/permission error: {str(e)}",
                }
            )
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            error_details.update(
                {
                    "category": "network_error",
                    "recoverable": True,
                    "details": f"Network/connection error: {str(e)}",
                }
            )
        else:
            error_details.update(
                {
                    "category": "unknown_error",
                    "recoverable": False,
                    "details": f"Unexpected error: {str(e)}",
                }
            )

        logger.exception(
            f"Error processing dataset {dataset_request.data_path}: {str(e)}"
        )

        # Clean up evaluation session on error
        if evaluation_session_id:
            try:
                embedding_manager = get_embedding_manager_from_config(config)
                embedding_manager.cleanup_question_session(evaluation_session_id)
            except Exception as cleanup_error:
                logger.warning(
                    f"Error cleaning up embeddings after failure: {cleanup_error}"
                )

        return {
            "success": False,
            "error": str(e),
            "error_details": error_details,
            "dataset_metadata": dataset_request.dict(),
            "embedding_session_id": evaluation_session_id,
        }


async def background_generate_bulk(
    task_id: str,
    request_data: BulkGenerateRequest,
    config: dict,
    data_generator: DataGenerator,
):
    """
    Background task for processing bulk dataset generation requests with Redis-based embedding management.

    This function is executed as a background task to generate multiple datasets in parallel, based on the list of
    DatasetRequest objects provided in the BulkGenerateRequest. It now includes Redis-based embedding management
    for better performance and memory management across datasets.

    Args:
        task_id (str): Unique identifier for this bulk generation task, used for status tracking.
        request_data (BulkGenerateRequest): The bulk request containing a list of dataset generation parameters.
        config (dict): Application configuration, including dataset generation and post-processing settings.
        data_generator (DataGenerator): Instance responsible for generating datasets from the provided parameters.

    Raises:
        Exception: Any error during processing is logged and updates the task status as failed.

    Notes:
        - Each dataset in the request is processed sequentially.
        - Uses Redis for embedding caching and management.
        - Automatically cleans up temporary embeddings based on evaluation results.
        - Supports cross-dataset aggregation or zipping if configured.
        - Internal results and metadata are managed for advanced post-processing.
    """
    try:
        # Process each dataset - COLLECT all output paths across ALL datasets
        all_results = []
        all_cross_dataset_output_paths = []
        all_dataset_metadata = []  # NEW: Store metadata for each dataset
        successful_count = 0
        failed_count = 0
        common_output_filename = None
        agency_name = None  # Initialize agency_id to None
        session_id = None  # Initialize session_id to None

        for i, dataset_request in enumerate(request_data.datasets):
            logger.info(
                f"Processing dataset {i + 1}/{len(request_data.datasets)}: {dataset_request.data_path}"
            )

            # Store the output_filename from first dataset for final aggregation
            if common_output_filename is None and dataset_request.output_filename:
                common_output_filename = dataset_request.output_filename

            # Store session_id from the first dataset (assuming all datasets in bulk have same session_id)
            if (
                session_id is None
                and hasattr(dataset_request, "session_id")
                and dataset_request.session_id
            ):
                session_id = dataset_request.session_id

            agency_name = (
                dataset_request.agency_name
                if hasattr(dataset_request, "agency_name")
                else None
            )

            result = await process_single_dataset(
                dataset_request, config, data_generator, task_id, agency_name
            )
            all_results.append(result)

            if result["success"]:
                successful_count += 1
                # Collect individual output paths using _internal_results
                if "_internal_results" in result:
                    for individual_result in result["_internal_results"]:
                        output_path = individual_result["output_path"]

                        if os.path.isdir(output_path):
                            # Look for faqs.json in the directory
                            json_file_path = os.path.join(output_path, "faqs.json")
                            if os.path.exists(json_file_path):
                                all_cross_dataset_output_paths.append(json_file_path)
                            else:
                                # Fallback: search for any JSON file
                                for root, dirs, files in os.walk(output_path):
                                    for file in files:
                                        if file.endswith(".json"):
                                            json_file_path = os.path.join(root, file)
                                            all_cross_dataset_output_paths.append(
                                                json_file_path
                                            )
                                            break
                        else:
                            # It's already a file
                            all_cross_dataset_output_paths.append(output_path)

                        # Extract metadata from dataset_request
                        metadata = (
                            dataset_request.model_dump()
                        )  # Use model_dump() for Pydantic v2
                        all_dataset_metadata.append(metadata)
            else:
                failed_count += 1

            # Update progress
            task_status_store[task_id]["completed_datasets"] = i + 1
            task_status_store[task_id]["results"] = all_results

        # Enhanced cross-dataset aggregation with metadata
        final_aggregated_path = None
        if (
            all_cross_dataset_output_paths
            and common_output_filename
            and config.get("dataset_generation", {}).get("post_processing")
            == "aggregation"
        ):
            logger.info(
                f"Performing cross-dataset aggregation for {len(all_cross_dataset_output_paths)} files"
            )

            # Create config for final aggregation with common output filename
            final_aggregation_config = config.copy()
            final_aggregation_config["dataset_generation"] = config.get(
                "dataset_generation", {}
            ).copy()

            if "aggregation" not in final_aggregation_config["dataset_generation"]:
                final_aggregation_config["dataset_generation"]["aggregation"] = {}
            else:
                final_aggregation_config["dataset_generation"]["aggregation"] = (
                    config.get("dataset_generation", {}).get("aggregation", {}).copy()
                )

            # Override with common output filename
            final_aggregation_config["dataset_generation"]["aggregation"][
                "output_filename"
            ] = common_output_filename

            # Perform final aggregation with metadata
            output_dir = config.get("directories", {}).get("output")
            base_output_dir = f"{output_dir}"

            final_post_processor = PostProcessorFactory.create_post_processor(
                final_aggregation_config
            )
            # Pass dataset metadata to the processor
            final_aggregated_path = final_post_processor.process(
                all_cross_dataset_output_paths, base_output_dir, all_dataset_metadata
            )

            logger.info(f"Cross-dataset aggregation completed: {final_aggregated_path}")

        # Final status update
        final_status = (
            "completed"
            if failed_count == 0
            else ("partial_success" if successful_count > 0 else "failed")
        )

        task_status_store[task_id].update(
            {
                "status": final_status,
                "message": f"Processing completed. {successful_count} successful, {failed_count} failed",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "successful_count": successful_count,
                "failed_count": failed_count,
                "results": all_results,
            }
        )

        logger.info(
            f"Background generation completed for task {task_id}: {successful_count} successful, {failed_count} failed"
        )

        # Send callback if configured
        callback_config = config.get("callback", {})
        callback_url = callback_config.get("url")
        callback_timeout = callback_config.get("timeout", 60.0)
        callback_retries = callback_config.get("retries", 3)

        if callback_url:
            # Clean up internal results before sending callback and add error summaries
            cleaned_results = []
            total_generation_errors = 0
            error_categories = {}

            for result in all_results:
                cleaned_result = result.copy()
                if "_internal_results" in cleaned_result:
                    del cleaned_result["_internal_results"]

                # Aggregate error information
                if "generation_errors" in result:
                    total_generation_errors += len(result["generation_errors"])

                if "error_details" in result:
                    category = result["error_details"].get("category", "unknown")
                    error_categories[category] = error_categories.get(category, 0) + 1

                cleaned_results.append(cleaned_result)

            # Enhanced callback payload with detailed error information
            callback_payload = {
                "task_id": session_id,
                "status": final_status,
                "message": task_status_store[task_id]["message"],
                "filePath": final_aggregated_path,
                "results": cleaned_results,
                "summary": {
                    "total_datasets": len(request_data.datasets),
                    "successful_datasets": successful_count,
                    "failed_datasets": failed_count,
                    "total_generation_errors": total_generation_errors,
                    "error_categories": error_categories,
                    "processing_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "has_partial_failures": total_generation_errors > 0
                    and successful_count > 0,
                },
            }

            if session_id is not None:
                callback_payload["session_id"] = session_id

            logger.info(
                f"Sending callback to {callback_url} for task {task_id} with {successful_count}/{len(request_data.datasets)} successful datasets"
            )

            # Send callback with enhanced error handling
            callback_result = await send_callback(
                callback_url, callback_payload, callback_retries, callback_timeout
            )

            if callback_result["success"]:
                logger.info(
                    f"Callback sent successfully for task {task_id} after {callback_result['attempts']} attempts"
                )
            else:
                logger.error(
                    f"Callback failed for task {task_id}: {callback_result['reason']} - {callback_result.get('last_error', 'Unknown error')}"
                )
                # Update task status to indicate callback failure
                task_status_store[task_id]["callback_status"] = "failed"
                task_status_store[task_id]["callback_error"] = callback_result.get(
                    "last_error", "Unknown callback error"
                )

            logger.info("Generation task completed")

    except Exception as e:
        logger.exception(f"Error in background generation for task {task_id}: {str(e)}")

        # Categorize the exception
        error_category = "unknown_error"
        if "timeout" in str(e).lower():
            error_category = "timeout_error"
        elif "memory" in str(e).lower() or "out of memory" in str(e).lower():
            error_category = "memory_error"
        elif "disk" in str(e).lower() or "space" in str(e).lower():
            error_category = "disk_error"
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            error_category = "network_error"

        task_status_store[task_id] = {
            "status": "failed",
            "message": f"Error generating bulk datasets: {str(e)}",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [],
            "error": str(e),
            "error_category": error_category,
        }

        # Send failure callback if configured
        callback_config = config.get("callback", {})
        callback_url = callback_config.get("url")

        if callback_url:
            failure_payload = {
                "task_id": task_id,
                "status": "failed",
                "message": f"Bulk generation failed: {str(e)}",
                "error": str(e),
                "error_category": error_category,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_datasets": len(request_data.datasets)
                    if "request_data" in locals()
                    else 0,
                    "successful_datasets": 0,
                    "failed_datasets": len(request_data.datasets)
                    if "request_data" in locals()
                    else 0,
                    "critical_failure": True,
                },
            }

            try:
                await send_callback(callback_url, failure_payload)
                logger.info(f"Failure callback sent for task {task_id}")
            except Exception as callback_error:
                logger.error(
                    f"Failed to send failure callback for task {task_id}: {str(callback_error)}"
                )


@router.post("/generate-bulk", response_model=GenerateResponse)
async def generate_bulk_dataset(
    request: BulkGenerateRequest,
    background_tasks: BackgroundTasks,
    fastapi_request: Request,
    data_generator: DataGenerator = Depends(get_data_generator),
):
    """
    Generate multiple datasets in bulk as a background task with Redis-based embedding management.

    Accepts a list of datasets with flexible metadata and returns immediately with a task_id.
    Now includes Redis-based embedding caching for improved performance and memory management.
    Supports callback notifications when processing is complete.
    """
    logger.info(
        "Received bulk dataset generation request with Redis embedding management"
    )
    try:
        logger.info(
            f"Received bulk generation request for {len(request.datasets)} datasets"
        )

        # Validate input
        if not request.datasets:
            raise HTTPException(
                status_code=400, detail="At least one dataset must be provided"
            )

        # Check Redis health before starting
        if not get_embedding_manager(fastapi_request).health_check():
            logger.error("Redis connection unhealthy")
            raise HTTPException(
                status_code=503, detail="Redis embedding cache unavailable"
            )

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Get configuration from app state
        config = fastapi_request.app.state.config
        logger.info(f"Loaded Configurations: {config}")

        # Initialize task status
        task_status_store[task_id] = {
            "status": "queued",
            "message": "Task queued for processing",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_datasets": len(request.datasets),
            "completed_datasets": 0,
            "results": [],
            "error": None,
            "redis_enabled": True,
        }

        # Add background task with unified embedding manager
        background_tasks.add_task(
            background_generate_bulk, task_id, request, config, data_generator
        )

        logger.info(f"Background task queued with ID: {task_id}")

        return GenerateResponse(
            task_id=task_id,
            status="accepted",
            message=f"Dataset generation task has been queued for {len(request.datasets)} datasets with unified embedding management. Use the task_id to check status.",
        )

    except Exception as e:
        logger.exception(f"Error queuing bulk generation task: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error queuing bulk generation task: {str(e)}"
        )


@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_status_store[task_id]


@router.get("/tasks")
async def list_tasks():
    """List all tasks and their statuses"""
    return {
        "tasks": [
            {"task_id": task_id, **status}
            for task_id, status in task_status_store.items()
        ]
    }


@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed task from the status store"""
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail="Task not found")

    task_status = task_status_store[task_id]["status"]
    if task_status in ["running", "queued"]:
        raise HTTPException(
            status_code=400, detail="Cannot delete running or queued tasks"
        )

    del task_status_store[task_id]
    return {"message": "Task deleted successfully"}


# === UNIFIED EMBEDDING MANAGEMENT ENDPOINTS ===


@router.get("/embeddings/cache/stats")
async def get_embedding_cache_stats(
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Get comprehensive embedding cache statistics"""
    try:
        stats = embedding_manager.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats,
            "redis_healthy": embedding_manager.health_check(),
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting cache stats: {str(e)}"
        )


@router.post("/embeddings/cache/preload")
async def preload_embeddings(
    agencies: List[str],
    topics: List[str],
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Manually preload topic embeddings for specified agencies and topics"""
    try:
        logger.info(
            f"Manual preload request for {agencies} agencies and {topics} topics"
        )

        # Use bulk loading for efficiency
        embeddings_by_context = embedding_manager.get_bulk_topic_embeddings(
            agencies, topics
        )

        if embeddings_by_context:
            return {
                "status": "success",
                "message": f"Successfully preloaded embeddings for {len(embeddings_by_context)} agency-topic combinations",
                "agencies": agencies,
                "topics": topics,
                "combinations_loaded": len(embeddings_by_context),
                "combinations": [f"{a}-{t}" for a, t in embeddings_by_context.keys()],
            }
        else:
            return {
                "status": "error",
                "message": "Failed to preload embeddings - no combinations found",
                "agencies": agencies,
                "topics": topics,
            }

    except Exception as e:
        logger.error(f"Error preloading embeddings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error preloading embeddings: {str(e)}"
        )


@router.get("/embeddings/cache/check")
async def check_embeddings_exist(
    agency: str,
    topic: str,
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Check if embeddings exist for specific agency-topic combination"""
    try:
        exists = embedding_manager.has_topic_embeddings(agency, topic)
        embeddings_count = 0

        if exists:
            cached_embeddings = embedding_manager.get_topic_embeddings(agency, topic)
            embeddings_count = len(cached_embeddings) if cached_embeddings else 0

        return {
            "agency": agency,
            "topic": topic,
            "exists": exists,
            "embeddings_count": embeddings_count,
        }

    except Exception as e:
        logger.error(f"Error checking embeddings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error checking embeddings: {str(e)}"
        )


@router.delete("/embeddings/cache/persistent")
async def clear_persistent_cache(
    agency: Optional[str] = None,
    topic: Optional[str] = None,
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Clear persistent embeddings cache (topic documents)"""
    try:
        if agency and topic:
            # Clear specific agency-topic combination
            success = embedding_manager.delete_topic_embeddings(agency, topic)
            message = f"Cleared topic embeddings for {agency}-{topic}"
        elif agency:
            # Clear all topics for specific agency
            success = embedding_manager.delete_topic_embeddings(agency)
            message = f"Cleared all topic embeddings for agency {agency}"
        else:
            # Clear all topic embeddings
            success = embedding_manager.clear_all_topic_embeddings()
            message = "Cleared all topic embeddings"

        if success:
            return {"status": "success", "message": message}
        else:
            return {
                "status": "error",
                "message": f"Failed to clear embeddings: {message}",
            }

    except Exception as e:
        logger.error(f"Error clearing persistent cache: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing persistent cache: {str(e)}"
        )


@router.delete("/embeddings/cache/temporary")
async def clear_temporary_cache(
    session_id: Optional[str] = None,
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Clear temporary embeddings cache (questions)"""
    try:
        if session_id:
            # Clear specific session
            success = embedding_manager.cleanup_question_session(session_id)
            message = f"Cleared question embeddings for session {session_id}"
        else:
            # Clear all question sessions
            success = embedding_manager.clear_all_question_sessions()
            message = "Cleared all question sessions"

        if success:
            return {"status": "success", "message": message}
        else:
            return {
                "status": "error",
                "message": f"Failed to clear embeddings: {message}",
            }

    except Exception as e:
        logger.error(f"Error clearing temporary cache: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing temporary cache: {str(e)}"
        )


@router.post("/embeddings/generate")
async def generate_embeddings_for_agency_topic(
    agency: str,
    topic: str,
    force_refresh: bool = False,
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Manually generate embeddings for specific agency-topic combination"""
    try:
        logger.info(
            f"Manual generation request for {agency}-{topic} (force_refresh={force_refresh})"
        )

        embeddings = embedding_manager.get_topic_embeddings(
            agency, topic, force_refresh=force_refresh
        )

        if embeddings:
            return {
                "status": "success",
                "message": f"Successfully generated embeddings for {agency}-{topic}",
                "agency": agency,
                "topic": topic,
                "embeddings_count": len(embeddings),
                "force_refresh": force_refresh,
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to generate embeddings for {agency}-{topic}",
                "agency": agency,
                "topic": topic,
            }

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {str(e)}"
        )


@router.get("/embeddings/health")
async def embedding_health_check(
    embedding_manager: UnifiedEmbeddingManager = Depends(get_embedding_manager),
):
    """Check unified embedding system health"""
    try:
        redis_healthy = embedding_manager.health_check()
        cache_stats = embedding_manager.get_cache_stats()

        return {
            "status": "healthy" if redis_healthy else "unhealthy",
            "redis_healthy": redis_healthy,
            "model_name": embedding_manager.model_name,
            "topic_documents_path": embedding_manager.topic_documents_path,
            "cache_stats": cache_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logger.error(f"Error in embedding health check: {e}")
        return {
            "status": "unhealthy",
            "redis_healthy": False,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


@router.get("/health")
async def health_check(request: Request):
    """Enhanced health check endpoint including Redis embedding system"""
    try:
        provider_name = request.app.state.config.get("provider", {}).get(
            "name", "unknown"
        )

        # Check Redis health
        try:
            embedding_manager = get_embedding_manager(request)
            redis_healthy = embedding_manager.health_check()
            embedding_stats = embedding_manager.get_cache_stats()
        except Exception as e:
            redis_healthy = False
            embedding_stats = {"error": str(e)}

        overall_healthy = redis_healthy

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "version": "1.0.0",
            "provider": provider_name,
            "unified_embedding_system": {
                "healthy": redis_healthy,
                "stats": embedding_stats,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
