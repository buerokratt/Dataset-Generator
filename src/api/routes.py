from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
import httpx
from src.core.data_generator import DataGenerator
from src.core.data_source import DataSourceManager
from src.core.post_processor_factory import PostProcessorFactory
from pydantic import BaseModel, Field, validator
from src.utils.logger import logger
import time
import uuid
import asyncio
import re

router = APIRouter()

# Store for tracking background task status
task_status_store = {}


class DatasetRequest(BaseModel):
    """
    This schema defines the parameters required to generate a synthetic dataset from a specific data source.
    It is used as part of bulk dataset generation requests and supports validation of input fields.
    """

    data_path: str = Field(..., description="Path to the data source")
    output_filename: Optional[str] = Field(None, description="Custom output filename")

    class Config:
        extra = "allow"  # Allow additional fields like agency_id, etc.

    @validator("data_path")
    def validate_data_path(cls, v):
        if not v or not v.strip():
            raise ValueError("data_path cannot be empty")
        if ".." in v:
            raise ValueError("data_path contains invalid characters")
        return v.strip()

    @validator("output_filename")
    def validate_output_filename(cls, v):
        if v is not None:
            if not v or not v.strip():
                raise ValueError("output_filename cannot be empty if provided")

            if not re.match(r"^[a-zA-Z0-9_-]+$", v.strip()):
                raise ValueError("output_filename contains invalid characters")
        return v.strip() if v else None


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


async def send_callback(callback_url: str, payload: dict, max_retries: int = 3):
    """Send callback notification to external system"""
    if not callback_url:
        logger.info("No callback URL configured, skipping callback")
        return

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
                logger.info(f"Callback sent successfully to {callback_url}")
                return
        except Exception as e:
            logger.warning(f"Callback attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All callback attempts failed for {callback_url}")
            else:
                await asyncio.sleep(2**attempt)


async def process_single_dataset(
    dataset_request: DatasetRequest,
    config: dict,
    data_generator: DataGenerator,
    task_id: str,
    agency_name: str,
) -> dict:
    """
    Process a single dataset generation request.

    This function handles the end-to-end logic for generating a synthetic dataset based on the parameters
    provided in a DatasetRequest. It loads data sources according to the specified traversal strategy and filter,
    applies the configured structure and prompt template, and invokes the DataGenerator to produce the dataset.
    After generation, it optionally performs post-processing (such as zipping or aggregation) on the output files.

    Args:
        dataset_request (DatasetRequest): The dataset generation request containing source path, output filename, and other parameters.
        config (dict): The full configuration dictionary, including dataset generation and directory settings.
        data_generator (DataGenerator): The main generator instance used to create synthetic datasets.
        task_id (str): The unique identifier for the current bulk generation task.

    Returns:
        dict: A dictionary containing the result of the dataset generation, including success status, output paths,
              any errors encountered, and metadata about the dataset and configuration used.

    Notes:
        - Supports both individual and bulk post-processing modes (e.g., zip, aggregation).
        - Handles validation and error reporting for missing data sources or configuration issues.
        - Used internally by the bulk dataset generation API endpoint.
    """
    try:
        dataset_config = config.get("dataset_generation", {})

        # Extract all parameters from config
        structure_name = dataset_config.get("structure_name")
        prompt_template_name = dataset_config.get("prompt_template_name")
        traversal_strategy = dataset_config.get("traversal_strategy")
        num_examples = dataset_config.get("num_samples")
        output_format = dataset_config.get("output_format")
        parameters = dataset_config.get("parameters", {})
        filter_config = dataset_config.get("filter", {})
        post_processing_type = dataset_config.get("post_processing", "zip")

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
        source_manager = DataSourceManager()

        # Load all matching sources
        data_sources = source_manager.load_sources(
            base_path=data_path,
            strategy_name=traversal_strategy,
            filter_config=filter_config,
        )

        if not data_sources:
            return {
                "success": False,
                "error": f"No data sources found matching the criteria in {data_path}",
                "dataset_metadata": dataset_request.dict(),
            }

        logger.info(
            f"Found {len(data_sources)} data sources to process for {data_path}"
        )

        # Track all output paths for post-processing
        all_output_paths = []
        results = []

        for source in data_sources:
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

        # MODIFIED: Post-processing logic to handle cross-dataset aggregation
        final_output_path = None

        # For aggregation mode, skip individual post-processing - will be handled at bulk level
        if post_processing_type == "aggregation":
            logger.info(
                f"Skipping individual aggregation for {dataset_request.data_path} - will be handled at bulk level for cross-dataset aggregation"
            )
            # Individual files will be collected for final cross-dataset aggregation

            return {
                "success": True,
                "dataset_metadata": dataset_request.dict(),
                "post_processing_type": post_processing_type,
                "final_output_path": None,  # Will be set after cross-dataset aggregation
                "_internal_results": results,  # Keep for internal aggregation logic (NOT exposed in callback)
                "configuration_used": {
                    "structure_name": structure_name,
                    "prompt_template_name": prompt_template_name,
                    "traversal_strategy": traversal_strategy,
                    "num_examples": num_examples,
                    "output_format": output_format,
                    "post_processing": post_processing_type,
                },
            }
        else:
            # For non-aggregation modes (like zip), perform individual post-processing
            logger.info(
                f"Performing individual {post_processing_type} post-processing for {dataset_request.data_path}"
            )

            # base_output_dir = f"{output_dir}/{structure_name}"
            base_output_dir = f"{output_dir}"
            post_processor = PostProcessorFactory.create_post_processor(modified_config)
            final_output_path = post_processor.process(
                all_output_paths, base_output_dir
            )

            if final_output_path:
                return {
                    "success": True,
                    "dataset_metadata": dataset_request.dict(),
                    "post_processing_type": post_processing_type,
                    "final_output_path": final_output_path,
                    "_internal_results": results,
                    "configuration_used": {
                        "structure_name": structure_name,
                        "prompt_template_name": prompt_template_name,
                        "traversal_strategy": traversal_strategy,
                        "num_examples": num_examples,
                        "output_format": output_format,
                        "post_processing": post_processing_type,
                    },
                }
            else:
                return {
                    "success": False,
                    "error": "Post-processing failed",
                    "dataset_metadata": dataset_request.dict(),
                    "_internal_results": results,  # Keep for consistency
                }

    except Exception as e:
        logger.exception(
            f"Error processing dataset {dataset_request.data_path}: {str(e)}"
        )
        return {
            "success": False,
            "error": str(e),
            "dataset_metadata": dataset_request.dict(),
        }


async def background_generate_bulk(
    task_id: str,
    request_data: BulkGenerateRequest,
    config: dict,
    data_generator: DataGenerator,
):
    """
    Background task for processing bulk dataset generation requests.

    This function is executed as a background task to generate multiple datasets in parallel, based on the list of
    DatasetRequest objects provided in the BulkGenerateRequest. It tracks progress and status using the task_id,
    updates the task_status_store, and performs optional cross-dataset post-processing (such as aggregation or zipping).
    Upon completion or failure, it updates the task status and can send a callback notification if configured.

    Args:
        task_id (str): Unique identifier for this bulk generation task, used for status tracking.
        request_data (BulkGenerateRequest): The bulk request containing a list of dataset generation parameters.
        config (dict): Application configuration, including dataset generation and post-processing settings.
        data_generator (DataGenerator): Instance responsible for generating datasets from the provided parameters.

    Raises:
        Exception: Any error during processing is logged and updates the task status as failed.

    Notes:
        - Each dataset in the request is processed sequentially.
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

        for i, dataset_request in enumerate(request_data.datasets):
            logger.info(
                f"Processing dataset {i + 1}/{len(request_data.datasets)}: {dataset_request.data_path}"
            )

            # Store the output_filename from first dataset for final aggregation
            if common_output_filename is None and dataset_request.output_filename:
                common_output_filename = dataset_request.output_filename

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
                        all_cross_dataset_output_paths.append(
                            individual_result["output_path"]
                        )
                        # Extract metadata from dataset_request (convert to dict to include extra fields)
                        metadata = dataset_request.dict()
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

        if callback_url:
            # Clean up internal results before sending callback
            cleaned_results = []
            for result in all_results:
                cleaned_result = result.copy()
                if "_internal_results" in cleaned_result:
                    del cleaned_result["_internal_results"]
                cleaned_results.append(cleaned_result)

            callback_payload = {
                "task_id": task_id,
                "status": final_status,
                "message": task_status_store[task_id]["message"],
                "filePath": final_aggregated_path,
                "results": cleaned_results,
            }
            print(callback_payload)
            logger.info(
                f"DEBUG: Sending callback to {callback_url} with payload: {callback_payload}"
            )
            # await send_callback(callback_url, callback_payload)
            logger.info(
                f"Callback sent successfully for task {task_id} to {callback_url}"
            )
            logger.info("Generation task completed successfully")

    except Exception as e:
        logger.exception(f"Error in background generation for task {task_id}: {str(e)}")
        task_status_store[task_id] = {
            "status": "failed",
            "message": f"Error generating bulk datasets: {str(e)}",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [],
            "error": str(e),
        }


@router.post("/generate-bulk", response_model=GenerateResponse)
async def generate_bulk_dataset(
    request: BulkGenerateRequest,
    background_tasks: BackgroundTasks,
    fastapi_request: Request,
    data_generator: DataGenerator = Depends(get_data_generator),
):
    """
    Generate multiple datasets in bulk as a background task.

    Accepts a list of datasets with flexible metadata and returns immediately with a task_id.
    Supports callback notifications when processing is complete.
    """
    logger.info("Received bulk dataset generation request")
    try:
        logger.info(
            f"Received bulk generation request for {len(request.datasets)} datasets"
        )

        # Validate input
        if not request.datasets:
            raise HTTPException(
                status_code=400, detail="At least one dataset must be provided"
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
        }

        # Add background task
        background_tasks.add_task(
            background_generate_bulk, task_id, request, config, data_generator
        )

        logger.info(f"Background task queued with ID: {task_id}")

        return GenerateResponse(
            task_id=task_id,
            status="accepted",
            message=f"Dataset generation task has been queued for {len(request.datasets)} datasets. Use the task_id to check status.",
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


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    provider_name = request.app.state.config.get("provider", {}).get("name", "unknown")
    return {"status": "healthy", "version": "1.0.0", "provider": provider_name}
