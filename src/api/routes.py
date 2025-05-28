from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Depends

from src.core.data_generator import DataGenerator
from src.core.data_source import DataSourceManager
from pydantic import BaseModel, Field
from src.utils.logger import logger
import os
import time
import zipfile
from pathlib import Path

router = APIRouter()

class GenerateRequest(BaseModel):
    dataset_structure_name: str
    prompt_template_name: str
    data_path: str
    traversal_strategy: str
    no_of_samples: int
    output_format: Optional[str] = "json"
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    filter: Optional[Dict[str, Any]] = Field(default_factory=dict)

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
        raise HTTPException(status_code=500, detail="Application configuration not found")
    return DataGenerator(request.app.state.config)

def create_dataset_zip(base_path: str, request: Request) -> Optional[str]:
    """
    Create a ZIP archive of a dataset directory
    
    Args:
        base_path: Path to the dataset directory
        request: FastAPI request object containing application configuration
        
    Returns:
        Path to the created ZIP file
    """
    try:
        # Get output directory from config
        config = request.app.state.config
        output_dir = config.get("directories", {}).get("output", "output_datasets")
        
        base_path = Path(base_path)
        if not base_path.exists():
            logger.error(f"Dataset path not found: {base_path}")
            return None
            
        # Extract the correct structure name (last part of the path)
        structure_name = base_path.name
        zip_filename = f"{structure_name}.zip"
        
        # Create the zip in configured output folder
        zip_path = os.path.join(output_dir, zip_filename)
        
        # Create the ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create arcname (path within the ZIP file)
                    arcname = os.path.relpath(file_path, start=base_path)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created ZIP archive: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Error creating ZIP archive: {e}")
        return None

@router.post("/generate-bulk")
async def generate_bulk_dataset(
    request: GenerateRequest,
    fastapi_request: Request,
    data_generator: DataGenerator = Depends(get_data_generator)):
    """
    Generate datasets in bulk by processing multiple data sources.
    
    This endpoint takes a dataset structure, prompt template, and a data path
    containing source files to process. It iterates through each matching source
    file according to the traversal strategy, generates a dataset for each source,
    and returns information about all generated datasets.
    
    Args:
        request (GenerateRequest): A request model containing dataset generation parameters
        data_generator (DataGenerator): Instance created via dependency injection
        fastapi_request (Request): The FastAPI request object for accessing app state
    
    Returns:
        dict: Response with status, results, and path to the ZIP archive
    """
    
    try:
        logger.info(f"Received bulk generation request: {request.dataset_structure_name}, {request.prompt_template_name}")
        
        # Extract parameters
        structure_name = request.dataset_structure_name
        prompt_template_name = request.prompt_template_name
        data_path = request.data_path
        traversal_strategy = request.traversal_strategy
        num_examples = request.no_of_samples
        output_format = request.output_format
        parameters = request.parameters
        filter_config = request.filter
        
        # Get output directory from config
        config = fastapi_request.app.state.config
        output_dir = config.get("directories", {}).get("output", "output_datasets")
        
        # Create data source manager
        source_manager = DataSourceManager()
        
        # Load all matching sources
        data_sources = source_manager.load_sources(
            base_path=data_path,
            strategy_name=traversal_strategy,
            filter_config=filter_config
        )
        
        if not data_sources:
            return {
                "status": "error",
                "message": f"No data sources found matching the criteria in {data_path}"
            }
            
        logger.info(f"Found {len(data_sources)} data sources to process")
        
        # Track all output paths for ZIP creation
        all_output_paths = []
        results = []
        
        for source in data_sources:
            # Extract metadata for parameters
            source_params = parameters.copy()
            
            # Add source metadata to parameters
            if traversal_strategy == "institutional":
                source_params["institution"] = source.metadata.get("institution", "unknown")  
                source_params["topic"] = source.metadata.get("topic", "unknown")
                source_params["topic_content"] = source.content
            else:
                # For other traversal strategies, use file path components
                source_params["file_path"] = source.path
                source_params["file_content"] = source.content
                source_params["file_name"] = source.name
                
                # Add any other metadata from the source
                for key, value in source.metadata.items():
                    if key not in source_params:
                        source_params[key] = value
            
            # Generate dataset for this source
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Determine output path based on metadata - use config output_dir
            if traversal_strategy == "institutional":
                institution = source.metadata.get("institution", "unknown")  
                topic = source.metadata.get("topic", "unknown")
                output_base_path = f"{output_dir}/{structure_name}/{institution}/{topic}_{timestamp}"
            else:
                # For other traversal strategies, use relative path
                rel_path = source.metadata.get("relative_path", source.name)
                output_base_path = f"{output_dir}/{structure_name}/{rel_path}_{timestamp}"
            
            logger.info(f"Generating dataset for source: {source.path} -> {output_base_path}")
            
            # Generate dataset
            result_path = data_generator.generate(
                structure_name=structure_name,
                prompt_template_name=prompt_template_name,
                output_base_path=output_base_path,
                num_examples=num_examples,
                output_format=output_format,
                parameters=source_params
            )
            
            all_output_paths.append(result_path)
            results.append({
                "source": source.path,
                "output_path": result_path
            })
        
        # Create a ZIP archive of the structure directory - use config output_dir
        base_output_dir = f"{output_dir}/{structure_name}"
        zip_path = create_dataset_zip(base_output_dir, fastapi_request)
        
        logger.info(f"Created ZIP archive of datasets: {zip_path}")
        
        return {
            "status": "success",
            "message": f"Generated datasets for {len(results)} sources",
            "results": results,
            "zip_path": zip_path
        }
    except Exception as e:
        logger.exception(f"Error generating bulk datasets: {str(e)}")
        return {
            "status": "error",
            "message": f"Error generating bulk datasets: {str(e)}"
        }

@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    provider_name = request.app.state.config.get("provider", {}).get("name", "unknown")
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "provider": provider_name
    }