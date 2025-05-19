import os
import requests
import json
import time
import yaml
from typing import Dict, Any
import random

# Add the project root directory to Python path
from src.utils.logger import logger, setup_logger
from src.utils.mlflow_tracking import MLflowTracker

setup_logger("synthetic-data-service", "INFO")

# Load configuration from YAML file
def load_config(config_path="config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

# Load configuration
CONFIG = load_config()

# API endpoint of the synthetic data service
SERVICE_URL = CONFIG["api"]["url"]
# Root directory containing source data files organized by institute
DATA_ROOT_DIR = CONFIG["directories"]["input"]
# Root directory where generated datasets will be saved
OUTPUT_ROOT_DIR = CONFIG["directories"]["output"]

def generate_dataset_for_topic(institute, topic_name, topic_file_path, dataset_type, mlflow_tracker=None, 
                              retry_count=0, max_retries=3, base_timeout=30):
    """
    Generate a synthetic dataset for a specific topic using the API service
    
    Args:
        institute: Institute name (organization that owns the content)
        topic_name: Name of the topic to generate data for
        topic_file_path: Path to the text file containing topic content
        dataset_type: Type of dataset to generate - either "faqs" or "conversations"
        mlflow_tracker: Optional MLflowTracker instance for logging
        retry_count: Current retry attempt count
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds, will increase with retries
    
    Returns:
        dict: Results containing success status, output path, etc.
    """
    # Calculate timeout with exponential backoff
    current_timeout = base_timeout * (2 ** retry_count)
    
    logger.info(f"Generating {dataset_type} for Institute: {institute}, Topic: {topic_name}" + 
               (f" (Retry {retry_count}/{max_retries}, timeout={current_timeout}s)" if retry_count > 0 else ""))
    
    results = {
        "success": False,
        "topic": topic_name,
        "institute": institute,
        "dataset_type": dataset_type,
        "output_path": None,
        "error": None,
        "duration_seconds": None,
        "retry_count": retry_count
    }
    
    start_time = time.time()
    
    try:
        # Read the content from the topic file
        with open(topic_file_path, 'r', encoding='utf-8') as f:
            topic_content = f.read()
            results["content_size_bytes"] = len(topic_content)
    except Exception as e:
        error_msg = f"Error reading topic file {topic_file_path}: {e}"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    # Use posix-style paths (forward slashes) for Docker compatibility
    target_output_base = os.path.join(OUTPUT_ROOT_DIR, institute, topic_name).replace('\\', '/')
    
    # Get configuration for the dataset type
    if dataset_type not in CONFIG:
        error_msg = f"Unknown dataset type: {dataset_type}"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
        
    dataset_config = CONFIG[dataset_type]
    structure_name = dataset_config["structure_name"]
    prompt_template_name = dataset_config["prompt_template_name"]
    num_examples = dataset_config["num_examples"]
    redundancy_factor = dataset_config["parameters"]["redundancy_factor"]
    
    # Prepare parameters by copying the default parameters from config
    # and adding topic-specific parameters
    parameters = dataset_config["parameters"].copy()
    parameters.update({
        "institute": institute,
        "topic": topic_name,
        "topic_content": topic_content,
        "redundancy_factor": redundancy_factor
    })
    
    # Ensure output directory exists
    os.makedirs(target_output_base, exist_ok=True)
    
    # Prepare API request payload with all necessary parameters
    payload = {
        "structure_name": structure_name,
        "prompt_template_name": prompt_template_name,
        "output_base_path": target_output_base,
        "parameters": parameters,
        "num_examples": num_examples
    }
    
    try:
        # Call the API to generate the dataset with the current timeout
        response = requests.post(f"{SERVICE_URL}/generate", json=payload, timeout=current_timeout)
        response.raise_for_status() # Raise exception for error status codes
        
        response_data = response.json()
        output_path = response_data.get('path')
        
        logger.info(f"Successfully generated {dataset_type} for {institute}/{topic_name} at {output_path}")
        
        results["success"] = True
        results["output_path"] = output_path
        results["num_examples"] = num_examples
        
        # Log to MLflow using the tracker if provided
        if mlflow_tracker and results["success"] and output_path:
            generation_time = time.time() - start_time
            metrics = {
                "num_examples": num_examples,
                "content_size_bytes": results["content_size_bytes"],
            }
            
            run_id = mlflow_tracker.log_generation(
                structure_name=structure_name,
                prompt_template_name=prompt_template_name,
                parameters=parameters,
                output_path=output_path,
                generation_time=generation_time,
                metrics=metrics
            )
            
            results["mlflow_run_id"] = run_id
        
    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
        error_msg = f"API call failed for {institute}/{topic_name}: {e}"
        logger.error(error_msg)
        results["error"] = error_msg
        
        # Implement retry logic
        if retry_count < max_retries:
            # Add a small random jitter to prevent all retries happening simultaneously
            jitter = random.uniform(0.5, 2.0)
            wait_time = (2 ** retry_count) + jitter
            
            logger.info(f"Retrying in {wait_time:.2f} seconds... (Attempt {retry_count+1}/{max_retries})")
            time.sleep(wait_time)
            
            # Recursive retry with incremented counter and timeout
            return generate_dataset_for_topic(
                institute, 
                topic_name, 
                topic_file_path, 
                dataset_type, 
                mlflow_tracker,
                retry_count + 1, 
                max_retries,
                base_timeout
            )
            
    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON response for {institute}/{topic_name}: {response.text}"
        logger.error(error_msg)
        results["error"] = error_msg
    finally:
        results["duration_seconds"] = time.time() - start_time
        
    return results

if __name__ == "__main__":
    # Initialize MLflow tracking
    mlflow_experiment_name = CONFIG.get("mlflow", {}).get("experiment_name", "synthetic_data_generation")
    mlflow_tracker = MLflowTracker(experiment_name=mlflow_experiment_name)
    
    # Track statistics for summary
    total_topics = 0
    successful_generations = 0
    failed_generations = 0
    total_duration = 0
    all_results = []
    
    # Create the output root directory if it doesn't exist
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    # Process all institutes and their topics in the data directory
    for institute_name in os.listdir(DATA_ROOT_DIR):
        institute_path = os.path.join(DATA_ROOT_DIR, institute_name)
        
        # Skip non-directories (only process institute folders)
        if not os.path.isdir(institute_path):
            continue
            
        # Clean up institute name if it already has the "output_" prefix
        # to avoid accumulating prefixes if script is run multiple times
        clean_institute_name = institute_name
        if institute_name.startswith("output_"):
            clean_institute_name = institute_name[7:]
            
        # Add "output_" prefix to institute name for the output directory
        target_institute_name = f"output_{clean_institute_name}"
        
        # Process each topic file in the institute directory
        for topic_filename in os.listdir(institute_path):
            if topic_filename.endswith(".txt"):  # Only process .txt files
                # Get topic name without extension
                topic_name_base = os.path.splitext(topic_filename)[0]
                # Full path to the topic file
                topic_full_path = os.path.join(institute_path, topic_filename)
                
                total_topics += 1
                
                # First generate QA pairs for this topic
                faq_results = generate_dataset_for_topic(
                    target_institute_name, 
                    topic_name_base, 
                    topic_full_path, 
                    "faqs",
                    mlflow_tracker,
                    max_retries=3,
                    base_timeout=30
                )
                
                all_results.append(faq_results)
                total_duration += faq_results.get("duration_seconds", 0)
                
                if faq_results["success"]:
                    successful_generations += 1
                else:
                    failed_generations += 1
                
                # Wait briefly to avoid overwhelming the API
                wait_time = CONFIG["processing"].get("wait_between_requests", 1)
                time.sleep(wait_time)
                
                # Then generate conversations for the same topic
                conv_results = generate_dataset_for_topic(
                    target_institute_name, 
                    topic_name_base, 
                    topic_full_path, 
                    "conversations",
                    mlflow_tracker,
                    max_retries=3,
                    base_timeout=30
                )
                
                all_results.append(conv_results)
                total_duration += conv_results.get("duration_seconds", 0)
                
                if conv_results["success"]:
                    successful_generations += 1
                else:
                    failed_generations += 1
    
    # Save detailed results as JSON artifact
    results_file = os.path.join(OUTPUT_ROOT_DIR, "generation_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Output summary
    logger.info(f"=== Generation Summary ===")
    logger.info(f"Total topics processed: {total_topics}")
    logger.info(f"Total generation attempts: {len(all_results)}")
    logger.info(f"Successful generations: {successful_generations}")
    logger.info(f"Failed generations: {failed_generations}")
    logger.info(f"Success rate: {successful_generations / max(1, len(all_results)):.2%}")
    logger.info(f"Total duration: {total_duration:.2f} seconds")