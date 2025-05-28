import os
import mlflow
import json
from src.utils.logger import logger

class MLflowTracker:
    """Utility class to track synthetic data generation experiments in MLflow."""
    
    def __init__(self, experiment_name="synthetic_data_generation"):
        """Initialize MLflow tracking with the specified experiment name."""
        # Set up MLflow tracking URI
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"Initializing MLflow tracking with URI: {mlflow_uri}")
        
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            self.is_active = True
            
            # Create a directory for temporary files that can be shared with Docker
            self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlflow_temp")
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.info(f"Using temporary directory for MLflow artifacts: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.is_active = False
    
    def log_generation(self, structure_name, prompt_template_name, parameters, 
                       output_path, generation_time=None, metrics=None):
        """
        Log a synthetic data generation run to MLflow.
        
        Args:
            structure_name (str): Name of the data structure
            prompt_template_name (str): Name of the prompt template used
            parameters (dict): Parameters used for generation
            output_path (str): Path to the generated file
            generation_time (float, optional): Time taken for generation in seconds
            metrics (dict, optional): Additional metrics to log
        
        Returns:
            str or None: MLflow run ID if successful, None otherwise
        """
        if not self.is_active:
            logger.warning("MLflow tracking is not active. Skipping logging.")
            return None
            
        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                
                # Log basic info as tags
                mlflow.set_tag("structure_name", structure_name)
                mlflow.set_tag("prompt_template_name", prompt_template_name)
                
                if "institute" in parameters and "topic" in parameters:
                    mlflow.set_tag("institute", parameters["institute"])
                    mlflow.set_tag("topic", parameters["topic"])
                
                # Log data structure and prompt template as parameters
                mlflow.log_param("structure_name", structure_name)
                mlflow.log_param("prompt_template_name", prompt_template_name)
                
                # Log parameters (flattening nested structures)
                self._log_parameters(parameters)
                
                # Log generation time
                if generation_time:
                    mlflow.log_metric("generation_time_seconds", generation_time)
                
                # Log additional metrics
                if metrics and isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                
                # Log output file as artifact
                if output_path and os.path.exists(output_path):
                    # Save the output file
                    mlflow.log_artifact(output_path)
                    
                    # Calculate and log file size
                    file_size_kb = os.path.getsize(output_path) / 1024
                    mlflow.log_metric("file_size_kb", round(file_size_kb, 2))
                
                # Save parameters as JSON using the shared Docker volume directory
                params_file = os.path.join(self.temp_dir, f"params_{run_id}.json")
                with open(params_file, 'w') as f:
                    json.dump(parameters, f, indent=2)
                
                try:
                    mlflow.log_artifact(params_file)
                except Exception as e:
                    logger.warning(f"Failed to log parameters file: {e}")
                finally:
                    # Clean up after successful logging
                    try:
                        os.remove(params_file)
                    except Exception:
                        pass
                
                print(f"🏃 View run {run_id} at: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")
                print(f"🧪 View experiment at: http://localhost:5000/#/experiments/{run.info.experiment_id}")
                
                logger.info(f"Logged generation run to MLflow with run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
            return None
    
    def _log_parameters(self, parameters):
        """Log parameters, handling nested dictionaries and limiting string lengths."""
        flat_params = {}
        
        def _flatten_dict(params, prefix=""):
            for key, value in params.items():
                if key == "topic_content" and len(str(value)) > 100:
                    # Skip or truncate very large content
                    flat_params[prefix + key] = str(value)[:100] + "..."
                    continue
                    
                if isinstance(value, dict):
                    _flatten_dict(value, f"{prefix}{key}_")
                else:
                    # Convert to string if not a primitive type
                    if not isinstance(value, (str, int, float, bool)) or value is None:
                        value = str(value)
                    
                    # Truncate strings if too long (MLflow has a limit)
                    if isinstance(value, str) and len(value) > 490:
                        value = value[:487] + "..."
                    
                    flat_params[prefix + key] = value
        
        _flatten_dict(parameters)
        
        # Log all parameters
        for name, value in flat_params.items():
            try:
                mlflow.log_param(name, value)
            except Exception as e:
                logger.warning(f"Could not log parameter {name}: {e}")