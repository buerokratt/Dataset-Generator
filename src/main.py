# Update your main.py to load configuration including the provider section
import argparse
import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import ConfigLoader
from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Dataset Generation Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the service on')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def create_app(config):
    """Create FastAPI application with routes and config"""
    app = FastAPI(
        title="Synthetic Dataset Generation Service",
        description="API for generating synthetic datasets using LLMs",
        version="1.0.0"
    )
    
    # Store config in app state
    app.state.config = config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    app.include_router(router)
    
    return app

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger("synthetic-data-service", log_level)
    
    # Load configuration with path from args
    config_paths = [args.config] if args.config else None
    config = ConfigLoader.load(paths=config_paths)
    
    if not config:
        logger.error("Failed to load configuration")
        return 1
    
    logger.info("Configuration loaded successfully")
    
    # Create FastAPI app
    app = create_app(config)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()