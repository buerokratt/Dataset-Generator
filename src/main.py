"""
Main entry point for the Synthetic Dataset Generation Service
"""
import argparse

from api.routes import app
from src.utils.logger import logger, setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Dataset Generation Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the service on')
    parser.add_argument('--config', type=str, default='config/service_config.yaml', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main entry point for the service"""
    args = parse_args()

    # Setup Loguru logger
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger("synthetic-data-service", log_level)

    logger.info(f"🚀 Starting Synthetic Dataset Generation Service on {args.host}:{args.port}")

    # Run the service
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level.lower())

if __name__ == "__main__":
    main()
