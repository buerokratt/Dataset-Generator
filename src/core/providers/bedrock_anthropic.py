"""
AWS Bedrock Anthropic provider for LLM inference.

This module implements the Bedrock Anthropic provider using LangChain's AWS Bedrock integration.
It provides seamless integration with Anthropic Claude models through AWS Bedrock while maintaining
the same interface as other providers in the system.

Features:
- Integration with AWS Bedrock Anthropic models (Claude-3, Claude-3.5)
- Rate limiting and retry mechanisms with exponential backoff
- Cost and token tracking through MLflow
- Parameter normalization between provider APIs
- Comprehensive error handling and logging
"""

import os
import time
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_aws import ChatBedrock
from botocore.exceptions import ClientError, EndpointConnectionError
from src.core.providers.base import ModelProvider
from src.utils.logger import logger
import json
import re

class BedrockConfigurationError(Exception):
    """Raised when there's a configuration or permission issue with Bedrock."""
    pass


class BedrockAPIError(Exception):
    """Raised when Bedrock API returns an error."""
    pass


class BedrockGenerationError(Exception):
    """Raised when text generation fails."""
    pass


class BedrockAnthropicProvider(ModelProvider):
    """
    AWS Bedrock Anthropic provider implementation.
    
    This provider uses LangChain's ChatBedrock to interface with Anthropic Claude models
    through AWS Bedrock. It handles authentication, rate limiting, retries, and parameter
    normalization while maintaining compatibility with the ModelProvider interface.
    
    Supported Models:
    - anthropic.claude-3-5-sonnet-20241022-v2:0
    - anthropic.claude-3-5-haiku-20241022-v1:0  
    - anthropic.claude-3-opus-20240229-v1:0
    - anthropic.claude-3-sonnet-20240229-v1:0
    - anthropic.claude-3-haiku-20240307-v1:0
    
    Configuration Parameters:
    - model_name: Bedrock model ID (default: anthropic.claude-3-5-sonnet-20241022-v2:0)
    - aws_region: AWS region (default: us-east-1)
    - temperature: Model temperature 0.0-1.0 (default: 0.7)
    - max_tokens: Maximum output tokens (default: 4096)
    - top_p: Nucleus sampling parameter (default: 1.0)
    - stop_sequences: List of stop sequences (optional)
    """
    
    # Default configuration values
    DEFAULT_MODEL = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
    DEFAULT_REGION = "eu-west-1" 
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TOP_P = 1.0
    
    # Rate limiting defaults (tokens per minute for Anthropic models)
    DEFAULT_TPM_LIMIT = 200000  # Conservative default
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Bedrock Anthropic provider.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model_name: Bedrock model identifier
                - aws_region: AWS region for Bedrock
                - temperature: Model temperature parameter
                - max_tokens: Maximum output tokens
                - top_p: Nucleus sampling parameter
                - stop_sequences: Optional stop sequences
                - tpm_limit: Tokens per minute rate limit
        """
        self.config = config
        self.model_name = config.get("model_name", self.DEFAULT_MODEL)
        self.aws_region = config.get("aws_region", self.DEFAULT_REGION)
        self.temperature = float(config.get("temperature", self.DEFAULT_TEMPERATURE))
        self.max_tokens = int(config.get("max_tokens", self.DEFAULT_MAX_TOKENS))
        self.top_p = float(config.get("top_p", self.DEFAULT_TOP_P))
        self.stop_sequences = config.get("stop_sequences", [])
        self.tpm_limit = int(config.get("tpm_limit", self.DEFAULT_TPM_LIMIT))
        
        # Rate limiting state
        self._last_request_time = 0
        self._token_count_window = []
        self._window_duration = 60  # 1 minute window
        
        # Initialize LangChain ChatBedrock client
        self._init_client()
        
        logger.info(f"Initialized BedrockAnthropicProvider with model {self.model_name} in region {self.aws_region}")
    
    def _init_client(self):
        """Initialize the LangChain ChatBedrock client with AWS credentials."""
        try:
            # Validate AWS credentials are available
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            if not aws_access_key or not aws_secret_key:
                logger.warning("AWS credentials not found in environment variables. "
                             "Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set, "
                             "or use IAM roles/instance profiles.")
            
            # Initialize ChatBedrock with configuration
            self.client = ChatBedrock(
                model_id=self.model_name,
                region_name=self.aws_region,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "stop_sequences": self.stop_sequences if self.stop_sequences else None
                }
            )
            
            logger.debug(f"ChatBedrock client initialized for model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatBedrock client: {str(e)}")
            raise
    
    def _rate_limit_check(self, estimated_tokens: int = 1000):
        """
        Enforce rate limiting based on tokens per minute.
        
        Args:
            estimated_tokens (int): Estimated tokens for the current request
        """
        current_time = time.time()
        
        # Remove tokens outside the current window
        self._token_count_window = [
            (timestamp, tokens) for timestamp, tokens in self._token_count_window
            if current_time - timestamp < self._window_duration
        ]
        
        # Calculate current window usage
        current_window_tokens = sum(tokens for _, tokens in self._token_count_window)
        
        # Check if adding this request would exceed the limit
        if current_window_tokens + estimated_tokens > self.tpm_limit:
            sleep_time = self._window_duration - (current_time - self._token_count_window[0][0])
            logger.warning(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            current_time = time.time()
        
        # Add current request to the window
        self._token_count_window.append((current_time, estimated_tokens))
        self._last_request_time = current_time
    
    def _normalize_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize request options to Bedrock Anthropic parameters.
        
        Args:
            options (Optional[Dict[str, Any]]): Request-specific options
            
        Returns:
            Dict[str, Any]: Normalized parameters for Bedrock
        """
        if not options:
            return {}
        
        normalized = {}
        
        # Map common parameters to Bedrock equivalents
        if "temperature" in options:
            normalized["temperature"] = float(options["temperature"])
        
        if "max_tokens" in options:
            normalized["max_tokens"] = int(options["max_tokens"])
        
        if "top_p" in options:
            normalized["top_p"] = float(options["top_p"])
        
        if "stop" in options or "stop_sequences" in options:
            stop_seqs = options.get("stop") or options.get("stop_sequences", [])
            if isinstance(stop_seqs, str):
                stop_seqs = [stop_seqs]
            normalized["stop_sequences"] = stop_seqs
        
        # Handle provider-specific parameters
        if "top_k" in options:
            # Note: Anthropic doesn't support top_k, log warning
            logger.warning("top_k parameter not supported by Anthropic models, ignoring")
        
        return normalized
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ClientError, EndpointConnectionError)),
        reraise=True
    )
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text using AWS Bedrock Anthropic models.
        
        Args:
            prompt (str): Input prompt for text generation
            options (Optional[Dict[str, Any]]): Request-specific parameters
            
        Returns:
            str: Generated text response
            
        Raises:
            Exception: If generation fails after retries
        """
        start_time = time.time()
        
        try:
            # Estimate tokens for rate limiting (rough approximation)
            estimated_input_tokens = len(prompt.split()) * 1.3  # Average tokens per word
            max_output_tokens = options.get("max_tokens", self.max_tokens) if options else self.max_tokens
            estimated_total_tokens = estimated_input_tokens + max_output_tokens
            
            # Apply rate limiting
            self._rate_limit_check(int(estimated_total_tokens))
            
            # Normalize options for this request
            normalized_options = self._normalize_options(options)
            
            # Update client model_kwargs if options provided
            if normalized_options:
                current_kwargs = self.client.model_kwargs.copy()
                current_kwargs.update(normalized_options)
                self.client.model_kwargs = current_kwargs

            logger.info("=" * 80)
            logger.info("PROMPT BEING SENT TO BEDROCK ANTHROPIC:")
            logger.info("=" * 80)
            logger.info(prompt)
            logger.info("=" * 80)
            logger.info(f"MODEL: {self.model_name}")
            logger.info(f"OPTIONS: {normalized_options}")
            logger.info("=" * 80)

            logger.info(f"Generating text with model {self.model_name}, prompt length: {len(prompt)} chars")

            # Generate response using LangChain
            response = self.client.invoke(prompt)
            generated_text = response.content
            
            # Calculate timing and log metrics
            duration = time.time() - start_time
            
            # Estimate actual token usage (approximation)
            output_tokens = len(generated_text.split()) * 1.3
            total_tokens = estimated_input_tokens + output_tokens
            
            logger.info(f"Text generated successfully. Duration: {duration:.2f}s, "
                       f"Est. tokens: {int(total_tokens)}, Model: {self.model_name}")
            
            return generated_text
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"AWS Bedrock API error {error_code}: {error_message}")
            
            # Handle specific error types
            if error_code in ['ThrottlingException', 'ServiceQuotaExceededException']:
                logger.warning("Rate limiting detected, retry will be attempted")
                raise  # Let tenacity handle the retry
            elif error_code in ['ValidationException', 'AccessDeniedException']:
                logger.error(f"Configuration or permission error: {error_message}")
                raise BedrockConfigurationError(f"Bedrock configuration error: {error_message}")
            else:
                raise BedrockAPIError(f"Bedrock API error: {error_message}")
                
        except EndpointConnectionError as e:
            logger.error(f"Network connectivity issue with Bedrock: {str(e)}")
            raise  # Let tenacity handle the retry
            
        except Exception as e:
            logger.error(f"Unexpected error during text generation: {str(e)}")
            raise BedrockGenerationError(f"Text generation failed: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if the Bedrock Anthropic provider is available and accessible.
        
        Returns:
            bool: True if the provider is healthy, False otherwise
        """
        try:
            # Simple test generation to verify connectivity
            test_prompt = "Hello"
            response = self.generate(test_prompt, {"max_tokens": 10})
            
            if response and len(response.strip()) > 0:
                logger.info("Bedrock Anthropic provider health check passed")
                return True
            else:
                logger.warning("Bedrock Anthropic provider returned empty response")
                return False
                
        except Exception as e:
            logger.error(f"Bedrock Anthropic provider health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Model information including name, region, and parameters
        """
        return {
            "provider": "bedrock-anthropic",
            "model_name": self.model_name,
            "aws_region": self.aws_region,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "tpm_limit": self.tpm_limit
        }
    
    def supports_batch_generation(self) -> bool:
        """Bedrock supports efficient batch generation via prompt engineering"""
        return True
    
    def generate_batch(self, prompt: str, num_samples: int, options: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate multiple samples efficiently using a single API call with batch prompt.
        
        Args:
            prompt: Base prompt for generation
            num_samples: Number of samples to generate  
            options: Generation options
            
        Returns:
            List of generated responses
        """
        
        logger.info(f"BATCH GENERATION: {num_samples} samples in 1 API call")
        
        # Create batch prompt that requests multiple samples
        batch_prompt = f"""I need you to execute the following task exactly {num_samples} times, each time producing a different result.

TASK TO EXECUTE {num_samples} TIMES:
```json
{{
  "prompt": "{prompt}",
  "num_samples": {num_samples},
  "options": {json.dumps(options)}
}}
{prompt}

INSTRUCTIONS:
- Execute the above task exactly {num_samples} times
- Each execution should produce a unique, different result
- Follow the original task requirements for each execution
- Return your results as a JSON array with exactly {num_samples} elements
- Each array element should contain the JSON object that the original task would produce

Example format for Estonian questions:
[
  {{"question": "Kas ma saan osta perioodipiletit veebis?"}},
  {{"question": "Kuidas ma saan kontrollida oma kaardi saldot?"}},
  {{"question": "Millal toimub järgmine avatud talude päev?"}}
]

Execute the task {num_samples} times now and return the JSON array:"""

        try:
            # Set higher max_tokens for batch generation
            batch_options = (options or {}).copy()
            original_max_tokens = batch_options.get("max_tokens", self.max_tokens)
            batch_options["max_tokens"] = min(original_max_tokens * num_samples, 200000)  # Cap at model limit
            
            logger.info("BATCH PROMPT:")
            logger.info("-" * 80)
            logger.info(batch_prompt[:500] + ("..." if len(batch_prompt) > 500 else ""))
            logger.info("-" * 80)
            
            # Single API call for all samples
            start_time = time.time()
            response = self.generate(batch_prompt, batch_options)
            duration = time.time() - start_time
            
            logger.info("BATCH RESPONSE:")
            logger.info("-" * 80) 
            logger.info(response[:500] + ("..." if len(response) > 500 else ""))
            logger.info("-" * 80)
            
            # Parse JSON array response with multiple extraction strategies
            json_str = None
            
            # Strategy 1: Look for JSON array with markdown code blocks
            markdown_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL | re.IGNORECASE)
            if markdown_match:
                json_str = markdown_match.group(1)
                logger.debug("Found JSON array in markdown code block")
            else:
                # Strategy 2: Look for JSON array directly
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    logger.debug("Found JSON array directly in response")
            
            if json_str:
                logger.debug(f"Extracted JSON string: {json_str[:200]}...")
                
                try:
                    samples = json.loads(json_str)
                    
                    if isinstance(samples, list):
                        if len(samples) == num_samples:
                            # Process each sample - they're already valid JSON objects
                            valid_samples = []
                            for i, sample in enumerate(samples):
                                if isinstance(sample, dict):
                                    # Convert dict to JSON string
                                    valid_samples.append(json.dumps(sample))
                                    logger.debug(f"BATCH SAMPLE {i+1}: {list(sample.keys()) if sample else 'empty'}")
                                elif isinstance(sample, str):
                                    # Handle case where samples are already strings
                                    valid_samples.append(sample)
                                    logger.debug(f"BATCH SAMPLE {i+1}: string format")
                                else:
                                    # Convert any other type to string
                                    valid_samples.append(json.dumps(sample))
                                    logger.debug(f"BATCH SAMPLE {i+1}: {type(sample).__name__} format")
                            
                            if valid_samples:
                                logger.info(f"BATCH SUCCESS: Generated {len(valid_samples)} valid samples in {duration:.2f}s")
                                return valid_samples
                        else:
                            logger.warning(f"Expected {num_samples} samples, got {len(samples)}. Falling back to individual calls.")
                            logger.debug(f"Sample types: {[type(s).__name__ for s in samples[:3]]}")
                    else:
                        logger.warning(f"Response is not a list but {type(samples)}. Falling back to individual calls.")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {e}. Falling back to individual calls.")
                    logger.debug(f"JSON string that failed: {json_str[:500]}...")
                    
            else:
                logger.warning("Could not find JSON array in batch response. Falling back to individual calls.")
                logger.debug(f"Full response: {response[:500]}...")
                
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}. Falling back to individual calls.")
        
        # Fallback to individual calls if batch fails
        logger.info("FALLBACK: Using individual API calls")
        return super().generate_batch(prompt, num_samples, options)
