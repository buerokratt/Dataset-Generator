"""
Azure OpenAI Provider Implementation

This module provides the Azure OpenAI provider implementation using LangChain
for the dataset generation system.
"""

import json
import os
import re
import time
from typing import Dict, Any, Optional, List

from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.providers.base import ModelProvider
from src.utils.logger import logger


class AzureOpenAIProvider(ModelProvider):
    """
    Azure OpenAI provider implementation using LangChain AzureChatOpenAI.
    
    Supports both individual and batch generation with automatic fallback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure OpenAI provider with configuration."""
        
        # Azure OpenAI specific configuration
        azure_config = config.get('azure_openai', {})
        
        # Required Azure OpenAI parameters
        self.azure_endpoint = azure_config.get('azure_endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = azure_config.get('api_key') or os.getenv('AZURE_OPENAI_API_KEY')
        self.api_version = azure_config.get('api_version', '2024-12-01-preview')
        self.deployment_name = azure_config.get('deployment_name') or os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        # Model parameters
        self.model_name = azure_config.get('model_name', 'gpt-4o')
        self.temperature = azure_config.get('temperature', 0.7)
        self.max_tokens = azure_config.get('max_tokens', 4096)
        self.top_p = azure_config.get('top_p', 1.0)
        
        # Rate limiting
        self.tpm_limit = azure_config.get('tpm_limit', 200000)
        self.rpm_limit = azure_config.get('rpm_limit', 6000)
        
        # Batch configuration
        batch_config = azure_config.get('batch_generation', {})
        self.batch_enabled = batch_config.get('enabled', True)
        self.max_batch_size = batch_config.get('max_batch_size', 10)
        self.max_tokens_per_batch = batch_config.get('max_tokens_per_batch', 20000)
        
        # Validate required parameters
        if not all([self.azure_endpoint, self.api_key, self.deployment_name]):
            missing = []
            if not self.azure_endpoint:
                missing.append('azure_endpoint/AZURE_OPENAI_ENDPOINT')
            if not self.api_key:
                missing.append('api_key/AZURE_OPENAI_API_KEY')
            if not self.deployment_name:
                missing.append('deployment_name/AZURE_OPENAI_DEPLOYMENT_NAME')
            raise ValueError(f"Missing required Azure OpenAI configuration: {', '.join(missing)}")
        
        # Initialize LangChain Azure OpenAI client
        try:
            self.client = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.deployment_name,
                openai_api_version=self.api_version,
                openai_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                request_timeout=60
            )
            
            logger.info(f"Initialized AzureOpenAIProvider with deployment {self.deployment_name} in region {self.azure_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def supports_batch_generation(self) -> bool:
        """Check if provider supports batch generation."""
        return self.batch_enabled
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text using Azure OpenAI.
        
        Args:
            prompt: The input prompt
            options: Optional generation parameters
            
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        try:
            # Apply any additional options
            client_options = {}
            if options:
                if 'max_tokens' in options:
                    client_options['max_tokens'] = options['max_tokens']
                if 'temperature' in options:
                    client_options['temperature'] = options['temperature']
            
            logger.debug(f"Generating text with deployment {self.deployment_name}, prompt length: {len(prompt)} chars")
            
            # Create temporary client with options if needed
            if client_options:
                temp_client = AzureChatOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    azure_deployment=self.deployment_name,
                    openai_api_version=self.api_version,
                    openai_api_key=self.api_key,
                    request_timeout=60,
                    **client_options
                )
                response = temp_client.invoke(prompt)
            else:
                response = self.client.invoke(prompt)
            
            duration = time.time() - start_time
            
            # Estimate tokens (rough approximation: 4 chars per token)
            estimated_tokens = len(response.content) // 4
            
            logger.info(f"Text generated successfully. Duration: {duration:.2f}s, Est. tokens: {estimated_tokens}, Deployment: {self.deployment_name}")
            
            return response.content
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Azure OpenAI generation failed after {duration:.2f}s: {e}")
            raise

    def generate_batch(self, prompt: str, num_samples: int, options: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate multiple samples using batch approach.
        
        Args:
            prompt: The base prompt to execute multiple times
            num_samples: Number of samples to generate
            options: Optional generation parameters
            
        Returns:
            List of generated responses
        """
        if not self.supports_batch_generation():
            logger.warning("Batch generation not supported, falling back to individual calls")
            return [self.generate(prompt, options) for _ in range(num_samples)]
        
        if num_samples > self.max_batch_size:
            logger.warning(f"Requested {num_samples} samples exceeds max batch size {self.max_batch_size}, falling back to individual calls")
            return [self.generate(prompt, options) for _ in range(num_samples)]
        
        start_time = time.time()
        
        # Create batch prompt that asks for multiple executions
        batch_prompt = f"""I need you to execute the following task exactly {num_samples} times, each time producing a different result.

TASK TO EXECUTE {num_samples} TIMES:
{prompt}

INSTRUCTIONS:
- Execute the above task exactly {num_samples} times
- Each execution should produce a unique, different result
- Follow the original task requirements for each execution
- Return your results as a JSON array with exactly {num_samples} elements
- Each array element should contain the JSON object that the original task would produce

Example format:
[
  {{"question": "First unique question in Estonian"}},
  {{"question": "Second unique question in Estonian"}},
  {{"question": "Third unique question in Estonian"}}
]

Execute the task {num_samples} times now and return the JSON array:"""

        logger.info(f"BATCH GENERATION: {num_samples} samples in 1 API call")
        logger.debug(f"BATCH PROMPT:\n{'-'*80}\n{batch_prompt[:500]}...\n{'-'*80}")
        
        try:
            # Set higher max_tokens for batch requests
            batch_options = options.copy() if options else {}
            batch_options['max_tokens'] = min(
                self.max_tokens_per_batch,
                batch_options.get('max_tokens', self.max_tokens)
            )
            
            logger.info(f"MODEL: {self.deployment_name}")
            logger.info(f"OPTIONS: {batch_options}")
            
            response = self.generate(batch_prompt, batch_options)
            duration = time.time() - start_time
            
            logger.info(f"BATCH RESPONSE:\n{'-'*80}\n{response[:500]}...\n{'-'*80}")
            
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
                                    logger.debug(f"✅ BATCH SAMPLE {i+1}: {type(sample).__name__} format")
                            
                            if valid_samples:
                                logger.info(f"BATCH SUCCESS: Generated {len(valid_samples)} valid samples in {duration:.2f}s")
                                return valid_samples
                        else:
                            logger.warning(f"Expected {num_samples} samples, got {len(samples)}. Falling back to individual calls.")
                    else:
                        logger.warning(f"Response is not a list but {type(samples)}. Falling back to individual calls.")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {e}. Falling back to individual calls.")
                    
            else:
                logger.warning("Could not find JSON array in batch response. Falling back to individual calls.")
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"BATCH GENERATION FAILED after {duration:.2f}s: {e}")
        
        # Fallback to individual generation
        logger.info(f"FALLING BACK to individual generation for {num_samples} samples")
        return [self.generate(prompt, options) for _ in range(num_samples)]
    
    def health_check(self) -> bool:
        """Check if Azure OpenAI service is accessible"""
        try:
            # Make a simple API call to check connectivity
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint
            )
            
            # Try to list models as a health check
            client.models.list()
            return True
            
        except Exception as e:
            logger.error(f"Azure OpenAI health check failed: {e}")
            return False
