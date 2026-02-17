"""Mistral AI client wrapper with retry logic and error handling."""

import logging
import time
from typing import List, Union, Optional, Dict, Any

import numpy as np
from mistralai import Mistral

from .config import Config


logger = logging.getLogger(__name__)


class MistralAPIError(Exception):
    """Raised when Mistral API calls fail."""
    pass


class MistralClient:
    """Wrapper for Mistral AI API with retry logic and error handling.
    
    Provides methods for text embeddings and chat completions with:
    - Exponential backoff retry logic for transient errors
    - Rate limit handling with Retry-After header support
    - Comprehensive logging of API calls and metrics
    - User-friendly error messages
    """
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        """Initialize Mistral client.
        
        Args:
            api_key: Mistral API key
            model: Model name to use for chat completions
            
        Raises:
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key
        self.model = model
        self.client = Mistral(api_key=api_key)
        
        logger.info(f"Initialized MistralClient with model: {model}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Embedding vector(s) as numpy array
            - Single text: shape (embedding_dim,)
            - Multiple texts: shape (num_texts, embedding_dim)
            
        Raises:
            MistralAPIError: If embedding generation fails
        """
        # Convert single string to list for uniform processing
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        if not text_list:
            raise ValueError("Cannot embed empty text list")
        
        # Retry logic with exponential backoff
        max_retries = 3
        base_wait_time = 2  # seconds
        
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                logger.debug(f"Generating embeddings for {len(text_list)} text(s) (attempt {attempt + 1}/{max_retries})")
                
                # Call Mistral embeddings API
                response = self.client.embeddings.create(
                    model="mistral-embed",
                    inputs=text_list
                )
                
                # Extract embeddings from response
                embeddings = [item.embedding for item in response.data]
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Log metrics
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Generated {len(text_list)} embedding(s) in {elapsed_time:.2f}s, "
                    f"dimension: {embeddings_array.shape[-1]}"
                )
                
                # Return single vector or array based on input
                if is_single:
                    return embeddings_array[0]
                return embeddings_array
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_str = str(e)
                
                # Check if rate limited (429) or service busy
                is_rate_limited = "429" in error_str or "rate_limit" in error_str.lower()
                is_busy = "503" in error_str or "500" in error_str or "busy" in error_str.lower()
                is_retryable = is_rate_limited or is_busy
                
                if attempt < max_retries - 1 and is_retryable:
                    # Calculate exponential backoff with jitter
                    wait_time = base_wait_time * (2 ** attempt)
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{max_retries}): {self._get_user_friendly_error(e)}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                
                # Final attempt failed or non-retryable error
                logger.error(
                    f"Embedding generation failed after {elapsed_time:.2f}s (attempt {attempt + 1}/{max_retries}): {error_str}",
                    exc_info=True
                )
                raise MistralAPIError(
                    f"Failed to generate embeddings: {self._get_user_friendly_error(e)}"
                ) from e
    
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            MistralAPIError: If chat completion fails
            ValueError: If messages format is invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys")
        
        start_time = time.time()
        
        try:
            logger.debug(
                f"Generating chat completion with model: {self.model}, "
                f"temperature: {temperature}, max_tokens: {max_tokens}"
            )
            
            # Call Mistral chat API
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            # Log metrics
            elapsed_time = time.time() - start_time
            tokens_used = getattr(response.usage, 'total_tokens', 'unknown')
            
            logger.info(
                f"Chat completion successful: model={self.model}, "
                f"tokens={tokens_used}, latency={elapsed_time:.2f}s"
            )
            
            return generated_text
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"Chat completion failed after {elapsed_time:.2f}s: {str(e)}",
                exc_info=True
            )
            raise MistralAPIError(
                f"Failed to generate chat completion: {self._get_user_friendly_error(e)}"
            ) from e
    
    def complete_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Chat completion with exponential backoff retry.
        
        Handles transient errors and rate limiting with intelligent retry logic:
        - Exponential backoff for transient errors
        - Respects Retry-After headers for rate limits
        - Logs all retry attempts
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            MistralAPIError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.chat_complete(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except MistralAPIError as e:
                last_exception = e
                
                # Don't retry on last attempt
                if attempt == max_retries:
                    break
                
                # Check if the underlying error is retryable
                underlying_error = e.__cause__ if e.__cause__ else e
                if not self._is_retryable_error(underlying_error):
                    logger.warning(f"Non-retryable error encountered: {str(e)}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = self._calculate_retry_delay(underlying_error, attempt, base_delay)
                
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s delay. "
                    f"Error: {str(e)}"
                )
                
                time.sleep(delay)
        
        # All retries exhausted
        logger.error(f"All {max_retries} retry attempts failed")
        raise MistralAPIError(
            f"Failed after {max_retries} retries: {self._get_user_friendly_error(last_exception)}"
        ) from last_exception
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The exception to check
            
        Returns:
            True if error is transient and should be retried
        """
        error_str = str(error).lower()
        
        # Rate limiting errors (429)
        if '429' in error_str or 'rate limit' in error_str:
            return True
        
        # Service unavailable (503)
        if '503' in error_str or 'service unavailable' in error_str:
            return True
        
        # Timeout errors
        if 'timeout' in error_str:
            return True
        
        # Connection errors
        if 'connection' in error_str:
            return True
        
        # Non-retryable errors: authentication (401), bad request (400), etc.
        if '401' in error_str or '400' in error_str or '403' in error_str:
            return False
        
        # Default to not retrying unknown errors
        return False
    
    def _calculate_retry_delay(
        self,
        error: Exception,
        attempt: int,
        base_delay: float
    ) -> float:
        """Calculate retry delay with exponential backoff.
        
        Respects Retry-After headers if present in rate limit errors.
        
        Args:
            error: The exception that triggered the retry
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay for exponential backoff
            
        Returns:
            Delay in seconds before next retry
        """
        # Check for Retry-After header in error message
        retry_after = self._extract_retry_after(error)
        if retry_after is not None:
            logger.debug(f"Using Retry-After header value: {retry_after}s")
            return retry_after
        
        # Exponential backoff: delay = base_delay * 2^attempt
        delay = base_delay * (2 ** attempt)
        
        # Add jitter to prevent thundering herd (±20%)
        import random
        jitter = delay * 0.2 * (2 * random.random() - 1)
        delay_with_jitter = delay + jitter
        
        # Cap maximum delay at 60 seconds
        return min(delay_with_jitter, 60.0)
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract Retry-After value from error if present.
        
        Args:
            error: The exception to check
            
        Returns:
            Retry-After value in seconds, or None if not found
        """
        error_str = str(error)
        
        # Look for "Retry-After: X" pattern
        import re
        match = re.search(r'retry[- ]after[:\s]+(\d+)', error_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _get_user_friendly_error(self, error: Exception) -> str:
        """Convert technical error to user-friendly message.
        
        Args:
            error: The exception to convert
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        # Rate limiting
        if '429' in error_str or 'rate limit' in error_str:
            return "Service is currently busy. Please try again in a moment."
        
        # Authentication
        if '401' in error_str or 'unauthorized' in error_str:
            return "API authentication failed. Please check your API key configuration."
        
        # Service unavailable
        if '503' in error_str or 'service unavailable' in error_str:
            return "Service is temporarily unavailable. Please try again later."
        
        # Timeout
        if 'timeout' in error_str:
            return "Request timed out. Please try again."
        
        # Connection errors
        if 'connection' in error_str:
            return "Unable to connect to the service. Please check your internet connection."
        
        # Bad request
        if '400' in error_str or 'bad request' in error_str:
            return "Invalid request. Please check your input and try again."
        
        # Generic error
        return "An unexpected error occurred. Please try again later."
    
    @classmethod
    def from_config(cls, config: Config) -> "MistralClient":
        """Create MistralClient from Config object.
        
        Args:
            config: Configuration object
            
        Returns:
            Configured MistralClient instance
        """
        return cls(api_key=config.MISTRAL_API_KEY, model=config.MISTRAL_MODEL)
