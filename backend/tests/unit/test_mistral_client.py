"""Unit tests for MistralClient."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.mistral_client import MistralClient, MistralAPIError
from src.config import Config


class TestMistralClientInitialization:
    """Test MistralClient initialization."""
    
    def test_init_with_valid_api_key(self):
        """Test initialization with valid API key."""
        client = MistralClient(api_key="test_api_key_12345", model="mistral-large-latest")
        assert client.api_key == "test_api_key_12345"
        assert client.model == "mistral-large-latest"
    
    def test_init_with_empty_api_key_raises_error(self):
        """Test initialization with empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            MistralClient(api_key="", model="mistral-large-latest")
    
    def test_init_with_default_model(self):
        """Test initialization uses default model when not specified."""
        client = MistralClient(api_key="test_api_key_12345")
        assert client.model == "mistral-large-latest"
    
    def test_from_config(self):
        """Test creating client from Config object."""
        with patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_key_from_config'}):
            config = Config()
            config.MISTRAL_API_KEY = "test_key_from_config"
            config.MISTRAL_MODEL = "mistral-large-latest"
            
            client = MistralClient.from_config(config)
            assert client.api_key == "test_key_from_config"
            assert client.model == "mistral-large-latest"


class TestMistralClientEmbed:
    """Test embedding generation."""
    
    @patch('src.mistral_client.Mistral')
    def test_embed_single_text(self, mock_mistral_class):
        """Test embedding generation for single text."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        client = MistralClient(api_key="test_key")
        result = client.embed("test text")
        
        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])
        mock_client.embeddings.create.assert_called_once_with(
            model="mistral-embed",
            inputs=["test text"]
        )
    
    @patch('src.mistral_client.Mistral')
    def test_embed_multiple_texts(self, mock_mistral_class):
        """Test embedding generation for multiple texts."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        client = MistralClient(api_key="test_key")
        result = client.embed(["text1", "text2"])
        
        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5, 0.6])
    
    @patch('src.mistral_client.Mistral')
    def test_embed_empty_list_raises_error(self, mock_mistral_class):
        """Test embedding empty list raises ValueError."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        client = MistralClient(api_key="test_key")
        
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            client.embed([])
    
    @patch('src.mistral_client.Mistral')
    def test_embed_api_error_raises_mistral_api_error(self, mock_mistral_class):
        """Test API error during embedding raises MistralAPIError."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        client = MistralClient(api_key="test_key")
        
        with pytest.raises(MistralAPIError, match="Failed to generate embeddings"):
            client.embed("test text")


class TestMistralClientChatComplete:
    """Test chat completion."""
    
    @patch('src.mistral_client.Mistral')
    def test_chat_complete_success(self, mock_mistral_class):
        """Test successful chat completion."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated response"))]
        mock_response.usage = Mock(total_tokens=100)
        mock_client.chat.complete.return_value = mock_response
        
        # Test
        client = MistralClient(api_key="test_key", model="mistral-large-latest")
        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat_complete(messages, temperature=0.7, max_tokens=4000)
        
        # Verify
        assert result == "Generated response"
        mock_client.chat.complete.assert_called_once_with(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )
    
    @patch('src.mistral_client.Mistral')
    def test_chat_complete_empty_messages_raises_error(self, mock_mistral_class):
        """Test chat completion with empty messages raises ValueError."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        client = MistralClient(api_key="test_key")
        
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            client.chat_complete([])
    
    @patch('src.mistral_client.Mistral')
    def test_chat_complete_invalid_message_format_raises_error(self, mock_mistral_class):
        """Test chat completion with invalid message format raises ValueError."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        client = MistralClient(api_key="test_key")
        
        with pytest.raises(ValueError, match="Each message must be a dict with 'role' and 'content' keys"):
            client.chat_complete([{"invalid": "format"}])
    
    @patch('src.mistral_client.Mistral')
    def test_chat_complete_api_error_raises_mistral_api_error(self, mock_mistral_class):
        """Test API error during chat completion raises MistralAPIError."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_client.chat.complete.side_effect = Exception("API Error")
        
        client = MistralClient(api_key="test_key")
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(MistralAPIError, match="Failed to generate chat completion"):
            client.chat_complete(messages)


class TestMistralClientRetryLogic:
    """Test retry logic with exponential backoff."""
    
    @patch('src.mistral_client.Mistral')
    @patch('src.mistral_client.time.sleep')
    def test_complete_with_retry_success_on_first_attempt(self, mock_sleep, mock_mistral_class):
        """Test successful completion on first attempt."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Success"))]
        mock_response.usage = Mock(total_tokens=50)
        mock_client.chat.complete.return_value = mock_response
        
        # Test
        client = MistralClient(api_key="test_key")
        messages = [{"role": "user", "content": "Hello"}]
        result = client.complete_with_retry(messages, max_retries=3)
        
        # Verify
        assert result == "Success"
        assert mock_client.chat.complete.call_count == 1
        mock_sleep.assert_not_called()
    
    @patch('src.mistral_client.Mistral')
    @patch('src.mistral_client.time.sleep')
    def test_complete_with_retry_success_after_retries(self, mock_sleep, mock_mistral_class):
        """Test successful completion after retries."""
        # Setup mock to fail twice then succeed
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Success"))]
        mock_response.usage = Mock(total_tokens=50)
        
        # First two calls fail with rate limit, third succeeds
        mock_client.chat.complete.side_effect = [
            Exception("429 rate limit exceeded"),
            Exception("503 service unavailable"),
            mock_response
        ]
        
        # Test
        client = MistralClient(api_key="test_key")
        messages = [{"role": "user", "content": "Hello"}]
        result = client.complete_with_retry(messages, max_retries=3, base_delay=0.1)
        
        # Verify
        assert result == "Success"
        assert mock_client.chat.complete.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('src.mistral_client.Mistral')
    @patch('src.mistral_client.time.sleep')
    def test_complete_with_retry_exhausts_retries(self, mock_sleep, mock_mistral_class):
        """Test retry exhaustion raises MistralAPIError."""
        # Setup mock to always fail
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_client.chat.complete.side_effect = Exception("429 rate limit exceeded")
        
        # Test
        client = MistralClient(api_key="test_key")
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(MistralAPIError, match="Failed after 3 retries"):
            client.complete_with_retry(messages, max_retries=3, base_delay=0.1)
        
        # Verify all retries were attempted
        assert mock_client.chat.complete.call_count == 4  # Initial + 3 retries
    
    @patch('src.mistral_client.Mistral')
    def test_complete_with_retry_non_retryable_error(self, mock_mistral_class):
        """Test non-retryable error raises immediately."""
        # Setup mock to fail with non-retryable error
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_client.chat.complete.side_effect = Exception("401 unauthorized")
        
        # Test
        client = MistralClient(api_key="test_key")
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(MistralAPIError):
            client.complete_with_retry(messages, max_retries=3)
        
        # Verify no retries were attempted
        assert mock_client.chat.complete.call_count == 1


class TestMistralClientErrorHandling:
    """Test error handling and user-friendly messages."""
    
    def test_is_retryable_error_rate_limit(self):
        """Test rate limit errors are retryable."""
        client = MistralClient(api_key="test_key")
        error = Exception("429 rate limit exceeded")
        assert client._is_retryable_error(error) is True
    
    def test_is_retryable_error_service_unavailable(self):
        """Test service unavailable errors are retryable."""
        client = MistralClient(api_key="test_key")
        error = Exception("503 service unavailable")
        assert client._is_retryable_error(error) is True
    
    def test_is_retryable_error_timeout(self):
        """Test timeout errors are retryable."""
        client = MistralClient(api_key="test_key")
        error = Exception("Request timeout")
        assert client._is_retryable_error(error) is True
    
    def test_is_retryable_error_unauthorized(self):
        """Test unauthorized errors are not retryable."""
        client = MistralClient(api_key="test_key")
        error = Exception("401 unauthorized")
        assert client._is_retryable_error(error) is False
    
    def test_is_retryable_error_bad_request(self):
        """Test bad request errors are not retryable."""
        client = MistralClient(api_key="test_key")
        error = Exception("400 bad request")
        assert client._is_retryable_error(error) is False
    
    def test_get_user_friendly_error_rate_limit(self):
        """Test user-friendly message for rate limit errors."""
        client = MistralClient(api_key="test_key")
        error = Exception("429 rate limit exceeded")
        message = client._get_user_friendly_error(error)
        assert "busy" in message.lower()
        assert "try again" in message.lower()
    
    def test_get_user_friendly_error_unauthorized(self):
        """Test user-friendly message for auth errors."""
        client = MistralClient(api_key="test_key")
        error = Exception("401 unauthorized")
        message = client._get_user_friendly_error(error)
        assert "authentication" in message.lower()
        assert "api key" in message.lower()
    
    def test_get_user_friendly_error_generic(self):
        """Test user-friendly message for unknown errors."""
        client = MistralClient(api_key="test_key")
        error = Exception("Unknown error")
        message = client._get_user_friendly_error(error)
        assert "unexpected error" in message.lower()


class TestMistralClientExponentialBackoff:
    """Test exponential backoff calculation."""
    
    def test_calculate_retry_delay_exponential(self):
        """Test exponential backoff calculation."""
        client = MistralClient(api_key="test_key")
        error = Exception("503 service unavailable")
        
        # Test exponential growth
        delay_0 = client._calculate_retry_delay(error, 0, 1.0)
        delay_1 = client._calculate_retry_delay(error, 1, 1.0)
        delay_2 = client._calculate_retry_delay(error, 2, 1.0)
        
        # Delays should roughly double (with jitter)
        assert 0.8 <= delay_0 <= 1.2  # ~1 second with jitter
        assert 1.6 <= delay_1 <= 2.4  # ~2 seconds with jitter
        assert 3.2 <= delay_2 <= 4.8  # ~4 seconds with jitter
    
    def test_calculate_retry_delay_respects_retry_after(self):
        """Test Retry-After header is respected."""
        client = MistralClient(api_key="test_key")
        error = Exception("429 rate limit exceeded. Retry-After: 5")
        
        delay = client._calculate_retry_delay(error, 0, 1.0)
        assert delay == 5.0
    
    def test_calculate_retry_delay_max_cap(self):
        """Test delay is capped at maximum."""
        client = MistralClient(api_key="test_key")
        error = Exception("503 service unavailable")
        
        # Large attempt number should be capped at 60 seconds
        delay = client._calculate_retry_delay(error, 10, 1.0)
        assert delay <= 60.0
    
    def test_extract_retry_after_from_error(self):
        """Test extracting Retry-After value from error."""
        client = MistralClient(api_key="test_key")
        
        error1 = Exception("Retry-After: 10")
        assert client._extract_retry_after(error1) == 10.0
        
        error2 = Exception("retry after 5 seconds")
        assert client._extract_retry_after(error2) == 5.0
        
        error3 = Exception("No retry info")
        assert client._extract_retry_after(error3) is None


class TestMistralClientLogging:
    """Test logging behavior."""
    
    @patch('src.mistral_client.Mistral')
    @patch('src.mistral_client.logger')
    def test_embed_logs_metrics(self, mock_logger, mock_mistral_class):
        """Test embedding generation logs metrics."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        client = MistralClient(api_key="test_key")
        client.embed("test text")
        
        # Verify logging
        assert mock_logger.info.called
        log_message = str(mock_logger.info.call_args)
        assert "embedding" in log_message.lower()
    
    @patch('src.mistral_client.Mistral')
    @patch('src.mistral_client.logger')
    def test_chat_complete_logs_metrics(self, mock_logger, mock_mistral_class):
        """Test chat completion logs metrics including tokens and latency."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(total_tokens=150)
        mock_client.chat.complete.return_value = mock_response
        
        # Test
        client = MistralClient(api_key="test_key", model="mistral-large-latest")
        messages = [{"role": "user", "content": "Hello"}]
        client.chat_complete(messages)
        
        # Verify logging includes model, tokens, and latency
        assert mock_logger.info.called
        log_message = str(mock_logger.info.call_args)
        assert "mistral-large-latest" in log_message
        assert "tokens" in log_message.lower()
        assert "latency" in log_message.lower()
    
    @patch('src.mistral_client.Mistral')
    @patch('src.mistral_client.logger')
    def test_api_key_never_logged(self, mock_logger, mock_mistral_class):
        """Test API key is never logged."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        api_key = "secret_api_key_12345"
        client = MistralClient(api_key=api_key)
        client.embed("test")
        
        # Verify API key is not in any log calls
        for call in mock_logger.info.call_args_list + mock_logger.debug.call_args_list:
            log_message = str(call)
            assert api_key not in log_message
