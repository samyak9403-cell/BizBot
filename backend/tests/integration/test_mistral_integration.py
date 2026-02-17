"""Integration tests for MistralClient with Config."""

import pytest
from unittest.mock import patch, Mock

from src.config import Config
from src.mistral_client import MistralClient


class TestMistralClientConfigIntegration:
    """Test MistralClient integration with Config."""
    
    @patch.dict('os.environ', {
        'MISTRAL_API_KEY': 'test_integration_key_12345',
        'MISTRAL_MODEL': 'mistral-large-latest'
    })
    def test_create_client_from_config(self):
        """Test creating MistralClient from Config object."""
        config = Config()
        config.MISTRAL_API_KEY = "test_integration_key_12345"
        config.MISTRAL_MODEL = "mistral-large-latest"
        
        client = MistralClient.from_config(config)
        
        assert client.api_key == "test_integration_key_12345"
        assert client.model == "mistral-large-latest"
    
    @patch('src.mistral_client.Mistral')
    @patch.dict('os.environ', {
        'MISTRAL_API_KEY': 'test_integration_key_12345',
        'MISTRAL_MODEL': 'mistral-large-latest'
    })
    def test_client_uses_config_model(self, mock_mistral_class):
        """Test that client uses model from config."""
        # Setup mock
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(total_tokens=50)
        mock_client.chat.complete.return_value = mock_response
        
        # Create client from config
        config = Config()
        config.MISTRAL_API_KEY = "test_integration_key_12345"
        config.MISTRAL_MODEL = "mistral-large-latest"
        
        client = MistralClient.from_config(config)
        
        # Make a chat completion
        messages = [{"role": "user", "content": "Hello"}]
        client.chat_complete(messages)
        
        # Verify the correct model was used
        mock_client.chat.complete.assert_called_once()
        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs['model'] == "mistral-large-latest"
