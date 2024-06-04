import pytest
import requests
from unittest.mock import patch, Mock, mock_open
import pandas as pd
from src.document_loader import DocumentLoader

@patch('document_loader.requests.get')
def test_fetch_document(mock_get):
    loader = DocumentLoader()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><p>Sample text</p></body></html>'
    mock_get.return_value = mock_response

    text = loader.fetch_document('http://example.com')
    assert text == 'Sample text'

    mock_get.side_effect = requests.exceptions.RequestException
    assert loader.fetch_document('http://example.com') is None


if __name__ == '__main__':
    pytest.main()
