"""Unit tests for Gemini LLM service with mocking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from google.genai.errors import APIError

from src.config import config
from src.llm_service import (
    GeminiAPIError,
    GeminiRateLimitError,
    GeminiService,
    MissingAPIKeyError,
)


def test_gemini_initialization_missing_key() -> None:
    """Verify GeminiService behavior when no API key is provided."""
    service = GeminiService(api_key="")
    assert not service.is_available()
    with pytest.raises(MissingAPIKeyError, match="Gemini API key is not configured"):
        _ = service.client

    with pytest.raises(MissingAPIKeyError):
        service.generate_answer("Who was Einstein?")


def test_gemini_initialization_with_key() -> None:
    """Verify initialization when API key is provided."""
    service = GeminiService(api_key="test_dummy_key_12345", model_name="gemini-2.5-flash")
    assert service.is_available()
    assert service.model_name == "gemini-2.5-flash"


def test_generate_answer_success() -> None:
    """Verify successful answer generation using a mocked Client."""
    service = GeminiService(api_key="test_dummy_key_12345")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Albert Einstein was a theoretical physicist."
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client

    answer = service.generate_answer("Who was Einstein?")
    assert answer == "Albert Einstein was a theoretical physicist."
    mock_client.models.generate_content.assert_called_once()


def test_generate_answer_empty_question() -> None:
    """Verify empty question returns immediately without API calls."""
    service = GeminiService(api_key="test_dummy_key_12345")
    assert service.generate_answer("") == ""
    assert service.generate_answer("   ") == ""


def test_extract_claims_raw_success() -> None:
    """Verify extract_claims_raw sends correct prompt and returns JSON string."""
    service = GeminiService(api_key="test_dummy_key_12345")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"claims": [{"claim_id": "claim_1", "claim_text": "Einstein was German."}]}'
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client

    raw_json = service.extract_claims_raw("Albert Einstein was German.")
    assert "claim_1" in raw_json
    assert "Einstein was German" in raw_json
    mock_client.models.generate_content.assert_called_once()


def test_retry_on_transient_error() -> None:
    """Verify retry logic recovers from transient API errors."""
    service = GeminiService(api_key="test_dummy_key_12345", max_retries=3)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Successful answer on attempt 3"
    mock_error = APIError(503, {"error": "Service temporarily unavailable"})

    service._client = mock_client

    with patch("time.sleep") as mock_sleep:
        # Fails twice with 503, succeeds on third attempt
        mock_client.models.generate_content.side_effect = [mock_error, mock_error, mock_response]

        result = service.generate_answer("Test question")
        assert result == "Successful answer on attempt 3"
        assert mock_client.models.generate_content.call_count == 3
        assert mock_sleep.call_count == 2


def test_rate_limit_error_handling() -> None:
    """Verify 429 rate limit errors trigger GeminiRateLimitError after retries."""
    service = GeminiService(api_key="test_dummy_key_12345", max_retries=2)
    mock_client = MagicMock()
    mock_error = APIError(429, {"error": "Resource Exhausted: Rate limit exceeded"})

    service._client = mock_client

    with patch("time.sleep"):
        mock_client.models.generate_content.side_effect = mock_error

        with pytest.raises(GeminiRateLimitError, match="Rate limit exceeded"):
            service.generate_answer("Rate limited question")


@pytest.mark.skipif(
    not config.is_gemini_configured(),
    reason="Optional integration test requires live GEMINI_API_KEY in .env",
)
def test_live_gemini_ping() -> None:
    """Optional live integration test only executed when user credentials are present."""
    service = GeminiService()
    assert service.test_connection() is True
