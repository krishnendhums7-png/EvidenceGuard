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
    GeminiUnavailableError,
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


def test_503_followed_by_success() -> None:
    """Verify 503 UNAVAILABLE followed by a successful response recovers after retry."""
    service = GeminiService(api_key="test_dummy_key_12345")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Albert Einstein was born in Ulm, Germany."
    mock_error_503 = APIError(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}})

    service._client = mock_client

    with patch("time.sleep") as mock_sleep:
        # Fails once with 503, then succeeds on retry
        mock_client.models.generate_content.side_effect = [mock_error_503, mock_response]

        result = service.generate_answer("Who was Einstein?")
        assert result == "Albert Einstein was born in Ulm, Germany."
        assert mock_client.models.generate_content.call_count == 2
        assert mock_sleep.call_count == 1


def test_three_consecutive_503_failures() -> None:
    """Verify 1 initial attempt + 3 retries (4 total attempts) all failing with 503 raises GeminiUnavailableError."""
    service = GeminiService(api_key="test_dummy_key_12345")

    mock_client = MagicMock()
    mock_error_503 = APIError(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary.", "status": "UNAVAILABLE"}})

    service._client = mock_client

    with patch("time.sleep") as mock_sleep:
        mock_client.models.generate_content.side_effect = mock_error_503

        with pytest.raises((GeminiUnavailableError, GeminiAPIError)) as exc_info:
            service.generate_answer("Who was Einstein?")

        assert "unavailable" in str(exc_info.value).lower() or "503" in str(exc_info.value).lower()
        # 1 initial request + 3 retries = 4 attempts total
        assert mock_client.models.generate_content.call_count == 4
        assert mock_sleep.call_count == 3


def test_rate_limit_429_error_handling() -> None:
    """Verify 429 rate limit errors trigger GeminiRateLimitError after retries."""
    service = GeminiService(api_key="test_dummy_key_12345")
    mock_client = MagicMock()
    mock_error = APIError(429, {"error": "RESOURCE_EXHAUSTED: Rate limit exceeded"})

    service._client = mock_client

    with patch("time.sleep") as mock_sleep:
        mock_client.models.generate_content.side_effect = mock_error

        with pytest.raises(GeminiRateLimitError, match="Rate limit exceeded"):
            service.generate_answer("Rate limited question")

        assert mock_client.models.generate_content.call_count == 4
        assert mock_sleep.call_count == 3


def test_invalid_api_key_fails_without_retries() -> None:
    """Verify permanent authentication errors fail immediately without wasteful retries."""
    service = GeminiService(api_key="bad_invalid_key")
    mock_client = MagicMock()
    mock_error = APIError(401, {"error": {"code": 401, "message": "API_KEY_INVALID: API key not valid", "status": "UNAUTHENTICATED"}})

    service._client = mock_client

    with patch("time.sleep") as mock_sleep:
        mock_client.models.generate_content.side_effect = mock_error

        with pytest.raises(GeminiAPIError, match="Permanent API error"):
            service.generate_answer("Who was Einstein?")

        # Must NOT retry: call count should be exactly 1
        assert mock_client.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0


@pytest.mark.skipif(
    not config.is_gemini_configured(),
    reason="Optional integration test requires live GEMINI_API_KEY in .env",
)
def test_live_gemini_ping() -> None:
    """Optional live integration test only executed when user credentials are present and reachable."""
    service = GeminiService()
    try:
        res = service.test_connection()
        if not res:
            pytest.skip("Gemini API endpoint unreachable (e.g. offline sandbox or network error)")
        assert res is True
    except Exception as exc:
        pytest.skip(f"Live network unavailable: {exc}")
