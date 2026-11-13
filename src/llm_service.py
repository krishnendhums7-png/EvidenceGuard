"""Gemini LLM integration service using the official Google Gen AI SDK.

Provides question-answering generation, structured claim extraction prompting,
exponential backoff retry logic, timeout handling, and graceful offline fallback.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import config
from src.logger import get_logger

logger = get_logger("llm_service")

T = TypeVar("T")

# Standard prompt for answering factual questions
FACTUAL_QA_SYSTEM_PROMPT = (
    "You are a factual, concise, and academically precise assistant. "
    "Provide clear, factual responses based strictly on established historical and scientific consensus. "
    "Do not include conversational filler, disclaimers, or speculation."
)

# Standard prompt for atomic claim decomposition
CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are an expert NLP fact-verification analyst.
Your task is to decompose the provided text into individual, atomic, independently verifiable factual claims.

Rules for atomic claim extraction:
1. Break compound sentences into standalone atomic statements.
2. Each claim must express exactly ONE verifiable proposition.
3. Replace pronouns (he, she, it, they) with their referenced entities wherever possible to make claims fully independent.
4. Preserve all specific dates, locations, measurements, and numerical figures from the original text.
5. Do NOT extrapolate, speculate, or introduce facts not explicitly stated in the source text.
6. Omit conversational filler, greetings, and purely subjective opinions.
7. Return your response as a valid JSON object matching this schema:
{
  "claims": [
    {
      "claim_id": "claim_1",
      "claim_text": "First independent factual claim.",
      "entities": ["Entity1", "Entity2"]
    }
  ]
}
"""


class GeminiAPIError(Exception):
    """Base exception for Gemini LLM service errors."""


class MissingAPIKeyError(GeminiAPIError):
    """Raised when an operation requires an API key that is not configured."""


class GeminiRateLimitError(GeminiAPIError):
    """Raised when API rate limits are exceeded."""


class GeminiTimeoutError(GeminiAPIError):
    """Raised when an API call exceeds the configured request timeout."""


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for exponential backoff retry on transient API failures."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_err: Optional[Exception] = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (APIError, TimeoutError, ConnectionError) as err:
                    last_err = err
                    err_msg = str(err)
                    is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg

                    if attempt == max_retries:
                        logger.error(
                            "Gemini API call failed after %d attempts: %s",
                            max_retries,
                            err_msg,
                        )
                        if is_rate_limit:
                            raise GeminiRateLimitError(f"Rate limit exceeded: {err_msg}") from err
                        raise GeminiAPIError(f"API request failed: {err_msg}") from err

                    logger.warning(
                        "Attempt %d/%d failed with error (%s). Retrying in %.1fs...",
                        attempt,
                        max_retries,
                        err_msg,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                except Exception as err:
                    # Non-retriable error
                    logger.error("Non-retriable error in Gemini API call: %s", err)
                    raise

            raise GeminiAPIError(f"Operation failed after {max_retries} retries: {last_err}")

        return wrapper

    return decorator


class GeminiService:
    """Service wrapper for Google Gemini API via official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize the Gemini service.

        Args:
            api_key: Optional override for the Gemini API key. Defaults to config.gemini_api_key.
            model_name: Model identifier. Defaults to config.gemini_model_name.
            max_retries: Maximum retry attempts for transient errors.
            timeout: Request timeout in seconds.
        """
        self.api_key: Optional[str] = api_key or config.gemini_api_key
        self.model_name: str = model_name or config.gemini_model_name
        self.max_retries: int = max_retries or config.gemini_max_retries
        self.timeout: float = timeout or config.gemini_request_timeout
        self._client: Optional[genai.Client] = None

        if self.is_available():
            self._init_client()

    def _init_client(self) -> None:
        """Instantiate the official google-genai Client."""
        if not self.api_key:
            return
        try:
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Initialized Google Gen AI Client with model '%s'", self.model_name)
        except Exception as err:
            logger.error("Failed to initialize Google Gen AI Client: %s", err)
            self._client = None

    def is_available(self) -> bool:
        """Check whether Gemini is configured with a valid API key."""
        return bool(self.api_key and self.api_key.strip())

    @property
    def client(self) -> genai.Client:
        """Access the underlying Client, raising an informative error if unconfigured."""
        if not self.is_available():
            raise MissingAPIKeyError(
                "Gemini API key is not configured. Add your GEMINI_API_KEY to the .env file. "
                "Alternatively, you can verify pasted LLM responses directly without Gemini."
            )
        if self._client is None:
            self._init_client()
        if self._client is None:
            raise GeminiAPIError("Failed to initialize Gemini API client.")
        return self._client

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def generate_answer(
        self,
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Generate a factual response for a user question.

        Args:
            question: User inquiry or prompt.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Output token limit.

        Returns:
            Generated response string.
        """
        if not question or not question.strip():
            return ""

        clean_q = question.strip()
        logger.info("Generating Gemini answer for question: '%s' (model=%s)", clean_q[:40], self.model_name)

        gen_config = types.GenerateContentConfig(
            system_instruction=FACTUAL_QA_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=clean_q,
            config=gen_config,
        )

        answer_text = response.text or ""
        logger.debug("Received answer (%d characters)", len(answer_text))
        return answer_text.strip()

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def extract_claims_raw(
        self,
        text: str,
        temperature: float = 0.0,
    ) -> str:
        """Request structured atomic claim extraction via Gemini.

        Args:
            text: Prose or response to decompose.
            temperature: Sampling temperature (0.0 for deterministic output).

        Returns:
            Raw JSON string from Gemini.
        """
        if not text or not text.strip():
            return '{"claims": []}'

        clean_text = text.strip()
        logger.info("Requesting Gemini atomic claim extraction for text (%d chars)", len(clean_text))

        gen_config = types.GenerateContentConfig(
            system_instruction=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=temperature,
        )

        prompt = f"Decompose the following text into atomic factual claims:\n\n{clean_text}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=gen_config,
        )

        output_text = response.text or ""
        return output_text.strip()

    def test_connection(self) -> bool:
        """Perform a lightweight test request to verify API key validity.

        Returns:
            True if connection and authentication succeed, False otherwise.
        """
        if not self.is_available():
            return False

        try:
            logger.info("Testing Gemini API connectivity with test ping...")
            _ = self.generate_answer("Ping. Respond with Pong.", max_tokens=10)
            logger.info("Gemini API connection test passed.")
            return True
        except Exception as err:
            logger.warning("Gemini API connectivity test failed: %s", err)
            return False


# Global singleton instance
_default_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Accessor for the global GeminiService instance."""
    global _default_gemini_service
    if _default_gemini_service is None:
        _default_gemini_service = GeminiService()
    return _default_gemini_service
