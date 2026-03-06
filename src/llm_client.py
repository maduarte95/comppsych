"""LLM client abstraction for multiple providers."""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from anthropic import Anthropic

from src.config import LLMParams

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    choice: str  # "A" or "B"
    raw_response: str
    parse_attempts: int


@dataclass
class ParseError:
    """Record of a parsing error."""

    raw_response: str
    error_message: str
    attempt: int


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a completion request to the LLM.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            Raw response text from the LLM
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    def complete_with_retry(
        self, system: str, user: str, max_retries: int = 3
    ) -> tuple[LLMResponse | None, list[ParseError]]:
        """Complete with retry logic for parse errors.

        Args:
            system: System prompt
            user: User prompt
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (LLMResponse or None, list of parse errors)
        """
        errors: list[ParseError] = []

        for attempt in range(1, max_retries + 1):
            raw_response = self.complete(system, user)
            choice = self._parse_choice(raw_response)

            if choice is not None:
                return (
                    LLMResponse(
                        choice=choice,
                        raw_response=raw_response,
                        parse_attempts=attempt,
                    ),
                    errors,
                )

            error = ParseError(
                raw_response=raw_response,
                error_message=f"Could not parse choice from response",
                attempt=attempt,
            )
            errors.append(error)
            logger.warning(
                f"Parse error (attempt {attempt}/{max_retries}): {raw_response[:100]}"
            )

        return None, errors

    def _parse_choice(self, response: str) -> str | None:
        """Parse choice from LLM response.

        Expects JSON like {"choice": "A"} or {"choice": "B"}

        Args:
            response: Raw LLM response

        Returns:
            "A" or "B" if successfully parsed, None otherwise
        """
        # Try to find JSON in response
        # First try direct JSON parse
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict) and "choice" in data:
                choice = data["choice"].upper()
                if choice in ("A", "B"):
                    return choice
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response (may have whitespace or newlines)
        json_patterns = [
            r'\{\s*"choice"\s*:\s*"([AB])"\s*\}',
            r'\{\s*\'choice\'\s*:\s*\'([AB])\'\s*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    # Available models (as of 2025)
    MODELS = [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ]

    # JSON schema for structured output - guarantees {"choice": "A"} or {"choice": "B"}
    CHOICE_SCHEMA = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": ["A", "B"],
            }
        },
        "required": ["choice"],
        "additionalProperties": False,
    }

    def __init__(self, params: LLMParams, api_key: str | None = None) -> None:
        """Initialize Anthropic client.

        Args:
            params: LLM parameters
            api_key: API key (if None, uses ANTHROPIC_API_KEY env var)
        """
        self.params = params
        self.client = Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        """Send a completion request to Claude with structured output.

        Uses output_config to guarantee valid JSON response.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            Raw response text (guaranteed to be valid JSON)
        """
        message = self.client.messages.create(
            model=self.params.model,
            max_tokens=self.params.max_tokens,
            temperature=self.params.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": self.CHOICE_SCHEMA,
                }
            },
        )

        # Extract text from response
        if message.content and len(message.content) > 0:
            return message.content[0].text
        return ""

    def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models."""
        return self.MODELS.copy()


class OpenRouterClient(LLMClient):
    """OpenRouter API client (uses OpenAI SDK with custom base URL)."""

    # Common models available via OpenRouter
    MODELS = [
        "anthropic/claude-3-5-haiku",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-opus",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-405b-instruct",
        "google/gemini-pro-1.5",
    ]

    def __init__(self, params: LLMParams, api_key: str | None = None) -> None:
        """Initialize OpenRouter client.

        Args:
            params: LLM parameters
            api_key: OpenRouter API key
        """
        # Import here to avoid dependency if not using OpenRouter
        from openai import OpenAI

        self.params = params
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def complete(self, system: str, user: str) -> str:
        """Send a completion request via OpenRouter.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            Raw response text
        """
        response = self.client.chat.completions.create(
            model=self.params.model,
            max_tokens=self.params.max_tokens,
            temperature=self.params.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""

    def get_available_models(self) -> list[str]:
        """Get list of common OpenRouter models."""
        return self.MODELS.copy()


def create_client(params: LLMParams, api_key: str | None = None) -> LLMClient:
    """Factory function to create appropriate LLM client.

    Args:
        params: LLM parameters (includes provider)
        api_key: API key for the provider

    Returns:
        LLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    if params.provider == "anthropic":
        return AnthropicClient(params, api_key)
    elif params.provider == "openrouter":
        return OpenRouterClient(params, api_key)
    else:
        raise ValueError(f"Unknown provider: {params.provider}")
