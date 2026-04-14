"""LLM client abstraction for multiple providers."""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from anthropic import Anthropic, AsyncAnthropic

from src.config import LLMParams

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    choice: str  # "A" or "B"
    raw_response: str
    parse_attempts: int
    thinking: str | None = None  # Thinking content if thinking mode enabled


@dataclass
class ParseError:
    """Record of a parsing error."""

    raw_response: str
    error_message: str
    attempt: int


@dataclass
class CompletionResult:
    """Result from a single LLM completion call."""

    text: str  # The text response (JSON with choice)
    thinking: str | None = None  # Thinking content if enabled


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, system: str, user: str) -> CompletionResult:
        """Send a completion request to the LLM.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            CompletionResult with text response and optional thinking
        """
        pass

    @abstractmethod
    async def acomplete(self, system: str, user: str) -> CompletionResult:
        """Async version of complete().

        Args:
            system: System prompt
            user: User prompt

        Returns:
            CompletionResult with text response and optional thinking
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
            # Debug: Print full prompt
            print("\n" + "=" * 80)
            print(f"LLM CALL (attempt {attempt}/{max_retries})")
            print("=" * 80)
            print("\n--- SYSTEM PROMPT ---")
            print(system)
            print("\n--- USER PROMPT ---")
            print(user)
            print("-" * 80)

            result = self.complete(system, user)

            # Debug: Print full response
            if result.thinking:
                print("\n--- THINKING ---")
                print(result.thinking)
            print("\n--- RESPONSE ---")
            print(result.text)
            print("=" * 80 + "\n")

            choice = self._parse_choice(result.text)

            if choice is not None:
                return (
                    LLMResponse(
                        choice=choice,
                        raw_response=result.text,
                        parse_attempts=attempt,
                        thinking=result.thinking,
                    ),
                    errors,
                )

            error = ParseError(
                raw_response=result.text,
                error_message="Could not parse choice from response",
                attempt=attempt,
            )
            errors.append(error)
            logger.warning(
                f"Parse error (attempt {attempt}/{max_retries}): {result.text[:100]}"
            )

        return None, errors

    async def acomplete_with_retry(
        self, system: str, user: str, max_retries: int = 3
    ) -> tuple[LLMResponse | None, list[ParseError]]:
        """Async version of complete_with_retry()."""
        errors: list[ParseError] = []

        for attempt in range(1, max_retries + 1):
            result = await self.acomplete(system, user)

            choice = self._parse_choice(result.text)

            if choice is not None:
                return (
                    LLMResponse(
                        choice=choice,
                        raw_response=result.text,
                        parse_attempts=attempt,
                        thinking=result.thinking,
                    ),
                    errors,
                )

            error = ParseError(
                raw_response=result.text,
                error_message="Could not parse choice from response",
                attempt=attempt,
            )
            errors.append(error)
            logger.warning(
                f"Parse error (attempt {attempt}/{max_retries}): {result.text[:100]}"
            )

        return None, errors

    def _parse_choice(self, response: str) -> str | None:
        """Parse choice from LLM response.

        Expects structured JSON output ({"choice": "A"} or {"choice": "B"}).

        Args:
            response: Raw LLM response

        Returns:
            "A" or "B" if successfully parsed, None otherwise
        """
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

    # Models that support adaptive thinking (4.6 models)
    ADAPTIVE_THINKING_MODELS = ["claude-sonnet-4-6", "claude-opus-4-6"]

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
        self.async_client = AsyncAnthropic(api_key=api_key)

    def _supports_adaptive_thinking(self) -> bool:
        """Check if current model supports adaptive thinking."""
        return self.params.model in self.ADAPTIVE_THINKING_MODELS

    def _get_thinking_config(self) -> dict | None:
        """Get thinking configuration based on params and model support.

        Returns:
            Thinking config dict or None if disabled
        """
        if self.params.thinking_mode == "disabled":
            return None

        if self.params.thinking_mode == "adaptive":
            if not self._supports_adaptive_thinking():
                # Fall back to enabled mode for older models
                logger.warning(
                    f"Model {self.params.model} doesn't support adaptive thinking, "
                    f"using enabled mode with budget_tokens={self.params.thinking_budget}"
                )
                return {
                    "type": "enabled",
                    "budget_tokens": self.params.thinking_budget,
                }
            return {"type": "adaptive"}

        # thinking_mode == "enabled"
        return {
            "type": "enabled",
            "budget_tokens": self.params.thinking_budget,
        }

    def complete(self, system: str, user: str) -> CompletionResult:
        """Send a completion request to Claude.

        Supports thinking modes and structured output.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            CompletionResult with text response and optional thinking
        """
        thinking_config = self._get_thinking_config()

        # Build API call kwargs
        kwargs: dict = {
            "model": self.params.model,
            "max_tokens": self.params.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Always use structured output - it's compatible with thinking mode
            # (grammar applies only to final text output, not thinking blocks)
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": self.CHOICE_SCHEMA,
                }
            },
        }

        # Add thinking config if enabled
        if thinking_config:
            kwargs["thinking"] = thinking_config
            # Thinking is incompatible with temperature modifications
            # (must use default temperature=1.0)
        else:
            # Only set temperature when thinking is disabled
            kwargs["temperature"] = self.params.temperature

        message = self.client.messages.create(**kwargs)

        # Extract thinking and text from response
        thinking_content = None
        text_content = ""

        for block in message.content:
            if block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "text":
                text_content = block.text

        return CompletionResult(text=text_content, thinking=thinking_content)

    async def acomplete(self, system: str, user: str) -> CompletionResult:
        """Async version of complete() using AsyncAnthropic."""
        thinking_config = self._get_thinking_config()

        kwargs: dict = {
            "model": self.params.model,
            "max_tokens": self.params.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": self.CHOICE_SCHEMA,
                }
            },
        }

        if thinking_config:
            kwargs["thinking"] = thinking_config
        else:
            kwargs["temperature"] = self.params.temperature

        message = await self.async_client.messages.create(**kwargs)

        thinking_content = None
        text_content = ""

        for block in message.content:
            if block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "text":
                text_content = block.text

        return CompletionResult(text=text_content, thinking=thinking_content)

    def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models."""
        return self.MODELS.copy()


class OpenRouterClient(LLMClient):
    """OpenRouter API client (uses OpenAI SDK with custom base URL)."""

    # Models available via OpenRouter
    MODELS = [
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
        "google/gemini-2.5-flash-lite",
        # Legacy models kept for reference:
        # "anthropic/claude-3-5-haiku",
        # "anthropic/claude-3.5-sonnet",
        # "anthropic/claude-3-opus",
        # "openai/gpt-4o",
        # "openai/gpt-4o-mini",
        # "meta-llama/llama-3.1-405b-instruct",
        # "google/gemini-pro-1.5",
    ]

    # Models that support the OpenRouter reasoning parameter
    REASONING_MODELS = [
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
        "google/gemini-2.5-flash-lite",
    ]

    # Response format for structured output (json_object is broadly supported across OpenRouter models)
    CHOICE_RESPONSE_FORMAT = {"type": "json_object"}

    def __init__(self, params: LLMParams, api_key: str | None = None) -> None:
        """Initialize OpenRouter client.

        Args:
            params: LLM parameters
            api_key: OpenRouter API key
        """
        from openai import AsyncOpenAI, OpenAI

        self.params = params
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _supports_reasoning(self) -> bool:
        """Check if the current model supports the reasoning parameter."""
        return self.params.model in self.REASONING_MODELS

    def _get_reasoning_config(self) -> dict | None:
        """Get OpenRouter reasoning config based on thinking_mode param.

        Returns:
            Reasoning config dict or None if disabled/unsupported
        """
        if self.params.thinking_mode == "disabled" or not self._supports_reasoning():
            return None
        if self.params.thinking_mode == "adaptive":
            return {"effort": "high"}
        # thinking_mode == "enabled"
        return {"max_tokens": self.params.thinking_budget}

    def complete(self, system: str, user: str) -> CompletionResult:
        """Send a completion request via OpenRouter.

        Supports structured outputs and reasoning for compatible models.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            CompletionResult with text response and optional reasoning
        """
        # json_object response_format requires "json" to appear in the messages.
        # Specify the key name without suggesting a value to avoid biasing the model's choice.
        system_with_json = system + '\nAlways respond with valid JSON using the key "choice".'

        kwargs: dict = {
            "model": self.params.model,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "messages": [
                {"role": "system", "content": system_with_json},
                {"role": "user", "content": user},
            ],
            "response_format": self.CHOICE_RESPONSE_FORMAT,
        }

        reasoning_config = self._get_reasoning_config()
        if reasoning_config:
            kwargs["extra_body"] = {"reasoning": reasoning_config}

        response = self.client.chat.completions.create(**kwargs)

        text = ""
        reasoning_text = None
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            msg = choice.message
            text = msg.content or ""
            # OpenRouter injects reasoning as an extra field not in the OpenAI SDK schema
            reasoning_text = getattr(msg, "reasoning", None) or (
                msg.model_extra.get("reasoning") if hasattr(msg, "model_extra") else None
            )
            if not text:
                logger.warning(f"Empty content from OpenRouter (finish_reason={choice.finish_reason})")

        return CompletionResult(text=text, thinking=reasoning_text)

    async def acomplete(self, system: str, user: str) -> CompletionResult:
        """Async version of complete() using AsyncOpenAI."""
        system_with_json = system + '\nAlways respond with valid JSON using the key "choice".'

        kwargs: dict = {
            "model": self.params.model,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "messages": [
                {"role": "system", "content": system_with_json},
                {"role": "user", "content": user},
            ],
            "response_format": self.CHOICE_RESPONSE_FORMAT,
        }

        reasoning_config = self._get_reasoning_config()
        if reasoning_config:
            kwargs["extra_body"] = {"reasoning": reasoning_config}

        response = await self.async_client.chat.completions.create(**kwargs)

        text = ""
        reasoning_text = None
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            msg = choice.message
            text = msg.content or ""
            reasoning_text = getattr(msg, "reasoning", None) or (
                msg.model_extra.get("reasoning") if hasattr(msg, "model_extra") else None
            )

        return CompletionResult(text=text, thinking=reasoning_text)

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
