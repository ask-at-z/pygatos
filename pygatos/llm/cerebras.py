"""Cerebras LLM backend using the Cerebras Cloud SDK."""

import json
import logging
import os
import re
import time
from typing import Optional

from pygatos.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class CerebrasBackend(BaseLLM):
    """
    Cerebras LLM backend using the Cerebras Cloud SDK.

    This backend communicates with the Cerebras Cloud API for fast inference
    on open-source models.

    Example:
        >>> llm = CerebrasBackend(model="llama-3.3-70b")
        >>> response = llm.generate("What is the capital of France?")
        >>> print(response)
        The capital of France is Paris.

    Available models (as of 2025):
        - llama-3.3-70b
        - llama-3.1-8b
        - llama-3.1-70b
        - qwen-3-32b

    Note:
        Requires the cerebras-cloud-sdk package: pip install cerebras-cloud-sdk
        Set CEREBRAS_API_KEY environment variable or pass api_key directly.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
        debug: bool = False,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize the Cerebras backend.

        Args:
            model: The Cerebras model name to use.
            api_key: Cerebras API key. If None, reads from CEREBRAS_API_KEY env var.
            temperature: Default sampling temperature.
            max_tokens: Default maximum tokens to generate.
            timeout: Request timeout in seconds.
            debug: If True, log all prompts and responses.
            rate_limit_delay: Seconds to wait between API calls (default 2.0 for 30 req/min free tier).
        """
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            raise ImportError(
                "cerebras-cloud-sdk is required for CerebrasBackend. "
                "Install it with: pip install cerebras-cloud-sdk"
            )

        self._model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug
        self.rate_limit_delay = rate_limit_delay

        # Initialize the Cerebras client
        resolved_api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Cerebras API key is required. Set CEREBRAS_API_KEY environment variable "
                "or pass api_key directly."
            )

        self.client = Cerebras(api_key=resolved_api_key)

    @classmethod
    def from_config(cls, config) -> "CerebrasBackend":
        """Create a CerebrasBackend from a configuration object."""
        return cls(
            model=config.model,
            api_key=getattr(config, "api_key", None),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            debug=config.debug,
            rate_limit_delay=getattr(config, "rate_limit_delay", 2.0),
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt to send to the model.
            system: Optional system prompt.
            temperature: Optional sampling temperature (overrides default).
            max_tokens: Optional maximum tokens to generate (overrides default).

        Returns:
            The generated text response.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        if self.debug:
            logger.debug("=" * 60)
            logger.debug("LLM REQUEST (Cerebras)")
            logger.debug("=" * 60)
            if system:
                logger.debug(f"SYSTEM:\n{system}")
            logger.debug(f"PROMPT:\n{prompt}")
            logger.debug("-" * 60)

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self._model,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )

        response_text = chat_completion.choices[0].message.content

        if self.debug:
            logger.debug(f"RESPONSE:\n{response_text}")
            logger.debug("=" * 60)

        # Rate limit delay to avoid hitting API limits (30 req/min on free tier)
        if self.rate_limit_delay > 0:
            logger.debug(f"Rate limit delay: {self.rate_limit_delay} seconds")
            time.sleep(self.rate_limit_delay)

        return response_text

    def generate_chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using the chat API format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum tokens to generate.

        Returns:
            The generated text response.
        """
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self._model,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )

        response_text = chat_completion.choices[0].message.content

        # Rate limit delay to avoid hitting API limits (30 req/min on free tier)
        if self.rate_limit_delay > 0:
            logger.debug(f"Rate limit delay: {self.rate_limit_delay} seconds")
            time.sleep(self.rate_limit_delay)

        return response_text

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate and parse a JSON response.

        The method attempts to extract JSON from the response, handling
        cases where the model wraps JSON in markdown code blocks.

        Args:
            prompt: The user prompt to send to the model.
            system: Optional system prompt.
            temperature: Optional sampling temperature (overrides default).
            max_tokens: Optional maximum tokens to generate (overrides default).

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        # Add JSON formatting instruction to system prompt
        json_system = system or ""
        if json_system:
            json_system += "\n\n"
        json_system += "You must respond with valid JSON only. No additional text or explanation."

        response = self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from a response string.

        Handles cases where JSON is wrapped in markdown code blocks.

        Args:
            response: The raw response string.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ValueError: If JSON cannot be extracted or parsed.
        """
        # Try direct parsing first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, response)
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object or array in the response
        json_patterns = [
            r"(\{[\s\S]*\})",  # JSON object
            r"(\[[\s\S]*\])",  # JSON array
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            if matches:
                # Try the longest match first (most likely to be complete)
                for match in sorted(matches, key=len, reverse=True):
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue

        raise ValueError(f"Could not parse JSON from response: {response[:500]}...")

    def is_available(self) -> bool:
        """
        Check if the Cerebras API is available.

        Returns:
            True if the API is reachable and the API key is valid, False otherwise.
        """
        try:
            # Make a minimal request to verify connectivity
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model=self._model,
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        List commonly available models on Cerebras.

        Returns:
            List of model names.

        Note:
            This returns a static list of known models as Cerebras
            may not provide a dynamic model listing endpoint.
        """
        return [
            "llama-3.3-70b",
            "llama-3.1-8b",
            "llama-3.1-70b",
            "qwen-3-32b",
        ]
