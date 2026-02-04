"""Code suggestion for clusters of text."""

import logging
from typing import Optional

from pygatos.llm.base import BaseLLM
from pygatos.core.codebook import Code
from pygatos.prompts import (
    CODE_SUGGESTION_SYSTEM,
    CODE_SUGGESTION_PROMPT,
    STARTER_CODES_SYSTEM,
    STARTER_CODES_PROMPT,
    format_texts_for_prompt,
    add_study_context,
)

logger = logging.getLogger(__name__)


class CodeSuggester:
    """
    Suggests codes for clusters of semantically similar text.

    Given a cluster of text excerpts (e.g., information extraction bullet points),
    this class uses an LLM to suggest one or more codes that describe the
    common theme(s) in the cluster.

    Example:
        >>> suggester = CodeSuggester(llm=ollama_backend)
        >>> codes = suggester.suggest_codes(["point 1", "point 2", "point 3"])
        >>> print(codes[0].name, codes[0].definition)
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_texts_per_prompt: int = 20,
        temperature: Optional[float] = None,
        study_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        Initialize the code suggester.

        Args:
            llm: The LLM backend to use for code suggestion.
            max_texts_per_prompt: Maximum number of texts to include in a single prompt.
            temperature: Optional temperature override for LLM generation.
            study_context: Optional context about the study/dataset to improve suggestions.
            system_prompt: Optional custom system prompt (defaults to CODE_SUGGESTION_SYSTEM).
            user_prompt: Optional custom user prompt template (defaults to CODE_SUGGESTION_PROMPT).
                         Must contain {texts} placeholder.
        """
        self.llm = llm
        self.max_texts_per_prompt = max_texts_per_prompt
        self.temperature = temperature
        self.study_context = study_context
        self.system_prompt = system_prompt or CODE_SUGGESTION_SYSTEM
        self.user_prompt = user_prompt or CODE_SUGGESTION_PROMPT

    def suggest_codes(
        self,
        cluster_texts: list[str],
        cluster_id: Optional[int] = None,
        verbose: bool = False,
    ) -> list[Code]:
        """
        Suggest codes for a cluster of texts.

        Args:
            cluster_texts: List of text excerpts in the cluster.
            cluster_id: Optional cluster ID for logging/tracking.
            verbose: If True, log the prompt and response.

        Returns:
            List of suggested Code objects.
        """
        if not cluster_texts:
            logger.warning("Empty cluster provided, returning empty list")
            return []

        # Format texts for the prompt
        texts_formatted = format_texts_for_prompt(
            cluster_texts,
            max_texts=self.max_texts_per_prompt
        )

        prompt = self.user_prompt.format(texts=texts_formatted)

        # Add study context to system prompt if available
        system = add_study_context(self.system_prompt, self.study_context)

        if verbose:
            logger.info(f"Suggesting codes for cluster {cluster_id} ({len(cluster_texts)} texts)")
            logger.debug(f"Prompt:\n{prompt}")

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            if verbose:
                logger.debug(f"Response: {response}")

            codes = self._parse_codes_response(response, cluster_id)

            if verbose:
                logger.info(f"Suggested {len(codes)} codes for cluster {cluster_id}")
                for code in codes:
                    logger.info(f"  - {code.name}: {code.definition[:50]}...")

            return codes

        except Exception as e:
            logger.error(f"Failed to suggest codes for cluster {cluster_id}: {e}")
            return []

    def suggest_codes_batch(
        self,
        clusters: dict[int, list[str]],
        verbose: bool = False,
    ) -> dict[int, list[Code]]:
        """
        Suggest codes for multiple clusters.

        Args:
            clusters: Dictionary mapping cluster ID to list of texts.
            verbose: If True, log progress.

        Returns:
            Dictionary mapping cluster ID to list of suggested codes.
        """
        results = {}

        for cluster_id, texts in clusters.items():
            codes = self.suggest_codes(
                cluster_texts=texts,
                cluster_id=cluster_id,
                verbose=verbose,
            )
            results[cluster_id] = codes

        return results

    def generate_starter_codes(
        self,
        topic: str,
        n_codes: int = 20,
        verbose: bool = False,
    ) -> list[Code]:
        """
        Generate speculative starter codes for a topic.

        These are hypothetical codes that might appear in a study of the given topic.
        They are used to initialize the codebook before iterative code generation.

        Args:
            topic: The research topic (e.g., "political attitudes", "workplace feedback").
            n_codes: Number of starter codes to generate.
            verbose: If True, log the prompt and response.

        Returns:
            List of starter Code objects.
        """
        prompt = STARTER_CODES_PROMPT.format(n_codes=n_codes, topic=topic)

        # Add study context to system prompt if available
        system = add_study_context(STARTER_CODES_SYSTEM, self.study_context)

        if verbose:
            logger.info(f"Generating {n_codes} starter codes for topic: {topic}")
            logger.debug(f"Prompt:\n{prompt}")

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            if verbose:
                logger.debug(f"Response: {response}")

            codes = self._parse_codes_response(response, source_cluster=None)

            if verbose:
                logger.info(f"Generated {len(codes)} starter codes")
                for code in codes:
                    logger.info(f"  - {code.name}")

            return codes

        except Exception as e:
            logger.error(f"Failed to generate starter codes: {e}")
            return []

    def _parse_codes_response(
        self,
        response: dict,
        source_cluster: Optional[int] = None,
    ) -> list[Code]:
        """
        Parse the LLM response into Code objects.

        Args:
            response: Parsed JSON response from LLM.
            source_cluster: Optional cluster ID to attach to codes.

        Returns:
            List of Code objects.
        """
        codes = []

        # Handle different response formats
        if isinstance(response, dict):
            # Expected format: {"codes": [{"name": ..., "definition": ...}, ...]}
            codes_list = response.get("codes", [])

            # Also try singular "code" key
            if not codes_list and "code" in response:
                codes_list = [response["code"]]

            # Also try if the response itself is a code
            if not codes_list and "name" in response and "definition" in response:
                codes_list = [response]

        elif isinstance(response, list):
            codes_list = response
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            return []

        for code_data in codes_list:
            if isinstance(code_data, dict):
                name = code_data.get("name", "").strip()
                definition = code_data.get("definition", "").strip()

                if name and definition:
                    code = Code(
                        name=name,
                        definition=definition,
                        source_cluster=source_cluster,
                    )
                    codes.append(code)
                else:
                    logger.warning(f"Skipping code with missing name or definition: {code_data}")

        return codes

    def __repr__(self) -> str:
        return f"CodeSuggester(llm={self.llm.model_name})"
