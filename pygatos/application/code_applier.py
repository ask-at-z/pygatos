"""Apply codebook to label text data."""

import logging
import warnings
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from pygatos.llm.base import BaseLLM
from pygatos.core.codebook import Code, Codebook
from pygatos.core.embedder import Embedder
from pygatos.config import ApplicationConfig
from pygatos.models import InformationPoint
from pygatos.prompts import (
    CODE_APPLICATION_SYSTEM,
    CODE_APPLICATION_PROMPT,
    CODE_APPLICATION_SYSTEM_V2,
    CODE_APPLICATION_PROMPT_V2,
    CODE_APPLICATION_SYSTEM_V3,
    CODE_APPLICATION_PROMPT_V3,
    format_codes_for_prompt,
    add_study_context,
)

logger = logging.getLogger(__name__)


@dataclass
class PointApplicationResult:
    """Result of applying codebook to a single information point."""

    information_point: str
    source_text: str
    applied_codes: list[Code]
    candidate_codes: list[Code]
    point_interpretation: Optional[str] = None
    analysis: Optional[str] = None
    point_embedding: Optional[np.ndarray] = None
    # Lineage tracking fields (populated when using apply_to_points)
    chunk_index: Optional[int] = None
    chunk_text: Optional[str] = None


@dataclass
class ApplicationResult:
    """Result of applying codebook to a single text (response-level)."""

    text: str
    applied_codes: list[Code]
    reasoning: str
    candidate_codes: list[Code]
    text_embedding: Optional[np.ndarray] = None
    # V2 fields for audit trail
    text_summary: Optional[str] = None
    analysis: Optional[str] = None
    # V3 fields for point-level coding
    information_points: list[str] = field(default_factory=list)
    point_results: list[PointApplicationResult] = field(default_factory=list)


class CodeApplier:
    """
    Applies a codebook to label text data.

    Uses a two-stage approach:
    1. Retrieval: Find top-K most similar codes based on embedding similarity
    2. Judgment: LLM decides which candidate codes actually apply

    This approach balances accuracy with efficiency—only relevant codes
    are sent to the LLM for judgment.

    Example:
        >>> applier = CodeApplier(llm, embedder, top_k=10)
        >>> result = applier.apply(text, codebook)
        >>> for code in result.applied_codes:
        ...     print(f"Applied: {code.name}")
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Embedder,
        top_k: int = 10,
        temperature: Optional[float] = None,
        study_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        Initialize the code applier.

        Args:
            llm: LLM backend for code judgment.
            embedder: Embedder for text/code embeddings.
            top_k: Number of candidate codes to retrieve for judgment.
            temperature: Optional temperature override for LLM.
            study_context: Optional context about the study/dataset to improve coding.
            system_prompt: Optional custom system prompt. If None, uses default V3 prompt.
            user_prompt: Optional custom user prompt. Must contain {information_point}, {source_text}, {codes}.
        """
        self.llm = llm
        self.embedder = embedder
        self.top_k = top_k
        self.temperature = temperature
        self.study_context = study_context
        self._system_prompt = system_prompt or CODE_APPLICATION_SYSTEM_V3
        self._user_prompt = user_prompt or CODE_APPLICATION_PROMPT_V3

    @classmethod
    def from_config(
        cls,
        config: ApplicationConfig,
        llm: BaseLLM,
        embedder: Embedder,
        study_context: Optional[str] = None,
    ) -> "CodeApplier":
        """Create a CodeApplier from configuration.

        Args:
            config: Application configuration.
            llm: LLM backend.
            embedder: Embedder.
            study_context: Optional context about the study/dataset.

        Returns:
            Configured CodeApplier instance.
        """
        return cls(
            llm=llm,
            embedder=embedder,
            top_k=config.top_k,
            study_context=study_context,
        )

    def apply(
        self,
        text: str,
        codebook: Codebook,
        verbose: bool = False,
    ) -> ApplicationResult:
        """
        Apply codebook to a single text.

        Args:
            text: Text to code.
            codebook: Codebook to apply.
            verbose: If True, log detailed information.

        Returns:
            ApplicationResult with applied codes.
        """
        if not codebook.accepted_codes:
            return ApplicationResult(
                text=text,
                applied_codes=[],
                reasoning="No codes in codebook",
                candidate_codes=[],
            )

        text = text.strip()

        if not text:
            return ApplicationResult(
                text=text,
                applied_codes=[],
                reasoning="Empty text",
                candidate_codes=[],
            )

        if verbose:
            logger.info(f"Applying codebook to text ({len(text)} chars)")

        # Step 1: Embed text
        text_embedding = self.embedder.embed(text)

        # Step 2: Retrieve candidate codes
        candidates = self._retrieve_candidates(text_embedding, codebook)

        if verbose:
            logger.info(f"  Retrieved {len(candidates)} candidate codes")

        if not candidates:
            return ApplicationResult(
                text=text,
                applied_codes=[],
                reasoning="No similar codes found",
                candidate_codes=[],
                text_embedding=text_embedding,
            )

        # Step 3: LLM judgment
        applied_codes, reasoning, text_summary, analysis = self._judge_codes(
            text, candidates, verbose
        )

        if verbose:
            logger.info(f"  Applied {len(applied_codes)} codes")

        return ApplicationResult(
            text=text,
            applied_codes=applied_codes,
            reasoning=reasoning,
            candidate_codes=candidates,
            text_embedding=text_embedding,
            text_summary=text_summary,
            analysis=analysis,
        )

    def apply_batch(
        self,
        texts: list[str],
        codebook: Codebook,
        verbose: bool = False,
    ) -> list[list[Code]]:
        """
        Apply codebook to multiple texts.

        Args:
            texts: List of texts to code.
            codebook: Codebook to apply.
            verbose: If True, log progress.

        Returns:
            List of applied code lists (one per text).
        """
        results = []

        text_iter = texts
        if verbose:
            text_iter = tqdm(texts, desc="Applying codebook", unit="text")

        for text in text_iter:
            result = self.apply(text, codebook, verbose=False)
            results.append(result.applied_codes)

        return results

    def apply_with_details(
        self,
        texts: list[str],
        codebook: Codebook,
        verbose: bool = False,
    ) -> list[ApplicationResult]:
        """
        Apply codebook to multiple texts with full details.

        Args:
            texts: List of texts to code.
            codebook: Codebook to apply.
            verbose: If True, log progress.

        Returns:
            List of ApplicationResult objects.
        """
        results = []

        text_iter = texts
        if verbose:
            text_iter = tqdm(texts, desc="Applying codebook", unit="text")

        for text in text_iter:
            result = self.apply(text, codebook, verbose=False)
            results.append(result)

        return results

    def _retrieve_candidates(
        self,
        text_embedding: np.ndarray,
        codebook: Codebook,
    ) -> list[Code]:
        """
        Retrieve top-K candidate codes based on embedding similarity.

        Args:
            text_embedding: Embedding of the text to code.
            codebook: Codebook to search.

        Returns:
            List of candidate codes (most similar first).
        """
        # Ensure all codes have embeddings
        self._ensure_code_embeddings(codebook)

        # Find similar codes
        similar = codebook.find_similar_codes(
            text_embedding,
            top_k=self.top_k,
            accepted_only=True,
        )

        return [code for code, similarity in similar]

    def _ensure_code_embeddings(self, codebook: Codebook) -> None:
        """Ensure all accepted codes have embeddings."""
        codes_to_embed = []
        texts_to_embed = []

        for code in codebook.accepted_codes:
            if code.embedding is None:
                codes_to_embed.append(code)
                texts_to_embed.append(f"{code.name}: {code.definition}")

        if texts_to_embed:
            embeddings = self.embedder.embed(texts_to_embed)
            for i, code in enumerate(codes_to_embed):
                code.embedding = embeddings[i]

    def _judge_codes(
        self,
        text: str,
        candidates: list[Code],
        verbose: bool = False,
    ) -> tuple[list[Code], str, Optional[str], Optional[str]]:
        """
        Use LLM to judge which candidate codes apply.

        Uses v2 reasoning-first prompt to prevent post-hoc rationalization.

        Args:
            text: Text being coded.
            candidates: Candidate codes to evaluate.
            verbose: If True, log detailed information.

        Returns:
            Tuple of (applied codes, reasoning, text_summary, analysis).
        """
        # Format codes for prompt
        codes_text = format_codes_for_prompt(candidates, include_definition=True)

        # Use v2 prompt for reasoning-first approach
        prompt = CODE_APPLICATION_PROMPT_V2.format(
            text=text,
            codes=codes_text,
        )

        # Add study context to system prompt if available
        system = add_study_context(CODE_APPLICATION_SYSTEM_V2, self.study_context)

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            applied_names = response.get("applied_codes", [])
            text_summary = response.get("text_summary", "")
            analysis = response.get("analysis", "")
            # Backwards compatibility: use analysis as reasoning if no reasoning key
            reasoning = response.get("reasoning", analysis)

            # Map names back to Code objects
            name_to_code = {c.name: c for c in candidates}
            applied = []

            for name in applied_names:
                # Try exact match first
                if name in name_to_code:
                    applied.append(name_to_code[name])
                else:
                    # Try case-insensitive match
                    name_lower = name.lower()
                    for code_name, code in name_to_code.items():
                        if code_name.lower() == name_lower:
                            applied.append(code)
                            break

            if verbose:
                logger.debug(f"  Applied: {[c.name for c in applied]}")
                logger.debug(f"  Text summary: {text_summary}")
                logger.debug(f"  Analysis: {analysis}")

            return applied, reasoning, text_summary, analysis

        except Exception as e:
            logger.error(f"  LLM judgment failed: {e}")
            return [], f"Error: {e}", None, None

    def apply_point(
        self,
        information_point: str,
        source_text: str,
        codebook: Codebook,
        verbose: bool = False,
        prompt_context: Optional[str] = None,
    ) -> PointApplicationResult:
        """
        Apply codebook to a single information point.

        Uses the information point's embedding for semantic matching,
        and provides source text as context for LLM judgment.

        Args:
            information_point: The extracted atomic idea to code.
            source_text: The original text the point was extracted from (stored for tracking).
            codebook: Codebook to apply.
            verbose: If True, log detailed information.
            prompt_context: Optional context to show the LLM instead of source_text.
                           If None, source_text is used in the prompt.
                           Use this to control what context the LLM sees while
                           still storing the true source_text for audit purposes.

        Returns:
            PointApplicationResult with applied codes.
        """
        if not codebook.accepted_codes:
            return PointApplicationResult(
                information_point=information_point,
                source_text=source_text,
                applied_codes=[],
                candidate_codes=[],
            )

        information_point = information_point.strip()
        if not information_point:
            return PointApplicationResult(
                information_point=information_point,
                source_text=source_text,
                applied_codes=[],
                candidate_codes=[],
            )

        # Step 1: Embed the information point (not the full source text)
        point_embedding = self.embedder.embed(information_point)

        # Step 2: Retrieve candidate codes based on point embedding
        candidates = self._retrieve_candidates(point_embedding, codebook)

        if not candidates:
            return PointApplicationResult(
                information_point=information_point,
                source_text=source_text,
                applied_codes=[],
                candidate_codes=[],
                point_embedding=point_embedding,
            )

        # Step 3: LLM judgment with source context
        # Use prompt_context if provided, otherwise fall back to source_text
        context_for_llm = prompt_context if prompt_context is not None else source_text
        applied_codes, point_interpretation, analysis = self._judge_point(
            information_point, context_for_llm, candidates, verbose
        )

        return PointApplicationResult(
            information_point=information_point,
            source_text=source_text,
            applied_codes=applied_codes,
            candidate_codes=candidates,
            point_interpretation=point_interpretation,
            analysis=analysis,
            point_embedding=point_embedding,
        )

    def _judge_point(
        self,
        information_point: str,
        source_text: str,
        candidates: list[Code],
        verbose: bool = False,
    ) -> tuple[list[Code], Optional[str], Optional[str]]:
        """
        Use LLM to judge which codes apply to an information point.

        Args:
            information_point: The atomic idea to code.
            source_text: Original text for context.
            candidates: Candidate codes to evaluate.
            verbose: If True, log detailed information.

        Returns:
            Tuple of (applied codes, point_interpretation, analysis).
        """
        codes_text = format_codes_for_prompt(candidates, include_definition=True)

        prompt = self._user_prompt.format(
            information_point=information_point,
            source_text=source_text,
            codes=codes_text,
        )

        # Add study context to system prompt if available
        system = add_study_context(self._system_prompt, self.study_context)

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            applied_names = response.get("applied_codes", [])
            point_interpretation = response.get("point_interpretation", "")
            analysis = response.get("analysis", "")

            # Map names back to Code objects
            name_to_code = {c.name: c for c in candidates}
            applied = []

            for name in applied_names:
                if name in name_to_code:
                    applied.append(name_to_code[name])
                else:
                    name_lower = name.lower()
                    for code_name, code in name_to_code.items():
                        if code_name.lower() == name_lower:
                            applied.append(code)
                            break

            if verbose:
                logger.debug(f"  Point: {information_point[:50]}...")
                logger.debug(f"  Applied: {[c.name for c in applied]}")

            return applied, point_interpretation, analysis

        except Exception as e:
            logger.error(f"  LLM judgment failed for point: {e}")
            return [], None, None

    def apply_to_points(
        self,
        structured_points: list[InformationPoint],
        codebook: Codebook,
        include_source_context: bool = False,
        verbose: bool = False,
    ) -> ApplicationResult:
        """
        Apply codebook to pre-extracted information points.

        This is the preferred method when information points have already been
        extracted during codebook generation. It avoids redundant extraction
        and ensures consistency between extraction and application phases.

        Args:
            structured_points: Pre-extracted information points with lineage.
                              Each point knows its source text and chunk origin.
            codebook: Codebook to apply.
            include_source_context: If True, use chunk/source text as context
                                   in LLM prompts for better judgment.
            verbose: Enable verbose logging.

        Returns:
            ApplicationResult with applied codes and point-level details.
        """
        if not codebook.accepted_codes:
            return ApplicationResult(
                text="",
                applied_codes=[],
                reasoning="No codes in codebook",
                candidate_codes=[],
            )

        if not structured_points:
            return ApplicationResult(
                text="",
                applied_codes=[],
                reasoning="No information points provided",
                candidate_codes=[],
            )

        # Get source text from first point (all points should share same source)
        source_text = structured_points[0].source_text

        if verbose:
            logger.info(f"Applying codes to {len(structured_points)} pre-extracted points")

        # Apply codes to each information point
        point_results = []
        all_applied_codes = []
        all_candidate_codes = set()

        for point in structured_points:
            # Determine what context to show the LLM
            # - include_source_context=True: show chunk text (or source if no chunking)
            # - include_source_context=False: indicate no context provided
            if include_source_context:
                # Use chunk text if available, otherwise source text
                prompt_context = None  # Use source_text in the prompt
                actual_source = point.context_text
            else:
                # Don't show source context in prompt, just indicate it's not provided
                prompt_context = "(Source context not included - code based on information point only)"
                actual_source = point.context_text  # Still store actual source for tracking

            # Use existing apply_point method
            point_result = self.apply_point(
                information_point=point.text,
                source_text=actual_source,  # Always store the real source
                codebook=codebook,
                verbose=False,
                prompt_context=prompt_context,  # Controls what LLM sees
            )

            # Add lineage tracking from the InformationPoint
            point_result.chunk_index = point.chunk_index
            point_result.chunk_text = point.chunk_text

            point_results.append(point_result)
            all_applied_codes.extend(point_result.applied_codes)
            all_candidate_codes.update(c.name for c in point_result.candidate_codes)

        # Deduplicate applied codes (a code should only appear once per response)
        seen_codes = set()
        unique_applied = []
        for code in all_applied_codes:
            if code.name not in seen_codes:
                seen_codes.add(code.name)
                unique_applied.append(code)

        # Build reasoning from point-level analyses
        reasoning_parts = []
        for i, pr in enumerate(point_results):
            if pr.applied_codes:
                codes_str = ", ".join(c.name for c in pr.applied_codes)
                point_preview = pr.information_point[:50]
                reasoning_parts.append(f"Point {i+1}: \"{point_preview}...\" -> {codes_str}")

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else "No codes applied"

        if verbose:
            logger.info(f"  Applied {len(unique_applied)} unique codes across {len(point_results)} points")

        return ApplicationResult(
            text=source_text,
            applied_codes=unique_applied,
            reasoning=reasoning,
            candidate_codes=[],  # Candidates are at point level
            information_points=[p.text for p in structured_points],
            point_results=point_results,
        )

    def apply_to_points_batch(
        self,
        extraction_results: dict,
        codebook: Codebook,
        include_source_context: bool = False,
        verbose: bool = False,
    ) -> list[ApplicationResult]:
        """
        Apply codebook to multiple texts using pre-extracted information points.

        This is the batch version of apply_to_points() for processing multiple
        texts efficiently when extraction results are already available.

        Args:
            extraction_results: Dict mapping ID to SummarizationResult.
                               The SummarizationResult must have structured_points populated.
            codebook: Codebook to apply.
            include_source_context: If True, include chunk/source context in prompts.
            verbose: If True, log progress.

        Returns:
            List of ApplicationResult objects in the same order as extraction_results.
        """
        # Import here to avoid circular dependency
        from pygatos.core.summarizer import SummarizationResult

        results = []
        items = list(extraction_results.items())

        item_iter = items
        if verbose:
            item_iter = tqdm(items, desc="Applying codebook (pre-extracted)", unit="text")

        for item_id, extraction_result in item_iter:
            # Handle both SummarizationResult and raw list of InformationPoints
            if isinstance(extraction_result, SummarizationResult):
                structured_points = extraction_result.structured_points
            elif isinstance(extraction_result, list):
                # Assume list of InformationPoint objects
                structured_points = extraction_result
            else:
                logger.warning(f"Unknown extraction result type for {item_id}: {type(extraction_result)}")
                results.append(ApplicationResult(
                    text="",
                    applied_codes=[],
                    reasoning=f"Invalid extraction result type: {type(extraction_result)}",
                    candidate_codes=[],
                ))
                continue

            if not structured_points:
                # No information points - create empty result
                results.append(ApplicationResult(
                    text=extraction_result.original_text if hasattr(extraction_result, 'original_text') else "",
                    applied_codes=[],
                    reasoning="No information points to code",
                    candidate_codes=[],
                ))
                continue

            result = self.apply_to_points(
                structured_points=structured_points,
                codebook=codebook,
                include_source_context=include_source_context,
                verbose=False,
            )
            results.append(result)

        return results

    def apply_with_extraction(
        self,
        text: str,
        codebook: Codebook,
        summarizer,  # Type hint avoided to prevent circular import
        verbose: bool = False,
    ) -> ApplicationResult:
        """
        Apply codebook to text by first extracting information points.

        .. deprecated::
            This method re-runs information extraction, which is wasteful when
            extraction has already been performed during codebook generation.
            Use :meth:`apply_to_points` with pre-extracted points instead.

        This method:
        1. Extracts atomic information points from the text
        2. Codes each information point individually
        3. Aggregates codes at the response level

        Args:
            text: The full text to code.
            codebook: Codebook to apply.
            summarizer: Summarizer instance for information extraction.
            verbose: If True, log detailed information.

        Returns:
            ApplicationResult with aggregated codes and point-level details.
        """
        warnings.warn(
            "apply_with_extraction() is deprecated and will be removed in v2.0. "
            "It re-runs information extraction unnecessarily. "
            "Use apply_to_points() with pre-extracted SummarizationResult.structured_points instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not codebook.accepted_codes:
            return ApplicationResult(
                text=text,
                applied_codes=[],
                reasoning="No codes in codebook",
                candidate_codes=[],
            )

        text = text.strip()
        if not text:
            return ApplicationResult(
                text=text,
                applied_codes=[],
                reasoning="Empty text",
                candidate_codes=[],
            )

        # Step 1: Extract information points
        summary_result = summarizer.summarize(text, verbose=False)
        information_points = summary_result.information_points

        if not information_points:
            # Fallback to treating full text as single point
            if verbose:
                logger.info("  No information points extracted, using full text")
            return self.apply(text, codebook, verbose=verbose)

        if verbose:
            logger.info(f"  Extracted {len(information_points)} information points")

        # Step 2: Code each information point
        point_results = []
        all_applied_codes = []
        all_candidate_codes = set()

        for point in information_points:
            point_result = self.apply_point(point, text, codebook, verbose=False)
            point_results.append(point_result)
            all_applied_codes.extend(point_result.applied_codes)
            all_candidate_codes.update(c.name for c in point_result.candidate_codes)

        # Step 3: Deduplicate applied codes (a code should only appear once per response)
        seen_codes = set()
        unique_applied = []
        for code in all_applied_codes:
            if code.name not in seen_codes:
                seen_codes.add(code.name)
                unique_applied.append(code)

        # Build reasoning from point-level analyses
        reasoning_parts = []
        for i, pr in enumerate(point_results):
            if pr.applied_codes:
                codes_str = ", ".join(c.name for c in pr.applied_codes)
                reasoning_parts.append(
                    f"Point {i+1}: \"{pr.information_point[:50]}...\" → {codes_str}"
                )

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else "No codes applied"

        if verbose:
            logger.info(f"  Applied {len(unique_applied)} unique codes across {len(point_results)} points")

        return ApplicationResult(
            text=text,
            applied_codes=unique_applied,
            reasoning=reasoning,
            candidate_codes=[],  # Candidates are at point level
            information_points=information_points,
            point_results=point_results,
        )

    def apply_with_extraction_batch(
        self,
        texts: list[str],
        codebook: Codebook,
        summarizer,
        verbose: bool = False,
    ) -> list[ApplicationResult]:
        """
        Apply codebook to multiple texts using information extraction.

        .. deprecated::
            This method re-runs information extraction for each text.
            Use :meth:`apply_to_points_batch` with pre-extracted points instead.

        Args:
            texts: List of texts to code.
            codebook: Codebook to apply.
            summarizer: Summarizer instance for information extraction.
            verbose: If True, log progress.

        Returns:
            List of ApplicationResult objects.
        """
        warnings.warn(
            "apply_with_extraction_batch() is deprecated and will be removed in v2.0. "
            "It re-runs information extraction unnecessarily. "
            "Use apply_to_points_batch() with pre-extracted results instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        results = []

        text_iter = texts
        if verbose:
            text_iter = tqdm(texts, desc="Applying codebook (with extraction)", unit="text")

        for text in text_iter:
            result = self.apply_with_extraction(text, codebook, summarizer, verbose=False)
            results.append(result)

        return results

    def get_code_frequencies(
        self,
        results: list[ApplicationResult],
    ) -> dict[str, int]:
        """
        Calculate code frequencies from application results.

        Args:
            results: List of ApplicationResult objects.

        Returns:
            Dict mapping code name to frequency count.
        """
        frequencies = {}

        for result in results:
            for code in result.applied_codes:
                frequencies[code.name] = frequencies.get(code.name, 0) + 1

        return dict(sorted(frequencies.items(), key=lambda x: -x[1]))

    def __repr__(self) -> str:
        return f"CodeApplier(top_k={self.top_k})"
