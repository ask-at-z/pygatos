"""Text summarization and information extraction for long texts."""

import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from pygatos.llm.base import BaseLLM
from pygatos.models import InformationPoint
from pygatos.config import SummarizationConfig
from pygatos.prompts import (
    GENERIC_SUMMARY_SYSTEM,
    GENERIC_SUMMARY_PROMPT,
    GENERIC_SUMMARY_WITH_CONTEXT_PROMPT,
    INFORMATION_EXTRACTION_SYSTEM,
    INFORMATION_EXTRACTION_PROMPT,
    INFORMATION_EXTRACTION_WITH_CONTEXT_PROMPT,
    add_study_context,
)

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """
    Result of summarizing a single text.

    Attributes:
        original_text: The original input text before any processing.
        chunks: List of text chunks (single element if no chunking needed).
        generic_summaries: Summary bullets for each chunk (used as context).
        information_points: Flat list of extracted info point strings (backward compat).
        structured_points: List of InformationPoint objects with full lineage tracking.
        n_chunks: Number of chunks the text was split into.
    """

    original_text: str
    chunks: list[str]
    generic_summaries: list[str]
    information_points: list[str]
    n_chunks: int
    structured_points: list[InformationPoint] = field(default_factory=list)


class Summarizer:
    """
    Extracts atomic information points from long texts using chunking and LLM.

    The GATOS methodology uses a two-phase summarization process:
    1. Generic summary: Create a brief overview of each chunk for context
    2. Information extraction: Extract specific, codable information points

    For short texts (below chunk_size), the text is processed directly without chunking.

    Example:
        >>> summarizer = Summarizer(llm, chunk_size=250, chunk_overlap=50)
        >>> result = summarizer.summarize(long_text)
        >>> for point in result.information_points:
        ...     print(f"- {point}")
    """

    def __init__(
        self,
        llm: BaseLLM,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        max_context_bullets: int = 8,
        bullets_per_chunk: int = 4,
        temperature: Optional[float] = None,
        study_context: Optional[str] = None,
    ):
        """
        Initialize the summarizer.

        Args:
            llm: LLM backend for summarization.
            chunk_size: Target size of each chunk in words (approximate).
            chunk_overlap: Number of words to overlap between chunks.
            max_context_bullets: Maximum prior summary bullets to include as context.
            bullets_per_chunk: Target number of generic summary bullets per chunk.
            temperature: Optional temperature override for LLM.
            study_context: Optional context about the study/dataset to improve extraction.
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_bullets = max_context_bullets
        self.bullets_per_chunk = bullets_per_chunk
        self.temperature = temperature
        self.study_context = study_context

    @classmethod
    def from_config(
        cls,
        config: SummarizationConfig,
        llm: BaseLLM,
        study_context: Optional[str] = None,
    ) -> "Summarizer":
        """Create a Summarizer from configuration.

        Args:
            config: Summarization configuration.
            llm: LLM backend.
            study_context: Optional context about the study/dataset.

        Returns:
            Configured Summarizer instance.
        """
        return cls(
            llm=llm,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_context_bullets=config.max_context_bullets,
            bullets_per_chunk=config.bullets_per_chunk,
            study_context=study_context,
        )

    def summarize(
        self,
        text: str,
        skip_chunking: bool = False,
        verbose: bool = False,
        extraction_context: Optional[str] = None,
    ) -> SummarizationResult:
        """
        Extract information points from a text.

        Args:
            text: The text to summarize (the actual source of information points).
            skip_chunking: If True, treat text as atomic (no chunking).
            verbose: If True, log detailed information.
            extraction_context: Optional context to help the LLM understand the text
                               (e.g., a question that prompted the answer). This context
                               is included in the extraction prompt but NOT stored as
                               the source - only `text` is tracked as the source.

        Returns:
            SummarizationResult with chunks, summaries, and information points.
        """
        text = text.strip()

        if not text:
            return SummarizationResult(
                original_text=text,
                chunks=[],
                generic_summaries=[],
                information_points=[],
                n_chunks=0,
                structured_points=[],
            )

        # Check if text is short enough to process directly
        word_count = len(text.split())

        if skip_chunking or word_count <= self.chunk_size:
            if verbose:
                logger.info(f"Processing text directly ({word_count} words)")

            # Extract information points directly (no chunking needed)
            # Pass extraction_context to help LLM understand the text
            info_points = self._extract_information_points(
                text, context=None, extraction_context=extraction_context
            )

            # Create structured points with lineage
            # Treat non-chunked text as a single chunk (chunk_index=0, chunk_text=text)
            # This ensures context_text always returns the actual source text
            structured = [
                InformationPoint(
                    text=point,
                    source_text=text,
                    chunk_index=0,
                    chunk_text=text,
                )
                for point in info_points
            ]

            return SummarizationResult(
                original_text=text,
                chunks=[text],
                generic_summaries=[],
                information_points=info_points,
                n_chunks=1,
                structured_points=structured,
            )

        # Chunk the text
        chunks = self._chunk_text(text)

        if verbose:
            logger.info(f"Split text into {len(chunks)} chunks ({word_count} words)")

        # Process each chunk
        all_generic_summaries = []
        all_information_points = []
        all_structured_points = []

        for i, chunk in enumerate(chunks):
            if verbose:
                logger.info(f"  Processing chunk {i+1}/{len(chunks)}")

            # Build context from previous summaries
            context = self._build_context(all_generic_summaries)

            # Phase 1: Generate generic summary for context
            generic_summary = self._generate_generic_summary(chunk, context)
            all_generic_summaries.extend(generic_summary)

            # Phase 2: Extract information points
            # Pass extraction_context to help LLM understand chunks in context
            info_points = self._extract_information_points(
                chunk, context, extraction_context=extraction_context
            )
            all_information_points.extend(info_points)

            # Create structured points with chunk lineage
            for point in info_points:
                all_structured_points.append(
                    InformationPoint(
                        text=point,
                        source_text=text,
                        chunk_index=i,
                        chunk_text=chunk,
                    )
                )

        if verbose:
            logger.info(f"  Extracted {len(all_information_points)} information points")

        return SummarizationResult(
            original_text=text,
            chunks=chunks,
            generic_summaries=all_generic_summaries,
            information_points=all_information_points,
            n_chunks=len(chunks),
            structured_points=all_structured_points,
        )

    def summarize_batch(
        self,
        texts: list[str],
        skip_chunking: bool = False,
        verbose: bool = False,
    ) -> list[SummarizationResult]:
        """
        Summarize multiple texts.

        Args:
            texts: List of texts to summarize.
            skip_chunking: If True, treat texts as atomic.
            verbose: If True, log detailed information.

        Returns:
            List of SummarizationResult objects.
        """
        results = []

        for i, text in enumerate(texts):
            if verbose:
                logger.info(f"Summarizing text {i+1}/{len(texts)}")

            result = self.summarize(text, skip_chunking=skip_chunking, verbose=verbose)
            results.append(result)

        return results

    def extract_all_points(
        self,
        texts: list[str],
        skip_chunking: bool = False,
        verbose: bool = False,
    ) -> list[str]:
        """
        Extract and flatten all information points from multiple texts.

        Args:
            texts: List of texts to process.
            skip_chunking: If True, treat texts as atomic.
            verbose: If True, log detailed information.

        Returns:
            Flat list of all information points.
        """
        all_points = []

        for i, text in enumerate(texts):
            if verbose:
                logger.info(f"Processing text {i+1}/{len(texts)}")

            result = self.summarize(text, skip_chunking=skip_chunking, verbose=verbose)
            all_points.extend(result.information_points)

        return all_points

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Uses sentence boundaries where possible to create natural breaks.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        # Split into sentences (rough approximation)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence exceeds chunk size, start a new chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                overlap_words = 0
                overlap_start = len(current_chunk)

                # Find overlap point
                for j in range(len(current_chunk) - 1, -1, -1):
                    overlap_words += len(current_chunk[j].split())
                    if overlap_words >= self.chunk_overlap:
                        overlap_start = j
                        break

                # Start new chunk with overlap sentences
                current_chunk = current_chunk[overlap_start:]
                current_word_count = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_word_count += sentence_words

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _build_context(self, prior_summaries: list[str]) -> Optional[str]:
        """
        Build context string from prior generic summaries.

        Args:
            prior_summaries: List of prior summary bullets.

        Returns:
            Context string or None if no prior summaries.
        """
        if not prior_summaries:
            return None

        # Take the most recent summaries up to max_context_bullets
        recent = prior_summaries[-self.max_context_bullets:]

        return '\n'.join([f"- {s}" for s in recent])

    def _generate_generic_summary(
        self,
        chunk: str,
        context: Optional[str] = None,
    ) -> list[str]:
        """
        Generate a generic summary for a chunk using JSON structured output.

        Args:
            chunk: The text chunk to summarize.
            context: Optional context from prior summaries.

        Returns:
            List of summary bullet points.
        """
        if context:
            prompt = GENERIC_SUMMARY_WITH_CONTEXT_PROMPT.format(
                context=context,
                text=chunk,
            )
        else:
            prompt = GENERIC_SUMMARY_PROMPT.format(text=chunk)

        # Add study context to system prompt if available
        system = add_study_context(GENERIC_SUMMARY_SYSTEM, self.study_context)

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            # Extract summary points from JSON response
            points = response.get("summary_points", [])

            # Ensure all points are strings and non-empty
            return [str(p).strip() for p in points if p and str(p).strip()]

        except Exception as e:
            logger.warning(f"Failed to generate generic summary: {e}")
            # Fallback: try plain text parsing if JSON fails
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system=system,
                    temperature=self.temperature,
                )
                return self._parse_bullets(response)
            except Exception:
                return []

    def _extract_information_points(
        self,
        chunk: str,
        context: Optional[str] = None,
        extraction_context: Optional[str] = None,
    ) -> list[str]:
        """
        Extract information points from a chunk using JSON structured output.

        Args:
            chunk: The text chunk to process.
            context: Optional context from prior summaries (for multi-chunk texts).
            extraction_context: Optional context to help understand the chunk
                               (e.g., a question that prompted an answer).
                               This is prepended to the chunk in the prompt.

        Returns:
            List of information point strings.
        """
        # Build the text to send to LLM
        # If extraction_context is provided, prepend it to help LLM understand the chunk
        if extraction_context:
            text_for_prompt = f"{extraction_context}\n\n{chunk}"
        else:
            text_for_prompt = chunk

        if context:
            prompt = INFORMATION_EXTRACTION_WITH_CONTEXT_PROMPT.format(
                context=context,
                text=text_for_prompt,
            )
        else:
            prompt = INFORMATION_EXTRACTION_PROMPT.format(text=text_for_prompt)

        # Add study context to system prompt if available
        system = add_study_context(INFORMATION_EXTRACTION_SYSTEM, self.study_context)

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            # Extract information points from JSON response
            points = response.get("information_points", [])

            # Ensure all points are strings and non-empty
            return [str(p).strip() for p in points if p and str(p).strip()]

        except Exception as e:
            logger.warning(f"Failed to extract information points: {e}")
            # Fallback: try plain text parsing if JSON fails
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system=system,
                    temperature=self.temperature,
                )
                return self._parse_bullets(response)
            except Exception:
                return []

    def _parse_bullets(self, response: str) -> list[str]:
        """
        Parse bullet points from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            List of cleaned bullet point strings.
        """
        lines = response.strip().split('\n')
        bullets = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove bullet markers
            if line.startswith('- '):
                line = line[2:]
            elif line.startswith('* '):
                line = line[2:]
            elif re.match(r'^\d+\.\s+', line):
                line = re.sub(r'^\d+\.\s+', '', line)

            line = line.strip()

            if line:
                bullets.append(line)

        return bullets

    def __repr__(self) -> str:
        return (
            f"Summarizer(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
