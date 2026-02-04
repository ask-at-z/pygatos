"""Two-stage novelty evaluation for codes."""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from pygatos.llm.base import BaseLLM
from pygatos.core.codebook import Code, Codebook
from pygatos.core.embedder import Embedder
from pygatos.prompts import (
    NOVELTY_EVALUATION_SYSTEM,
    NOVELTY_EVALUATION_PROMPT,
    NOVELTY_EVALUATION_SYSTEM_V2,
    NOVELTY_EVALUATION_PROMPT_V2,
    format_codes_for_prompt,
    add_study_context,
)

logger = logging.getLogger(__name__)


@dataclass
class NoveltyResult:
    """Result of a novelty evaluation."""

    code: Code
    is_novel: bool
    stage: str  # "stage1_auto_reject", "stage1_pass", "stage2_accept", "stage2_reject"
    max_similarity: float
    most_similar_code: Optional[Code]
    similar_to_accepted: bool  # Was the most similar code from accepted group?
    reasoning: Optional[str] = None


class NoveltyEvaluator:
    """
    Two-stage novelty evaluator for codes.

    Stage 1 (Semantic Similarity Gate):
        - Embed the new code and compare to ALL codes (accepted + rejected)
        - If similarity > threshold: auto-reject (skip Stage 2)
        - Rationale: similar to rejected = would also fail; similar to accepted = redundant

    Stage 2 (LLM Evaluation):
        - Use RAG to get top-K similar codes from accepted codes only
        - LLM evaluates whether the new code is truly novel
        - Returns accept or reject decision

    Example:
        >>> evaluator = NoveltyEvaluator(llm, embedder, similarity_threshold=0.8)
        >>> result = evaluator.evaluate(new_code, codebook)
        >>> if result.is_novel:
        ...     codebook.add_code(new_code, accepted=True)
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Embedder,
        similarity_threshold: float = 0.8,
        top_k_rag: int = 8,
        temperature: Optional[float] = None,
        prompt_version: int = 1,
        study_context: Optional[str] = None,
        include_rejected_in_rag: bool = True,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        Initialize the novelty evaluator.

        Args:
            llm: LLM backend for Stage 2 evaluation.
            embedder: Embedder for generating code embeddings.
            similarity_threshold: Cosine similarity threshold for Stage 1 auto-rejection.
            top_k_rag: Number of similar codes to retrieve for Stage 2 RAG.
            temperature: Optional temperature override for LLM.
            prompt_version: Which prompt version to use (1=original, 2=stricter).
            study_context: Optional context about the study/dataset to improve evaluation.
            include_rejected_in_rag: Whether to include rejected codes in Stage 2 RAG context.
            system_prompt: Optional custom system prompt. If provided, overrides prompt_version.
            user_prompt: Optional custom user prompt. Must contain {code_name}, {code_definition}, {existing_codes}.
        """
        self.llm = llm
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.top_k_rag = top_k_rag
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.study_context = study_context
        self.include_rejected_in_rag = include_rejected_in_rag
        self._evaluation_counter = 0  # Track evaluation order

        # Use custom prompts if provided, otherwise select based on version
        if system_prompt is not None:
            self._system_prompt = system_prompt
        elif prompt_version == 2:
            self._system_prompt = NOVELTY_EVALUATION_SYSTEM_V2
        else:
            self._system_prompt = NOVELTY_EVALUATION_SYSTEM

        if user_prompt is not None:
            self._user_prompt = user_prompt
        elif prompt_version == 2:
            self._user_prompt = NOVELTY_EVALUATION_PROMPT_V2
        else:
            self._user_prompt = NOVELTY_EVALUATION_PROMPT

    def evaluate(
        self,
        code: Code,
        codebook: Codebook,
        verbose: bool = False,
    ) -> NoveltyResult:
        """
        Evaluate whether a code is novel.

        Args:
            code: The code to evaluate.
            codebook: The current codebook (with accepted and rejected codes).
            verbose: If True, log detailed information.

        Returns:
            NoveltyResult with the decision and metadata.
        """
        if verbose:
            logger.info(f"Evaluating novelty of code: {code.name}")

        # Ensure the code has an embedding
        if code.embedding is None:
            code_text = f"{code.name}: {code.definition}"
            code.embedding = self.embedder.embed(code_text)

        # Increment evaluation counter
        self._evaluation_counter += 1
        code.evaluation_order = self._evaluation_counter

        # Check if codebook is empty (first code is auto-accepted)
        if len(codebook.accepted_codes) == 0 and len(codebook.rejected_codes) == 0:
            if verbose:
                logger.info(f"  First code - auto-accepting: {code.name}")

            # Set code metadata
            code.novelty_stage = "first_code"
            code.novelty_reasoning = "First code in codebook - automatically accepted"
            code.similarity_score = 0.0
            code.similar_to = None

            return NoveltyResult(
                code=code,
                is_novel=True,
                stage="first_code",
                max_similarity=0.0,
                most_similar_code=None,
                similar_to_accepted=False,
                reasoning="First code in codebook - automatically accepted",
            )

        # Stage 1: Semantic similarity check against ALL codes
        stage1_result = self._stage1_similarity_check(code, codebook, verbose)

        if not stage1_result.is_novel:
            # Auto-rejected in Stage 1 - set code metadata
            code.novelty_stage = stage1_result.stage
            code.novelty_reasoning = stage1_result.reasoning
            code.similarity_score = stage1_result.max_similarity
            code.similar_to = stage1_result.most_similar_code.name if stage1_result.most_similar_code else None
            return stage1_result

        # Stage 2: LLM evaluation with RAG
        stage2_result = self._stage2_llm_evaluation(code, codebook, verbose)

        # Set code metadata from Stage 2 result
        code.novelty_stage = stage2_result.stage
        code.novelty_reasoning = stage2_result.reasoning
        code.similarity_score = stage2_result.max_similarity
        code.similar_to = stage2_result.most_similar_code.name if stage2_result.most_similar_code else None

        return stage2_result

    def _stage1_similarity_check(
        self,
        code: Code,
        codebook: Codebook,
        verbose: bool = False,
    ) -> NoveltyResult:
        """
        Stage 1: Check semantic similarity against all codes.

        Args:
            code: The code to evaluate.
            codebook: The current codebook.
            verbose: If True, log detailed information.

        Returns:
            NoveltyResult. If is_novel=False, the code was auto-rejected.
        """
        if verbose:
            logger.info("  Stage 1: Semantic similarity check")

        max_similarity, most_similar, is_from_accepted = codebook.check_similarity_against_all(
            code.embedding
        )

        if verbose:
            if most_similar:
                logger.info(f"    Max similarity: {max_similarity:.3f} with '{most_similar.name}'")
                logger.info(f"    Similar code is from: {'accepted' if is_from_accepted else 'rejected'} group")
            else:
                logger.info("    No similar codes found")

        if max_similarity > self.similarity_threshold:
            # Auto-reject
            reason = (
                f"Similar to {'accepted' if is_from_accepted else 'rejected'} code "
                f"'{most_similar.name}' (similarity: {max_similarity:.3f})"
            )

            if verbose:
                logger.info(f"    AUTO-REJECT: {reason}")

            return NoveltyResult(
                code=code,
                is_novel=False,
                stage="stage1_auto_reject",
                max_similarity=max_similarity,
                most_similar_code=most_similar,
                similar_to_accepted=is_from_accepted,
                reasoning=reason,
            )

        # Passed Stage 1
        if verbose:
            logger.info(f"    PASSED: Similarity {max_similarity:.3f} <= threshold {self.similarity_threshold}")

        return NoveltyResult(
            code=code,
            is_novel=True,  # Preliminary - will be updated by Stage 2
            stage="stage1_pass",
            max_similarity=max_similarity,
            most_similar_code=most_similar,
            similar_to_accepted=is_from_accepted,
            reasoning="Passed semantic similarity check",
        )

    def _stage2_llm_evaluation(
        self,
        code: Code,
        codebook: Codebook,
        verbose: bool = False,
    ) -> NoveltyResult:
        """
        Stage 2: LLM evaluation with RAG.

        Args:
            code: The code to evaluate.
            codebook: The current codebook.
            verbose: If True, log detailed information.

        Returns:
            NoveltyResult with final decision.
        """
        if verbose:
            logger.info("  Stage 2: LLM novelty evaluation")

        # Get top-K similar codes from accepted codes
        similar_accepted = codebook.find_similar_codes(
            code.embedding,
            top_k=self.top_k_rag,
            accepted_only=True,
        )

        # Optionally get similar rejected codes too
        similar_rejected = []
        if self.include_rejected_in_rag and codebook.rejected_codes:
            similar_rejected = codebook.find_similar_codes(
                code.embedding,
                top_k=self.top_k_rag // 2,  # Fewer rejected codes than accepted
                accepted_only=False,
            )
            # Filter to only rejected codes (find_similar_codes with accepted_only=False returns all)
            similar_rejected = [
                (c, s) for c, s in similar_rejected
                if c in codebook.rejected_codes
            ][:self.top_k_rag // 2]

        if verbose:
            logger.info(f"    Retrieved {len(similar_accepted)} similar accepted codes for comparison")
            for sim_code, sim_score in similar_accepted:
                logger.info(f"      - {sim_code.name} (similarity: {sim_score:.3f})")
            if similar_rejected:
                logger.info(f"    Retrieved {len(similar_rejected)} similar rejected codes for context")
                for sim_code, sim_score in similar_rejected:
                    logger.info(f"      - {sim_code.name} (similarity: {sim_score:.3f}) [REJECTED]")

        # Format codes for prompt
        existing_codes_text = self._format_similar_codes_with_status(
            similar_accepted, similar_rejected
        )

        prompt = self._user_prompt.format(
            code_name=code.name,
            code_definition=code.definition,
            existing_codes=existing_codes_text,
        )

        # Add study context to system prompt if available
        system = add_study_context(self._system_prompt, self.study_context)

        if verbose:
            logger.debug(f"    Prompt (v{self.prompt_version}):\n{prompt}")

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            if verbose:
                logger.debug(f"    LLM Response: {response}")

            is_novel = response.get("is_novel", False)
            reasoning = response.get("reasoning", "No reasoning provided")
            similar_to = response.get("similar_to", None)

            stage = "stage2_accept" if is_novel else "stage2_reject"

            if verbose:
                logger.info(f"    Decision: {'ACCEPT' if is_novel else 'REJECT'}")
                logger.info(f"    Reasoning: {reasoning}")
                if similar_to:
                    logger.info(f"    Similar to: {similar_to}")

            # Get the most similar code for the result (from accepted codes)
            most_similar_code = similar_accepted[0][0] if similar_accepted else None
            max_similarity = similar_accepted[0][1] if similar_accepted else 0.0

            return NoveltyResult(
                code=code,
                is_novel=is_novel,
                stage=stage,
                max_similarity=max_similarity,
                most_similar_code=most_similar_code,
                similar_to_accepted=True,  # Stage 2 only compares to accepted
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"    LLM evaluation failed: {e}")
            # Default to rejecting on error (conservative)
            return NoveltyResult(
                code=code,
                is_novel=False,
                stage="stage2_error",
                max_similarity=0.0,
                most_similar_code=None,
                similar_to_accepted=False,
                reasoning=f"LLM evaluation failed: {e}",
            )

    def _format_similar_codes(self, similar_codes: list[Tuple[Code, float]]) -> str:
        """Format similar codes for the prompt (legacy method)."""
        if not similar_codes:
            return "No similar codes in codebook yet."

        lines = []
        for i, (code, similarity) in enumerate(similar_codes, 1):
            lines.append(f"{i}. {code.name} (similarity: {similarity:.2f})")
            lines.append(f"   Definition: {code.definition}")

        return "\n".join(lines)

    def _format_similar_codes_with_status(
        self,
        accepted_codes: list[Tuple[Code, float]],
        rejected_codes: list[Tuple[Code, float]],
    ) -> str:
        """Format similar codes for the prompt, distinguishing accepted and rejected."""
        if not accepted_codes and not rejected_codes:
            return "No similar codes in codebook yet."

        lines = []

        # Format accepted codes
        if accepted_codes:
            lines.append("ACCEPTED CODES IN CODEBOOK:")
            for i, (code, similarity) in enumerate(accepted_codes, 1):
                lines.append(f"{i}. {code.name} (similarity: {similarity:.2f})")
                lines.append(f"   Definition: {code.definition}")

        # Format rejected codes (if any)
        if rejected_codes:
            if lines:
                lines.append("")  # Blank line separator
            lines.append("PREVIOUSLY REJECTED SIMILAR CODES (for context):")
            for i, (code, similarity) in enumerate(rejected_codes, 1):
                lines.append(f"{i}. {code.name} (similarity: {similarity:.2f}) [REJECTED]")
                lines.append(f"   Definition: {code.definition}")
                if code.novelty_reasoning:
                    # Truncate long reasoning
                    reason = code.novelty_reasoning[:150]
                    if len(code.novelty_reasoning) > 150:
                        reason += "..."
                    lines.append(f"   Rejection reason: {reason}")

        return "\n".join(lines)

    def evaluate_batch(
        self,
        codes: list[Code],
        codebook: Codebook,
        verbose: bool = False,
    ) -> list[NoveltyResult]:
        """
        Evaluate multiple codes and update codebook.

        This processes codes sequentially, updating the codebook after each
        decision so subsequent evaluations see the updated state.

        Args:
            codes: List of codes to evaluate.
            codebook: The codebook to update.
            verbose: If True, log detailed information.

        Returns:
            List of NoveltyResult objects.
        """
        results = []

        for code in codes:
            result = self.evaluate(code, codebook, verbose=verbose)
            results.append(result)

            # Update codebook based on result
            if result.is_novel:
                codebook.add_code(code, accepted=True)
                if verbose:
                    logger.info(f"  Added to codebook: {code.name}")
            else:
                codebook.add_code(code, accepted=False)
                if verbose:
                    logger.info(f"  Rejected from codebook: {code.name}")

        return results

    def embed_code(self, code: Code) -> Code:
        """
        Ensure a code has an embedding.

        Args:
            code: The code to embed.

        Returns:
            The same code with embedding populated.
        """
        if code.embedding is None:
            code_text = f"{code.name}: {code.definition}"
            code.embedding = self.embedder.embed(code_text)
        return code

    def embed_codes(self, codes: list[Code]) -> list[Code]:
        """
        Ensure all codes have embeddings.

        Args:
            codes: List of codes to embed.

        Returns:
            The same codes with embeddings populated.
        """
        texts = []
        codes_to_embed = []

        for code in codes:
            if code.embedding is None:
                texts.append(f"{code.name}: {code.definition}")
                codes_to_embed.append(code)

        if texts:
            embeddings = self.embedder.embed(texts)
            for i, code in enumerate(codes_to_embed):
                code.embedding = embeddings[i]

        return codes

    def __repr__(self) -> str:
        return (
            f"NoveltyEvaluator(threshold={self.similarity_threshold}, "
            f"top_k={self.top_k_rag}, prompt_v{self.prompt_version})"
        )
