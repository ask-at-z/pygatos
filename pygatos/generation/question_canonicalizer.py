"""Question canonicalization for focus group questions."""

import logging
import os
import re
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pygatos.llm.base import BaseLLM
from pygatos.core.embedder import Embedder
from pygatos.core.clusterer import Clusterer
from pygatos.prompts import (
    QUESTION_CANONICAL_SYSTEM,
    QUESTION_CANONICAL_PROMPT,
    QUESTION_ASSIGNMENT_SYSTEM,
    QUESTION_ASSIGNMENT_PROMPT,
    add_study_context,
)

logger = logging.getLogger(__name__)


@dataclass
class CanonicalQuestion:
    """A canonical (standardized) question."""

    canonical_question: str
    topic: str
    original_questions: list[str] = field(default_factory=list)
    original_indices: list[int] = field(default_factory=list)
    cluster_id: int = -1
    embedding: Optional[np.ndarray] = None


@dataclass
class QuestionMapping:
    """Mapping from an original question to its canonical form."""

    original_question: str
    canonical_question: str
    topic: str
    date: str
    group_name: str
    similarity_score: float = 0.0
    # Audit fields for LLM-based assignment
    assignment_method: str = "embedding"  # "embedding" or "llm"
    llm_reasoning: Optional[str] = None
    llm_confidence: Optional[str] = None
    candidates_considered: Optional[str] = None  # JSON string of top candidates


@dataclass
class ClusterGenerationAudit:
    """Audit record for canonical question generation from a cluster."""

    cluster_id: int
    num_original_questions: int
    original_questions: list[str]  # The questions sent to the LLM
    num_canonicals_generated: int
    canonicals_generated: list[dict]  # List of {canonical_question, topic, covers_questions}
    raw_llm_response: Optional[dict] = None  # The full parsed JSON response
    error: Optional[str] = None  # Error message if generation failed


class QuestionCanonicalizer:
    """
    Canonicalizes focus group questions by clustering and LLM-based labeling.

    This class:
    1. Loads questions from CSV files
    2. Embeds and clusters semantically similar questions
    3. Uses an LLM to generate 1-N canonical questions per cluster
    4. Assigns each original question to its best-fit canonical form

    Example:
        >>> canonicalizer = QuestionCanonicalizer(llm, embedder)
        >>> canonicalizer.load_questions("data/FG_Qs_Asked")
        >>> canonicalizer.cluster_questions(n_clusters=50)
        >>> canonicalizer.generate_canonical_questions()
        >>> mappings = canonicalizer.get_mappings()
        >>> canonicalizer.export("output/questions_codebooked.csv")
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Embedder,
        clusterer: Optional[Clusterer] = None,
        temperature: Optional[float] = None,
        study_context: Optional[str] = None,
    ):
        """
        Initialize the question canonicalizer.

        Args:
            llm: LLM backend for generating canonical questions.
            embedder: Embedder for question embeddings.
            clusterer: Optional clusterer instance. If None, uses default agglomerative.
            temperature: Optional temperature override for LLM.
            study_context: Optional context about the study/dataset.
        """
        self.llm = llm
        self.embedder = embedder
        self.clusterer = clusterer or Clusterer(method="agglomerative")
        self.temperature = temperature
        self.study_context = study_context

        # Data storage
        self._questions_df: Optional[pd.DataFrame] = None
        self._unique_questions: list[str] = []
        self._unique_questions_cleaned: list[str] = []  # Without number prefixes
        self._question_embeddings: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._canonical_questions: list[CanonicalQuestion] = []
        self._mappings: list[QuestionMapping] = []
        self._cluster_generation_audit: list[ClusterGenerationAudit] = []

    @staticmethod
    def strip_question_number(question: str) -> str:
        """
        Strip leading number prefix from a question.

        Handles patterns like:
        - "3. What do you think..."
        - "14. How would you..."
        - "Q5. Why did..."

        Args:
            question: The original question text.

        Returns:
            Question text without the leading number prefix.
        """
        if not question:
            return question

        # Pattern: optional 'Q' or 'q', followed by digits, followed by '.', optional space
        # Examples: "3. ", "14. ", "Q5. ", "q12. "
        pattern = r'^[Qq]?\d+\.\s*'
        cleaned = re.sub(pattern, '', question.strip())
        return cleaned

    def load_questions(
        self,
        folder_path: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Load all questions from a folder of CSV files.

        Args:
            folder_path: Path to folder containing question CSV files.
            verbose: If True, log progress.

        Returns:
            DataFrame with all questions and metadata.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in: {folder_path}")

        if verbose:
            logger.info(f"Loading questions from {len(csv_files)} files")

        dfs = []
        for csv_file in tqdm(csv_files, desc="Loading files", disable=not verbose):
            try:
                df = pd.read_csv(csv_file)
                # Ensure expected columns exist
                if "Question" in df.columns:
                    dfs.append(df)
                else:
                    logger.warning(f"Skipping {csv_file.name}: no 'Question' column")
            except Exception as e:
                logger.warning(f"Error loading {csv_file.name}: {e}")

        if not dfs:
            raise ValueError("No valid question files loaded")

        self._questions_df = pd.concat(dfs, ignore_index=True)

        # Get unique questions for clustering (original text)
        self._unique_questions = self._questions_df["Question"].dropna().unique().tolist()

        # Create cleaned versions (without number prefixes) for embedding/clustering
        self._unique_questions_cleaned = [
            self.strip_question_number(q) for q in self._unique_questions
        ]

        if verbose:
            logger.info(
                f"Loaded {len(self._questions_df)} question instances, "
                f"{len(self._unique_questions)} unique questions"
            )

        return self._questions_df

    def load_questions_from_dataframe(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Load questions from an existing DataFrame.

        Args:
            df: DataFrame with 'Question', 'Date', 'Group Name' columns.
            verbose: If True, log progress.

        Returns:
            The stored DataFrame.
        """
        if "Question" not in df.columns:
            raise ValueError("DataFrame must have 'Question' column")

        self._questions_df = df.copy()
        self._unique_questions = df["Question"].dropna().unique().tolist()

        # Create cleaned versions (without number prefixes) for embedding/clustering
        self._unique_questions_cleaned = [
            self.strip_question_number(q) for q in self._unique_questions
        ]

        if verbose:
            logger.info(
                f"Loaded {len(df)} question instances, "
                f"{len(self._unique_questions)} unique questions"
            )

        return self._questions_df

    def embed_questions(self, verbose: bool = False) -> np.ndarray:
        """
        Generate embeddings for all unique questions.

        Uses cleaned questions (without number prefixes) for embedding.

        Args:
            verbose: If True, log progress.

        Returns:
            Array of question embeddings.
        """
        if not self._unique_questions_cleaned:
            raise ValueError("No questions loaded. Call load_questions() first.")

        if verbose:
            logger.info(f"Embedding {len(self._unique_questions_cleaned)} unique questions (cleaned)")

        self._question_embeddings = self.embedder.embed_batch(
            self._unique_questions_cleaned,
            show_progress=verbose,
        )

        return self._question_embeddings

    def cluster_questions(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Cluster questions by semantic similarity.

        Args:
            n_clusters: Number of clusters. If None, auto-determine.
            distance_threshold: Distance threshold for agglomerative clustering.
            verbose: If True, log progress.

        Returns:
            Array of cluster labels.
        """
        if self._question_embeddings is None:
            self.embed_questions(verbose=verbose)

        # Update clusterer settings if provided
        if n_clusters is not None:
            self.clusterer.n_clusters = n_clusters
        if distance_threshold is not None:
            self.clusterer.distance_threshold = distance_threshold

        if verbose:
            logger.info("Clustering questions...")

        self._cluster_labels = self.clusterer.fit(self._question_embeddings)

        if verbose:
            stats = self.clusterer.get_cluster_stats(self._cluster_labels)
            logger.info(
                f"Created {stats['n_clusters']} clusters "
                f"(avg size: {stats['mean_cluster_size']:.1f})"
            )

        return self._cluster_labels

    def generate_canonical_questions(
        self,
        verbose: bool = False,
    ) -> list[CanonicalQuestion]:
        """
        Generate canonical questions for each cluster using LLM.

        Args:
            verbose: If True, log progress.

        Returns:
            List of CanonicalQuestion objects.
        """
        if self._cluster_labels is None:
            raise ValueError("Questions not clustered. Call cluster_questions() first.")

        # Group questions by cluster
        cluster_indices = self.clusterer.get_cluster_indices(self._cluster_labels)

        if verbose:
            logger.info(f"Generating canonical questions for {len(cluster_indices)} clusters")

        self._canonical_questions = []

        clusters_iter = cluster_indices.items()
        if verbose:
            clusters_iter = tqdm(clusters_iter, desc="Processing clusters", total=len(cluster_indices))

        for cluster_id, indices in clusters_iter:
            # Use cleaned questions for LLM prompt
            cluster_questions = [self._unique_questions_cleaned[i] for i in indices]

            # Generate canonical question(s) for this cluster
            canonicals = self._generate_canonical_for_cluster(
                cluster_id=cluster_id,
                questions=cluster_questions,
                indices=indices,
                verbose=verbose,
            )
            self._canonical_questions.extend(canonicals)

        if verbose:
            logger.info(f"Generated {len(self._canonical_questions)} canonical questions")

        return self._canonical_questions

    def _generate_canonical_for_cluster(
        self,
        cluster_id: int,
        questions: list[str],
        indices: list[int],
        verbose: bool = False,
    ) -> list[CanonicalQuestion]:
        """
        Generate canonical question(s) for a single cluster.

        Args:
            cluster_id: The cluster ID.
            questions: List of questions in this cluster.
            indices: Original indices of questions.
            verbose: If True, log details.

        Returns:
            List of CanonicalQuestion objects (1 or more per cluster).
        """
        # Format questions for prompt
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        prompt = QUESTION_CANONICAL_PROMPT.format(questions=questions_text)
        system = add_study_context(QUESTION_CANONICAL_SYSTEM, self.study_context)

        # Log full prompt and system message
        logger.debug(f"=== Cluster {cluster_id} ({len(questions)} questions) ===")
        logger.debug(f"SYSTEM:\n{system}")
        logger.debug(f"PROMPT:\n{prompt}")

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            # Log full response
            logger.debug(f"RESPONSE:\n{response}")

            # Parse response - new format with multiple canonical questions
            canonical_list = response.get("canonical_questions", [])

            # Fallback for old single-question format
            if not canonical_list and "canonical_question" in response:
                canonical_list = [{
                    "canonical_question": response["canonical_question"],
                    "topic": response.get("topic", "Unknown"),
                    "covers_questions": list(range(1, len(questions) + 1)),
                }]

            canonicals = []
            canonicals_for_audit = []
            for item in canonical_list:
                canonical_q = item.get("canonical_question", "")
                topic = item.get("topic", "Unknown")
                covers = item.get("covers_questions", list(range(1, len(questions) + 1)))

                # Convert 1-indexed to 0-indexed
                covered_indices = [i - 1 for i in covers if 1 <= i <= len(questions)]

                canonical = CanonicalQuestion(
                    canonical_question=canonical_q,
                    topic=topic,
                    original_questions=[questions[i] for i in covered_indices],
                    original_indices=[indices[i] for i in covered_indices],
                    cluster_id=cluster_id,
                )

                # Generate embedding for the canonical question
                canonical.embedding = self.embedder.embed(canonical_q)

                canonicals.append(canonical)
                canonicals_for_audit.append({
                    "canonical_question": canonical_q,
                    "topic": topic,
                    "covers_questions": covers,
                })

            # Store audit record
            self._cluster_generation_audit.append(ClusterGenerationAudit(
                cluster_id=cluster_id,
                num_original_questions=len(questions),
                original_questions=questions,
                num_canonicals_generated=len(canonicals),
                canonicals_generated=canonicals_for_audit,
                raw_llm_response=response,
                error=None,
            ))

            return canonicals

        except Exception as e:
            logger.error(f"Error generating canonical for cluster {cluster_id}: {e}")

            # Store audit record for error case
            self._cluster_generation_audit.append(ClusterGenerationAudit(
                cluster_id=cluster_id,
                num_original_questions=len(questions),
                original_questions=questions,
                num_canonicals_generated=1,
                canonicals_generated=[{
                    "canonical_question": questions[0],
                    "topic": "Unknown",
                    "covers_questions": list(range(1, len(questions) + 1)),
                }],
                raw_llm_response=None,
                error=str(e),
            ))

            # Fallback: use most representative question
            return [CanonicalQuestion(
                canonical_question=questions[0],
                topic="Unknown",
                original_questions=questions,
                original_indices=indices,
                cluster_id=cluster_id,
            )]

    def assign_questions_to_canonicals(
        self,
        verbose: bool = False,
        use_llm: bool = False,
        low_confidence_threshold: float = 0.8,
        ambiguity_threshold: float = 0.9,
        close_match_gap: float = 0.05,
        top_k_candidates: int = 5,
    ) -> list[QuestionMapping]:
        """
        Assign each original question to its best-fit canonical question.

        Supports hybrid mode where LLM is used for:
        - Low confidence matches (similarity < low_confidence_threshold)
        - Ambiguous matches (multiple candidates above ambiguity_threshold)
        - Close matches (top 2 candidates within close_match_gap of each other)

        Args:
            verbose: If True, log progress.
            use_llm: If True, use LLM for low-confidence and ambiguous cases.
            low_confidence_threshold: Use LLM when best match is below this.
            ambiguity_threshold: Use LLM when multiple matches are above this.
            close_match_gap: Use LLM when top 2 matches are within this gap.
            top_k_candidates: Number of candidates to present to LLM.

        Returns:
            List of QuestionMapping objects.
        """
        if not self._canonical_questions:
            raise ValueError("No canonical questions. Call generate_canonical_questions() first.")

        if self._questions_df is None:
            raise ValueError("No questions loaded. Call load_questions() first.")

        # Build canonical embeddings matrix
        canonical_embeddings = np.vstack([
            cq.embedding if cq.embedding is not None else self.embedder.embed(cq.canonical_question)
            for cq in self._canonical_questions
        ])

        if verbose:
            logger.info(f"Assigning {len(self._questions_df)} questions to {len(self._canonical_questions)} canonicals")
            if use_llm:
                logger.info(f"  Hybrid mode: LLM for similarity < {low_confidence_threshold}, multiple > {ambiguity_threshold}, or gap < {close_match_gap}")

        self._mappings = []
        llm_calls = 0
        embedding_only = 0

        for _, row in tqdm(
            self._questions_df.iterrows(),
            desc="Assigning questions",
            total=len(self._questions_df),
            disable=not verbose,
        ):
            question = row.get("Question", "")
            date = str(row.get("Date", ""))
            group_name = str(row.get("Group Name", ""))

            if pd.isna(question) or not question.strip():
                # Skip empty questions
                continue

            # Find best matching canonical via embeddings
            q_embedding = self.embedder.embed(question)
            similarities = np.dot(canonical_embeddings, q_embedding.T).flatten()

            # Get top-k indices sorted by similarity
            top_k_indices = np.argsort(similarities)[-top_k_candidates:][::-1]
            best_idx = top_k_indices[0]
            best_score = float(similarities[best_idx])

            # Determine if we need LLM judgment
            needs_llm = False
            if use_llm:
                # Case 1: Low confidence - best match is below threshold
                if best_score < low_confidence_threshold:
                    needs_llm = True
                    logger.debug(f"LLM needed (low confidence {best_score:.3f}): {question[:50]}...")

                # Case 2: Ambiguous - multiple high-scoring candidates
                high_scoring = [i for i in top_k_indices if similarities[i] >= ambiguity_threshold]
                if len(high_scoring) > 1:
                    needs_llm = True
                    logger.debug(f"LLM needed (ambiguous, {len(high_scoring)} candidates > {ambiguity_threshold}): {question[:50]}...")

                # Case 3: Close match - top 2 candidates are very close in score
                if len(top_k_indices) >= 2 and not needs_llm:
                    second_best_score = float(similarities[top_k_indices[1]])
                    gap = best_score - second_best_score
                    if gap < close_match_gap:
                        needs_llm = True
                        logger.debug(f"LLM needed (close match, gap {gap:.3f} < {close_match_gap}): {question[:50]}...")

            # Initialize audit fields
            assignment_method = "embedding"
            llm_reasoning = None
            llm_confidence = None
            candidates_considered = None

            if needs_llm:
                # Use LLM to judge
                result = self._llm_judge_assignment(
                    question=question,
                    candidate_indices=top_k_indices,
                    similarities=similarities,
                )
                llm_calls += 1

                if result is not None:
                    assignment_method = "llm"
                    llm_reasoning = result.reasoning
                    llm_confidence = result.confidence
                    candidates_considered = result.candidates_json

                    if result.selected_index is not None:
                        best_idx = result.selected_index
                        best_score = result.similarity_score
                    # If LLM returns None index, fall back to embedding result
            else:
                embedding_only += 1

            best_canonical = self._canonical_questions[best_idx]

            self._mappings.append(QuestionMapping(
                original_question=question,
                canonical_question=best_canonical.canonical_question,
                topic=best_canonical.topic,
                date=date,
                group_name=group_name,
                similarity_score=best_score,
                assignment_method=assignment_method,
                llm_reasoning=llm_reasoning,
                llm_confidence=llm_confidence,
                candidates_considered=candidates_considered,
            ))

        if verbose:
            logger.info(f"Created {len(self._mappings)} question mappings")
            if use_llm:
                logger.info(f"  Embedding-only: {embedding_only}, LLM-judged: {llm_calls}")

        return self._mappings

    @dataclass
    class LLMJudgmentResult:
        """Result from LLM judgment including audit information."""
        selected_index: Optional[int]
        similarity_score: float
        reasoning: str
        confidence: str
        candidates_json: str

    def _llm_judge_assignment(
        self,
        question: str,
        candidate_indices: np.ndarray,
        similarities: np.ndarray,
    ) -> Optional["QuestionCanonicalizer.LLMJudgmentResult"]:
        """
        Use LLM to judge which canonical question best matches.

        Args:
            question: The original question to assign.
            candidate_indices: Indices of candidate canonicals (sorted by similarity).
            similarities: Full similarity array.

        Returns:
            LLMJudgmentResult with selection and audit info, or None on error.
        """
        import json

        # Format candidates for prompt and for audit
        candidates_text = []
        candidates_audit = []
        for i, idx in enumerate(candidate_indices):
            cq = self._canonical_questions[idx]
            sim = float(similarities[idx])  # Convert numpy float to Python float
            candidates_text.append(f"{i+1}. [{cq.topic}] {cq.canonical_question} (similarity: {sim:.3f})")
            candidates_audit.append({
                "rank": i + 1,
                "topic": cq.topic,
                "canonical": cq.canonical_question,
                "similarity": round(sim, 4),
            })

        candidates_str = "\n".join(candidates_text)
        candidates_json = json.dumps(candidates_audit)

        prompt = QUESTION_ASSIGNMENT_PROMPT.format(
            original_question=question,
            candidates=candidates_str,
        )
        system = add_study_context(QUESTION_ASSIGNMENT_SYSTEM, self.study_context)

        # Log prompt
        logger.debug(f"=== LLM Assignment Judgment ===")
        logger.debug(f"ORIGINAL: {question}")
        logger.debug(f"CANDIDATES:\n{candidates_str}")

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            logger.debug(f"RESPONSE: {response}")

            selected_idx = response.get("selected_index")
            reasoning = response.get("reasoning", "")
            confidence = response.get("confidence", "unknown")

            if selected_idx is None:
                # LLM said no good match - return result with None index
                logger.debug(f"LLM: No good match - {reasoning}")
                return QuestionCanonicalizer.LLMJudgmentResult(
                    selected_index=None,
                    similarity_score=0.0,
                    reasoning=reasoning,
                    confidence=confidence,
                    candidates_json=candidates_json,
                )

            # Convert 1-indexed to actual index
            if 1 <= selected_idx <= len(candidate_indices):
                actual_idx = int(candidate_indices[selected_idx - 1])
                actual_sim = float(similarities[actual_idx])
                logger.debug(f"LLM selected candidate {selected_idx} (index {actual_idx}, sim {actual_sim:.3f})")
                return QuestionCanonicalizer.LLMJudgmentResult(
                    selected_index=actual_idx,
                    similarity_score=actual_sim,
                    reasoning=reasoning,
                    confidence=confidence,
                    candidates_json=candidates_json,
                )
            else:
                logger.warning(f"LLM returned invalid index {selected_idx}")
                return QuestionCanonicalizer.LLMJudgmentResult(
                    selected_index=None,
                    similarity_score=0.0,
                    reasoning=f"Invalid index {selected_idx}",
                    confidence="error",
                    candidates_json=candidates_json,
                )

        except Exception as e:
            logger.error(f"LLM judgment failed: {e}")
            return None

    def get_mappings(self) -> list[QuestionMapping]:
        """Return the question mappings."""
        return self._mappings

    def get_canonical_questions(self) -> list[CanonicalQuestion]:
        """Return the canonical questions."""
        return self._canonical_questions

    def load_canonical_questions(
        self,
        csv_path: str,
        verbose: bool = False,
    ) -> list[CanonicalQuestion]:
        """
        Load canonical questions from a previously exported CSV file.

        This allows running just the assignment step without regenerating canonicals.

        Args:
            csv_path: Path to canonical_questions.csv file.
            verbose: If True, log progress.

        Returns:
            List of CanonicalQuestion objects.
        """
        if verbose:
            logger.info(f"Loading canonical questions from {csv_path}")

        df = pd.read_csv(csv_path)

        self._canonical_questions = []
        for _, row in df.iterrows():
            cq = CanonicalQuestion(
                canonical_question=row["Canonical Question"],
                topic=row.get("Topic", "Unknown"),
                cluster_id=int(row.get("Cluster ID", -1)),
                original_questions=[],  # Not stored in CSV
                original_indices=[],
            )
            self._canonical_questions.append(cq)

        # Generate embeddings for all canonical questions
        if verbose:
            logger.info(f"Generating embeddings for {len(self._canonical_questions)} canonical questions...")

        canonical_texts = [cq.canonical_question for cq in self._canonical_questions]
        embeddings = self.embedder.embed_batch(canonical_texts, show_progress=verbose)

        for i, cq in enumerate(self._canonical_questions):
            cq.embedding = embeddings[i]

        if verbose:
            logger.info(f"Loaded {len(self._canonical_questions)} canonical questions")

        return self._canonical_questions

    def export(
        self,
        output_path: str,
        verbose: bool = False,
        include_audit: bool = True,
    ) -> pd.DataFrame:
        """
        Export the codebooked questions to a CSV file.

        Args:
            output_path: Path to output CSV file.
            verbose: If True, log progress.
            include_audit: If True, include audit fields (assignment method, LLM reasoning, etc).

        Returns:
            DataFrame with original and canonical questions.
        """
        if not self._mappings:
            raise ValueError("No mappings. Call assign_questions_to_canonicals() first.")

        # Create output DataFrame
        data = []
        for mapping in self._mappings:
            row = {
                "Question": mapping.original_question,
                "Date": mapping.date,
                "Group Name": mapping.group_name,
                "Codebooked Question": mapping.canonical_question,
                "Topic": mapping.topic,
                "Similarity Score": mapping.similarity_score,
            }
            if include_audit:
                row["Assignment Method"] = mapping.assignment_method
                row["LLM Reasoning"] = mapping.llm_reasoning
                row["LLM Confidence"] = mapping.llm_confidence
                row["Candidates Considered"] = mapping.candidates_considered
            data.append(row)

        df = pd.DataFrame(data)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False)

        if verbose:
            logger.info(f"Exported {len(df)} codebooked questions to {output_path}")

        return df

    def export_canonical_questions(
        self,
        output_path: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Export just the canonical questions to a CSV file.

        Args:
            output_path: Path to output CSV file.
            verbose: If True, log progress.

        Returns:
            DataFrame with canonical questions and metadata.
        """
        if not self._canonical_questions:
            raise ValueError("No canonical questions. Call generate_canonical_questions() first.")

        data = []
        for i, cq in enumerate(self._canonical_questions):
            data.append({
                "ID": i + 1,
                "Canonical Question": cq.canonical_question,
                "Topic": cq.topic,
                "Cluster ID": cq.cluster_id,
                "Original Question Count": len(cq.original_questions),
            })

        df = pd.DataFrame(data)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False)

        if verbose:
            logger.info(f"Exported {len(df)} canonical questions to {output_path}")

        return df

    def export_cluster_generation_audit(
        self,
        output_path: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Export the cluster generation audit log to a CSV file.

        Each row represents one cluster that was sent to the LLM, with columns for:
        - Cluster ID and size
        - Original questions (as JSON array)
        - Number of canonicals generated
        - Each canonical's question, topic, and covered questions
        - Raw LLM response (as JSON)
        - Any errors that occurred

        Args:
            output_path: Path to output CSV file.
            verbose: If True, log progress.

        Returns:
            DataFrame with cluster generation audit data.
        """
        import json

        if not self._cluster_generation_audit:
            raise ValueError("No cluster generation audit data. Call generate_canonical_questions() first.")

        data = []
        for audit in self._cluster_generation_audit:
            row = {
                "Cluster ID": audit.cluster_id,
                "Num Original Questions": audit.num_original_questions,
                "Original Questions": json.dumps(audit.original_questions),
                "Num Canonicals Generated": audit.num_canonicals_generated,
                "Error": audit.error,
            }

            # Flatten canonicals into separate columns
            for i, canonical in enumerate(audit.canonicals_generated):
                prefix = f"Canonical {i+1}"
                row[f"{prefix} Question"] = canonical.get("canonical_question", "")
                row[f"{prefix} Topic"] = canonical.get("topic", "")
                row[f"{prefix} Covers"] = json.dumps(canonical.get("covers_questions", []))

            # Store raw LLM response
            row["Raw LLM Response"] = json.dumps(audit.raw_llm_response) if audit.raw_llm_response else None

            data.append(row)

        df = pd.DataFrame(data)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False)

        if verbose:
            logger.info(f"Exported {len(df)} cluster generation audit records to {output_path}")

        return df

    def export_by_group(
        self,
        output_folder: str,
        verbose: bool = False,
        include_audit: bool = True,
    ) -> dict[str, str]:
        """
        Export codebooked questions as separate files per group.

        Args:
            output_folder: Folder to write output files.
            verbose: If True, log progress.
            include_audit: If True, include audit fields (assignment method, LLM reasoning, etc).

        Returns:
            Dictionary mapping group name to output file path.
        """
        if not self._mappings:
            raise ValueError("No mappings. Call assign_questions_to_canonicals() first.")

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Group mappings by group name
        by_group: dict[str, list[QuestionMapping]] = {}
        for mapping in self._mappings:
            group = mapping.group_name
            if group not in by_group:
                by_group[group] = []
            by_group[group].append(mapping)

        output_files = {}
        for group_name, mappings in by_group.items():
            # Clean filename
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in group_name)
            output_path = os.path.join(output_folder, f"QsAsked__{safe_name}_codebooked.csv")

            data = []
            for mapping in mappings:
                row = {
                    "Question": mapping.original_question,
                    "Date": mapping.date,
                    "Group Name": mapping.group_name,
                    "Codebooked Question": mapping.canonical_question,
                    "Topic": mapping.topic,
                    "Similarity Score": mapping.similarity_score,
                }
                if include_audit:
                    row["Assignment Method"] = mapping.assignment_method
                    row["LLM Reasoning"] = mapping.llm_reasoning
                    row["LLM Confidence"] = mapping.llm_confidence
                    row["Candidates Considered"] = mapping.candidates_considered
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            output_files[group_name] = output_path

        if verbose:
            logger.info(f"Exported {len(output_files)} group-level files to {output_folder}")

        return output_files

    def run_full_pipeline(
        self,
        folder_path: str,
        output_path: str,
        distance_threshold: float = 0.9,
        n_clusters: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full canonicalization pipeline.

        Args:
            folder_path: Path to folder containing question CSV files.
            output_path: Path to output CSV file.
            distance_threshold: Distance threshold for agglomerative clustering.
                For normalized embeddings with ward linkage:
                - 0.8 → ~68% cosine similarity
                - 0.9 → ~60% cosine similarity (default)
                - 1.0 → ~50% cosine similarity
                Lower values = tighter clusters with fewer questions each.
            n_clusters: Fixed number of clusters. If provided, overrides distance_threshold.
            verbose: If True, log progress.

        Returns:
            DataFrame with codebooked questions.
        """
        if verbose:
            logger.info("Starting question canonicalization pipeline")

        # Step 1: Load questions
        self.load_questions(folder_path, verbose=verbose)

        # Step 2: Embed questions
        self.embed_questions(verbose=verbose)

        # Step 3: Cluster questions (prefer distance_threshold over n_clusters)
        if n_clusters is not None:
            self.cluster_questions(n_clusters=n_clusters, verbose=verbose)
        else:
            self.cluster_questions(distance_threshold=distance_threshold, verbose=verbose)

        # Step 4: Generate canonical questions
        self.generate_canonical_questions(verbose=verbose)

        # Step 5: Assign questions to canonicals
        self.assign_questions_to_canonicals(verbose=verbose)

        # Step 6: Export results
        result_df = self.export(output_path, verbose=verbose)

        if verbose:
            logger.info("Question canonicalization complete")

        return result_df

    def __repr__(self) -> str:
        n_questions = len(self._unique_questions) if self._unique_questions else 0
        n_canonical = len(self._canonical_questions) if self._canonical_questions else 0
        return f"QuestionCanonicalizer(questions={n_questions}, canonical={n_canonical})"
