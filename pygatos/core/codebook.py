"""Codebook data structures and operations."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd


@dataclass
class Code:
    """
    A single code in the codebook.

    Attributes:
        name: Short name/label for the code.
        definition: Longer description of what the code represents.
        embedding: Optional pre-computed embedding vector.
        source_cluster: Optional cluster ID that generated this code.
        theme: Optional theme this code belongs to.
        created_at: Timestamp when the code was created.
        evaluation_order: Order in which this code was evaluated (for visualization).
        novelty_stage: Stage at which novelty was determined (first_code, stage1_auto_reject, stage2_accept, etc.).
        novelty_reasoning: LLM's reasoning for the novelty decision.
        similarity_score: Max similarity score during novelty evaluation.
        similar_to: Name of most similar code during evaluation.
        metadata: Optional additional metadata.
    """

    name: str
    definition: str
    embedding: Optional[np.ndarray] = None
    source_cluster: Optional[int] = None
    theme: Optional[str] = None
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    evaluation_order: Optional[int] = None
    novelty_stage: Optional[str] = None
    novelty_reasoning: Optional[str] = None
    similarity_score: Optional[float] = None
    similar_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure embedding is a numpy array if provided."""
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)

    def to_dict(self) -> dict:
        """Convert the code to a dictionary (excludes embedding)."""
        return {
            "name": self.name,
            "definition": self.definition,
            "source_cluster": self.source_cluster,
            "theme": self.theme,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evaluation_order": self.evaluation_order,
            "novelty_stage": self.novelty_stage,
            "novelty_reasoning": self.novelty_reasoning,
            "similarity_score": self.similarity_score,
            "similar_to": self.similar_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Code":
        """Create a Code from a dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            name=data["name"],
            definition=data["definition"],
            source_cluster=data.get("source_cluster"),
            theme=data.get("theme"),
            created_at=created_at,
            evaluation_order=data.get("evaluation_order"),
            novelty_stage=data.get("novelty_stage"),
            novelty_reasoning=data.get("novelty_reasoning"),
            similarity_score=data.get("similarity_score"),
            similar_to=data.get("similar_to"),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self):
        """Make Code hashable by name."""
        return hash(self.name)

    def __eq__(self, other):
        """Two codes are equal if they have the same name."""
        if not isinstance(other, Code):
            return False
        return self.name == other.name

    def __repr__(self):
        return f"Code(name='{self.name}', theme='{self.theme}')"


@dataclass
class Theme:
    """
    A theme grouping multiple related codes.

    Attributes:
        name: Short name/label for the theme.
        definition: Description of what the theme represents.
        codes: List of codes belonging to this theme.
        embedding: Optional pre-computed embedding vector.
        created_at: Timestamp when the theme was created.
        source_cluster: Optional cluster ID that generated this theme.
        evaluation_order: Order in which this theme was evaluated (for visualization).
        novelty_stage: Stage at which novelty was determined.
        novelty_reasoning: LLM's reasoning for the novelty decision.
        similarity_score: Max similarity score during novelty evaluation.
        similar_to: Name of most similar theme during evaluation.
        metadata: Optional additional metadata.
    """

    name: str
    definition: str
    codes: list[Code] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    source_cluster: Optional[int] = None
    evaluation_order: Optional[int] = None
    novelty_stage: Optional[str] = None
    novelty_reasoning: Optional[str] = None
    similarity_score: Optional[float] = None
    similar_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure embedding is a numpy array if provided."""
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)

    def add_code(self, code: Code) -> None:
        """Add a code to this theme."""
        code.theme = self.name
        if code not in self.codes:
            self.codes.append(code)

    def to_dict(self) -> dict:
        """Convert the theme to a dictionary."""
        return {
            "name": self.name,
            "definition": self.definition,
            "codes": [c.to_dict() for c in self.codes],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_cluster": self.source_cluster,
            "evaluation_order": self.evaluation_order,
            "novelty_stage": self.novelty_stage,
            "novelty_reasoning": self.novelty_reasoning,
            "similarity_score": self.similarity_score,
            "similar_to": self.similar_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Theme":
        """Create a Theme from a dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        codes = [Code.from_dict(c) for c in data.get("codes", [])]

        return cls(
            name=data["name"],
            definition=data["definition"],
            codes=codes,
            created_at=created_at,
            source_cluster=data.get("source_cluster"),
            evaluation_order=data.get("evaluation_order"),
            novelty_stage=data.get("novelty_stage"),
            novelty_reasoning=data.get("novelty_reasoning"),
            similarity_score=data.get("similarity_score"),
            similar_to=data.get("similar_to"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self):
        return f"Theme(name='{self.name}', n_codes={len(self.codes)})"


class Codebook:
    """
    A collection of codes organized into themes.

    The Codebook maintains both accepted and rejected codes, allowing
    for tracking of the novelty evaluation process.

    Example:
        >>> codebook = Codebook()
        >>> code = Code(name="Economic Concerns", definition="Mentions of economic issues")
        >>> codebook.add_code(code, accepted=True)
        >>> similar = codebook.find_similar_codes(query_embedding, top_k=5)
    """

    def __init__(self):
        """Initialize an empty codebook."""
        self.accepted_codes: list[Code] = []
        self.rejected_codes: list[Code] = []
        self.themes: list[Theme] = []  # Accepted themes
        self.rejected_themes: list[Theme] = []  # Rejected themes during novelty eval
        self.created_at: datetime = datetime.now()
        self.metadata: dict = {}

        # Cache for embeddings
        self._accepted_embeddings_cache: Optional[np.ndarray] = None
        self._rejected_embeddings_cache: Optional[np.ndarray] = None

    def add_code(self, code: Code, accepted: bool = True) -> None:
        """
        Add a code to the codebook.

        Args:
            code: The code to add.
            accepted: If True, add to accepted codes; if False, add to rejected.
        """
        if accepted:
            if code not in self.accepted_codes:
                self.accepted_codes.append(code)
                self._accepted_embeddings_cache = None  # Invalidate cache
        else:
            if code not in self.rejected_codes:
                self.rejected_codes.append(code)
                self._rejected_embeddings_cache = None  # Invalidate cache

    def add_theme(self, theme: Theme, accepted: bool = True) -> None:
        """
        Add a theme to the codebook.

        Args:
            theme: The theme to add.
            accepted: If True, add to accepted themes; if False, add to rejected.
        """
        if accepted:
            if theme not in self.themes:
                self.themes.append(theme)
                # Update theme reference in codes
                for code in theme.codes:
                    code.theme = theme.name
        else:
            if theme not in self.rejected_themes:
                self.rejected_themes.append(theme)

    def get_code_by_name(self, name: str, accepted_only: bool = True) -> Optional[Code]:
        """
        Get a code by its name.

        Args:
            name: The code name to search for.
            accepted_only: If True, only search accepted codes.

        Returns:
            The matching Code or None if not found.
        """
        for code in self.accepted_codes:
            if code.name == name:
                return code

        if not accepted_only:
            for code in self.rejected_codes:
                if code.name == name:
                    return code

        return None

    def get_codes_by_theme(self, theme_name: str) -> list[Code]:
        """Get all codes belonging to a theme."""
        return [c for c in self.accepted_codes if c.theme == theme_name]

    def get_code_embeddings(self, accepted_only: bool = True) -> np.ndarray:
        """
        Get embeddings for all codes.

        Args:
            accepted_only: If True, return only accepted code embeddings.

        Returns:
            Numpy array of shape (n_codes, embedding_dim).
        """
        if accepted_only:
            if self._accepted_embeddings_cache is None:
                embeddings = [c.embedding for c in self.accepted_codes if c.embedding is not None]
                if embeddings:
                    self._accepted_embeddings_cache = np.vstack(embeddings)
                else:
                    return np.array([])
            return self._accepted_embeddings_cache
        else:
            all_codes = self.accepted_codes + self.rejected_codes
            embeddings = [c.embedding for c in all_codes if c.embedding is not None]
            if embeddings:
                return np.vstack(embeddings)
            return np.array([])

    def find_similar_codes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        accepted_only: bool = True,
        threshold: Optional[float] = None,
    ) -> list[tuple[Code, float]]:
        """
        Find the most similar codes to a query embedding.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top matches to return.
            accepted_only: If True, only search accepted codes.
            threshold: Optional minimum similarity threshold.

        Returns:
            List of (Code, similarity) tuples, sorted by similarity descending.
        """
        codes = self.accepted_codes if accepted_only else (self.accepted_codes + self.rejected_codes)
        codes_with_embeddings = [(c, c.embedding) for c in codes if c.embedding is not None]

        if not codes_with_embeddings:
            return []

        code_list, embeddings_list = zip(*codes_with_embeddings)
        embeddings = np.vstack(embeddings_list)

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarities (assumes normalized embeddings)
        similarities = np.dot(query_embedding, embeddings.T).flatten()

        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [(code_list[i], float(similarities[i])) for i in top_indices]

        # Apply threshold if provided
        if threshold is not None:
            results = [(c, s) for c, s in results if s >= threshold]

        return results

    def check_similarity_against_all(
        self,
        query_embedding: np.ndarray,
    ) -> tuple[float, Optional[Code], bool]:
        """
        Check similarity against all codes (accepted and rejected).

        Used for Stage 1 of novelty evaluation.

        Args:
            query_embedding: The query embedding vector.

        Returns:
            Tuple of (max_similarity, most_similar_code, is_accepted).
            is_accepted indicates whether the most similar code is in accepted group.
        """
        max_sim = -1.0
        most_similar = None
        is_accepted = True

        # Check accepted codes
        for code in self.accepted_codes:
            if code.embedding is not None:
                sim = self._cosine_similarity(query_embedding, code.embedding)
                if sim > max_sim:
                    max_sim = sim
                    most_similar = code
                    is_accepted = True

        # Check rejected codes
        for code in self.rejected_codes:
            if code.embedding is not None:
                sim = self._cosine_similarity(query_embedding, code.embedding)
                if sim > max_sim:
                    max_sim = sim
                    most_similar = code
                    is_accepted = False

        return max_sim, most_similar, is_accepted

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.ndim > 1:
            a = a.flatten()
        if b.ndim > 1:
            b = b.flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def to_dataframe(self, include_rejected: bool = False, sort_by_order: bool = True) -> pd.DataFrame:
        """
        Convert the codebook to a pandas DataFrame.

        Args:
            include_rejected: If True, include rejected codes with a 'status' column.
            sort_by_order: If True, sort by evaluation_order for visualization.

        Returns:
            DataFrame with code information.
        """
        records = []

        for code in self.accepted_codes:
            records.append({
                "evaluation_order": code.evaluation_order,
                "name": code.name,
                "definition": code.definition,
                "theme": code.theme,
                "source_cluster": code.source_cluster,
                "novelty_stage": code.novelty_stage,
                "novelty_reasoning": code.novelty_reasoning,
                "similarity_score": code.similarity_score,
                "similar_to": code.similar_to,
                "created_at": code.created_at,
                "status": "accepted",
            })

        if include_rejected:
            for code in self.rejected_codes:
                records.append({
                    "evaluation_order": code.evaluation_order,
                    "name": code.name,
                    "definition": code.definition,
                    "theme": code.theme,
                    "source_cluster": code.source_cluster,
                    "novelty_stage": code.novelty_stage,
                    "novelty_reasoning": code.novelty_reasoning,
                    "similarity_score": code.similarity_score,
                    "similar_to": code.similar_to,
                    "created_at": code.created_at,
                    "status": "rejected",
                })

        df = pd.DataFrame(records)

        # Sort by evaluation order if requested and the column exists
        if sort_by_order and len(df) > 0 and "evaluation_order" in df.columns:
            df = df.sort_values("evaluation_order", na_position="last").reset_index(drop=True)

        return df

    def to_dict(self) -> dict:
        """Convert the codebook to a dictionary."""
        return {
            "accepted_codes": [c.to_dict() for c in self.accepted_codes],
            "rejected_codes": [c.to_dict() for c in self.rejected_codes],
            "themes": [t.to_dict() for t in self.themes],
            "rejected_themes": [t.to_dict() for t in self.rejected_themes],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Codebook":
        """Create a Codebook from a dictionary."""
        codebook = cls()

        codebook.accepted_codes = [Code.from_dict(c) for c in data.get("accepted_codes", [])]
        codebook.rejected_codes = [Code.from_dict(c) for c in data.get("rejected_codes", [])]
        codebook.themes = [Theme.from_dict(t) for t in data.get("themes", [])]
        codebook.rejected_themes = [Theme.from_dict(t) for t in data.get("rejected_themes", [])]

        created_at = data.get("created_at")
        if created_at:
            codebook.created_at = datetime.fromisoformat(created_at)

        codebook.metadata = data.get("metadata", {})

        return codebook

    def to_json(self, path: str) -> None:
        """Save the codebook to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> "Codebook":
        """Load a codebook from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_csv(self, path: str, include_rejected: bool = False) -> None:
        """Save the codebook to a CSV file."""
        df = self.to_dataframe(include_rejected=include_rejected)
        df.to_csv(path, index=False)

    def __len__(self) -> int:
        """Return the number of accepted codes."""
        return len(self.accepted_codes)

    def __repr__(self) -> str:
        return (
            f"Codebook(accepted={len(self.accepted_codes)}, "
            f"rejected={len(self.rejected_codes)}, themes={len(self.themes)})"
        )
