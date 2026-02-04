"""Theme generation from accepted codes."""

import logging
from typing import Optional

import numpy as np

from pygatos.llm.base import BaseLLM
from pygatos.core.codebook import Code, Theme, Codebook
from pygatos.core.embedder import Embedder
from pygatos.core.clusterer import Clusterer
from pygatos.generation.novelty_evaluator import NoveltyEvaluator
from pygatos.prompts import (
    THEME_SUGGESTION_SYSTEM,
    THEME_SUGGESTION_PROMPT,
    format_codes_for_prompt,
    add_study_context,
)

logger = logging.getLogger(__name__)


class ThemeGenerator:
    """
    Generates themes by clustering and consolidating accepted codes.

    The theme generation process mirrors the code generation process:
    1. Embed all accepted codes
    2. Cluster similar codes together
    3. For each cluster, suggest a theme
    4. Run novelty evaluation on themes to remove redundancy

    Example:
        >>> generator = ThemeGenerator(llm, embedder, clusterer, novelty_evaluator)
        >>> themes = generator.generate_themes(codebook)
        >>> for theme in themes:
        ...     print(f"{theme.name}: {len(theme.codes)} codes")
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Embedder,
        clusterer: Optional[Clusterer] = None,
        novelty_evaluator: Optional[NoveltyEvaluator] = None,
        n_clusters: Optional[int] = None,
        temperature: Optional[float] = None,
        study_context: Optional[str] = None,
    ):
        """
        Initialize the theme generator.

        Args:
            llm: LLM backend for theme suggestion.
            embedder: Embedder for code embeddings.
            clusterer: Optional clusterer (creates default if not provided).
            novelty_evaluator: Optional novelty evaluator for theme deduplication.
            n_clusters: Optional number of clusters (auto-determined if None).
            temperature: Optional temperature override for LLM.
            study_context: Optional context about the study/dataset to improve suggestions.
        """
        self.llm = llm
        self.embedder = embedder
        self.clusterer = clusterer or Clusterer(method="agglomerative")
        self.novelty_evaluator = novelty_evaluator
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.study_context = study_context

    def generate_themes(
        self,
        codebook: Codebook,
        verbose: bool = False,
    ) -> list[Theme]:
        """
        Generate themes from accepted codes in the codebook.

        Args:
            codebook: Codebook with accepted codes.
            verbose: If True, log detailed information.

        Returns:
            List of Theme objects.
        """
        accepted_codes = codebook.accepted_codes

        if len(accepted_codes) < 2:
            logger.warning("Not enough codes to generate themes (need at least 2)")
            if accepted_codes:
                # Create a single theme with all codes
                theme = Theme(
                    name="General",
                    definition="General theme containing all codes",
                    codes=accepted_codes.copy(),
                )
                return [theme]
            return []

        if verbose:
            logger.info(f"Generating themes from {len(accepted_codes)} accepted codes")

        # Step 1: Ensure all codes have embeddings
        self._embed_codes(accepted_codes)

        # Step 2: Get embeddings matrix
        embeddings = np.vstack([c.embedding for c in accepted_codes])

        # Step 3: Cluster codes
        n_clusters = self.n_clusters or self._estimate_n_clusters(len(accepted_codes))

        if verbose:
            logger.info(f"Clustering codes into {n_clusters} clusters")

        self.clusterer.n_clusters = n_clusters
        labels = self.clusterer.fit(embeddings)
        code_clusters = self._group_codes_by_cluster(accepted_codes, labels)

        if verbose:
            stats = self.clusterer.get_cluster_stats(labels)
            logger.info(f"Created {stats['n_clusters']} clusters")

        # Step 4: Suggest themes for each cluster
        suggested_themes = []
        for cluster_id, codes in code_clusters.items():
            if cluster_id == -1:  # Skip noise cluster
                continue

            theme = self._suggest_theme_for_cluster(codes, cluster_id, verbose)
            if theme:
                suggested_themes.append(theme)

        if verbose:
            logger.info(f"Suggested {len(suggested_themes)} themes")

        # Step 5: Optional novelty evaluation on themes
        if self.novelty_evaluator and len(suggested_themes) > 1:
            themes = self._deduplicate_themes(suggested_themes, codebook, verbose)
        else:
            # Mark all as accepted with first_theme stage
            for i, theme in enumerate(suggested_themes):
                theme.evaluation_order = i + 1
                if i == 0:
                    theme.novelty_stage = "first_theme"
                else:
                    theme.novelty_stage = "no_dedup"
            themes = suggested_themes

        # Step 6: Update codebook
        for theme in themes:
            codebook.add_theme(theme)

        if verbose:
            logger.info(f"Final theme count: {len(themes)}")
            for theme in themes:
                logger.info(f"  - {theme.name}: {len(theme.codes)} codes")

        return themes

    def _embed_codes(self, codes: list[Code]) -> None:
        """Ensure all codes have embeddings."""
        texts_to_embed = []
        codes_to_embed = []

        for code in codes:
            if code.embedding is None:
                texts_to_embed.append(f"{code.name}: {code.definition}")
                codes_to_embed.append(code)

        if texts_to_embed:
            embeddings = self.embedder.embed(texts_to_embed)
            for i, code in enumerate(codes_to_embed):
                code.embedding = embeddings[i]

    def _estimate_n_clusters(self, n_codes: int) -> int:
        """Estimate number of theme clusters based on code count."""
        # Rough heuristic: sqrt(n) to n/3 themes
        import math
        min_clusters = max(2, int(math.sqrt(n_codes)))
        max_clusters = max(3, n_codes // 3)
        return min(min_clusters + 2, max_clusters)

    def _group_codes_by_cluster(
        self,
        codes: list[Code],
        labels: np.ndarray,
    ) -> dict[int, list[Code]]:
        """Group codes by their cluster labels."""
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(codes[i])
        return clusters

    def _suggest_theme_for_cluster(
        self,
        codes: list[Code],
        cluster_id: int,
        verbose: bool = False,
    ) -> Optional[Theme]:
        """Suggest a theme for a cluster of codes."""
        if not codes:
            return None

        # Format codes for prompt
        codes_text = format_codes_for_prompt(codes, include_definition=True)

        prompt = THEME_SUGGESTION_PROMPT.format(codes=codes_text)

        # Add study context to system prompt if available
        system = add_study_context(THEME_SUGGESTION_SYSTEM, self.study_context)

        if verbose:
            logger.info(f"  Suggesting theme for cluster {cluster_id} ({len(codes)} codes)")

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system=system,
                temperature=self.temperature,
            )

            name = response.get("name", f"Theme {cluster_id}").strip()
            definition = response.get("definition", "").strip()

            if not definition:
                definition = f"Theme containing codes related to {name}"

            theme = Theme(
                name=name,
                definition=definition,
                codes=codes.copy(),
            )

            # Set theme reference in codes
            for code in codes:
                code.theme = name

            if verbose:
                logger.info(f"    Theme: {name}")

            return theme

        except Exception as e:
            logger.error(f"  Failed to suggest theme for cluster {cluster_id}: {e}")
            # Create a fallback theme
            return Theme(
                name=f"Theme {cluster_id}",
                definition=f"Auto-generated theme for cluster {cluster_id}",
                codes=codes.copy(),
            )

    def _deduplicate_themes(
        self,
        themes: list[Theme],
        codebook: Codebook,
        verbose: bool = False,
    ) -> list[Theme]:
        """
        Remove redundant themes using novelty evaluation.

        This treats themes like codes and applies the same novelty process.
        Stores rejected themes with evaluation metadata in the codebook.
        """
        if verbose:
            logger.info("Deduplicating themes...")

        # Create a temporary codebook for theme novelty evaluation
        theme_codebook = Codebook()

        # Convert themes to codes for novelty evaluation
        accepted_themes = []
        eval_order = 1

        for theme in themes:
            theme.evaluation_order = eval_order
            eval_order += 1

            # Create a pseudo-code for the theme
            theme_code = Code(
                name=theme.name,
                definition=theme.definition,
            )

            # Embed the theme
            theme_text = f"{theme.name}: {theme.definition}"
            theme_code.embedding = self.embedder.embed(theme_text)
            theme.embedding = theme_code.embedding

            # Run novelty evaluation
            result = self.novelty_evaluator.evaluate(
                theme_code,
                theme_codebook,
                verbose=verbose,
            )

            # Store novelty metadata on the theme
            theme.novelty_stage = result.stage
            theme.novelty_reasoning = result.reasoning
            theme.similarity_score = result.max_similarity
            if result.most_similar_code:
                theme.similar_to = result.most_similar_code.name

            if result.is_novel:
                theme_codebook.add_code(theme_code, accepted=True)
                accepted_themes.append(theme)

                if verbose:
                    logger.info(f"    Accepted theme: {theme.name}")
            else:
                # Store rejected theme in the main codebook
                codebook.add_theme(theme, accepted=False)

                # Merge codes into the most similar accepted theme
                if result.most_similar_code:
                    similar_theme_name = result.most_similar_code.name
                    for accepted in accepted_themes:
                        if accepted.name == similar_theme_name:
                            accepted.codes.extend(theme.codes)
                            for code in theme.codes:
                                code.theme = accepted.name
                            if verbose:
                                logger.info(f"    Merged '{theme.name}' into '{similar_theme_name}'")
                            break

                if verbose and not result.most_similar_code:
                    logger.info(f"    Rejected theme: {theme.name}")

        return accepted_themes

    def __repr__(self) -> str:
        return f"ThemeGenerator(n_clusters={self.n_clusters})"
