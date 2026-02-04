"""High-level pipeline orchestration for GATOS workflow."""

import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from pygatos.config import GATOSConfig
from pygatos.core import (
    Embedder,
    Clusterer,
    Codebook,
    Summarizer,
    DimensionalityReducer,
)
from pygatos.llm.base import BaseLLM
from pygatos.cache import CacheManager
from pygatos.generation import CodeSuggester, NoveltyEvaluator, ThemeGenerator

logger = logging.getLogger(__name__)


class GATOSPipeline:
    """
    High-level orchestration of the GATOS workflow.

    The pipeline handles the full codebook generation process:
    1. Load and preprocess data
    2. Summarize long texts (optional)
    3. Embed texts/information points
    4. Reduce dimensionality (optional)
    5. Cluster similar items
    6. Generate codes for each cluster
    7. Evaluate novelty and deduplicate
    8. Generate themes (optional)

    Example:
        >>> pipeline = GATOSPipeline()
        >>> codebook = pipeline.generate_codebook(
        ...     data="responses.csv",
        ...     text_column="response",
        ... )
        >>> print(f"Generated {len(codebook.accepted_codes)} codes")
    """

    def __init__(
        self,
        config: Optional[GATOSConfig] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the GATOS pipeline.

        Args:
            config: Full configuration object. If not provided, uses defaults.
            llm_model: Override LLM model name.
            embedding_model: Override embedding model name.
            cache_dir: Override cache directory.
        """
        self.config = config or GATOSConfig()

        # Apply overrides
        if llm_model:
            self.config.llm.model = llm_model
        if embedding_model:
            self.config.embedding.model_name = embedding_model
        if cache_dir:
            self.config.cache.cache_dir = cache_dir

        # Initialize components lazily
        self._llm = None
        self._embedder = None
        self._cache = None
        self._summarizer = None
        self._reducer = None
        self._code_suggester = None
        self._novelty_evaluator = None
        self._theme_generator = None

    @property
    def llm(self) -> BaseLLM:
        """Get or create LLM backend based on config."""
        if self._llm is None:
            self._llm = self.config.llm.create_backend()
            if not self._llm.is_available():
                backend = self.config.llm.backend
                if backend == "ollama":
                    raise RuntimeError(
                        f"LLM not available at {self.config.llm.base_url}. "
                        "Make sure Ollama is running: ollama serve"
                    )
                elif backend == "cerebras":
                    raise RuntimeError(
                        "Cerebras API not available. "
                        "Check your CEREBRAS_API_KEY environment variable."
                    )
                else:
                    raise RuntimeError(f"LLM backend '{backend}' not available.")
        return self._llm

    @property
    def embedder(self) -> Embedder:
        """Get or create embedder."""
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self.config.embedding.model_name}")
            self._embedder = Embedder.from_config(self.config.embedding)
        return self._embedder

    @property
    def cache(self) -> CacheManager:
        """Get or create cache manager."""
        if self._cache is None:
            self._cache = CacheManager.from_config(self.config.cache)
        return self._cache

    @property
    def summarizer(self) -> Summarizer:
        """Get or create summarizer."""
        if self._summarizer is None:
            self._summarizer = Summarizer.from_config(
                self.config.summarization,
                llm=self.llm,
                study_context=self.config.study_context,
            )
        return self._summarizer

    @property
    def reducer(self) -> DimensionalityReducer:
        """Get or create dimensionality reducer."""
        if self._reducer is None:
            self._reducer = DimensionalityReducer.from_config(
                self.config.dim_reduction,
                random_state=self.config.random_seed,
            )
        return self._reducer

    @property
    def code_suggester(self) -> CodeSuggester:
        """Get or create code suggester."""
        if self._code_suggester is None:
            self._code_suggester = CodeSuggester(
                llm=self.llm,
                study_context=self.config.study_context,
                system_prompt=self.config.code_suggestion_system_prompt,
                user_prompt=self.config.code_suggestion_user_prompt,
            )
        return self._code_suggester

    @property
    def novelty_evaluator(self) -> NoveltyEvaluator:
        """Get or create novelty evaluator."""
        if self._novelty_evaluator is None:
            self._novelty_evaluator = NoveltyEvaluator(
                llm=self.llm,
                embedder=self.embedder,
                similarity_threshold=self.config.novelty.similarity_threshold,
                top_k_rag=self.config.novelty.top_k_rag,
                prompt_version=2,  # Use v2 (stricter) by default
                study_context=self.config.study_context,
                include_rejected_in_rag=self.config.novelty.include_rejected_in_rag,
                system_prompt=self.config.novelty_evaluation_system_prompt,
                user_prompt=self.config.novelty_evaluation_user_prompt,
            )
        return self._novelty_evaluator

    @property
    def theme_generator(self) -> ThemeGenerator:
        """Get or create theme generator."""
        if self._theme_generator is None:
            self._theme_generator = ThemeGenerator(
                llm=self.llm,
                embedder=self.embedder,
                novelty_evaluator=self.novelty_evaluator,
                study_context=self.config.study_context,
            )
        return self._theme_generator

    def generate_codebook(
        self,
        data: Union[str, Path, pd.DataFrame],
        text_column: str,
        id_column: Optional[str] = None,
        skip_summarization: bool = False,
        skip_dim_reduction: bool = False,
        generate_themes: bool = True,
        max_clusters: Optional[int] = None,
        verbose: bool = True,
    ) -> Codebook:
        """
        Generate a codebook from text data.

        This is the main entry point for the GATOS workflow.

        Args:
            data: Path to CSV file or DataFrame with text data.
            text_column: Name of column containing text to analyze.
            id_column: Optional name of ID column.
            skip_summarization: If True, treat texts as atomic (no chunking).
                Use this for short texts like questions or survey responses.
            skip_dim_reduction: If True, skip PCA+UMAP before clustering.
                Use this for smaller datasets or when clustering on raw embeddings is preferred.
            generate_themes: If True, generate themes from accepted codes.
            max_clusters: Maximum number of clusters to process (for testing).
            verbose: If True, log progress information.

        Returns:
            Codebook with accepted codes (and themes if generated).
        """
        start_time = datetime.now()

        if verbose:
            logger.info("=" * 60)
            logger.info("GATOS Codebook Generation Pipeline")
            logger.info("=" * 60)

        # Step 1: Load data
        if verbose:
            logger.info("\n[Step 1] Loading data...")

        df = self._load_data(data)
        texts = df[text_column].astype(str).tolist()
        ids = df[id_column].tolist() if id_column else list(range(len(texts)))

        if verbose:
            logger.info(f"  Loaded {len(texts)} texts")

        # Step 2: Summarization (optional)
        if not skip_summarization:
            if verbose:
                logger.info("\n[Step 2] Extracting information points...")

            all_points = []
            point_to_source = {}

            for i, text in enumerate(texts):
                result = self.summarizer.summarize(text, verbose=False)

                for point in result.information_points:
                    point_to_source[len(all_points)] = ids[i]
                    all_points.append(point)

            if verbose:
                logger.info(f"  Extracted {len(all_points)} information points")

            texts_to_embed = all_points
        else:
            if verbose:
                logger.info("\n[Step 2] Skipping summarization (atomic texts)")

            texts_to_embed = texts
            point_to_source = {i: ids[i] for i in range(len(texts))}

        # Step 3: Embedding
        if verbose:
            logger.info("\n[Step 3] Embedding texts...")

        # Check cache
        cached = self.cache.get_embedding(
            texts_to_embed,
            self.config.embedding.model_name,
        )

        if cached is not None:
            embeddings = cached
            if verbose:
                logger.info(f"  Loaded from cache: {embeddings.shape}")
        else:
            embeddings = self.embedder.embed(texts_to_embed, show_progress=verbose)
            self.cache.set_embedding(
                texts_to_embed,
                embeddings,
                self.config.embedding.model_name,
            )
            if verbose:
                logger.info(f"  Generated embeddings: {embeddings.shape}")

        # Step 4: Dimensionality reduction (optional)
        if not skip_dim_reduction and len(texts_to_embed) > 50:
            if verbose:
                logger.info("\n[Step 4] Reducing dimensionality...")

            reduced_embeddings = self.reducer.fit_transform(embeddings)

            if verbose:
                logger.info(f"  Reduced: {embeddings.shape[1]} -> {reduced_embeddings.shape[1]} dims")

            cluster_embeddings = reduced_embeddings
        else:
            if verbose:
                logger.info("\n[Step 4] Skipping dimensionality reduction")

            cluster_embeddings = embeddings

        # Step 5: Clustering
        if verbose:
            logger.info("\n[Step 5] Clustering texts...")

        n_clusters = self.config.clustering.compute_n_clusters(len(texts_to_embed))
        self.config.clustering.n_clusters = n_clusters

        clusterer = Clusterer.from_config(self.config.clustering)
        labels = clusterer.fit(cluster_embeddings)
        clusters = clusterer.get_cluster_contents(labels, texts_to_embed)

        if max_clusters:
            # Limit clusters for testing
            clusters = dict(list(clusters.items())[:max_clusters])

        if verbose:
            stats = clusterer.get_cluster_stats(labels)
            logger.info(f"  Created {stats['n_clusters']} clusters")
            logger.info(f"  Cluster size range: {stats['min_cluster_size']}-{stats['max_cluster_size']}")

        # Step 6: Code generation
        if verbose:
            logger.info("\n[Step 6] Generating codes...")

        codebook = Codebook()
        all_suggested = []

        # Filter out noise cluster
        valid_clusters = {k: v for k, v in clusters.items() if k != -1}

        cluster_iter = valid_clusters.items()
        if verbose:
            cluster_iter = tqdm(
                cluster_iter,
                desc="  Generating codes",
                total=len(valid_clusters),
                unit="cluster",
            )

        for cluster_id, cluster_texts in cluster_iter:
            codes = self.code_suggester.suggest_codes(
                cluster_texts=cluster_texts,
                cluster_id=cluster_id,
                verbose=False,
            )

            all_suggested.extend(codes)

        if verbose:
            logger.info(f"  Suggested {len(all_suggested)} codes from {len(valid_clusters)} clusters")

        # Step 7: Novelty evaluation
        if verbose:
            logger.info("\n[Step 7] Evaluating novelty...")

        accepted = 0
        rejected = 0

        code_iter = all_suggested
        if verbose:
            code_iter = tqdm(
                all_suggested,
                desc="  Evaluating codes",
                unit="code",
            )

        for code in code_iter:
            result = self.novelty_evaluator.evaluate(code, codebook, verbose=False)

            if result.is_novel:
                codebook.add_code(code, accepted=True)
                accepted += 1
            else:
                codebook.add_code(code, accepted=False)
                rejected += 1

        if verbose:
            logger.info(f"  Accepted: {accepted}, Rejected: {rejected}")
            dedup_rate = rejected / len(all_suggested) * 100 if all_suggested else 0
            logger.info(f"  Deduplication rate: {dedup_rate:.1f}%")

        # Step 8: Theme generation (optional)
        if generate_themes and len(codebook.accepted_codes) >= 2:
            if verbose:
                logger.info("\n[Step 8] Generating themes...")

            themes = self.theme_generator.generate_themes(codebook, verbose=False)

            if verbose:
                logger.info(f"  Generated {len(themes)} themes")
        else:
            if verbose:
                logger.info("\n[Step 8] Skipping theme generation")

        # Summary
        duration = (datetime.now() - start_time).total_seconds()

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Duration: {duration:.1f} seconds")
            logger.info(f"  Texts processed: {len(texts)}")
            logger.info(f"  Codes accepted: {len(codebook.accepted_codes)}")
            logger.info(f"  Themes: {len(codebook.themes)}")

        return codebook

    def apply_codebook(
        self,
        data: Union[str, Path, pd.DataFrame],
        codebook: Codebook,
        text_column: str,
        id_column: Optional[str] = None,
        use_extraction: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Apply a codebook to label data.

        Args:
            data: Path to CSV or DataFrame.
            codebook: Codebook to apply.
            text_column: Column containing text to code.
            id_column: Optional ID column.
            use_extraction: If True, extract information points and code each one.
                          If False, code the full text directly.
            verbose: If True, log progress.

        Returns:
            DataFrame with applied codes.
        """
        # Import here to avoid circular dependency
        from pygatos.application import CodeApplier

        df = self._load_data(data)
        texts = df[text_column].astype(str).tolist()
        ids = df[id_column].tolist() if id_column else list(range(len(texts)))

        applier = CodeApplier(
            llm=self.llm,
            embedder=self.embedder,
            top_k=self.config.application.top_k,
            study_context=self.config.study_context,
        )

        if use_extraction:
            results = applier.apply_with_extraction_batch(
                texts, codebook, self.summarizer, verbose=verbose
            )
            applied_codes_list = [r.applied_codes for r in results]
        else:
            applied_codes_list = applier.apply_batch(texts, codebook, verbose=verbose)

        # Build result DataFrame
        rows = []
        for i, applied_codes in enumerate(applied_codes_list):
            row = {
                "id": ids[i],
                "text": texts[i],
                "codes": [c.name for c in applied_codes],
                "n_codes": len(applied_codes),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def apply_codebook_with_details(
        self,
        data: Union[str, Path, pd.DataFrame],
        codebook: Codebook,
        text_column: str,
        id_column: Optional[str] = None,
        use_extraction: bool = True,
        extraction_results: Optional[dict] = None,
        include_source_context: bool = False,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, list]:
        """
        Apply a codebook to label data, returning full details.

        Returns both a simple DataFrame and full ApplicationResult objects
        for audit/transparency purposes.

        Args:
            data: Path to CSV or DataFrame.
            codebook: Codebook to apply.
            text_column: Column containing text to code.
            id_column: Optional ID column.
            use_extraction: If True, use information points for coding.
                          If False, code the full text directly.
            extraction_results: Optional dict mapping ID to SummarizationResult.
                              If provided with use_extraction=True, these pre-extracted
                              points will be reused instead of re-extracting.
                              This avoids redundant LLM calls and ensures consistency.
            include_source_context: If True, include chunk/source text as context
                                   when coding information points (use_extraction only).
            verbose: If True, log progress.

        Returns:
            Tuple of (DataFrame with applied codes, list of ApplicationResult objects).

        Example:
            # Reuse extraction results from codebook generation
            >>> extraction_results = {i: result for i, result in enumerate(summarization_results)}
            >>> df, results = pipeline.apply_codebook_with_details(
            ...     data=my_df,
            ...     codebook=codebook,
            ...     text_column="text",
            ...     extraction_results=extraction_results,
            ... )
        """
        # Import here to avoid circular dependency
        from pygatos.application import CodeApplier

        df = self._load_data(data)
        texts = df[text_column].astype(str).tolist()
        ids = df[id_column].tolist() if id_column else list(range(len(texts)))

        applier = CodeApplier(
            llm=self.llm,
            embedder=self.embedder,
            top_k=self.config.application.top_k,
            study_context=self.config.study_context,
        )

        if use_extraction:
            if extraction_results is not None:
                # Use pre-extracted points (preferred - no redundant extraction)
                if verbose:
                    logger.info("Using pre-extracted information points for code application")
                results = applier.apply_to_points_batch(
                    extraction_results=extraction_results,
                    codebook=codebook,
                    include_source_context=include_source_context,
                    verbose=verbose,
                )
            else:
                # Fall back to re-extraction (deprecated behavior)
                if verbose:
                    logger.warning(
                        "No extraction_results provided. Falling back to re-extraction. "
                        "This is deprecated - pass extraction_results for better efficiency."
                    )
                results = applier.apply_with_extraction_batch(
                    texts, codebook, self.summarizer, verbose=verbose
                )
        else:
            # Use direct text application (legacy behavior)
            results = applier.apply_with_details(texts, codebook, verbose=verbose)

        # Build result DataFrame
        rows = []
        for i, result in enumerate(results):
            row = {
                "id": ids[i],
                "text": texts[i],
                "codes": [c.name for c in result.applied_codes],
                "n_codes": len(result.applied_codes),
            }
            rows.append(row)

        return pd.DataFrame(rows), results

    def _load_data(self, data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from path or return DataFrame directly."""
        if isinstance(data, pd.DataFrame):
            return data

        path = Path(data)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif path.suffix.lower() == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def save_codebook(
        self,
        codebook: Codebook,
        output_dir: Union[str, Path],
        prefix: str = "codebook",
    ) -> dict[str, Path]:
        """
        Save codebook to multiple formats.

        Args:
            codebook: Codebook to save.
            output_dir: Output directory.
            prefix: Filename prefix.

        Returns:
            Dict mapping format to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # JSON
        json_path = output_dir / f"{prefix}.json"
        codebook.to_json(json_path)
        paths["json"] = json_path

        # CSV
        csv_path = output_dir / f"{prefix}.csv"
        codebook.to_csv(csv_path, include_rejected=True)
        paths["csv"] = csv_path

        return paths

    def __repr__(self) -> str:
        return (
            f"GATOSPipeline(llm={self.config.llm.model}, "
            f"embedding={self.config.embedding.model_name})"
        )
