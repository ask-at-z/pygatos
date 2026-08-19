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
from pygatos import provenance as _prov

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

        # Populated by generate_codebook(collect_provenance=True); see save_provenance().
        self.provenance: Optional[dict] = None

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
                temperature=self.config.novelty.temperature,
                prompt_version=2,  # Use v2 (stricter) by default
                study_context=self.config.study_context,
                include_rejected_in_rag=self.config.novelty.include_rejected_in_rag,
                system_prompt=self.config.novelty_evaluation_system_prompt,
                user_prompt=self.config.novelty_evaluation_user_prompt,
                stage1_compare_accepted_only=self.config.novelty.stage1_compare_accepted_only,
                policy=self.config.novelty.policy,
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
        collect_provenance: bool = False,
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
            collect_provenance: If True, retain the full extraction->cluster->code chain on
                ``self.provenance`` and stamp each code's metadata with the source documents and
                information points that produced it. Serialize with ``save_provenance()``. Off by
                default (unchanged behavior + no extra memory for callers that don't need audit trails).

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
            extraction_by_id = {}          # essay_id -> SummarizationResult (provenance)
            point_structured = {}          # global point idx -> InformationPoint (lineage)

            for i, text in enumerate(texts):
                with _prov.stage("summarization", item=ids[i]):
                    result = self.summarizer.summarize(text, verbose=False)
                if collect_provenance:
                    extraction_by_id[ids[i]] = result
                structured = result.structured_points or []
                for j, point in enumerate(result.information_points):
                    idx = len(all_points)
                    point_to_source[idx] = ids[i]
                    if collect_provenance and j < len(structured):
                        point_structured[idx] = structured[j]
                    all_points.append(point)

            if verbose:
                logger.info(f"  Extracted {len(all_points)} information points")

            texts_to_embed = all_points
        else:
            if verbose:
                logger.info("\n[Step 2] Skipping summarization (atomic texts)")

            texts_to_embed = texts
            point_to_source = {i: ids[i] for i in range(len(texts))}
            all_points = texts
            extraction_by_id = {}
            point_structured = {}

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
        cluster_outcomes = []   # provenance: per-cluster code-generation outcome

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
            with _prov.stage("code_suggestion", item=cluster_id):
                codes = self.code_suggester.suggest_codes(
                    cluster_texts=cluster_texts,
                    cluster_id=cluster_id,
                    verbose=False,
                )

            cluster_outcomes.append({
                "cluster_id": int(cluster_id),
                "n_points": len(cluster_texts),
                "n_codes_returned": len(codes),
                "code_names": [c.name for c in codes],
                "status": "ok" if codes else "no_codes_returned",
            })
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
            with _prov.stage("novelty_evaluation", item=code.name):
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

        # Step 7b: Provenance — link every code back to its source points and documents.
        if collect_provenance:
            labels_list = [int(x) for x in labels]
            cluster_points: dict[int, list] = {}
            for idx, lab in enumerate(labels_list):
                cluster_points.setdefault(lab, []).append(idx)
            for code in list(codebook.accepted_codes) + list(codebook.rejected_codes):
                cid = code.source_cluster
                pt_idx = cluster_points.get(cid, []) if cid is not None else []
                essays = sorted({point_to_source.get(i) for i in pt_idx
                                 if point_to_source.get(i) is not None}, key=str)
                if not code.metadata:
                    code.metadata = {}
                code.metadata["source_essays"] = essays
                code.metadata["source_point_indices"] = pt_idx
                code.metadata["n_source_points"] = len(pt_idx)
            n_noise = sum(1 for l in labels_list if l == -1)
            self.provenance = {
                "extraction_by_id": extraction_by_id,      # essay_id -> SummarizationResult
                "point_to_source": point_to_source,        # global point idx -> essay_id
                "point_structured": point_structured,      # global point idx -> InformationPoint
                "all_points": all_points,                  # global point idx -> text
                "labels": labels_list,                     # global point idx -> cluster_id
                "ids": ids,
                # Numerical checkpoints. Re-embedding is NOT bit-identical across processes or
                # devices, so the exact partition cannot be reproduced without these arrays.
                "embeddings": embeddings,
                "cluster_embeddings": cluster_embeddings,
                "pipeline_state": {
                    "n_information_points": len(all_points),
                    "embedding_model": self.config.embedding.model_name,
                    "embedding_device": getattr(self.config.embedding, "device", None),
                    "embedding_dim": int(embeddings.shape[1]) if hasattr(embeddings, "shape") else None,
                    "dim_reduction_applied": bool(
                        hasattr(cluster_embeddings, "shape") and hasattr(embeddings, "shape")
                        and cluster_embeddings.shape[1] != embeddings.shape[1]),
                    "reduced_dim": int(cluster_embeddings.shape[1]) if hasattr(cluster_embeddings, "shape") else None,
                    "skip_dim_reduction_arg": skip_dim_reduction,
                    "clustering": {
                        "method": getattr(self.config.clustering, "method", None),
                        "n_clusters_resolved": int(n_clusters) if n_clusters is not None else None,
                        "n_clusters_found": len(set(labels_list) - {-1}),
                        "n_noise_points": n_noise,
                        "max_clusters_arg": max_clusters,
                    },
                    "clusters": cluster_outcomes,
                    "n_clusters_yielding_no_codes": sum(
                        1 for c in cluster_outcomes if c["status"] != "ok"),
                    "random_seed": self.config.random_seed,
                    "llm_seed": getattr(self.config.llm, "seed", None),
                    "consolidation_settings": _prov.consolidation_settings(self),
                },
            }
            if verbose:
                logger.info(f"  Provenance captured for {len(extraction_by_id)} documents, "
                            f"{len(all_points)} information points")

        # Step 8: Theme generation (optional)
        if generate_themes and len(codebook.accepted_codes) >= 2:
            if verbose:
                logger.info("\n[Step 8] Generating themes...")

            with _prov.stage("theme_generation"):
                themes = self.theme_generator.generate_themes(codebook, verbose=False)

            if verbose:
                logger.info(f"  Generated {len(themes)} themes")

            # Step 8b: optional per-code best-theme (re)assignment + validation.
            if self.config.theme.refine_after_generation:
                if verbose:
                    logger.info("[Step 8b] Refining code -> theme assignments...")
                report = self.reassign_code_themes(codebook, verbose=False)
                if verbose:
                    logger.info(f"  Reassigned {report['n_moves']}/{report['n_codes']} codes "
                                f"({report['n_llm_calls']} LLM calls)")
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

    def save_provenance(self, codebook: Codebook, output_dir: Union[str, Path]) -> dict[str, Path]:
        """
        Serialize the generation provenance (the document -> window -> summary -> point -> cluster ->
        code chain) captured when generate_codebook(collect_provenance=True) was run.

        Writes two files into ``output_dir``:

        - ``extraction.json`` — per document: chunks (windows), per-window generic summaries, and the
          structured information points with full lineage (text, chunk index, chunk text). Rehydrate
          with ``load_extraction_results()`` to reuse these exact points during code application.
        - ``code_grounding.json`` — per code (accepted and rejected): the source cluster, the source
          documents, and the actual information-point texts that induced it — the audit trail linking
          every code back to the evidence it came from.

        NOTE: both files contain text derived from the source documents and must be handled with the
        same confidentiality as the source corpus (do not commit / egress for sensitive data).

        Args:
            codebook: The codebook returned by the matching generate_codebook(collect_provenance=True).
            output_dir: Directory to write the two JSON files into.

        Returns a dict mapping name -> written path. Raises if no provenance was collected.
        """
        import json
        if not self.provenance:
            raise RuntimeError(
                "No provenance available. Run generate_codebook(collect_provenance=True) first."
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prov = self.provenance

        # 1) extraction.json — per-document windows/summaries/points (rehydratable)
        extraction = {
            str(essay_id): result.to_dict(include_original_text=False)
            for essay_id, result in prov["extraction_by_id"].items()
        }
        extraction_path = output_dir / "extraction.json"
        extraction_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False))

        # 2) code_grounding.json — code -> source documents + the information points that induced it
        all_points = prov["all_points"]
        point_to_source = prov["point_to_source"]
        point_structured = prov.get("point_structured", {})

        def _points_for(indices: list) -> list:
            out = []
            for i in indices:
                sp = point_structured.get(i)
                out.append({
                    "point_index": i,
                    "text": all_points[i] if i < len(all_points) else None,
                    "essay_id": point_to_source.get(i),
                    "chunk_index": getattr(sp, "chunk_index", None) if sp else None,
                    "chunk_text": getattr(sp, "chunk_text", None) if sp else None,
                })
            return out

        grounding = []
        for accepted, code in ([(True, c) for c in codebook.accepted_codes] +
                               [(False, c) for c in codebook.rejected_codes]):
            md = code.metadata or {}
            grounding.append({
                "code": code.name,
                "accepted": accepted,
                "theme": getattr(code, "theme", None),
                "source_cluster": code.source_cluster,
                "n_source_points": md.get("n_source_points", 0),
                "source_essays": md.get("source_essays", []),
                "source_information_points": _points_for(md.get("source_point_indices", [])),
            })
        grounding_path = output_dir / "code_grounding.json"
        grounding_path.write_text(json.dumps(grounding, indent=2, ensure_ascii=False))

        written = {"extraction": extraction_path, "code_grounding": grounding_path}

        # 3) pipeline_state.json — the reduction/clustering parameters actually used plus the
        #    per-cluster code-generation outcome (clusters that yielded no codes are otherwise
        #    invisible in every artifact).
        state = prov.get("pipeline_state")
        if state is not None:
            state_path = output_dir / "pipeline_state.json"
            state_path.write_text(json.dumps(state, indent=2, default=str))
            written["pipeline_state"] = state_path

        # 4) embeddings.npz — the vectors and the reduced matrix actually handed to the clusterer,
        #    with labels and the point->document index. Re-embedding is NOT bit-identical across
        #    processes/devices, so the recorded partition cannot be reproduced from text alone.
        emb = prov.get("embeddings")
        if emb is not None:
            import hashlib as _hashlib
            import numpy as _np
            red = prov.get("cluster_embeddings")
            n = len(all_points)
            emb_path = output_dir / "embeddings.npz"
            _np.savez_compressed(
                emb_path,
                vectors=_np.asarray(emb, dtype=_np.float32),
                reduced=(_np.asarray(red, dtype=_np.float32) if red is not None
                         else _np.empty((0, 0), dtype=_np.float32)),
                labels=_np.asarray(prov.get("labels", []), dtype=_np.int32),
                point_index=_np.arange(n, dtype=_np.int32),
                essay_id=_np.array([str(point_to_source.get(i, "")) for i in range(n)]),
                point_sha256=_np.array([
                    _hashlib.sha256(str(all_points[i]).encode("utf-8")).hexdigest()
                    for i in range(n)
                ]),
            )
            written["embeddings"] = emb_path

        return written

    @staticmethod
    def load_extraction_results(path: Union[str, Path]) -> dict:
        """
        Load ``extraction.json`` (written by save_provenance) back into a dict of
        ``{essay_id: SummarizationResult}`` with structured_points rehydrated, suitable for
        ``apply_codebook_with_details(extraction_results=...)`` to reuse the exact information points
        from generation (no re-extraction).
        """
        import json
        from pygatos.core.summarizer import SummarizationResult
        data = json.loads(Path(path).read_text())
        return {essay_id: SummarizationResult.from_dict(d) for essay_id, d in data.items()}

    def reassign_code_themes(
        self,
        codebook: Codebook,
        verbose: bool = True,
    ) -> dict:
        """
        (Re)assign each accepted code to its best-fitting theme (per-code assignment +
        LLM validation), using the settings in ``config.theme``.

        Runs on a codebook that already has themes (e.g. one just generated, or one loaded
        from disk). Mutates codes' ``theme`` and rebuilds theme membership in place.

        Args:
            codebook: Codebook with accepted codes and themes.
            verbose: If True, log each reassignment.

        Returns:
            Report dict from ThemeGenerator.assign_codes_to_themes (moves, sizes, counts).
        """
        tc = self.config.theme
        return self.theme_generator.assign_codes_to_themes(
            codebook,
            top_k=tc.assignment_top_k,
            confidence_threshold=tc.assignment_confidence_threshold,
            validate=tc.validate_assignments,
            verbose=verbose,
        )

    @staticmethod
    def save_application_provenance(
        results: list,
        ids: list,
        output_dir: Union[str, Path],
        include_text: bool = True,
    ) -> dict[str, Path]:
        """
        Persist the per-information-point application record and the document x code incidence.

        ``apply_codebook`` returns only marginal counts, but margins do not determine the joint
        distribution: without the incidence matrix there is no co-occurrence, no per-document unit
        of analysis, no confidence interval, and no way to trace a prevalence figure back to the
        excerpt that produced it. Judgment failures are also recorded separately — an empty
        applied-code list from a failed call is otherwise indistinguishable from "no codes apply"
        and silently deflates every count.

        Writes:
        - ``prevalence_assignments.csv`` — long-form document x code incidence (ids + code names
          only, no text; safe to share alongside a paper).
        - ``application_points.csv`` — one row per information point: candidates considered, codes
          applied, and judgment status (no text).
        - ``application_points_restricted.jsonl`` — the same rows plus point text, interpretation
          and analysis. SENSITIVE: inherits the corpus's confidentiality. Omitted when
          ``include_text=False``.

        Args:
            results: ApplicationResult objects (from ``apply_codebook_with_details``).
            ids: Document ids aligned with ``results``.
            output_dir: Destination directory.
            include_text: If False, skip the restricted text-bearing file.
        """
        import csv
        import json as _json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        inc_path = output_dir / "prevalence_assignments.csv"
        pts_path = output_dir / "application_points.csv"
        restricted_path = output_dir / "application_points_restricted.jsonl"

        n_points = n_failed = 0
        with inc_path.open("w", newline="") as f_inc, pts_path.open("w", newline="") as f_pts:
            w_inc = csv.writer(f_inc)
            w_inc.writerow(["document_id", "code", "theme"])
            w_pts = csv.writer(f_pts)
            w_pts.writerow(["document_id", "point_index", "n_candidates", "candidate_codes",
                            "n_applied", "applied_codes", "judgment_failed"])
            f_res = restricted_path.open("w", encoding="utf-8") if include_text else None
            try:
                for doc_id, res in zip(ids, results):
                    if res is None:
                        continue
                    # document-level incidence (a code counts once per document)
                    seen = set()
                    for c in getattr(res, "applied_codes", []) or []:
                        if c.name not in seen:
                            seen.add(c.name)
                            w_inc.writerow([doc_id, c.name, getattr(c, "theme", None)])
                    for pi, pr in enumerate(getattr(res, "point_results", []) or []):
                        n_points += 1
                        failed = bool(getattr(pr, "judgment_failed", False))
                        n_failed += int(failed)
                        applied = [c.name for c in (pr.applied_codes or [])]
                        cands = [c.name for c in (pr.candidate_codes or [])]
                        w_pts.writerow([doc_id, pi, len(cands), "|".join(cands),
                                        len(applied), "|".join(applied), failed])
                        if f_res is not None:
                            f_res.write(_json.dumps({
                                "document_id": doc_id, "point_index": pi,
                                "information_point": pr.information_point,
                                "candidate_codes": cands, "applied_codes": applied,
                                "judgment_failed": failed,
                                "point_interpretation": pr.point_interpretation,
                                "analysis": pr.analysis,
                                "chunk_index": getattr(pr, "chunk_index", None),
                                "chunk_text": getattr(pr, "chunk_text", None),
                            }, ensure_ascii=False, default=str) + "\n")
            finally:
                if f_res is not None:
                    f_res.close()

        summary = {"n_documents": len(results), "n_points": n_points,
                   "n_judgment_failures": n_failed}
        (output_dir / "application_summary.json").write_text(_json.dumps(summary, indent=2))
        out = {"incidence": inc_path, "points": pts_path,
               "summary": output_dir / "application_summary.json"}
        if include_text:
            out["restricted"] = restricted_path
        if n_failed:
            logger.warning(
                f"{n_failed}/{n_points} information points had LLM judgment failures; these are "
                f"recorded (judgment_failed=True) and must NOT be read as 'no codes apply'.")
        return out

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
