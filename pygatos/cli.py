"""Command-line interface for pygatos."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from pygatos import GATOSPipeline, GATOSConfig
from pygatos import provenance
from pygatos.io import (
    load_csv,
    load_data,
    export_codebook_json,
    export_code_evaluation,
    export_theme_evaluation,
    export_codebook_simple,
    export_coded_data_audit,
    export_coded_data_simple,
    export_coded_data_points,
    export_run_metadata,
)
from pygatos.visualization import (
    plot_code_frequencies,
    plot_codebook_growth,
    plot_saturation_analysis,
)

logger = logging.getLogger(__name__)


def _pygatos_version() -> str:
    try:
        from importlib.metadata import version
        return version("pygatos")
    except Exception:
        return "unknown"


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging for CLI.

    Args:
        verbose: If True, show INFO level logs.
        debug: If True, show DEBUG level logs (includes LLM prompts/responses).
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def create_output_dir(base_dir: Path, timestamp: datetime) -> Path:
    """Create timestamped output directory."""
    dirname = timestamp.strftime("%Y%m%d-%H%M")
    output_dir = base_dir / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the full GATOS pipeline."""
    start_time = datetime.now()

    setup_logging(verbose=args.verbose, debug=args.debug)

    # Create output directory
    output_base = Path(args.output_dir)
    output_dir = create_output_dir(output_base, start_time)

    logger.info("=" * 70)
    logger.info("PYGATOS - Qualitative Codebook Generation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("")
    logger.info("Loading data...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    try:
        df = load_data(data_path, text_column=args.text_column, id_column=args.id_column)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Filter out empty/very short responses
    n_raw_rows = len(df)
    df = df[df[args.text_column].notna()]
    df = df[df[args.text_column].astype(str).str.len() > 5]

    logger.info(f"Loaded {len(df)} valid responses from {data_path.name}")

    # Configure pipeline
    logger.info("")
    logger.info("Configuring pipeline...")

    config = GATOSConfig()

    if args.embedding_model:
        config.embedding.model_name = args.embedding_model
    if args.llm_model:
        config.llm.model = args.llm_model
    if args.cluster_size:
        config.clustering.target_cluster_size = args.cluster_size
    if args.similarity_threshold:
        config.novelty.similarity_threshold = args.similarity_threshold
    if args.top_k:
        config.application.top_k = args.top_k
    if args.debug:
        config.llm.debug = True
    if args.context:
        config.study_context = args.context
    if args.novelty_policy:
        config.novelty.policy = args.novelty_policy

    logger.info(f"  Embedding model: {config.embedding.model_name}")
    logger.info(f"  LLM model: {config.llm.model}")
    logger.info(f"  Cluster size: {config.clustering.target_cluster_size}")
    logger.info(f"  Similarity threshold: {config.novelty.similarity_threshold}")
    logger.info(f"  Novelty policy: {config.novelty.policy}")
    logger.info(f"  Top-K for application: {config.application.top_k}")
    if config.study_context:
        logger.info(f"  Study context: {config.study_context[:50]}...")

    # Initialize pipeline
    pipeline = GATOSPipeline(config=config)

    # Check LLM availability
    if not pipeline.llm.is_available():
        logger.error("LLM not available! Make sure Ollama is running.")
        return 1

    logger.info("Pipeline initialized successfully")

    # Forward-path provenance: record every LLM HTTP attempt, and write a run manifest.
    # The call log contains the corpus text sent to the model, so it inherits the corpus's
    # confidentiality. Disable with --no-llm-log; the manifest records which way it ran.
    llm_recorder = None
    if not args.no_llm_log:
        llm_recorder = provenance.LLMCallRecorder(output_dir / "llm_calls.jsonl")
        llm_recorder.attach(pipeline.llm)
        logger.info(f"  LLM call log: {output_dir / 'llm_calls.jsonl'}")

    manifest_path = output_dir / "manifest.json"

    def write_manifest(**extra) -> None:
        manifest = {
            "pygatos_version": _pygatos_version(),
            "package_git": provenance.git_metadata(Path(__file__).resolve().parents[1]),
            "environment": provenance.environment_metadata(),
            "corpus": provenance.corpus_metadata(
                df, args.text_column, args.id_column, data_path, n_raw=n_raw_rows,
                filter_description="dropped rows with missing text or <= 5 characters",
            ),
            "llm_backend": {
                "backend": config.llm.backend,
                "model": config.llm.model,
                "temperature": config.llm.temperature,
                "seed": config.llm.seed,
            },
            "consolidation_settings": provenance.consolidation_settings(pipeline),
            "resolved_prompts": provenance.resolved_prompts(pipeline),
            "llm_calls_logged": llm_recorder is not None,
        }
        if config.llm.backend == "ollama":
            manifest["ollama"] = provenance.ollama_metadata(config.llm.base_url, config.llm.model)
        manifest.update(extra)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Written now so a killed run still leaves its settings on disk; rewritten with
    # stats when the run completes.
    write_manifest(status="in_progress")
    logger.info(f"  Saved: {manifest_path.name}")

    # Generate codebook
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: Generating Codebook")
    logger.info("=" * 70)

    codebook = pipeline.generate_codebook(
        data=df,
        text_column=args.text_column,
        id_column=args.id_column,
        skip_summarization=args.skip_summarization,
        skip_dim_reduction=args.skip_dim_reduction,
        generate_themes=not args.skip_themes,
        verbose=args.verbose,
    )

    logger.info("")
    logger.info(f"Codebook generated:")
    logger.info(f"  Accepted codes: {len(codebook.accepted_codes)}")
    logger.info(f"  Rejected codes: {len(codebook.rejected_codes)}")
    logger.info(f"  Themes: {len(codebook.themes)}")

    # Save codebook outputs
    logger.info("")
    logger.info("Saving codebook outputs...")

    json_path = export_codebook_json(
        codebook, output_dir / "codebook_full.json", include_rejected=True
    )
    logger.info(f"  Saved: {json_path.name}")

    code_eval_path = export_code_evaluation(codebook, output_dir / "code_evaluation.csv")
    logger.info(f"  Saved: {code_eval_path.name}")

    theme_eval_path = export_theme_evaluation(codebook, output_dir / "theme_evaluation.csv")
    logger.info(f"  Saved: {theme_eval_path.name}")

    codebook_path = export_codebook_simple(codebook, output_dir / "codebook.csv")
    logger.info(f"  Saved: {codebook_path.name}")

    # Apply codebook
    application_results = None

    if not args.skip_apply:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Applying Codebook")
        logger.info("=" * 70)

        use_extraction = not args.skip_apply_extraction
        if use_extraction:
            logger.info("Using extraction-based coding (information points)")
        else:
            logger.info("Using direct text coding (legacy mode)")

        try:
            coded_df, application_results = pipeline.apply_codebook_with_details(
                data=df,
                codebook=codebook,
                text_column=args.text_column,
                id_column=args.id_column,
                use_extraction=use_extraction,
                verbose=args.verbose,
            )

            logger.info(f"Applied codebook to {len(coded_df)} responses")

            # Save coded data
            export_coded_data_audit(application_results, output_dir / "coded_data_audit.csv")
            logger.info(f"  Saved: coded_data_audit.csv")

            export_coded_data_simple(
                application_results, output_dir / "coded_data.csv", include_candidates=True
            )
            logger.info(f"  Saved: coded_data.csv")

            # Export point-level coding details if extraction was used
            if use_extraction:
                export_coded_data_points(application_results, output_dir / "coded_data_points.csv")
                logger.info(f"  Saved: coded_data_points.csv")

        except Exception as e:
            logger.error(f"Code application failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Generate visualizations
    if not args.skip_viz:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 3: Generating Visualizations")
        logger.info("=" * 70)

        try:
            # Frequency chart (only if we have application results)
            if application_results:
                code_frequencies = {}
                for result in application_results:
                    for code in result.applied_codes:
                        code_frequencies[code.name] = code_frequencies.get(code.name, 0) + 1
                code_frequencies = dict(sorted(code_frequencies.items(), key=lambda x: -x[1]))

                plot_code_frequencies(
                    code_frequencies,
                    title="Code Frequencies",
                    top_n=15,
                    save_path=output_dir / "frequencies.png",
                )
                logger.info(f"  Saved: frequencies.png")

            # Growth chart
            plot_codebook_growth(
                codebook,
                title="Codebook Growth",
                save_path=output_dir / "growth.png",
            )
            logger.info(f"  Saved: growth.png")

            # Saturation analysis
            plot_saturation_analysis(
                codebook,
                title="Codebook Saturation Analysis",
                save_path=output_dir / "saturation.png",
            )
            logger.info(f"  Saved: saturation.png")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Save run metadata
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    run_config = {
        "data_file": str(data_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "embedding_model": config.embedding.model_name,
        "llm_model": config.llm.model,
        "cluster_size": config.clustering.target_cluster_size,
        "similarity_threshold": config.novelty.similarity_threshold,
        "top_k": config.application.top_k,
        "study_context": config.study_context,
        "skip_summarization": args.skip_summarization,
        "skip_dim_reduction": args.skip_dim_reduction,
        "skip_themes": args.skip_themes,
        "skip_apply": args.skip_apply,
    }

    run_results = {
        "n_responses": len(df),
        "n_accepted_codes": len(codebook.accepted_codes),
        "n_rejected_codes": len(codebook.rejected_codes),
        "n_themes": len(codebook.themes),
        "deduplication_rate": (
            len(codebook.rejected_codes)
            / (len(codebook.accepted_codes) + len(codebook.rejected_codes))
            * 100
            if (len(codebook.accepted_codes) + len(codebook.rejected_codes)) > 0
            else 0
        ),
    }

    export_run_metadata(
        output_dir=output_dir,
        config=run_config,
        results=run_results,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"  Saved: run_metadata.json")

    write_manifest(
        status="complete",
        llm_call_stats=llm_recorder.stats() if llm_recorder else None,
    )
    if llm_recorder:
        llm_recorder.close()

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Responses processed: {len(df)}")
    logger.info(f"Codes accepted: {len(codebook.accepted_codes)}")
    logger.info(f"Codes rejected: {len(codebook.rejected_codes)}")
    logger.info(f"Themes generated: {len(codebook.themes)}")
    logger.info(f"Outputs saved to: {output_dir}")

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pygatos",
        description="PYGATOS - Python library for GATOS qualitative codebook generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pygatos run -d survey.csv -t response_text
  pygatos run -d data.csv -t text_col -i id_col --cluster-size 6
  pygatos run -d data.xlsx -t comments --llm-model llama3:8b --skip-summarization
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full GATOS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    run_parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to input data file (CSV, Excel, or JSON)",
    )
    run_parser.add_argument(
        "-t", "--text-column",
        required=True,
        help="Name of column containing text to analyze",
    )

    # Optional arguments
    run_parser.add_argument(
        "-i", "--id-column",
        help="Name of ID column (optional)",
    )
    run_parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Output directory (default: ./output)",
    )

    # Model configuration
    run_parser.add_argument(
        "-m", "--llm-model",
        help="LLM model name for Ollama (default: from config)",
    )
    run_parser.add_argument(
        "-e", "--embedding-model",
        help="Embedding model name (default: from config)",
    )

    # Pipeline configuration
    run_parser.add_argument(
        "-c", "--cluster-size",
        type=int,
        help="Target cluster size for clustering (default: 10)",
    )
    run_parser.add_argument(
        "-s", "--similarity-threshold",
        type=float,
        help="Similarity threshold for novelty evaluation (default: 0.8)",
    )
    run_parser.add_argument(
        "-k", "--top-k",
        type=int,
        help="Number of candidate codes for application (default: 10)",
    )

    # Study context
    run_parser.add_argument(
        "--context",
        type=str,
        help="Brief description of the dataset/study to improve LLM understanding (e.g., 'Survey responses about inflation concerns from US adults')",
    )

    run_parser.add_argument(
        "--novelty-policy",
        choices=["reject-unless-distinct", "keep-unless-duplicate"],
        help=(
            "Instructed policy for the Step-7 consolidation decision. "
            "'reject-unless-distinct' (default) is the published behavior; "
            "'keep-unless-duplicate' is the validated higher-recall repair "
            "(larger codebook; audit duplication with pygatos.diagnostics)"
        ),
    )
    run_parser.add_argument(
        "--no-llm-log",
        action="store_true",
        help="Do not write llm_calls.jsonl (the per-call prompt/response record)",
    )

    # Skip options
    run_parser.add_argument(
        "--skip-summarization",
        action="store_true",
        help="Skip information extraction (use for short texts)",
    )
    run_parser.add_argument(
        "--skip-dim-reduction",
        action="store_true",
        help="Skip PCA+UMAP dimensionality reduction",
    )
    run_parser.add_argument(
        "--skip-themes",
        action="store_true",
        help="Skip theme generation",
    )
    run_parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="Skip code application step",
    )
    run_parser.add_argument(
        "--skip-apply-extraction",
        action="store_true",
        help="Skip extraction during code application (code full text directly instead of extracted points)",
    )
    run_parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    # Output options
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with progress bars",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (shows all LLM prompts and responses)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return run_pipeline(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
