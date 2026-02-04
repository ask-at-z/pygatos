"""Data export utilities for pygatos."""

import json
import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import pandas as pd

from pygatos.core.codebook import Codebook, Code, Theme

logger = logging.getLogger(__name__)


def export_codebook_json(
    codebook: Codebook,
    path: Union[str, Path],
    include_rejected: bool = False,
    include_embeddings: bool = False,
    indent: int = 2,
) -> Path:
    """
    Export codebook to JSON format.

    Args:
        codebook: Codebook to export.
        path: Output file path.
        include_rejected: If True, include rejected codes.
        include_embeddings: If True, include embeddings (large!).
        indent: JSON indentation.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    # Build dict manually to support options
    def code_to_dict(code: Code) -> dict:
        d = {
            "name": code.name,
            "definition": code.definition,
            "source_cluster": code.source_cluster,
            "theme": code.theme,
        }
        # Add metadata if present
        if hasattr(code, "novelty_stage") and code.novelty_stage:
            d["novelty_stage"] = code.novelty_stage
        if hasattr(code, "novelty_reasoning") and code.novelty_reasoning:
            d["novelty_reasoning"] = code.novelty_reasoning
        if hasattr(code, "similarity_score") and code.similarity_score is not None:
            d["similarity_score"] = code.similarity_score
        if hasattr(code, "similar_to") and code.similar_to:
            d["similar_to"] = code.similar_to
        if hasattr(code, "evaluation_order") and code.evaluation_order:
            d["evaluation_order"] = code.evaluation_order
        if include_embeddings and code.embedding is not None:
            d["embedding"] = code.embedding.tolist()
        return d

    data = {
        "accepted_codes": [code_to_dict(c) for c in codebook.accepted_codes],
        "themes": [t.to_dict() for t in codebook.themes],
        "metadata": codebook.metadata,
    }

    if include_rejected:
        data["rejected_codes"] = [code_to_dict(c) for c in codebook.rejected_codes]

    # Add export metadata
    data["_export_info"] = {
        "exported_at": datetime.now().isoformat(),
        "pygatos_version": "0.1.0",
        "include_rejected": include_rejected,
        "include_embeddings": include_embeddings,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    logger.info(f"Exported codebook to {path}")

    return path


def export_codebook_csv(
    codebook: Codebook,
    path: Union[str, Path],
    include_rejected: bool = True,
    include_metadata: bool = True,
) -> Path:
    """
    Export codebook to CSV format.

    Args:
        codebook: Codebook to export.
        path: Output file path.
        include_rejected: If True, include rejected codes.
        include_metadata: If True, include novelty metadata columns.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    codebook.to_csv(path, include_rejected=include_rejected)

    logger.info(f"Exported codebook CSV to {path}")

    return path


def export_coded_data(
    results: list,  # List of ApplicationResult or similar
    path: Union[str, Path],
    format: str = "csv",
) -> Path:
    """
    Export coded data results.

    Args:
        results: List of application results.
        path: Output file path.
        format: Output format ('csv' or 'json').

    Returns:
        Path to exported file.
    """
    path = Path(path)

    rows = []
    for result in results:
        row = {
            "text": result.text if hasattr(result, "text") else str(result),
            "codes": [c.name for c in getattr(result, "applied_codes", [])],
            "n_codes": len(getattr(result, "applied_codes", [])),
            "reasoning": getattr(result, "reasoning", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if format == "csv":
        # Convert codes list to string for CSV
        df["codes"] = df["codes"].apply(lambda x: "; ".join(x))
        df.to_csv(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Exported {len(results)} coded records to {path}")

    return path


def export_code_frequencies(
    frequencies: dict[str, int],
    path: Union[str, Path],
    include_percentages: bool = True,
) -> Path:
    """
    Export code frequency counts.

    Args:
        frequencies: Dict mapping code name to count.
        path: Output file path.
        include_percentages: If True, include percentage column.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    total = sum(frequencies.values())

    rows = []
    for name, count in frequencies.items():
        row = {"code": name, "count": count}
        if include_percentages:
            row["percentage"] = round(count / total * 100, 1) if total > 0 else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("count", ascending=False)

    df.to_csv(path, index=False)

    logger.info(f"Exported code frequencies to {path}")

    return path


def export_themes_hierarchy(
    codebook: Codebook,
    path: Union[str, Path],
    format: str = "markdown",
) -> Path:
    """
    Export theme hierarchy.

    Args:
        codebook: Codebook with themes.
        path: Output file path.
        format: Output format ('markdown', 'json', or 'csv').

    Returns:
        Path to exported file.
    """
    path = Path(path)

    if format == "markdown":
        lines = ["# Codebook Theme Hierarchy", ""]

        for theme in codebook.themes:
            lines.append(f"## {theme.name}")
            lines.append(f"_{theme.definition}_")
            lines.append("")

            for code in theme.codes:
                lines.append(f"- **{code.name}**: {code.definition}")

            lines.append("")

        # Add uncategorized codes
        themed_code_names = set()
        for theme in codebook.themes:
            for code in theme.codes:
                themed_code_names.add(code.name)

        uncategorized = [
            c for c in codebook.accepted_codes
            if c.name not in themed_code_names
        ]

        if uncategorized:
            lines.append("## Uncategorized Codes")
            lines.append("")
            for code in uncategorized:
                lines.append(f"- **{code.name}**: {code.definition}")

        content = "\n".join(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    elif format == "json":
        data = {
            "themes": [
                {
                    "name": t.name,
                    "definition": t.definition,
                    "codes": [{"name": c.name, "definition": c.definition} for c in t.codes],
                }
                for t in codebook.themes
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif format == "csv":
        rows = []
        for theme in codebook.themes:
            for code in theme.codes:
                rows.append({
                    "theme": theme.name,
                    "theme_definition": theme.definition,
                    "code": code.name,
                    "code_definition": code.definition,
                })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Exported theme hierarchy to {path}")

    return path


def export_code_evaluation(
    codebook: Codebook,
    path: Union[str, Path],
) -> Path:
    """
    Export code evaluation results showing all suggested codes with novelty decisions.

    Args:
        codebook: Codebook with accepted and rejected codes.
        path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    records = []

    # Add accepted codes
    for code in codebook.accepted_codes:
        records.append({
            "evaluation_order": code.evaluation_order,
            "name": code.name,
            "definition": code.definition,
            "status": "accepted",
            "novelty_stage": code.novelty_stage,
            "similarity_score": code.similarity_score,
            "similar_to": code.similar_to,
            "novelty_reasoning": code.novelty_reasoning,
            "source_cluster": code.source_cluster,
            "theme": code.theme,
        })

    # Add rejected codes
    for code in codebook.rejected_codes:
        records.append({
            "evaluation_order": code.evaluation_order,
            "name": code.name,
            "definition": code.definition,
            "status": "rejected",
            "novelty_stage": code.novelty_stage,
            "similarity_score": code.similarity_score,
            "similar_to": code.similar_to,
            "novelty_reasoning": code.novelty_reasoning,
            "source_cluster": code.source_cluster,
            "theme": code.theme,
        })

    df = pd.DataFrame(records)

    # Sort by evaluation order
    if len(df) > 0 and "evaluation_order" in df.columns:
        df = df.sort_values("evaluation_order", na_position="last").reset_index(drop=True)

    df.to_csv(path, index=False)

    logger.info(f"Exported code evaluation to {path}")

    return path


def export_theme_evaluation(
    codebook: Codebook,
    path: Union[str, Path],
) -> Path:
    """
    Export theme evaluation results showing all suggested themes with novelty decisions.

    Args:
        codebook: Codebook with accepted and rejected themes.
        path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    records = []

    # Add accepted themes
    for theme in codebook.themes:
        records.append({
            "evaluation_order": theme.evaluation_order,
            "name": theme.name,
            "definition": theme.definition,
            "status": "accepted",
            "novelty_stage": theme.novelty_stage,
            "similarity_score": theme.similarity_score,
            "similar_to": theme.similar_to,
            "novelty_reasoning": theme.novelty_reasoning,
            "n_codes": len(theme.codes),
            "codes": "; ".join([c.name for c in theme.codes]),
        })

    # Add rejected themes
    for theme in codebook.rejected_themes:
        records.append({
            "evaluation_order": theme.evaluation_order,
            "name": theme.name,
            "definition": theme.definition,
            "status": "rejected",
            "novelty_stage": theme.novelty_stage,
            "similarity_score": theme.similarity_score,
            "similar_to": theme.similar_to,
            "novelty_reasoning": theme.novelty_reasoning,
            "n_codes": len(theme.codes),
            "codes": "; ".join([c.name for c in theme.codes]),
        })

    df = pd.DataFrame(records)

    # Sort by evaluation order
    if len(df) > 0 and "evaluation_order" in df.columns:
        df = df.sort_values("evaluation_order", na_position="last").reset_index(drop=True)

    df.to_csv(path, index=False)

    logger.info(f"Exported theme evaluation to {path}")

    return path


def export_codebook_simple(
    codebook: Codebook,
    path: Union[str, Path],
) -> Path:
    """
    Export a simple codebook with just label, definition, and type.

    This creates a clean codebook file with all accepted codes and themes
    in a simple three-column format.

    Args:
        codebook: Codebook with accepted codes and themes.
        path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    records = []

    # Add accepted codes
    for code in codebook.accepted_codes:
        records.append({
            "label": code.name,
            "definition": code.definition,
            "type": "code",
        })

    # Add accepted themes
    for theme in codebook.themes:
        records.append({
            "label": theme.name,
            "definition": theme.definition,
            "type": "theme",
        })

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

    logger.info(f"Exported simple codebook to {path}")

    return path


def export_coded_data_audit(
    results: list,  # List of ApplicationResult
    path: Union[str, Path],
) -> Path:
    """
    Export coded data with full audit trail.

    Includes text, candidate codes, LLM analysis, and applied codes.
    This is the verbose version for transparency and debugging.

    For extraction-based coding, also includes information points and
    point-level coding details.

    Args:
        results: List of ApplicationResult objects.
        path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    rows = []
    for result in results:
        # Check if this was extraction-based coding
        info_points = getattr(result, "information_points", [])
        point_results = getattr(result, "point_results", [])

        row = {
            "text": result.text if hasattr(result, "text") else str(result),
            "applied_codes": "; ".join([c.name for c in getattr(result, "applied_codes", [])]),
            "n_applied": len(getattr(result, "applied_codes", [])),
            "reasoning": getattr(result, "reasoning", "") or "",
        }

        if info_points:
            # Extraction-based coding
            row["n_info_points"] = len(info_points)
            row["information_points"] = " | ".join(info_points)

            # Build point-level summary
            point_summaries = []
            for i, pr in enumerate(point_results):
                codes = ", ".join(c.name for c in pr.applied_codes) if pr.applied_codes else "(none)"
                point_summaries.append(f"[{i+1}] {pr.information_point[:80]}... → {codes}")
            row["point_coding"] = " || ".join(point_summaries)
        else:
            # Direct text coding (legacy)
            row["n_info_points"] = 0
            row["information_points"] = ""
            row["point_coding"] = ""
            row["candidate_codes"] = "; ".join([c.name for c in getattr(result, "candidate_codes", [])])
            row["text_summary"] = getattr(result, "text_summary", "") or ""
            row["analysis"] = getattr(result, "analysis", "") or ""

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    logger.info(f"Exported {len(results)} coded records (audit) to {path}")

    return path


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def export_coded_data_points(
    results: list,  # List of ApplicationResult
    path: Union[str, Path],
    truncate_text: int = 200,
) -> Path:
    """
    Export point-level coding details with full lineage tracking.

    Each row is one information point with its associated codes and
    chunk lineage (chunk_index, chunk_text) for full traceability.

    Args:
        results: List of ApplicationResult objects.
        path: Output file path.
        truncate_text: Max characters for text fields (for readability).

    Returns:
        Path to exported file.
    """
    path = Path(path)

    rows = []
    for resp_idx, result in enumerate(results):
        source_text = result.text if hasattr(result, "text") else str(result)
        point_results = getattr(result, "point_results", [])

        if point_results:
            # Extraction-based: one row per point
            for point_idx, pr in enumerate(point_results):
                # Get chunk lineage if available
                chunk_index = getattr(pr, "chunk_index", None)
                chunk_text = getattr(pr, "chunk_text", None)

                row = {
                    "response_idx": resp_idx,
                    "point_idx": point_idx,
                    "source_text": _truncate(source_text, truncate_text),
                    "chunk_index": chunk_index,
                    "chunk_text": _truncate(chunk_text, truncate_text) if chunk_text else None,
                    "information_point": pr.information_point,
                    "candidate_codes": "; ".join([c.name for c in pr.candidate_codes]),
                    "applied_codes": "; ".join([c.name for c in pr.applied_codes]),
                    "n_applied": len(pr.applied_codes),
                    "point_interpretation": pr.point_interpretation or "",
                    "analysis": pr.analysis or "",
                }
                rows.append(row)
        else:
            # Direct coding: single row for the full text
            row = {
                "response_idx": resp_idx,
                "point_idx": 0,
                "source_text": _truncate(source_text, truncate_text),
                "chunk_index": None,
                "chunk_text": None,
                "information_point": "(full text)",
                "candidate_codes": "; ".join([c.name for c in getattr(result, "candidate_codes", [])]),
                "applied_codes": "; ".join([c.name for c in getattr(result, "applied_codes", [])]),
                "n_applied": len(getattr(result, "applied_codes", [])),
                "point_interpretation": getattr(result, "text_summary", "") or "",
                "analysis": getattr(result, "analysis", "") or "",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    logger.info(f"Exported {len(rows)} point-level coding records to {path}")

    return path


def export_coded_data_simple(
    results: list,  # List of ApplicationResult
    path: Union[str, Path],
    include_candidates: bool = False,
) -> Path:
    """
    Export coded data in simple format.

    Just text and applied codes, for easy downstream analysis.

    Args:
        results: List of ApplicationResult objects.
        path: Output file path.
        include_candidates: If True, also include candidate codes column.

    Returns:
        Path to exported file.
    """
    path = Path(path)

    rows = []
    for result in results:
        row = {
            "text": result.text if hasattr(result, "text") else str(result),
            "applied_codes": "; ".join([c.name for c in getattr(result, "applied_codes", [])]),
        }
        if include_candidates:
            row["candidate_codes"] = "; ".join([c.name for c in getattr(result, "candidate_codes", [])])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    logger.info(f"Exported {len(results)} coded records (simple) to {path}")

    return path


def export_extraction_lineage(
    extraction_results: dict,
    path: Union[str, Path],
    truncate_text: int = 200,
) -> Path:
    """
    Export extraction lineage showing full traceability from source to information points.

    Creates a CSV with columns showing: source_id, source_text, chunk_index,
    chunk_text, and information_point. This enables full auditability -
    any information point can be traced back to its source text and chunk.

    Args:
        extraction_results: Dict mapping ID to SummarizationResult.
        path: Output file path.
        truncate_text: Max characters for text fields (for readability).

    Returns:
        Path to exported file.
    """
    path = Path(path)

    rows = []
    for source_id, result in extraction_results.items():
        # Handle SummarizationResult objects
        if hasattr(result, 'structured_points'):
            structured_points = result.structured_points
            source_text = result.original_text
        else:
            # Skip if not a SummarizationResult
            logger.warning(f"Skipping {source_id}: not a SummarizationResult")
            continue

        if not structured_points:
            # Source had no information points extracted
            rows.append({
                "source_id": source_id,
                "source_text": _truncate(source_text, truncate_text),
                "source_text_length": len(source_text),
                "n_chunks": result.n_chunks if hasattr(result, 'n_chunks') else 0,
                "chunk_index": None,
                "chunk_text": None,
                "info_point_idx": None,
                "information_point": "(no points extracted)",
            })
            continue

        for point_idx, point in enumerate(structured_points):
            rows.append({
                "source_id": source_id,
                "source_text": _truncate(point.source_text, truncate_text),
                "source_text_length": len(point.source_text),
                "n_chunks": result.n_chunks if hasattr(result, 'n_chunks') else 1,
                "chunk_index": point.chunk_index,
                "chunk_text": _truncate(point.chunk_text, truncate_text) if point.chunk_text else None,
                "info_point_idx": point_idx,
                "information_point": point.text,
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    logger.info(f"Exported extraction lineage ({len(rows)} rows) to {path}")

    return path


def export_run_metadata(
    output_dir: Union[str, Path],
    config: dict,
    results: dict,
    start_time: datetime,
    end_time: datetime,
    notes: str = "",
) -> Path:
    """
    Export run metadata for reproducibility.

    Args:
        output_dir: Output directory.
        config: Configuration dictionary.
        results: Results dictionary.
        start_time: Run start time.
        end_time: Run end time.
        notes: Optional notes.

    Returns:
        Path to metadata file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = (end_time - start_time).total_seconds()

    metadata = {
        "run_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
        },
        "config": config,
        "results": results,
        "notes": notes,
    }

    path = output_dir / "run_metadata.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Exported run metadata to {path}")

    return path
