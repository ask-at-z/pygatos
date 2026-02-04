"""Data loading utilities for pygatos."""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(
    path: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None,
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file for processing.

    Args:
        path: Path to CSV file.
        text_column: Name of column containing text.
        id_column: Optional name of ID column.
        encoding: File encoding.
        **kwargs: Additional arguments passed to pd.read_csv.

    Returns:
        DataFrame with validated columns.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, encoding=encoding, **kwargs)

    # Validate text column
    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate ID column if specified
    if id_column and id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Loaded {len(df)} rows from {path.name}")

    return df


def load_excel(
    path: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None,
    sheet_name: Union[str, int] = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Load an Excel file for processing.

    Args:
        path: Path to Excel file.
        text_column: Name of column containing text.
        id_column: Optional name of ID column.
        sheet_name: Sheet name or index.
        **kwargs: Additional arguments passed to pd.read_excel.

    Returns:
        DataFrame with validated columns.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    # Validate text column
    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if id_column and id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Loaded {len(df)} rows from {path.name}")

    return df


def load_json(
    path: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a JSON file for processing.

    Args:
        path: Path to JSON file.
        text_column: Name of field containing text.
        id_column: Optional name of ID field.
        **kwargs: Additional arguments passed to pd.read_json.

    Returns:
        DataFrame with validated columns.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_json(path, **kwargs)

    # Validate text column
    if text_column not in df.columns:
        raise ValueError(
            f"Text field '{text_column}' not found. "
            f"Available fields: {list(df.columns)}"
        )

    if id_column and id_column not in df.columns:
        raise ValueError(
            f"ID field '{id_column}' not found. "
            f"Available fields: {list(df.columns)}"
        )

    logger.info(f"Loaded {len(df)} records from {path.name}")

    return df


def load_data(
    path: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load data from file, auto-detecting format.

    Supports CSV, Excel (.xlsx, .xls), and JSON files.

    Args:
        path: Path to data file.
        text_column: Name of column/field containing text.
        id_column: Optional name of ID column/field.
        **kwargs: Additional arguments passed to format-specific loader.

    Returns:
        DataFrame with validated columns.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path, text_column, id_column, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        return load_excel(path, text_column, id_column, **kwargs)
    elif suffix == ".json":
        return load_json(path, text_column, id_column, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .csv, .xlsx, .xls, .json"
        )


def load_texts(
    path: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None,
    **kwargs,
) -> tuple[list[str], list]:
    """
    Load data and extract text and IDs.

    Convenience function that returns texts and IDs directly.

    Args:
        path: Path to data file.
        text_column: Name of column containing text.
        id_column: Optional name of ID column.
        **kwargs: Additional arguments passed to loader.

    Returns:
        Tuple of (texts list, ids list).
    """
    df = load_data(path, text_column, id_column, **kwargs)

    texts = df[text_column].astype(str).tolist()

    if id_column:
        ids = df[id_column].tolist()
    else:
        ids = list(range(len(texts)))

    return texts, ids


def combine_csvs(
    paths: list[Union[str, Path]],
    text_column: str,
    id_column: Optional[str] = None,
    add_source_column: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Combine multiple CSV files into a single DataFrame.

    Args:
        paths: List of paths to CSV files.
        text_column: Name of column containing text.
        id_column: Optional name of ID column.
        add_source_column: If True, add a 'source_file' column.
        **kwargs: Additional arguments passed to pd.read_csv.

    Returns:
        Combined DataFrame.
    """
    dfs = []

    for path in paths:
        path = Path(path)
        df = load_csv(path, text_column, id_column, **kwargs)

        if add_source_column:
            df["source_file"] = path.name

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    logger.info(f"Combined {len(paths)} files: {len(combined)} total rows")

    return combined
