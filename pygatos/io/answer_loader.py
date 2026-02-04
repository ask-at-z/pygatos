"""Data loader for focus group Q&A files."""

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class AnswerDataset:
    """Container for loaded answer data with metadata."""

    df: pd.DataFrame
    total_instances: int
    total_files: int
    na_count: int
    has_iid: bool
    date_range: tuple[str, str]  # (earliest, latest)
    group_names: list[str]

    @property
    def non_na_count(self) -> int:
        """Count of non-NA answers."""
        return self.total_instances - self.na_count


class AnswerLoader:
    """
    Loads focus group Q&A data from CSV files.

    Handles the Longwell FG_Q_and_A folder structure:
    - Filenames encode Date and Group Name: `QandA__YY.MM.DD Group Name.csv`
    - Columns: Question, Respondent, Answer, [optional Iid]
    - NA values should be preserved (not coded)

    Example:
        >>> loader = AnswerLoader()
        >>> dataset = loader.load("data/FG_Q_and_A")
        >>> print(f"Loaded {dataset.total_instances} answers")
        >>> print(f"Non-NA answers: {dataset.non_na_count}")
    """

    # Regex to extract date and group name from filename
    # Format: QandA__YY.MM.DD Group Name.csv or QandA__YY.M.DD Group Name.csv
    # Some files have single-digit months (e.g., 24.4.16 instead of 24.04.16)
    # Some files have extra prefix like QandA__QsAsked__YY.MM.DD
    FILENAME_PATTERN = re.compile(
        r"QandA__(?:QsAsked__)?(\d{2})\.(\d{1,2})\.(\d{1,2})\s+(.+)\.csv$"
    )

    def __init__(
        self,
        na_values: Optional[list[str]] = None,
        preserve_na_string: bool = True,
    ):
        """
        Initialize the answer loader.

        Args:
            na_values: List of strings to treat as NA. Default: ["NA"]
            preserve_na_string: If True, keep "NA" as string instead of NaN.
                               This allows distinguishing between missing data
                               and explicit "NA" responses.
        """
        self.na_values = na_values or ["NA"]
        self.preserve_na_string = preserve_na_string

    def parse_filename(self, filename: str) -> tuple[str, str]:
        """
        Extract date and group name from filename.

        Args:
            filename: Filename like "QandA__23.02.08 Flippers.csv"

        Returns:
            Tuple of (date_str, group_name) like ("2023-02-08", "Flippers")
            Returns ("unknown", filename) if pattern doesn't match.
        """
        match = self.FILENAME_PATTERN.match(filename)
        if match:
            year, month, day, group_name = match.groups()
            # Convert YY to YYYY (assume 2000s) and zero-pad month/day
            full_year = f"20{year}"
            date_str = f"{full_year}-{int(month):02d}-{int(day):02d}"
            return date_str, group_name.strip()
        else:
            # Try alternative patterns
            # Some files might have slightly different formats
            logger.warning(f"Could not parse filename: {filename}")
            return "unknown", Path(filename).stem

    def load(
        self,
        folder_path: Union[str, Path],
        verbose: bool = False,
    ) -> AnswerDataset:
        """
        Load all Q&A data from a folder.

        Args:
            folder_path: Path to folder containing Q&A CSV files.
            verbose: If True, show progress bar and log details.

        Returns:
            AnswerDataset with combined data and metadata.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in: {folder_path}")

        if verbose:
            logger.info(f"Loading Q&A data from {len(csv_files)} files")

        dfs = []
        has_iid_any = False
        dates = []
        group_names = []

        file_iter = tqdm(csv_files, desc="Loading files", disable=not verbose)
        for csv_file in file_iter:
            try:
                # Load with custom NA handling
                if self.preserve_na_string:
                    # Don't treat "NA" as NaN - keep it as string
                    df = pd.read_csv(csv_file, na_values=[], keep_default_na=False)
                else:
                    df = pd.read_csv(csv_file, na_values=self.na_values)

                # Validate required columns
                required_cols = ["Question", "Respondent", "Answer"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    logger.warning(
                        f"Skipping {csv_file.name}: missing columns {missing}"
                    )
                    continue

                # Check for Iid column
                if "Iid" in df.columns:
                    has_iid_any = True

                # Extract metadata from filename
                date_str, group_name = self.parse_filename(csv_file.name)
                df["Date"] = date_str
                df["Group Name"] = group_name
                df["Source File"] = csv_file.name

                dates.append(date_str)
                group_names.append(group_name)
                dfs.append(df)

            except Exception as e:
                logger.warning(f"Error loading {csv_file.name}: {e}")

        if not dfs:
            raise ValueError("No valid Q&A files loaded")

        # Combine all dataframes
        combined = pd.concat(dfs, ignore_index=True)

        # Ensure Iid column exists (even if empty) for consistency
        if "Iid" not in combined.columns:
            combined["Iid"] = None

        # Reorder columns for clarity
        column_order = [
            "Question",
            "Respondent",
            "Answer",
            "Iid",
            "Date",
            "Group Name",
            "Source File",
        ]
        combined = combined[column_order]

        # Count NA values
        na_count = self._count_na_values(combined)

        # Sort dates and get range
        valid_dates = [d for d in dates if d != "unknown"]
        if valid_dates:
            date_range = (min(valid_dates), max(valid_dates))
        else:
            date_range = ("unknown", "unknown")

        dataset = AnswerDataset(
            df=combined,
            total_instances=len(combined),
            total_files=len(dfs),
            na_count=na_count,
            has_iid=has_iid_any,
            date_range=date_range,
            group_names=sorted(set(group_names)),
        )

        if verbose:
            logger.info(
                f"Loaded {dataset.total_instances} answer instances "
                f"from {dataset.total_files} files"
            )
            logger.info(
                f"NA answers: {dataset.na_count} ({dataset.na_count/dataset.total_instances*100:.1f}%)"
            )
            logger.info(f"Date range: {date_range[0]} to {date_range[1]}")

        return dataset

    def _count_na_values(self, df: pd.DataFrame) -> int:
        """Count answers that are NA (string or NaN)."""
        answer_col = df["Answer"]

        if self.preserve_na_string:
            # Count string "NA" values
            na_mask = answer_col.astype(str).str.strip().str.upper() == "NA"
        else:
            # Count NaN values
            na_mask = answer_col.isna()

        # Also count empty strings
        empty_mask = answer_col.astype(str).str.strip() == ""

        return int((na_mask | empty_mask).sum())

    def get_non_na_answers(
        self,
        dataset: AnswerDataset,
    ) -> pd.DataFrame:
        """
        Filter dataset to only non-NA answers.

        Args:
            dataset: Loaded AnswerDataset.

        Returns:
            DataFrame with only non-NA answers.
        """
        df = dataset.df.copy()
        answer_col = df["Answer"].astype(str).str.strip()

        # Filter out NA and empty answers
        mask = (answer_col.str.upper() != "NA") & (answer_col != "")

        return df[mask].reset_index(drop=True)

    def get_answers_by_group(
        self,
        dataset: AnswerDataset,
    ) -> dict[str, pd.DataFrame]:
        """
        Split dataset by focus group.

        Args:
            dataset: Loaded AnswerDataset.

        Returns:
            Dict mapping group name to DataFrame.
        """
        groups = {}
        for group_name, group_df in dataset.df.groupby("Group Name"):
            groups[group_name] = group_df.reset_index(drop=True)
        return groups

    def get_unique_questions(
        self,
        dataset: AnswerDataset,
    ) -> list[str]:
        """
        Get list of unique questions in the dataset.

        Args:
            dataset: Loaded AnswerDataset.

        Returns:
            List of unique question strings.
        """
        return dataset.df["Question"].dropna().unique().tolist()

    def summary(self, dataset: AnswerDataset) -> str:
        """
        Generate a summary string of the dataset.

        Args:
            dataset: Loaded AnswerDataset.

        Returns:
            Formatted summary string.
        """
        lines = [
            "=== Answer Dataset Summary ===",
            f"Total answer instances: {dataset.total_instances:,}",
            f"Non-NA answers: {dataset.non_na_count:,} ({dataset.non_na_count/dataset.total_instances*100:.1f}%)",
            f"NA answers: {dataset.na_count:,} ({dataset.na_count/dataset.total_instances*100:.1f}%)",
            f"Files loaded: {dataset.total_files}",
            f"Has Iid column: {dataset.has_iid}",
            f"Date range: {dataset.date_range[0]} to {dataset.date_range[1]}",
            f"Unique groups: {len(dataset.group_names)}",
            f"Unique questions: {len(self.get_unique_questions(dataset))}",
            f"Unique respondents: {dataset.df['Respondent'].nunique()}",
        ]
        return "\n".join(lines)

    def link_canonical_questions(
        self,
        dataset: AnswerDataset,
        canonical_mapping_path: Union[str, Path],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Link answers to their canonical question forms.

        DEPRECATED: Use link_question_codes() instead for the new GATOS pipeline output.

        Uses the question canonicalization mapping from Phase 1 to add
        canonical question and topic columns to the answer data.

        Args:
            dataset: Loaded AnswerDataset.
            canonical_mapping_path: Path to the all_questions_codebooked.csv
                                   file from Phase 1.
            verbose: If True, log progress.

        Returns:
            DataFrame with added 'Canonical Question' and 'Question Topic' columns.
        """
        import warnings
        warnings.warn(
            "link_canonical_questions() is deprecated. Use link_question_codes() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        canonical_path = Path(canonical_mapping_path)
        if not canonical_path.exists():
            raise FileNotFoundError(
                f"Canonical mapping file not found: {canonical_mapping_path}"
            )

        # Load canonical mapping
        canonical_df = pd.read_csv(canonical_path)

        # Create lookup dict: original question -> (canonical question, topic)
        question_to_canonical = {}
        for _, row in canonical_df.iterrows():
            orig = row["Question"]
            canonical = row["Codebooked Question"]
            topic = row.get("Topic", "")
            question_to_canonical[orig] = (canonical, topic)

        if verbose:
            logger.info(
                f"Loaded {len(question_to_canonical)} canonical mappings"
            )

        # Add canonical columns to answer data
        df = dataset.df.copy()

        def get_canonical(question):
            if question in question_to_canonical:
                return question_to_canonical[question]
            return (None, None)

        canonical_data = df["Question"].apply(get_canonical)
        df["Canonical Question"] = canonical_data.apply(lambda x: x[0])
        df["Question Topic"] = canonical_data.apply(lambda x: x[1])

        # Count matches
        matched = df["Canonical Question"].notna().sum()
        total = len(df)

        if verbose:
            logger.info(
                f"Linked {matched:,}/{total:,} answers to canonical questions "
                f"({matched/total*100:.1f}%)"
            )
            unmatched = total - matched
            if unmatched > 0:
                logger.warning(
                    f"{unmatched:,} answers could not be linked to canonical questions"
                )

        return df

    def link_question_codes(
        self,
        dataset: AnswerDataset,
        labeled_questions_path: Union[str, Path],
        codebook_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Link answers to their question topic codes.

        Uses the labeled_questions.csv from the GATOS pipeline to add
        question codes and themes to the answer data. Supports one-to-many
        relationship (each question can have multiple codes).

        Args:
            dataset: Loaded AnswerDataset.
            labeled_questions_path: Path to labeled_questions.csv from the
                                   question codebook generation pipeline.
            codebook_path: Optional path to question_codebook_full.json for
                          code-to-theme mapping. If not provided, themes
                          will not be added.
            verbose: If True, log progress.

        Returns:
            DataFrame with added columns:
                - 'question_codes': List of all codes for this question
                - 'n_question_codes': Number of codes
                - 'question_themes': List of themes (if codebook provided)
        """
        labeled_path = Path(labeled_questions_path)
        if not labeled_path.exists():
            raise FileNotFoundError(
                f"Labeled questions file not found: {labeled_questions_path}"
            )

        # Load labeled questions
        labeled_df = pd.read_csv(labeled_path)

        # Parse the codes column (stored as string representation of list)
        def parse_codes(codes_str):
            """Parse codes column from string to list."""
            if pd.isna(codes_str) or codes_str == "" or codes_str == "[]":
                return []
            try:
                return ast.literal_eval(codes_str)
            except (ValueError, SyntaxError):
                return []

        labeled_df["codes_parsed"] = labeled_df["codes"].apply(parse_codes)

        # Create lookup dict: question text -> list of codes
        question_to_codes = {}
        for _, row in labeled_df.iterrows():
            question = row["Question"]
            codes = row["codes_parsed"]
            question_to_codes[question] = codes

        if verbose:
            logger.info(
                f"Loaded {len(question_to_codes)} question-to-codes mappings"
            )
            codes_counts = [len(c) for c in question_to_codes.values()]
            avg_codes = sum(codes_counts) / len(codes_counts) if codes_counts else 0
            logger.info(f"Average codes per question: {avg_codes:.2f}")

        # Load code-to-theme mapping if codebook provided
        code_to_theme = {}
        if codebook_path:
            cb_path = Path(codebook_path)
            if cb_path.exists():
                with open(cb_path) as f:
                    codebook = json.load(f)
                for code in codebook.get("accepted_codes", []):
                    code_to_theme[code["name"]] = code.get("theme", "")
                if verbose:
                    logger.info(
                        f"Loaded {len(code_to_theme)} code-to-theme mappings"
                    )
            else:
                logger.warning(f"Codebook not found: {codebook_path}")

        # Add code columns to answer data
        df = dataset.df.copy()

        def get_codes(question):
            return question_to_codes.get(question, [])

        def get_themes(codes):
            if not code_to_theme:
                return []
            themes = set()
            for code in codes:
                theme = code_to_theme.get(code)
                if theme:
                    themes.add(theme)
            return sorted(list(themes))

        df["question_codes"] = df["Question"].apply(get_codes)
        df["n_question_codes"] = df["question_codes"].apply(len)

        if code_to_theme:
            df["question_themes"] = df["question_codes"].apply(get_themes)
            df["n_question_themes"] = df["question_themes"].apply(len)

        # Count matches
        matched = df["n_question_codes"].gt(0).sum()
        total = len(df)

        if verbose:
            logger.info(
                f"Linked {matched:,}/{total:,} answers to question codes "
                f"({matched/total*100:.1f}%)"
            )
            unmatched = total - matched
            if unmatched > 0:
                logger.warning(
                    f"{unmatched:,} answers could not be linked to question codes"
                )

            # Show code distribution
            all_codes = []
            for codes in df["question_codes"]:
                all_codes.extend(codes)
            unique_codes = set(all_codes)
            logger.info(
                f"Answers cover {len(unique_codes)} unique question codes"
            )

        return df

    def get_answers_by_code(
        self,
        df: pd.DataFrame,
        code: str,
    ) -> pd.DataFrame:
        """
        Get all answers to questions that have a specific code.

        Args:
            df: DataFrame with 'question_codes' column (from link_question_codes).
            code: The code to filter by.

        Returns:
            DataFrame with only answers where the question has the specified code.
        """
        if "question_codes" not in df.columns:
            raise ValueError(
                "DataFrame must have 'question_codes' column. "
                "Call link_question_codes() first."
            )

        mask = df["question_codes"].apply(lambda codes: code in codes)
        return df[mask].reset_index(drop=True)

    def get_answers_by_theme(
        self,
        df: pd.DataFrame,
        theme: str,
    ) -> pd.DataFrame:
        """
        Get all answers to questions that have a specific theme.

        Args:
            df: DataFrame with 'question_themes' column (from link_question_codes).
            theme: The theme to filter by.

        Returns:
            DataFrame with only answers where the question has the specified theme.
        """
        if "question_themes" not in df.columns:
            raise ValueError(
                "DataFrame must have 'question_themes' column. "
                "Call link_question_codes() with codebook_path first."
            )

        mask = df["question_themes"].apply(lambda themes: theme in themes)
        return df[mask].reset_index(drop=True)

    def get_code_answer_counts(
        self,
        df: pd.DataFrame,
    ) -> dict[str, int]:
        """
        Get count of answers per question code.

        Args:
            df: DataFrame with 'question_codes' column.

        Returns:
            Dict mapping code name to answer count, sorted by count descending.
        """
        if "question_codes" not in df.columns:
            raise ValueError("DataFrame must have 'question_codes' column.")

        counts = {}
        for codes in df["question_codes"]:
            for code in codes:
                counts[code] = counts.get(code, 0) + 1

        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_theme_answer_counts(
        self,
        df: pd.DataFrame,
    ) -> dict[str, int]:
        """
        Get count of answers per question theme.

        Args:
            df: DataFrame with 'question_themes' column.

        Returns:
            Dict mapping theme name to answer count, sorted by count descending.
        """
        if "question_themes" not in df.columns:
            raise ValueError("DataFrame must have 'question_themes' column.")

        counts = {}
        for themes in df["question_themes"]:
            for theme in themes:
                counts[theme] = counts.get(theme, 0) + 1

        return dict(sorted(counts.items(), key=lambda x: -x[1]))


def load_qa_data(
    folder_path: Union[str, Path],
    verbose: bool = False,
) -> AnswerDataset:
    """
    Convenience function to load Q&A data.

    Args:
        folder_path: Path to folder containing Q&A CSV files.
        verbose: If True, show progress.

    Returns:
        AnswerDataset with combined data.
    """
    loader = AnswerLoader()
    return loader.load(folder_path, verbose=verbose)
