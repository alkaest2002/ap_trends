from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable


def check_columns_(df: pd.DataFrame, columns: list[str]) -> None:
    """Check if specified columns are present in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to check.
        columns (list[str]): List of column names to check for.

    Raises:
        ValueError: If any of the specified columns are not present in the DataFrame.

    """
    # Check for missing columns
    missing_cols = set(columns) - set(df.columns)

    # Raise error if any columns are missing
    if missing_cols:
        error_msg: str = f"The following columns are missing from the DataFrame: {', '.join(missing_cols)}"
        raise ValueError(error_msg)


def make_excerpt(
        df: pd.DataFrame,
        column: str = "abstract",
        num_paragraphs: int = 3,
    ) -> pd.Series:
    """Create excerpt from column by cleaning up common abbreviations and limiting to first few paragraphs.

    Args:
        df (pd.DataFrame): DataFrame containing the abstracts.
        column (str): Name of the column containing the abstracts (default is "abstract").
        num_paragraphs (int): Number of paragraphs to include in the excerpt (default is 3).

    Returns:
        pd.Series: Series containing the excerpts.

    Raises:
        ValueError: If the specified column is not present in the DataFrame.

    """
    # Defualt column to abstract if not specified
    if not column:
        column = "abstract"

    # Check if column is present in df
    check_columns_(df, [column])

    # Dictionary of abbreviations to clean up
    abbrev_dict: dict[str, str | Callable] = {
        r"\bet al\.": "et al",
        r"\be\.g\.": "eg",
        r"\bi\.e\.": "ie",
        r"\bcf\.": "cf",
        r"\bviz\.": "viz",
        r"\bvs\.": "vs",
        r"\bca\.": "ca",
        r"\bc\.": "c",
        r"\bibid\.": "ibid",
        r"\bop\. cit\.": "op cit",
        r"\bloc\. cit\.": "loc cit",
        r"\bq\.v\.": "qv",
        r"\b[Nn]\.?[Bb]\.": "NB",
        r"\b[Pp]\.?[Ss]\.": "PS",
        r"\bff\.": "ff",
        r"\bpp\.": "pp",
        r"\bvols?\.": lambda m: m.group().replace(".", ""),
        r"\beds?\.": lambda m: m.group().replace(".", ""),
        r"\btrans\.": "trans",
        r"\brev\.": "rev",
        r"\brepr\.": "repr"
    }

    # Fill NaN values with empty strings
    excerpt: pd.Series = df[column].replace("[No abstract available]", np.nan).fillna("")

    # Apply abbreviation replacements
    for pattern, replacement in abbrev_dict.items():
        excerpt = excerpt.str.replace(pattern, replacement, regex=True)

    return (
       excerpt
            .str.split(r"\.\s+", regex=True)
            .str[:num_paragraphs]
            .str.join(". ")
            .add(".")
            .str.replace(r"^\.$", "", regex=True)
    )


def make_text_to_embed(df: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """Prepare text for embedding by filling NaN values.

    Args:
        df (pd.DataFrame): DataFrame containing the abstracts.
        columns (list[str]): List of column names to be combined for embedding.

    Returns:
        pd.Series: Series containing the text ready for embedding.

    Raises:
        ValueError: If any of the specified columns are not present in the DataFrame.

    """
    # If no columns specified
    if not columns:
        # Default columns to title and abstract if none specified
        columns = ["title", "excerpt"]

    # Raise error if columns are not present in df (use pd.index intersection)
    check_columns_(df, columns)

    # Fill NaN values with empty strings
    text_to_embed: pd.DataFrame = df[columns].fillna("")

    # Iterate over columns
    for col in columns:
        # Wrap the text in tags
        text_to_embed[col] = f"<{col}>" + text_to_embed[col] + f"</{col}>"
        # Remove empty tags
        text_to_embed[col] = text_to_embed[col].str.replace(rf"<{col}>\.?</{col}>", "", regex=True)

    return text_to_embed.agg(" ".join, axis=1).str.strip()


def get_topics_in_period(
        df: pd.DataFrame,
        topics_info: pd.DataFrame,
        period: tuple[int, int],
        max_topics: int = 5
    ) -> pd.DataFrame:
    """Get the most prevalent topics in a specified time period.

    Args:
        df (pd.DataFrame): DataFrame containing the topic assignments.
        topics_info (pd.DataFrame): DataFrame containing the topic information.
        period (tuple[int, int]): Tuple specifying the start and end year of the period.
        max_topics (int): Maximum number of topics to return (default is 5).

    Returns:
        pd.DataFrame: DataFrame containing the topics in the specified period.

    """
    # Define period mask
    period_mask: pd.Series[np.bool_] = df.year.between(*period)

    # Filter df by period
    df_period: pd.DataFrame = df[period_mask]

    # Find n largest topics
    topics_in_period: list[int] = (
        df_period
            .groupby("topic")
            .size()
            # Get the most prevalent topics
            # Increase by one, in case -1 topic (uncategorized) is present
            .nlargest(max_topics + 1)
            .drop(-1, errors="ignore")
            .index
            .to_list()
    )

    # Get topic info for topics in period
    # Exclude -1 Topic (uncategorized)
    return (
        topics_info
            .loc[topics_info.Topic.isin(topics_in_period), :]
            .iloc[:max_topics, :]
    )
