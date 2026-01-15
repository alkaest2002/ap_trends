from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable


def make_excerpt(df: pd.DataFrame, column: str = "abstract", num_paragraphs: int = 3) -> pd.Series:
    """Create excerpt from column by cleaning up common abbreviations and limiting to first few paragraphs.

    Args:
        df (pd.DataFrame): DataFrame containing the abstracts.
        column (str): Name of the column containing the abstracts (default is "abstract").
        num_paragraphs (int): Number of paragraphs to include in the excerpt (default is 3).

    Returns:
        pd.Series: Series containing the excerpts.

    """
    # Defualt column to abstract if not specified
    if not column:
        column = "abstract"

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
    excerpt: pd.Series = df[column].fillna("")

    # Apply abbreviation replacements
    for pattern, replacement in abbrev_dict.items():
        excerpt = excerpt.str.replace(pattern, replacement, regex=True)

    return (
       excerpt
            .str.split(r"\.\s+", regex=True)
            .str[:num_paragraphs]
            .str.join(". ")
            .add(".")
    )
