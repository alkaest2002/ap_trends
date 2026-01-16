from typing import TYPE_CHECKING

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


def make_excerpt(df: pd.DataFrame, column: str = "abstract", num_paragraphs: int = 3) -> pd.Series:
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


def get_psychology_sections_list() -> list[str]:
    """Get a list of psychology-related sections.

    Returns:
        list[str]: List of psychology-related section names.

    Notes:
        see https://www.apa.org/pubs/databases/training/class-codes

    """
    return [
        "general psychology history & systems",
        "psychometrics & statistics & methodology tests & testing",
        "psychometrics & statistics & methodology sensory & motor testing",
        "psychometrics & statistics & methodology developmental scales & schedules",
        "psychometrics & statistics & methodology personality scales & inventories",
        "psychometrics & statistics & methodology clinical psychological testing",
        "psychometrics & statistics & methodology neuropsychological assessment",
        "psychometrics & statistics & methodology health psychology testing",
        "psychometrics & statistics & methodology educational measurement",
        "psychometrics & statistics & methodology occupational & employment testing",
        "psychometrics & statistics & methodology consumer opinion & attitude testing",
        "psychometrics & statistics & methodology statistics & mathematics",
        "psychometrics & statistics & methodology research methods & experimental design",
        "human experimental psychology sensory perception",
        "human experimental psychology visual perception",
        "human experimental psychology auditory & speech perception",
        "human experimental psychology motor processes",
        "human experimental psychology cognitive processes",
        "human experimental psychology learning & memory",
        "human experimental psychology attention",
        "human experimental psychology motivation & emotion",
        "human experimental psychology consciousness states",
        "human experimental psychology psychic phenomenon & paranormal activities",
        "animal experimental & comparative psychology learning & motivation",
        "animal experimental & comparative psychology social & instinctive behavior",
        "physiological psychology & neuroscience genetics",
        "physiological psychology & neuroscience neuropsychology & neurology",
        "physiological psychology & neuroscience electrophysiology",
        "physiological psychology & neuroscience physiological processes",
        "physiological psychology & neuroscience psychophysiology",
        "physiological psychology & neuroscience psychopharmacology",
        "psychology & the humanities literature & fine arts",
        "psychology & the humanities philosophy",
        "communication systems linguistics & language & speech",
        "communication systems mass media communications",
        "developmental psychology cognitive & perceptual development",
        "developmental psychology psychosocial & personality development",
        "developmental psychology aging & older adult development",
        "social processes & social issues social structure & organization",
        "social processes & social issues religion",
        "social processes & social issues culture & ethnology",
        "social processes & social issues marriage & family",
        "social processes & social issues divorce & remarriage",
        "social processes & social issues childrearing & child care",
        "social processes & social issues political processes & political issues",
        "social processes & social issues sex & gender roles",
        "social processes & social issues sexual behavior & sexual orientation",
        "social processes & social issues drug & alcohol usage (legal)",
        "social psychology group & interpersonal processes",
        "social psychology social perception & cognition",
        "personality psychology personality traits & processes",
        "personality psychology personality theory",
        "personality psychology psychoanalytic theory",
        "psychological & physical disorders psychological disorders",
        "psychological & physical disorders affective disorders",
        "psychological & physical disorders schizophrenia & psychotic states",
        "psychological & physical disorders anxiety disorders",
        "psychological & physical disorders personality disorders",
        "psychological & physical disorders behavior disorders & antisocial behavior",
        "psychological & physical disorders substance abuse & addiction",
        "psychological & physical disorders criminal behavior & juvenile delinquency",
        "psychological & physical disorders neurodevelopmental & autism spectrum disorders",
        "psychological & physical disorders learning disorders",
        "psychological & physical disorders intellectual developmental disorders",
        "psychological & physical disorders eating disorders",
        "psychological & physical disorders communication disorders",
        "psychological & physical disorders environmental toxins & health",
        "psychological & physical disorders physical & somatic disorders",
        "psychological & physical disorders immunological disorders",
        "psychological & physical disorders cancer",
        "psychological & physical disorders cardiovascular disorders",
        "psychological & physical disorders neurological disorders & brain damage",
        "psychological & physical disorders vision & hearing & sensory disorders",
        "health & mental health treatment & prevention psychotherapy & psychotherapeutic counseling",
        "health & mental health treatment & prevention cognitive therapy",
        "health & mental health treatment & prevention behavior therapy & behavior modification",
        "health & mental health treatment & prevention group & family therapy",
        "health & mental health treatment & prevention interpersonal & client centered & humanistic therapy",
        "health & mental health treatment & prevention psychoanalytic therapy",
        "health & mental health treatment & prevention clinical psychopharmacology",
        "health & mental health treatment & prevention specialized interventions",
        "health & mental health treatment & prevention clinical hypnosis",
        "health & mental health treatment & prevention self help groups",
        "health & mental health treatment & prevention lay & paraprofessional & pastoral counseling",
        "health & mental health treatment & prevention art & music & movement therapy",
        "health & mental health treatment & prevention health psychology & medicine",
        "health & mental health treatment & prevention behavioral & psychological treatment of physical illness",
        "health & mental health treatment & prevention medical treatment of physical illness",
        "health & mental health treatment & prevention promotion & maintenance of health & wellness",
        "health & mental health treatment & prevention health & mental health services",
        "health & mental health treatment & prevention outpatient services",
        "health & mental health treatment & prevention community & social services",
        "health & mental health treatment & prevention home care & hospice",
        "health & mental health treatment & prevention nursing homes & residential care",
        "health & mental health treatment & prevention inpatient & hospital services",
        "health & mental health treatment & prevention rehabilitation",
        "health & mental health treatment & prevention drug & alcohol rehabilitation",
        "health & mental health treatment & prevention occupational & vocational rehabilitation",
        "health & mental health treatment & prevention speech & language therapy",
        "health & mental health treatment & prevention criminal rehabilitation & penology",
        "health & mental health personnel issues professional education & training",
        "health & mental health personnel issues professional personnel attitudes & characteristics",
        "health & mental health personnel issues professional ethics & standards & liability",
        "health & mental health personnel issues professional impairment",
        "educational & school psychology educational administration & personnel",
        "educational & school psychology curriculum & programs & teaching methods",
        "educational & school psychology academic learning & achievement",
        "educational & school psychology classroom dynamics & student adjustment & attitudes",
        "educational & school psychology special & compensatory education",
        "educational & school psychology gifted & talented",
        "educational & school psychology educational/vocational counseling & student services",
        "organizational psychology & human resources occupational interests & guidance",
        "organizational psychology & human resources personnel management & selection & training",
        "organizational psychology & human resources personnel evaluation & job performance",
        "organizational psychology & human resources management & management training",
        "organizational psychology & human resources personnel attitudes & job satisfaction",
        "organizational psychology & human resources organizational behavior",
        "organizational psychology & human resources working conditions & industrial safety",
        "sport psychology & leisure sports & exercise",
        "sport psychology & leisure recreation & leisure",
        "consumer psychology consumer attitudes & behavior",
        "consumer psychology marketing & advertising",
        "engineering & environmental psychology human factors engineering",
        "engineering & environmental psychology lifespace & institutional design",
        "engineering & environmental psychology community & environmental planning",
        "engineering & environmental psychology environmental issues & attitudes",
        "engineering & environmental psychology transportation",
        "cognitive psychology & intelligent systems artificial intelligence & expert systems",
        "cognitive psychology & intelligent systems robotics",
        "cognitive psychology & intelligent systems neural networks",
        "forensic psychology & legal issues civil rights & civil law",
        "forensic psychology & legal issues criminal law & criminal adjudication",
        "forensic psychology & legal issues mediation & conflict resolution",
        "forensic psychology & legal issues crime prevention",
        "forensic psychology & legal issues police & legal personnel"
    ]
