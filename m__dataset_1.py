import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    return Path, pd


@app.cell
def _(Path, pd):
    # Init metadata object
    metdata = {}

    # Dataset folder
    dataset_folder = Path("./datasets/dataset_1/")

    # Load dataset
    df = pd.read_csv(dataset_folder / "psycarticles.csv")

    # Add info to metadada
    metdata["size_after_loading"] = df.shape[0]

    # Lowercase column names
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()

    # Lowercase journal names
    df.publication = df.publication.str.lower()

    # Drop articles without either abstract or journal name
    df = df.dropna(subset=["abstract", "publication"], how="any")

    # Add info to metadada
    metdata["size_after_dropping_empty_abstract_or_journal"] = df.shape[0]

    # Clean abstract
    df.abstract = df.abstract.str.replace("abstract|Abstract", "", regex=True)

    # Create publication year
    df["year"] = df.alphadate.str.extract(r"(\d{4})").astype(int)

    # Limit year of publication
    df = df[df.year.ge(1900)]

    # Add dataset original size to metadada
    metdata["size_after_limiting_year_of_publication"] = df.shape[0]

    # Limit journals to most prolific ones (number of publications ge 50) or to those which have aviation in the title
    # Note: decided to drop sustainability journal after inspection of titles
    all_journals = df.publication.value_counts()
    list_most_prolific = all_journals[all_journals.ge(20)].index.drop("sustainability").to_list()
    list_aviation = all_journals[all_journals.index.str.contains("aviation")].index.to_list()
    df = df[df.publication.isin([*list_most_prolific, *list_aviation])]

    # Add dataset original size to metadada
    metdata["size_after_keeping_most_prolific_journals_or_sector_journals"] = df.shape[0]

    # Filter columns
    df = df.loc[:, ["year", "publication", "title","abstract"]].reset_index(drop=True)

    # Dictionary of abbreviations to clean up
    abbrev_dict = {
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

    # Apply all cleanups
    df["excerpt"] = df["abstract"].fillna("")
    for pattern, replacement in abbrev_dict.items():
        df["excerpt"] = df["excerpt"].str.replace(pattern, replacement, regex=True)

    # Create excerpt (hopefully the first few abstract paragraphs)
    NUNBER_OF_PARA_TO_KEEP = 3
    df["excerpt"] = (
        df["excerpt"]
            .str.split(r"\.\s+", regex=True)
            .str[:NUNBER_OF_PARA_TO_KEEP]
            .str.join(". ")
            .add(".")
    )

    # Create text to embed (title + excerpt + publication)
    df["text_to_embed"] = (
        df.title.radd("<title>").add("</title>") 
            + df.excerpt.radd("<excerpt>").add("</excerpt>")
            + df.publication.radd("<journal>").add("</journal>")
    )

    df.info()
    return dataset_folder, df, metdata


@app.cell
def _(df):
    df.sample(15)
    return


@app.cell
def _(metdata):
    metdata
    return


@app.cell
def _(dataset_folder, df):
    df.to_csv(dataset_folder / "psycarticles_cleaned.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
