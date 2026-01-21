import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import orjson
    import pycountry
    import spacy
    import numpy as np
    import pandas as pd
    from lib.utils_base import extract_countries
    from lib.utils_pandas import make_excerpt, make_text_to_embed
    from langdetect import detect

    nlp = spacy.load("en_core_web_lg")
    return (
        Path,
        extract_countries,
        make_excerpt,
        make_text_to_embed,
        nlp,
        orjson,
        pd,
    )


@app.cell
def _(Path):
    DATASET_FOLDER = Path("./datasets/dataset_2/")
    return (DATASET_FOLDER,)


@app.cell
def _(
    DATASET_FOLDER,
    extract_countries,
    make_excerpt,
    make_text_to_embed,
    nlp,
    pd,
):
    # Init metadata object
    metadata = {
        "size_before_processing": None,
        "size_after_processing": None,
        "lossy_ops": []
    }

    df = pd.read_csv(DATASET_FOLDER / "scopus.csv")

    metadata["size_before_processing"] = df.shape[0]

    # Lowercase columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df["country"] = df.affiliations.apply(extract_countries, nlp_model=nlp)

    # add lowercased title
    df["title_lowercase"] = df.title.str.lower().str.extract(r"^([^\.]+)\.?$")

    # drop duplicated titles
    df = df.drop_duplicates(subset="title_lowercase")
    metadata["lossy_ops"].append(("Drop duplicate title", df.shape[0]))

    # Make excerpt
    df["excerpt"] = make_excerpt(df, "abstract")

    # Make doc
    df["doc"] = make_text_to_embed(df, ["title","excerpt"])

    # Filter columns
    df = df.loc[:, ["year","country","affiliations","title","doc"]]

    metadata["size_after_processing"] = df.shape[0]
    return df, metadata


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(DATASET_FOLDER, Path, df, metadata, orjson):
    # Persist
    df.to_csv(DATASET_FOLDER / "dataset.csv", index=False)
    with Path(DATASET_FOLDER / "cleanup_recap.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
