import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    from pathlib import Path
    import orjson
    import spacy
    import pandas as pd
    from lib.utils_base import extract_countries, get_or_create_folders
    from lib.utils_pandas import make_excerpt, make_text_to_embed

    nlp = spacy.load("en_core_web_lg")
    return (
        Path,
        extract_countries,
        get_or_create_folders,
        make_excerpt,
        make_text_to_embed,
        nlp,
        orjson,
        pd,
    )


@app.cell
def _(get_or_create_folders):
    # Define paths
    TYPE_OF_DOC = "title_with_abstract"
    NUM_OF_PARAGRAPH = -1
    [DATASET_FOLDER] = get_or_create_folders(TYPE_OF_DOC)
    DATASET_FOLDER
    return DATASET_FOLDER, NUM_OF_PARAGRAPH


@app.cell
def _(
    NUM_OF_PARAGRAPH,
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

    # Load original dataset
    df = pd.read_csv("./dataset/scopus.csv")
    metadata["size_before_processing"] = df.shape[0]

    # Lowercase all columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Add lowercased title
    df["title_lowercase"] = df.title.str.lower().str.extract(r"^([^\.]+)\.?$")

    # Drop duplicated titles
    df = df.drop_duplicates(subset="title_lowercase")
    metadata["lossy_ops"].append(("Drop duplicate titles", df.shape[0]))  # ty:ignore[possibly-missing-attribute]

    # Compute country
    df["country"] = df.affiliations.apply(extract_countries, nlp_model=nlp)

    # Make excerpt
    df["excerpt"] = make_excerpt(df, column="abstract", num_paragraphs=NUM_OF_PARAGRAPH)

    # Make doc
    df["doc"] = make_text_to_embed(df, ["title", "excerpt"])

    # Filter columns
    metadata["size_after_processing"] = df.shape[0]
    return df, metadata


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(DATASET_FOLDER, Path, df, metadata, orjson):
    # Persist
    df.loc[:, ["year","country","title","doc"]]\
        .to_csv(DATASET_FOLDER / "dataset.csv", index=False)

    with Path(DATASET_FOLDER / "cleanup_recap.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
