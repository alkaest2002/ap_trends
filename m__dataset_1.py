import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import orjson
    from lib.utils import make_excerpt, make_text_to_embed
    return Path, make_excerpt, make_text_to_embed, orjson, pd


@app.cell
def _(Path, make_excerpt, make_text_to_embed, pd):
    # Init metadata object
    metadata = {
        "size_before_processing": None,
        "size_after_processing": None,
        "lossy_ops": []
    }

    # Dataset folder
    DATASET_FOLDER = Path("./datasets/dataset_1/")

    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "psycarticles.csv")

    # Add info to metadada
    metadata["size_before_processing"] = df.shape[0]

    # Lowercase column names
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()

    # Lowercase journal names
    df.publication = df.publication.str.lower()

    # Drop articles without either abstract or journal name
    df = df.dropna(subset=["abstract", "publication"], how="any")
    metadata["lossy_ops"].append(["drop empty abstract or empty publication info", df.shape[0]])

    # Clean abstract
    df.abstract = df.abstract.str.replace("abstract|Abstract", "", regex=True)

    # Create publication year
    df["year"] = df.alphadate.str.extract(r"(\d{4})").astype(int)

    # Limit year of publication
    df = df[df.year.ge(1900)]
    metadata["lossy_ops"].append(["publication date greater than 1900", df.shape[0]])

    # Limit journals to most prolific ones (number of publications ge 50) or to those which have aviation in the title
    # Note: decided to drop sustainability journal after inspection of titles
    all_journals = df.publication.value_counts()
    list_most_prolific = all_journals[all_journals.ge(20)].index.drop("sustainability").to_list()
    list_aviation = all_journals[all_journals.index.str.contains("aviation")].index.to_list()
    df = df[df.publication.isin([*list_most_prolific, *list_aviation])]
    metadata["lossy_ops"].append(["limit to most prolific journals or sector journals", df.shape[0]])

    # Filter columns
    df = df.loc[:, ["year", "publication", "title","abstract"]].reset_index(drop=True)

    # Compute excerpt
    df["excerpt"] = make_excerpt(df)

    # Create text to embed (title + excerpt + publication)
    df["doc"] = make_text_to_embed(df=df)

    # Compute final dataset size
    metadata["size_after_processing"] = df.shape[0]

    df.info()
    return DATASET_FOLDER, df, metadata


@app.cell
def _(df):
    df.sample(15, random_state=42)
    return


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(DATASET_FOLDER, Path, df, metadata, orjson):
    # Persist
    df.to_csv(DATASET_FOLDER / "psycarticles_cleaned.csv", index=False)
    with Path(DATASET_FOLDER / "cleanup_recap.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
