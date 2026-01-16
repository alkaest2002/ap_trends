import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import orjson
    import pandas as pd
    from lib.utils import make_excerpt, make_text_to_embed
    return Path, make_excerpt, make_text_to_embed, orjson, pd


@app.cell
def _(Path):
    DATASET_FOLDER = Path("./datasets/dataset_3/")
    return (DATASET_FOLDER,)


@app.cell
def _(DATASET_FOLDER, make_excerpt, make_text_to_embed, pd):
    # Init metadata object
    metadata = {
        "size_before_processing": None,
        "size_after_processing": None,
        "lossy_ops": []
    }

    # Load datasets
    s1 = pd.read_csv(DATASET_FOLDER / "scopus_1.csv")
    s2 = pd.read_csv(DATASET_FOLDER / "scopus_2.csv")
    p1 = pd.read_csv(DATASET_FOLDER / "psycarticles.csv")

    # Standardize columns
    s1 = s1.loc[:, ["Year", "Source title","Title", "Abstract"]]
    s2 = s2.loc[:, ["Year", "Source title","Title", "Abstract"]]
    s1.columns = "year","publication","title","abstract"
    s2.columns = "year","publication","title","abstract"

    p1 = p1.loc[:, ["AlphaDate","Publication","Title","Abstract"]]
    p1.columns = ["year","publication","title","abstract"]
    p1["year"] = p1.year.str.extract(r"(\d{4})").astype(int)

    # Combine datasets
    df = pd.concat([s1,s2,p1])
    metadata["size_before_processing"] = df.shape[0]

    # Drop duplicate titles
    df = df.drop_duplicates(subset="title")
    metadata["lossy_ops"].append(("drop duplicate titles", df.shape[0]))

    # replace [No abstract available] with None
    df.abstract = df.abstract.replace("[No abstract available]", None)

    # Create excerpt
    df["excerpt"] = make_excerpt(df)

    # Create text to embed
    df["doc"] = make_text_to_embed(df, ["title", "excerpt"])

    # Update metadata
    metadata["size_after_processing"] = df.shape[0]
    return df, metadata


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(df):
    df.sample(5)
    return


@app.cell
def _(DATASET_FOLDER, Path, df, metadata, orjson):
    # Persist
    df.to_csv(DATASET_FOLDER / "dataset_cleaned.csv")
    with Path(DATASET_FOLDER / "cleanup_recap.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
