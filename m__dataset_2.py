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
def _(Path, make_excerpt, make_text_to_embed, pd):
    # Init metadata object
    metadata: dict = {
        "size_before_processing": None,
        "size_after_processing": None,
        "lossy_ops": []
    }

    # Dataset folder
    DATASET_FOLDER = Path("./datasets/dataset_2/")

    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "psycarticles.csv")

    # Add info to metadada
    metadata["size_before_processing"] = df.shape[0]

    # Lowercase columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Create publication year
    df["year"] = df.alphadate.str.extract(r"(\d{4})").astype(int)

    # Drop University Wire publication
    df = df[~df.publication.eq("University Wire")]
    metadata["lossy_ops"].append(["drop University Wire publication", df.shape[0]])

    # Filter columns
    df = df.loc[:, ["year","publication","title","abstract"]]

    # Create excerpt
    df["excerpt"] = make_excerpt(df)

    # Create text to embed
    df["doc"] = make_text_to_embed(df, ["title", "excerpt"])

    # Add info to metadada
    metadata["size_after_processing"] = df.shape[0]
    return DATASET_FOLDER, df, metadata


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.sample(5, random_state=42)
    return


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(DATASET_FOLDER, Path, df, metadata, orjson):
    # Persist
    df.to_csv(DATASET_FOLDER / "psycarticles_cleaned.csv")
    with Path(DATASET_FOLDER / "cleanup_recap.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
