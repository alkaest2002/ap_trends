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
    metadata = {
        "size_before_processing": None,
        "size_after_processing": None,
        "lossy_ops": []
    }

    # Dataset folder
    dataset_folder = Path("./datasets/dataset_2/")

    # Load dataset
    df = pd.read_csv(dataset_folder / "psycarticles.csv")

    # Add info to metadada
    metadata["size_before_processing"] = df.shape[0]

    # Lowercase columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Create publication year
    df["year"] = df.alphadate.str.extract(r"(\d{4})").astype(int)

    # Filter columns
    df = df.loc[:, ["year","publication","title","abstract"]]

    # Create excerpt
    df["excerpt"] = make_excerpt(df)

    # Create text to embed
    df["text_to_embed"] = make_text_to_embed(df, ["title", "excerpt"])

    # Add info to metadada
    metadata["size_after_processing"] = df.shape[0]
    return dataset_folder, df, metadata


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
def _(Path, dataset_folder, df, metadata, orjson):
    # Persist
    df.to_csv(dataset_folder / "psycarticles_cleaned.csv")
    with Path(dataset_folder / "cleanup_recap.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
