import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import orjson
    import pycountry
    import spacy
    import pandas as pd
    from lib.utils_pandas import make_excerpt, make_text_to_embed
    from langdetect import detect
    return Path, pd


@app.cell
def _(Path):
    DATASET_FOLDER = Path("./datasets/dataset_2/")
    return (DATASET_FOLDER,)


@app.cell
def _(DATASET_FOLDER, pd):
    # Init metadata object
    metadata = {
        "size_before_processing": None,
        "size_after_processing": None,
        "lossy_ops": []
    }

    df = pd.read_csv(DATASET_FOLDER / "scopus.csv")

    # Lowercase columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
