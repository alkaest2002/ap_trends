import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    from lib.utils import make_excerpt
    return Path, make_excerpt, pd


@app.cell
def _(Path, make_excerpt, pd):
    # Init metadata object
    metdata = {}

    # Dataset folder
    dataset_folder = Path("./datasets/dataset_2/")

    # Load dataset
    df = pd.read_csv(dataset_folder / "psycarticles.csv")

    # Add info to metadada
    metdata["size_after_loading"] = df.shape[0]

    # Lowercase columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Create publication year
    df["year"] = df.alphadate.str.extract(r"(\d{4})").astype(int)

    # Filter columns
    df = df.loc[:, ["year","publication","title","abstract"]]

    # Create excerpt
    make_excerpt(df)
    return df, metdata


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(metdata):
    metdata
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
