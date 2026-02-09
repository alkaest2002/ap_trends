import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    from pathlib import Path

    import pandas as pd
    import numpy as np

    from lib.utils_embeddings import get_sentence_transformer
    from lib.utils_base import get_or_create_folders
    return Path, get_or_create_folders, get_sentence_transformer, np, pd


@app.cell
def _(get_or_create_folders):
    # Define Paths
    TYPE_OF_DOC = "title_with_excerpt_2"
    EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
    [DATASET_FOLDER, EMBEDDINGS_FOLDER] = get_or_create_folders(TYPE_OF_DOC, EMBEDDINGS_MODEL_NAME)[:2]
    DATASET_FOLDER, EMBEDDINGS_FOLDER
    return DATASET_FOLDER, EMBEDDINGS_FOLDER, EMBEDDINGS_MODEL_NAME


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.shape
    return (df,)


@app.cell
def _(df):
    # Sample dataset
    df.sample(5)
    return


@app.cell
def _(EMBEDDINGS_MODEL_NAME, df, get_sentence_transformer):
    # Compute docs embeddings 
    embeddings = get_sentence_transformer(df.doc.to_list(), EMBEDDINGS_MODEL_NAME)
    return (embeddings,)


@app.cell
def _(EMBEDDINGS_FOLDER, EMBEDDINGS_MODEL_NAME, Path, embeddings, np):
    # Persist embeddings
    embedding_model_name_filepath = Path(EMBEDDINGS_FOLDER / "embedding_model_name.txt")
    with embedding_model_name_filepath.open("w") as f:
        f.write(EMBEDDINGS_MODEL_NAME)

    embeddings_filepath = Path(EMBEDDINGS_FOLDER / "embeddings.npy")
    np.save(embeddings_filepath, np.array(embeddings))
    return


@app.cell
def _(embeddings, np):
    # Show embeddings dim
    np.array(embeddings).shape
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
