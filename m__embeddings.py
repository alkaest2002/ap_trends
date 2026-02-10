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
    TYPE_OF_FAMILY_MODEL = "sentence_transformers"
    TYPE_OF_EMBEDDINGS_MODEL = "BAAI/bge-small-en-v1.5"

    [DATASET_FOLDER, EMBEDDINGS_FOLDER] = get_or_create_folders(TYPE_OF_DOC, TYPE_OF_FAMILY_MODEL, TYPE_OF_EMBEDDINGS_MODEL)[:2]
    DATASET_FOLDER, EMBEDDINGS_FOLDER
    return DATASET_FOLDER, EMBEDDINGS_FOLDER, TYPE_OF_EMBEDDINGS_MODEL


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
def _(TYPE_OF_EMBEDDINGS_MODEL, df, get_sentence_transformer):
    # Compute docs embeddings 
    embeddings = get_sentence_transformer(df.doc.to_list(), TYPE_OF_EMBEDDINGS_MODEL)
    return (embeddings,)


@app.cell
def _(EMBEDDINGS_FOLDER, Path, TYPE_OF_EMBEDDINGS_MODEL, embeddings, np):
    # Persist embeddings
    embedding_model_name_filepath = Path(EMBEDDINGS_FOLDER / "embedding_model_name.txt")
    with embedding_model_name_filepath.open("w") as f:
        f.write(TYPE_OF_EMBEDDINGS_MODEL)

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
