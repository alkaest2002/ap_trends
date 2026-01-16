import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import pandas as pd
    import numpy as np
    from lib.embeddings import get_batch_embeddings
    return Path, get_batch_embeddings, np, pd


@app.cell
def _(Path):
    DATASET_FOLDER = Path("./datasets/dataset_3/")
    EMBEDDINGS_FOLDER = DATASET_FOLDER / "openai-small" / "embeddings"
    return DATASET_FOLDER, EMBEDDINGS_FOLDER


@app.cell
def _(DATASET_FOLDER, pd):
    EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"

    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.shape
    return EMBEDDINGS_MODEL_NAME, df


@app.cell
def _(df):
    df.sample(5, random_state=42)
    return


@app.cell
def _(EMBEDDINGS_MODEL_NAME, df, get_batch_embeddings):
    texts_to_embed = df.doc.to_list()
    _, embeddings = get_batch_embeddings(texts_to_embed, embedding_model_name=EMBEDDINGS_MODEL_NAME)
    return (embeddings,)


@app.cell
def _(EMBEDDINGS_FOLDER, EMBEDDING_MODEL_NAME, Path, embeddings, np):
    embedding_model_name_filepath = Path(EMBEDDINGS_FOLDER / "embedding_model_name.txt")
    with embedding_model_name_filepath.open("w") as f:
        f.write(EMBEDDING_MODEL_NAME)

    embeddings_filepath = Path(EMBEDDINGS_FOLDER / "embeddings.npy")
    np.save(embeddings_filepath, np.array(embeddings))
    return


@app.cell
def _(embeddings, np):
    np.array(embeddings).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
