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
def _(pd):
    EMBEDDING_MODEL_NAME = "text-embedding-3-large"
    df = pd.read_csv("./data/psycarticles_cleaned.csv")
    df.shape
    return EMBEDDING_MODEL_NAME, df


@app.cell
def _(EMBEDDING_MODEL_NAME, df, get_batch_embeddings):
    texts_to_embed = df.doc.to_list()
    _, embeddings = get_batch_embeddings(texts_to_embed, embedding_model_name=EMBEDDING_MODEL_NAME)
    return (embeddings,)


@app.cell
def _(EMBEDDING_MODEL_NAME, Path, embeddings, np):
    embedding_model_name_filepath = Path("./data/embeddings/embedding_model_name.txt")
    with embedding_model_name_filepath.open("w") as f:
        f.write(EMBEDDING_MODEL_NAME)

    embeddings_filepath = Path("./data/embeddings/embeddings.npy")
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
