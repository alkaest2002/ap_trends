import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import pandas as pd
    import numpy as np
    from lib.utils_embeddings import get_sentence_transformer, normalize_model_name
    return Path, get_sentence_transformer, normalize_model_name, np, pd


@app.cell
def _():
    BASE_MODEL = "all-MiniLM-L6-v2"
    SPECTER2_PUBMED = "wwydmanski/specter2_pubmed-v0.7-full"
    return (SPECTER2_PUBMED,)


@app.cell
def _(Path, SPECTER2_PUBMED, normalize_model_name):
    DATASET_FOLDER = Path("./dataset/titles_with_excerpts_2/")
    EMBEDDINGS_MODEL_NAME = SPECTER2_PUBMED

    EMBEDDINGS_FOLDER = Path("out") / "sentence_transformers" / normalize_model_name(EMBEDDINGS_MODEL_NAME) /  "embeddings"
    if not EMBEDDINGS_FOLDER.exists():
        EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)
    return DATASET_FOLDER, EMBEDDINGS_FOLDER, EMBEDDINGS_MODEL_NAME


@app.cell
def _(DATASET_FOLDER, pd):
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.shape
    return (df,)


@app.cell
def _(df):
    df.sample(5, random_state=42)
    return


@app.cell
def _(EMBEDDINGS_MODEL_NAME, df, get_sentence_transformer):
    texts_to_embed = df.doc.to_list()
    embeddings = get_sentence_transformer(texts_to_embed, EMBEDDINGS_MODEL_NAME)
    return (embeddings,)


@app.cell
def _(EMBEDDINGS_FOLDER, EMBEDDINGS_MODEL_NAME, Path, embeddings, np):
    embedding_model_name_filepath = Path(EMBEDDINGS_FOLDER / "embedding_model_name.txt")
    with embedding_model_name_filepath.open("w") as f:
        f.write(EMBEDDINGS_MODEL_NAME)

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
