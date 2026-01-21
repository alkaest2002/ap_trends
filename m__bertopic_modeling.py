import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from lib.bertopic.dataset_2.openai.model_small import get_bertopic_model, default_bertopic_settings
    return Path, get_bertopic_model, np, pd


@app.cell
def _(Path):
    DATASET_FOLDER = Path("./datasets/dataset_2/")
    MODEL_FOLDER = DATASET_FOLDER / "openai_small"
    EMBEDDING_FOLDER = MODEL_FOLDER / "embeddings"
    return DATASET_FOLDER, EMBEDDING_FOLDER, MODEL_FOLDER


@app.cell
def _(DATASET_FOLDER, pd):
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.sample(5)
    return (df,)


@app.cell
def _(df):
    docs = df.doc.to_list()
    docs[0]
    return (docs,)


@app.cell
def _(EMBEDDING_FOLDER):
    embedding_model_name = ""
    with (EMBEDDING_FOLDER / "embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()
    embedding_model_name
    return


@app.cell
def _(EMBEDDING_FOLDER, np):
    embeddings = np.load((EMBEDDING_FOLDER / "embeddings.npy"))
    embeddings.shape
    return (embeddings,)


@app.cell
def _(MODEL_FOLDER, docs, embeddings, get_bertopic_model):
    # Get model
    topic_model = get_bertopic_model()

    # Run model
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Persist model
    topic_model.save(path=MODEL_FOLDER / "bertopic", serialization="safetensors")
    return (topic_model,)


@app.cell
def _(DATASET_FOLDER, df, topic_model):
    # Add topic clusters
    df["topic"] = topic_model.topics_
    df.to_csv(DATASET_FOLDER / "dataset_topic.csv", index=False)
    return


@app.cell
def _(topic_model):
    t = topic_model.get_topic_info()
    return (t,)


@app.cell
def _(t):
    t.iloc[:,:]
    return


@app.cell
def _(df):
    df.loc[df.topic.eq(3)]
    return


@app.cell
def _(df, t):
    t.loc[:, ["Count"]].sum().rdiv(df.topic.eq(-1).sum())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
