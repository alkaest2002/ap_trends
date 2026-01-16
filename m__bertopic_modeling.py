import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from lib.bertopic.model_dataset_3 import get_bertopic_model, default_bertopic_settings
    return Path, default_bertopic_settings, get_bertopic_model, np, pd


@app.cell
def _(Path, pd):
    DATASET_FOLDER = Path("./datasets/dataset_3/")
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.head()
    return DATASET_FOLDER, df


@app.cell
def _(df):
    docs = df.doc.to_list()
    docs[0]
    return (docs,)


@app.cell
def _(DATASET_FOLDER):
    embedding_model_name = ""
    with (DATASET_FOLDER / "embeddings/embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()
    embedding_model_name
    return


@app.cell
def _(DATASET_FOLDER, np):
    embeddings = np.load((DATASET_FOLDER / "embeddings/embeddings.npy"))
    embeddings.shape
    return (embeddings,)


@app.cell
def _(default_bertopic_settings):
    default_bertopic_settings
    return


@app.cell
def _(DATASET_FOLDER, docs, embeddings, get_bertopic_model):
    # Get model
    topic_model = get_bertopic_model()

    # Run model
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Persist model
    topic_model.save(path=DATASET_FOLDER / "bertopic", serialization="safetensors")
    return (topic_model,)


@app.cell
def _(DATASET_FOLDER, df, topic_model):
    # Add topic clusters
    df["topic"] = topic_model.topics_
    df.to_csv(DATASET_FOLDER / "dataset_topic.csv", index=False)
    return


@app.cell
def _(topic_model):
    topic_model.get_topic_info()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
