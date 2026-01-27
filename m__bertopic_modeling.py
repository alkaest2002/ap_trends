import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from lib.bertopic.dataset_2.openai.model_small import get_bertopic_model, default_bertopic_settings
    return Path, get_bertopic_model, np, pd


@app.cell
def _(Path):
    # Define paths
    DATASET_FOLDER = Path("./datasets/dataset_2/openai_small/titles_with_excerpts_2/")
    EMBEDDINGS_FOLDER = DATASET_FOLDER / "embeddings"
    BERTOPIC_FOLDER = DATASET_FOLDER / "bertopic"
    return BERTOPIC_FOLDER, DATASET_FOLDER, EMBEDDINGS_FOLDER


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.sample(5)
    return (df,)


@app.cell
def _(EMBEDDINGS_FOLDER):
    # Get embedding_model_name
    with (EMBEDDINGS_FOLDER / "embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()
    embedding_model_name
    return


@app.cell
def _(EMBEDDINGS_FOLDER, np):
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FOLDER / "embeddings.npy")
    embeddings.shape
    return (embeddings,)


@app.cell
def _(df):
    # Get Docs
    docs = df.doc.to_list()
    return


@app.cell
def _(df, embeddings, get_bertopic_model):
    # Get BERTopic model
    topic_model = get_bertopic_model()

    # Fit BERTopic model
    topics, probs = topic_model.fit_transform(df.doc.to_list(), embeddings=embeddings)
    return probs, topic_model, topics


@app.cell
def _(BERTOPIC_FOLDER, DATASET_FOLDER, df, np, probs, topic_model, topics):
    # Persist BERTopic model
    topic_model.save(path=BERTOPIC_FOLDER, serialization="safetensors")

    # Persist probabilities
    np.save(BERTOPIC_FOLDER / "probs.npy", probs)

    # Add topics to dataset
    df["topic"] = topics

    # Persist dataset with topics
    df.to_csv(DATASET_FOLDER / "dataset_topic.csv", index=False)

    # Persist topics info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(BERTOPIC_FOLDER / "topic_info.csv", index=False)
    return (topic_info,)


@app.cell
def _(topic_info):
    # Show topics 
    topic_info.sort_values(by="Topic")
    return


@app.cell
def _(df, topic_info):
    # Compute number of uncategorized articles
    topic_info.loc[:, ["Count"]].sum().rdiv(df.topic.eq(-1).sum()).squeeze()
    return


@app.cell
def _(df):
    df[df.doc.str.contains("suic")]
    return


@app.cell
def _(df):
    df[df.topic.eq(65)]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
