import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from lib.utils_embeddings import normalize_model_name
    from lib.bertopic.sentence_transformers.model_base import get_bertopic_model
    return Path, get_bertopic_model, np, pd


@app.cell
def _(Path):
    # Define paths
    DATASET_FOLDER = Path("./dataset/titles_with_excerpts_2/")
    OUT_FOLDER = Path("out") / "sentence_transformers" / "all_MiniLM_L6_v2"
    EMBEDDINGS_FOLDER = OUT_FOLDER / "embeddings"
    BERTOPIC_FOLDER = OUT_FOLDER / "bertopic"

    for folder in {EMBEDDINGS_FOLDER, BERTOPIC_FOLDER}:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
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
def _(topic_info):
    # Count Number of clusters
    "Number of meaningful clusters", topic_info.shape[0]-1
    return


@app.cell
def _(df, topic_info):
    # Compute number of uncategorized articles
    topic_info.loc[:, ["Count"]].sum().rdiv(df.topic.eq(-1).sum()).squeeze()
    return


@app.cell
def _(df):
    # Explore words
    df[df.doc.str.contains("suic")]
    return


@app.cell
def _(df):
    # Explore topics
    df[df.topic.eq(0)]
    return


@app.cell
def _(topic_info):
    topic_info[topic_info.Representation.str[0].str.contains("Attitude")]
    return


if __name__ == "__main__":
    app.run()
