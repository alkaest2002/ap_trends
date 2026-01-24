import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import itertools
    from os import getenv
    from pathlib import Path

    from bertopic import BERTopic
    from bertopic.backend import OpenAIBackend
    from dotenv import load_dotenv
    from openai import OpenAI
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from lib.utils_pandas import get_topics_in_period

    load_dotenv();
    return (
        BERTopic,
        OpenAI,
        OpenAIBackend,
        Path,
        get_topics_in_period,
        getenv,
        np,
        pd,
    )


@app.cell
def _(Path):
    # Define paths
    DATASET_FOLDER = Path("./datasets/dataset_2/")
    MODEL_FOLDER = DATASET_FOLDER / "openai_small"
    EMBEDDING_FOLDER = MODEL_FOLDER / "embeddings"
    BERTOPIC_FOLDER = MODEL_FOLDER / "bertopic"

    # Define other constants
    REDUCE_UNCATEGORIZED = False
    return (
        BERTOPIC_FOLDER,
        DATASET_FOLDER,
        EMBEDDING_FOLDER,
        REDUCE_UNCATEGORIZED,
    )


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset_topic.csv")
    docs = df.doc.to_list()
    df.sample(5)
    return df, docs


@app.cell
def _(EMBEDDING_FOLDER, OpenAI, OpenAIBackend, getenv):
    # Get embedding_model_name
    with (EMBEDDING_FOLDER / "embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()

    # Init embedding model
    client = OpenAI(api_key=getenv("OPENAI_APIKEY"))
    embedding_model = OpenAIBackend(client=client, embedding_model="text-embedding-3-small")
    return (embedding_model,)


@app.cell
def _(
    BERTOPIC_FOLDER,
    BERTopic,
    REDUCE_UNCATEGORIZED,
    docs,
    embedding_model,
    np,
    pd,
):
    # Load BERTopic related files
    topic_model = BERTopic.load(BERTOPIC_FOLDER, embedding_model=embedding_model)
    probs = np.load(file=BERTOPIC_FOLDER / "probs.npy")
    topic_model.update_topics(docs)
    topics = topic_model.topics_

    if REDUCE_UNCATEGORIZED:
        new_topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=0.01)
        topic_model.update_topics(docs, topics=new_topics)
        topics_info = topic_model.get_topic_info()
    else:
        topics_info = pd.read_csv(BERTOPIC_FOLDER / "topic_info.csv")
    return topic_model, topics_info


@app.cell
def _(topic_model, topics_info):
    # Show topics
    topic_labels = topic_model.generate_topic_labels(nr_words=4,topic_prefix=False, separator=",")
    topics_info.insert(4,"Representation_2", topic_labels)
    topics_info.sort_values(by="Topic")
    return


@app.cell
def _(topics_info):
    # Compute number of uncategorized articles
    topics_info.loc[topics_info.Topic.eq(-1), "Count"].div(topics_info.Count.sum()).squeeze()
    return


@app.cell
def _(df, get_topics_in_period, topics_info):
    get_topics_in_period(df, topics_info,(1920,1950), 8)
    return


@app.cell
def _(topics_info):
    topics_info.Representation.str.contains("suicide").sum()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
