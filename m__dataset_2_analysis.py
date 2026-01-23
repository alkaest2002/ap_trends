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
    from sklearn.feature_extraction.text import CountVectorizer

    import numpy as np
    import pandas as pd

    load_dotenv();
    return BERTopic, OpenAI, OpenAIBackend, Path, getenv, np, pd


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
    return (topics_info,)


@app.cell
def _(topics_info):
    # Show topics 
    topics_info.sort_values(by="Topic")
    return


@app.cell
def _(topics_info):
    # Compute number of uncategorized articles
    topics_info.loc[topics_info.Topic.eq(-1), "Count"].div(topics_info.Count.sum()).squeeze()
    return


@app.cell
def _(np, pd):
    def get_topics_in_period(df: pd.DataFrame, topics_info: pd.DataFrame, period: tuple[int, int], max_topics: int=5):

        # Define period mask
        period_mask: pd.Series[np.bool_] = df.year.between(*period)

        # Filter df by period
        df_period = df[period_mask]

        # Find n largest topics
        topics_in_period: list[int] = (
            df_period
                .groupby("topic")
                .size()
                .nlargest(max_topics +1)
                .index
                .to_list()
        )

        final = topics_info.loc[topics_info.Topic.isin(topics_in_period), :].sort_values(by="Topic")

        return final[final.Topic.ne(-1)].iloc[:max_topics, :]
    return (get_topics_in_period,)


@app.cell
def _(df, get_topics_in_period, topics_info):
    get_topics_in_period(df, topics_info,(2016,2025), 10)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
