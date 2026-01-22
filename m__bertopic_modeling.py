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
    return probs, topic_model, topics


@app.cell
def _(DATASET_FOLDER, df, docs, probs, topic_model, topics):
    reduce_outliers = True
    if reduce_outliers:
        new_topics_ = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=0.01)
        topic_model.update_topics(docs, topics=new_topics_)
    else:
        topic_model.update_topics(docs, topics=topics)
   


    # Add topic clusters
    df["topic"] = topic_model.topics_
    df.to_csv(DATASET_FOLDER / "dataset_topic.csv", index=False)

    # See topics info
    t = topic_model.get_topic_info()
    return (t,)


@app.cell
def _(t):
    t.sort_values(by="Topic")
    return


@app.cell
def _(df, t):
    # Compute the number of uncategorized articles
    t.loc[:, ["Count"]].sum().rdiv(df.topic.eq(-1).sum())
    return


@app.cell
def _(df, np, pd):
    def get_topics_in_period(period: tuple[int, int], max_topics: int=5):
    
        # Define period mask
        period_mask: pd.Series[np.bool_] = df.year.between(*period)

        # Filter df by period
        df_period = df[period_mask]

        # Find n largest topics
        topics_in_period: list[int] = (
            df_period
                .groupby("topic")
                .size()
                .nlargest(max_topics)
                .index
                .to_list()
        )
    
        # Return articole with topic
        return df_period.loc[df_period.topic.isin(topics_in_period), :].sort_values(by="topic")
    return (get_topics_in_period,)


@app.cell
def _(get_topics_in_period):
    get_topics_in_period((2022,2025), 5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
