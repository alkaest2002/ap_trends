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
    return (docs,)


@app.cell
def _(docs, embeddings, get_bertopic_model):
    # Get BERTopic model
    topic_model = get_bertopic_model()

    # Fit BERTopic model
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return (topic_model,)


@app.cell
def _(docs, topic_model):
    # Get original topic info
    topic_info_original = topic_model.get_topic_info()
    topic_info_original["theme"] = topic_info_original["Representation"].str[0]

    # define consolidation function
    def consoldation_fn(x, topic_model, docs):
        topics =  [t for t in x if t != -1]
        if len(topics) > 1:
            print(f"consolidating {topics}")
            topic_model.merge_topics(docs, topics)

    # Consolidate duplicated themes
    topcis_list = topic_info_original.groupby("theme").Topic.agg(list)
    topcis_list[topcis_list.str.len().gt(1)].apply(consoldation_fn, topic_model=topic_model, docs=docs)

    # Get updated topic info
    topic_info = topic_model.get_topic_info()
    topic_info.head()
    return (topic_info,)


@app.cell
def _(topic_info):
    # Count Number of clusters
    "Numero di temi", topic_info.shape[0]-1, "Articoli non categorizzati", topic_info.iloc[0].Count / topic_info.Count.sum()
    return


@app.cell
def _(df, topic_model):
    # update topics in df
    df["topic"] = topic_model.topics_
    return


@app.cell
def _(topic_info):
    topic_info[topic_info.Representation.str[0].str.contains('Pilot', na=False)]
    return


@app.cell
def _(df):
    # Explore words
    df[df.doc.str.contains("suic")]
    return


@app.cell
def _(df):
    # Explore topics
    df[df.topic.isin([18])]
    return


@app.cell
def _(BERTOPIC_FOLDER, np, topic_model):
    # Persist BERTopic model
    topic_model.save(path=BERTOPIC_FOLDER, serialization="safetensors")

    # Persist probabilities
    np.save(BERTOPIC_FOLDER / "probs.npy", topic_model.probabilities_)
    return


@app.cell
def _(BERTOPIC_FOLDER, DATASET_FOLDER, df, topic_model):
    # Persist dataset with topics
    df.to_csv(DATASET_FOLDER / "dataset_topic.csv", index=False)

    # Persist topics info
    topic_info_final = topic_model.get_topic_info()
    topic_info_final.to_csv(BERTOPIC_FOLDER / "topic_info.csv", index=False)
    return


@app.cell
def _():
    print("finish")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
