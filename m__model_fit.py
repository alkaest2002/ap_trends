import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from importlib import import_module
    from lib.utils_base import get_or_create_folders, normalize_model_name
    from lib.bertopic.model_base import get_bertopic_model, get_bertopic_settings
    return Path, get_bertopic_model, get_or_create_folders, json, np, pd


@app.cell
def _(get_or_create_folders):
    # Define paths
    TYPE_OF_DOC = "title_with_excerpt_2"
    TYPE_OF_FAMILY_MODEL = "sentence_transformers"
    TYPE_OF_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    [
        DATASET_FOLDER,
        OUT_FOLDER,
        EMBEDDINGS_FOLDER,
        BERTOPIC_FOLDER
    ] = get_or_create_folders(TYPE_OF_DOC,TYPE_OF_FAMILY_MODEL, TYPE_OF_EMBEDDINGS_MODEL)[:4]

    DATASET_FOLDER, OUT_FOLDER, EMBEDDINGS_FOLDER, BERTOPIC_FOLDER
    return (
        BERTOPIC_FOLDER,
        DATASET_FOLDER,
        EMBEDDINGS_FOLDER,
        OUT_FOLDER,
        TYPE_OF_EMBEDDINGS_MODEL,
        TYPE_OF_FAMILY_MODEL,
    )


@app.cell
def _(DATASET_FOLDER, pd):
    # Load dataset
    df = pd.read_csv(DATASET_FOLDER / "dataset.csv")
    df.sample(5)
    return (df,)


@app.cell
def _(EMBEDDINGS_FOLDER, TYPE_OF_EMBEDDINGS_MODEL):
    # Get embedding_model_name
    with (EMBEDDINGS_FOLDER / "embedding_model_name.txt").open("r") as f:
        embedding_model_name = f.read()
    embedding_model_name, TYPE_OF_EMBEDDINGS_MODEL
    return


@app.cell
def _(EMBEDDINGS_FOLDER, np):
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FOLDER / "embeddings.npy")
    embeddings.shape
    return (embeddings,)


@app.cell
def _(TYPE_OF_EMBEDDINGS_MODEL, TYPE_OF_FAMILY_MODEL, get_bertopic_model):
    topic_model = get_bertopic_model(TYPE_OF_FAMILY_MODEL, TYPE_OF_EMBEDDINGS_MODEL)
    return (topic_model,)


@app.cell
def _(df):
    # Get Docs
    docs = df.doc.to_list()
    return (docs,)


@app.cell
def _(docs, embeddings, topic_model):
    # Fit BERTopic model
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Get original topic info
    topic_info_original = topic_model.get_topic_info()

    # Add theme
    topic_info_original["theme"] = topic_info_original["Representation"].str[0].str.lower().str.strip()
    return (topic_info_original,)


@app.cell
def _(BERTOPIC_FOLDER, docs, np, topic_info_original, topic_model):
    # List-aggregate by theme
    theme_lists = topic_info_original.groupby("theme").Topic.agg(list)

    # Get duplicated themes (theme lists longhe than 1)
    duplicated_themes = theme_lists[theme_lists.str.len().gt(1)].to_list()

    # Omit -1 in any theme list
    duplicated_themes = map(lambda x: [t for t in x if t != -1], duplicated_themes)

    # Keep theme lists longer than 1
    duplicated_themes = [l for l in duplicated_themes if len(l)>1]

    # Update topics
    topic_model.merge_topics(docs, duplicated_themes)

    # Persist topics info
    topic_info = topic_model.get_topic_info()

    # Add theme
    topic_info["theme"] = topic_info_original["Representation"].str[0].str.lower().str.strip()

    # Persist
    topic_model.save(path=BERTOPIC_FOLDER, serialization="safetensors")
    topic_info.to_csv(BERTOPIC_FOLDER / "topic_info.csv", index=False)
    np.save(BERTOPIC_FOLDER / "probs.npy", topic_model.probabilities_)

    topic_info
    return (topic_info,)


@app.cell
def _(topic_info):
    # Count Number of clusters
    "Numero di temi", topic_info.shape[0]-1, "Articoli non categorizzati", topic_info.iloc[0].Count / topic_info.Count.sum()
    return


@app.cell
def _(BERTOPIC_FOLDER, OUT_FOLDER, Path, df, json, topic_info, topic_model):
    def update_df(df, topic_model):

        # Add topics to df
        df["topic"] = topic_model.topics_

        # Persist topic stats
        topics_stats = dict(clusters=topic_info.shape[0]-1, uncategorized_articles=topic_info.iloc[0].Count / topic_info.Count.sum())

        with Path(OUT_FOLDER / "topic_stats.json").open("w") as fout:
            fout.write(json.dumps(topics_stats))

        # Persist dataset with topics
        df.to_csv(BERTOPIC_FOLDER / "dataset_topic.csv", index=False)

    update_df(df, topic_model)
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
