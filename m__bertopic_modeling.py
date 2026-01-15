import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from os import getenv
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from dotenv import load_dotenv
    from openai import OpenAI

    from bertopic import BERTopic
    from bertopic.backend import OpenAIBackend
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import MaximalMarginalRelevance

    # Load env vars
    load_dotenv()

    # Initiate openai client
    client = OpenAI(api_key=getenv("OPENAI_APIKEY"))
    return (
        BERTopic,
        ClassTfidfTransformer,
        CountVectorizer,
        HDBSCAN,
        MaximalMarginalRelevance,
        OpenAIBackend,
        Path,
        UMAP,
        client,
        np,
        pd,
    )


@app.cell
def _(Path, pd):
    DATASET_FOLDER = Path("./datasets/dataset_1/")
    df = pd.read_csv(DATASET_FOLDER / "psycarticles_cleaned.csv")
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
def _(
    BERTopic,
    ClassTfidfTransformer,
    CountVectorizer,
    HDBSCAN,
    MaximalMarginalRelevance,
    OpenAIBackend,
    UMAP,
    client,
):
    # Step 1 - Extract embeddings
    embedding_model = OpenAIBackend(client=client)

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(n_neighbors=15, n_components=30, min_dist=0.0, metric='cosine')

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic representations with
    # a `bertopic.representation` model
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    # All steps together
    topic_model = BERTopic(
        embedding_model=embedding_model,          # Step 1 - Extract embeddings
        umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
        representation_model=representation_model
    )
    return (topic_model,)


@app.cell
def _(docs, embeddings, topic_model):
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return


@app.cell
def _(topic_model):
    topic_model.get_topic_info()
    return


@app.cell
def _(DATASET_FOLDER, topic_model):
    # Persist
    topic_model.save(path=DATASET_FOLDER / "bertopic", serialization="safetensors")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
