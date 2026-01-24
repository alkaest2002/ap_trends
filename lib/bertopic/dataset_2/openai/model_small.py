from os import getenv
from typing import Any

from bertopic import BERTopic
from bertopic.backend import OpenAIBackend
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

from lib.utils_base import get_psychology_sections_list
from openai import OpenAI

zero_shot_topics = get_psychology_sections_list()

stop_words = ENGLISH_STOP_WORDS.union({
    "<title>", "</title>", "title", "<excerpt>", "</excerpt>", "excerpt",
})

# Load env vars
load_dotenv()


# Initiate openai client
client = OpenAI(api_key=getenv("OPENAI_APIKEY"))


# Default BERTopic settings for topic modeling
default_bertopic_settings: dict[str, Any] = {
    "umap": {
        "n_neighbors": 5,
        "n_components": 30,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42
    },
    "hdbscan": {
        "min_cluster_size": 3,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    },
    "vectorizer": {
        "stop_words": list(stop_words),
        "ngram_range":  (1, 3),
        "max_df": .35,
    },
    "ctfidf": {
        "bm25_weighting": True,
        "reduce_frequent_words": True,
    },
    "representation": {
        "maximal_marginal_relevance": {
            "diversity": 0.1
        },
    }
}


def get_bertopic_model(overrides: dict[str, Any] | None = None) -> Any:
    """Create a BERTopic model."""
    # Apply overrides to default settings via update
    if overrides:
        for key, value in overrides.items():
            if key in default_bertopic_settings:
                default_bertopic_settings[key].update(value)

    # Step 1 - Embedder
    embedding_model = OpenAIBackend(client=client, embedding_model="text-embedding-3-small")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(**default_bertopic_settings["umap"])

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(**default_bertopic_settings["hdbscan"])

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(**default_bertopic_settings["vectorizer"])

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer(**default_bertopic_settings["ctfidf"])

    # Step 6 - (Optional) Fine-tune topic representations
    representation_model: list = [
        MaximalMarginalRelevance(
            **default_bertopic_settings["representation"]["maximal_marginal_relevance"]
        ),
    ]

    # All steps together
    return BERTopic(
        calculate_probabilities=True,
        top_n_words=15,
        embedding_model=embedding_model,           # Step 1 - Extract embeddings
        umap_model=umap_model,                     # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,               # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,         # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                 # Step 5 - Extract topic words
        representation_model=representation_model  # Step 6 - Fine-tune topic representations  # ty:ignore[invalid-argument-type]
    )
