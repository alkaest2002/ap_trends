from os import getenv
from typing import Any

import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

from sentence_transformers import SentenceTransformer

stop_words = ENGLISH_STOP_WORDS.union({
    "<title>", "</title>", "title", "<excerpt>", "</excerpt>", "excerpt",
})

load_dotenv()

client = openai.OpenAI(api_key=getenv("OPENAI_APIKEY"))


# Default BERTopic settings for topic modeling
default_bertopic_settings: dict[str, Any] = {
    "umap": {
        "n_neighbors": 3,
        "n_components": 8,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42,
        "n_jobs": 1
    },
    "hdbscan": {
        "min_cluster_size": 5,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    },
    "vectorizer": {
        "stop_words": list(stop_words),
        "ngram_range":  (1, 3),
        "min_df": .3,
        "max_df": .7,
    },
    "ctfidf": {
        "bm25_weighting": True,
        "reduce_frequent_words": True,
    },
    "representation": {
        "KeyBERTInspired": {
            "top_n_words": 20,
        },
        "maximal_marginal_relevance": {
            "diversity": 0.3
        },
        "openai": {
            "model": "gpt-4o-mini",
            "temperature": 0,
        }
    }
}


def get_bertopic_settings() -> dict[str, Any]:
    """Get the default BERTopic settings."""
    return default_bertopic_settings


def get_bertopic_model(overrides: dict[str, Any] | None = None) -> Any:
    """Create a BERTopic model."""
    # Apply overrides to default settings via update
    if overrides:
        for key, value in overrides.items():
            if key in default_bertopic_settings:
                default_bertopic_settings[key].update(value)

    # Step 1 - Embedder
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token=getenv("HF_TOKEN"))

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
        KeyBERTInspired(
            **default_bertopic_settings["representation"]["KeyBERTInspired"]
        ),
        MaximalMarginalRelevance(
            **default_bertopic_settings["representation"]["maximal_marginal_relevance"]
        ),
        OpenAI(client=client, **default_bertopic_settings["representation"]["openai"]),
    ]

    # All steps together
    return BERTopic(
        calculate_probabilities=True,
        top_n_words=5,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model  # ty:ignore[invalid-argument-type]
    )
