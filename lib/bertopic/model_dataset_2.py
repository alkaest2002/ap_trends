from os import getenv
from typing import Any

from bertopic import BERTopic
from bertopic.backend import OpenAIBackend
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Load env vars
load_dotenv()

# Initiate openai client
client = OpenAI(api_key=getenv("OPENAI_APIKEY"))

# Initiate embedding model
embedding_model = OpenAIBackend(client=client)

# Default BERTopic settings for topic modeling
default_bertopic_settings: dict[str, Any] = {
    "umap": {
        "n_neighbors": 15,
        "n_components": 15,
        "min_dist": 0.0,
        "metric": "cosine"
    },
    "hdbscan": {
        "min_cluster_size": 4,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True
    },
    "vectorizer": {
        "stop_words": "english"
    },
    "representation": {
        "diversity": 0.3
    }
}


def get_bertopic_model() -> Any:
    """Create a BERTopic model."""
    # Step 1 - Embedder
    # Done at module level to avoid multiple instantiations

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(**default_bertopic_settings["umap"])

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(**default_bertopic_settings["hdbscan"])

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(**default_bertopic_settings["vectorizer"])

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic representations
    representation_model = MaximalMarginalRelevance(**default_bertopic_settings["representation"])

    # All steps together
    return BERTopic(
        embedding_model=embedding_model,           # Step 1 - Extract embeddings
        umap_model=umap_model,                     # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,               # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,         # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                 # Step 5 - Extract topic words
        representation_model=representation_model  # Step 6 - (Optional) Fine-tune topic representations
    )
