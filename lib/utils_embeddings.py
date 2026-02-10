import time
from os import getenv
from typing import Any

from dotenv import load_dotenv
from numpy.typing import NDArray
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load env vars
load_dotenv()

# Initiate openai client
client = OpenAI(api_key=getenv("OPENAI_APIKEY"))


def validate_docs_(docs: list[str]):
    """Validate that the docs list is not empty.

    Args:
        docs (list[str]): List of documents to validate.

    Returns:
        None

    Raises:
        ValueError: If the 'docs' list is empty.

    """
    # Raise error if docs is empty
    if not docs:
        error_msg: str = "The 'docs' list is empty. Please provide at least one text."
        raise ValueError(error_msg)


def get_openai_embeddings(
    docs: list[str],
    embedding_model_name: str = "text-embedding-3-large",
    batch_size: int = 100,
    delay_between_batches: float = 1.0
) -> tuple[str, list[float] | list[list[float]]]:
    """Get embeddings for a given text or list of texts using OpenAI API with batch processing.

    Args:
        docs (list[str]): List of documents to get embeddings for.
        embedding_model_name (str, optional): Embedding model to use. Defaults to "text-embedding-3-large".
        batch_size (int, optional): Number of texts to process in each batch. Defaults to 100.
        delay_between_batches (float, optional): Delay in seconds between batches. Defaults to 1.0.

    Returns:
        tuple[str, list[float] | list[list[float]]]: Embedding model name and embedding vector(s).
        single text input, or list of lists for multiple texts.

    Raises:
        ValueError: If the 'texts' list is empty.
        Exception: For any errors during API calls.

    """
    # Raise error if texts is empty
    validate_docs_(docs)

    # Remove newlines from texts to improve consistency
    cleaned_texts: list[str] = [text.replace("\n", " ") for text in docs]

    # Initialize list to hold all embeddings
    all_embeddings: list[list[float]] = []

    # Process in batches
    for i in range(0, len(cleaned_texts), batch_size):

        # Get the current batch
        batch: list[str] = cleaned_texts[i:i + batch_size]

        try:
            # Call OpenAI API for the batch
            response = client.embeddings.create(input=batch, model=embedding_model_name)

            # Extract embeddings from response
            batch_embeddings: list[list[float]] = [data.embedding for data in response.data]

            # Append batch embeddings to all embeddings
            all_embeddings.extend(batch_embeddings)

            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(cleaned_texts):
                time.sleep(delay_between_batches)

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            raise

    return embedding_model_name, all_embeddings


def get_sentence_transformer(
    docs: list[str],
    transformer_name: str = "all-MiniLM-L6-v2",
    sentence_transformer_kwargs: dict[str, Any] | None = None,
    sentence_model_kwargs: dict[str, Any] | None = None
) -> NDArray:
    """Get embeddings for a list of texts using SentenceTransformer.

    Args:
        docs (list[str]): List of documents to get embeddings for.
        transformer_name (str): Name of the SentenceTransformer model.
        sentence_transformer_kwargs (dict[str, Any], optional): Additional keyword arguments for the SentenceTransformer. Defaults to None.
        sentence_model_kwargs (dict[str, Any], optional): Additional keyword arguments for the SentenceTransformer model. Defaults to None.

    Returns:
        NDArray: Array of embedding vectors.

    """
    # Raise error if texts is empty
    validate_docs_(docs)

    # Load embedding model
    sentence_model = SentenceTransformer(transformer_name, **(sentence_transformer_kwargs or {}))

    # Compute embeddings
    return sentence_model.encode(docs, show_progress_bar=True, **(sentence_model_kwargs or {}))
