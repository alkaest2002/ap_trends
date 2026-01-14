import time
from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

# Load env vars
load_dotenv()

# Initiate openai client
client = OpenAI(api_key=getenv("OPENAI_APIKEY"))


def get_batch_embeddings(
    texts: str | list[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 100,
    delay_between_batches: float = 1.0
) -> list[float] | list[list[float]]:
    """Get embeddings for a given text or list of texts using OpenAI API with batch processing.

    Args:
        texts (str | list[str]): Text or list of texts to get embeddings for.
        model (str, optional): Model to use. Defaults to "text-embedding-3-large".
        batch_size (int, optional): Number of texts to process in each batch. Defaults to 100.
        delay_between_batches (float, optional): Delay in seconds between batches. Defaults to 1.0.

    Returns:
        Union[List[float], List[List[float]]]: Embedding vector(s). Returns a single list for
        single text input, or list of lists for multiple texts.

    """
    # Handle single text input
    if isinstance(texts, str):
        # Replace newlines, which can negatively affect performance.
        cleaned_text = texts.replace("\n", " ")
        return client.embeddings.create(input=[cleaned_text], model=model).data[0].embedding

    # Handle list of texts
    if not texts:
        return []

    # Clean all texts
    cleaned_texts = [text.replace("\n", " ") for text in texts]

    all_embeddings = []

    # Process in batches
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]

        try:
            # Call OpenAI API for the batch
            response = client.embeddings.create(input=batch, model=model)

            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(cleaned_texts):
                time.sleep(delay_between_batches)

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            # You might want to implement retry logic here
            raise

    return all_embeddings


def get_embedding(text: str, model: str = "text-embedding-3-large") -> list[float] | list[list[float]]:
    """Get embedding for a single text using OpenAI API.

    This function is kept for backward compatibility.

    Args:
        text (str): Text to get embedding for.
        model (str, optional): Model to use. Defaults to "text-embedding-3-large".

    Returns:
        list[float] | list[list[float]]: Embedding vector(s).

    """
    return get_batch_embeddings(text, model)
