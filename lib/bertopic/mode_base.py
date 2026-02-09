from importlib import import_module
from pathlib import Path
from typing import Any

from lib.utils_base import normalize_model_name


def get_bertopic_model(
        type_of_family_name: str,
        embedding_model_name: str,
) -> Any:
    """Get the BERTopic model based on the type of family and embedding model name.

    Args:
        type_of_family_name (str): The type of family (e.g., "sentence_transformers").
        embedding_model_name (str): The name of the embedding model (e.g., "all-MiniLM-L6-v2").

    Returns:
        Any: The BERTopic model instance.

    """
    # Get normalized model name
    model_name = normalize_model_name(embedding_model_name)

    # Python module path (used by importlib)
    base_module = f"lib.bertopic.{type_of_family_name}.model_base"
    specialized_module = f"lib.bertopic.{type_of_family_name}.{model_name}.model_base"

    # Filesystem path (used to check existence)
    specialized_dir = Path("./lib/bertopic") / type_of_family_name / model_name

    if specialized_dir.exists():
        return import_module(specialized_module).get_bertopic_model()

    return import_module(base_module).get_bertopic_model()
