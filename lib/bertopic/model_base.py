from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from lib.utils_base import normalize_model_name

if TYPE_CHECKING:
    from collections.abc import Callable


def _resolve_model_base_module(
    type_of_family_name: str,
    embedding_model_name: str,
) -> ModuleType:
    """Resolve and import the appropriate module for the given model.

    Checks for a specialized implementation first, falling back to the base
    implementation if no specialized version exists.

    Args:
        type_of_family_name: The model family (e.g., "sentence_transformers", "openai")
        embedding_model_name: The specific model name (e.g., "bert", "gpt")

    Returns:
        The imported module containing model configuration functions.

    Raises:
        ValueError: If either argument is empty or None.

    """
    if not type_of_family_name or not embedding_model_name:
        error_msg = "Both type_of_family_name and embedding_model_name are required"
        raise ValueError(error_msg)

    # Normalize model name for consistent file naming
    model_name = normalize_model_name(embedding_model_name)

    # Define potential module paths
    base_module = f"lib.bertopic.{type_of_family_name}.model_base"
    specialized_module = f"lib.bertopic.{type_of_family_name}.{model_name}.model_base"

    # Define the path to check for the specialized module
    specialized_dir = Path(__file__).parent / type_of_family_name / model_name

    # If the specialized directory exists, use the specialized module; otherwise, use the base module
    module_path = specialized_module if specialized_dir.exists() else base_module

    return import_module(module_path)


def _call_from_resolved_module(
    func_name: str,
    type_of_family_name: str,
    embedding_model_name: str,
) -> "Callable[..., Any]":
    """Resolve correct module and retrieve the specified function from it.

    Args:
        func_name: The name of the function to retrieve from the module.
        type_of_family_name: The model family (e.g., "sentence_transformers", "openai")
        embedding_model_name: The specific model name (e.g., "bert", "gpt")

    Returns:
        The requested function from the resolved module.

    Raises:
        AttributeError: If the function is not defined in the resolved module.

    """
    # Resolve the correct module based on the family and model name
    module = _resolve_model_base_module(type_of_family_name, embedding_model_name)

    try:
        # Get the function from the resolved module
        func: Callable[..., Any] = getattr(module, func_name)
    except AttributeError as e:
        error_msg = (
            f"Function '{func_name}' not found in module '{module.__name__}' for "
            f"family='{type_of_family_name}', model='{embedding_model_name}'"
        )
        raise AttributeError(error_msg) from e

    return func


def get_bertopic_settings(
    type_of_family_name: str,
    embedding_model_name: str,
) -> dict[str, Any]:
    """Get the default BERTopic settings.

    Args:
        type_of_family_name: A string indicating the family of the model (e.g., "sentence_transformers", "openai")
        embedding_model_name: A string indicating the type of model (e.g., "bert", "gpt").

    Returns:
        A dictionary containing the default BERTopic settings.

    """
    func = _call_from_resolved_module(
        "get_bertopic_settings",
        type_of_family_name,
        embedding_model_name,
    )
    return func()


def get_bertopic_model(
    type_of_family_name: str,
    embedding_model_name: str,
) -> Any:
    """Create a BERTopic model.

    Args:
        type_of_family_name: A string indicating the family of the model (e.g., "sentence_transformers", "openai")
        embedding_model_name: A string indicating the type of model (e.g., "bert", "gpt").

    Returns:
        A BERTopic model instance.

    """
    func = _call_from_resolved_module(
        "get_bertopic_model",
        type_of_family_name,
        embedding_model_name,
    )
    return func()
