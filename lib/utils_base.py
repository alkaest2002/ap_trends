import contextlib
import json
import re
import shutil
from hashlib import md5
from pathlib import Path
from typing import Any

import pycountry
import spacy


def get_eu_countries() -> set[str]:
    """Return a set of EU member country names (as of 2026)."""
    return {
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
        "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
        "Slovenia", "Spain", "Sweden"
    }


def extract_countries(text: str, nlp_model: spacy.language.Language) -> str | None:
    """Extract country names from text using spaCy NER and pycountry validation.

    Args:
        text: Input text to extract countries from
        nlp_model: Pre-loaded spaCy model

    Returns:
        Comma-separated string of unique country names found in the text, or None if no countries are found.
        Adds "EU" if any found country belongs to the European Union.

    """
    # Validate input
    if not isinstance(text, str):
        return None

    # EU member countries (as of 2026)
    eu_countries = get_eu_countries()

    # Preprocess text to remove state abbreviations (e.g., ", CA")
    cleaned = re.sub(r",\s+[A-Z]{2}\b", "", text)

    # Process text with spaCy NER
    doc = nlp_model(cleaned)

    # Set to hold unique country names
    countries = set()

    # Flag to check if any EU country is found
    has_eu_country: bool = False

    # Common name mappings for problematic countries
    country_mappings: dict[str, str] = {
        "turkey": "Turkey",
        "south korea": "Korea, Republic of",
        "north korea": "Korea, Democratic People's Republic of",
        "usa": "United States",
        "united states": "United States",
        "uk": "United Kingdom",
        "britain": "United Kingdom",
        "great britain": "United Kingdom",
        "russia": "Russian Federation",
        "iran": "Iran, Islamic Republic of",
        "syria": "Syrian Arab Republic",
        "venezuela": "Venezuela, Bolivarian Republic of",
        "bolivia": "Bolivia, Plurinational State of",
        "vatican": "Holy See (Vatican City State)",
        "congo": "Congo",
        "czech republic": "Czechia",
    }

    def find_country_(entity_text: str) -> str | None:
        """Try multiple methods to find a country match."""
        entity_lower = entity_text.lower().strip()

        # 1. Check mappings first
        if entity_lower in country_mappings:
            return country_mappings[entity_lower]

        # 2. Try exact name match
        with contextlib.suppress(KeyError, LookupError):
            country = pycountry.countries.get(name=entity_text)
            if country:
                return country.name

        # 3. Try alternative names (common_name, official_name)
        for country in pycountry.countries:
            # Check common name
            if hasattr(country, "common_name") and country.common_name.lower() == entity_lower:
                return country.name
            # Check if entity is part of official name
            if entity_lower in country.name.lower():
                return country.name

        return None

    # Extract entities labeled as GPE or LOC
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "NORP"}:  # Added NORP for nationalities
            country_name = find_country_(ent.text)
            if country_name:
                countries.add(country_name)
                # Check if this country is in the EU
                if country_name in eu_countries:
                    has_eu_country = True

    # Additional check for multi-word countries that might be split
    # Look for common patterns like "South Korea", "North Korea", etc.
    text_lower = text.lower()
    multi_word_countries = {
        "south korea": "Korea, Republic of",
        "north korea": "Korea, Democratic People's Republic of",
        "united states": "United States",
        "united kingdom": "United Kingdom",
        "great britain": "United Kingdom",
        "new zealand": "New Zealand",
        "saudi arabia": "Saudi Arabia",
        "south africa": "South Africa",
    }

    for pattern, official_name in multi_word_countries.items():
        if pattern in text_lower:
            countries.add(official_name)
            # Check if this country is in the EU
            if official_name in eu_countries:
                has_eu_country = True

    # Add EU if any country found is an EU member
    if has_eu_country:
        countries.add("EU")

    return " - ".join(sorted(countries)) if countries else None


def normalize_model_name(model_name: str) -> str:
    """Normalize the embedding model name for consistent file naming.

    Args:
        model_name (str): Original model name.

    Returns:
        str: Normalized model name.

    """
    return model_name.replace(".", "_").replace("/", "__").replace("-", "_")


def get_or_create_folders(
        type_of_doc: str,
        type_of_family_model: str = "sentence_transformers",
        type_of_model: str | None = None) -> list[Path]:
    """Get or create folders for storing documents and metadata.

    Args:
        type_of_doc: A string indicating the type of document (e.g., "accidents", "incidents")
        type_of_family_model: A string indicating the family of the model (e.g., "sentence_transformers", "openai")
        type_of_model: A string indicating the type of model (e.g., "bert", "gpt")

    Returns:
        A list containing the paths to the documents folder and the metadata folder.

    """
    # Define paths
    base_dataset_path = Path("dataset")
    dataset_path = base_dataset_path / type_of_doc

    # If type_of_model is provided, create subfolders for that model's outputs
    if type_of_model is not None:
        normalized_model_name = normalize_model_name(type_of_model)
        out_path = Path("out") / type_of_family_model / normalized_model_name / type_of_doc
        out_bertopic_path = out_path / "bertopic"
        out_embeddings_path = out_path / "embeddings"
        out_imgs_path = out_path / "imgs"
        out_other_path = out_path / "other"

        # Create folders if they don't exist
        for path in [dataset_path, out_bertopic_path, out_embeddings_path, out_imgs_path, out_other_path]:
            path.mkdir(parents=True, exist_ok=True)

        return [
            dataset_path,
            out_path,
            out_embeddings_path,
            out_bertopic_path,
            out_imgs_path,
            out_other_path
        ]

    dataset_path.mkdir(parents=True, exist_ok=True)
    return [dataset_path]


def archive_results(params: dict[str, Any]) -> str:
    """Determine whether to archive results based on environment variable.

    Args:
        params: A dictionary of BERTopic parameters.

    Returns:
        str: Path to the archive folder if correctly archived, empty string otherwise.

    """
    # MD5 dict for BERTopic parameters
    params_str = str(sorted(params.items())).encode()
    params_hash = md5(params_str, usedforsecurity=False).hexdigest()
    archive_folder = Path("acme/results_archive") / params_hash
    archive_folder.mkdir(parents=True, exist_ok=True)

    # Delete existing files in the archive folder
    for file in archive_folder.glob("*"):
        if file.is_file():
            file.unlink()

    # Get model_name and type_of_doc from dict
    type_of_model_family = params.get("type_of_model_family", "unknown_family")
    type_of_embedding_model = params.get("type_of_embedding_model", "unknown_model")
    type_of_doc = params.get("type_of_doc", "unknown_doc")

    # Set base path for archiving results
    base_path = (
        Path("out")
            / type_of_model_family
            / normalize_model_name(type_of_embedding_model)
            / type_of_doc
    )

    # Define list of files to archive
    files_to_archive = [
        base_path / "topic_stats.json",
        base_path / "bertopic" / "topic_info.csv",
        base_path / "bertopic" / "topics.json",
        base_path / "imgs" / "img_elbow.svg",
        base_path / "imgs" / "img_topic_trajectories.svg",
        base_path / "other" / "china_topics.txt",
        base_path / "other" / "consolidated_topics.txt",
        base_path / "other" / "emerging_topics.txt",
        base_path / "other" / "eu_topics.txt",
        base_path / "other" / "eu_china_usa_common_topics.txt",
        base_path / "other" / "united states_topics.txt",
    ]

    try:
        for file in files_to_archive:
            if file.exists():
                print(f"Archiving {file} to {archive_folder}")
                shutil.copy(file, archive_folder / file.name)

        # Save settigns json with Path
        settings_path = archive_folder / "settings.json"
        with settings_path.open("w") as f:
            json.dump(params, f, indent=4)

    except Exception as e:
        print(f"Error archiving results: {e}")
        return ""

    return str(archive_folder)
