
import contextlib
import re

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
