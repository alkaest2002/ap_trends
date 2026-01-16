from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable


def check_columns_(df: pd.DataFrame, columns: list[str]) -> None:
    """Check if specified columns are present in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to check.
        columns (list[str]): List of column names to check for.

    Raises:
        ValueError: If any of the specified columns are not present in the DataFrame.

    """
    # Check for missing columns
    missing_cols = set(columns) - set(df.columns)

    # Raise error if any columns are missing
    if missing_cols:
        error_msg: str = f"The following columns are missing from the DataFrame: {', '.join(missing_cols)}"
        raise ValueError(error_msg)


def make_excerpt(df: pd.DataFrame, column: str = "abstract", num_paragraphs: int = 3) -> pd.Series:
    """Create excerpt from column by cleaning up common abbreviations and limiting to first few paragraphs.

    Args:
        df (pd.DataFrame): DataFrame containing the abstracts.
        column (str): Name of the column containing the abstracts (default is "abstract").
        num_paragraphs (int): Number of paragraphs to include in the excerpt (default is 3).

    Returns:
        pd.Series: Series containing the excerpts.

    Raises:
        ValueError: If the specified column is not present in the DataFrame.

    """
    # Defualt column to abstract if not specified
    if not column:
        column = "abstract"

    # Check if column is present in df
    check_columns_(df, [column])

    # Dictionary of abbreviations to clean up
    abbrev_dict: dict[str, str | Callable] = {
        r"\bet al\.": "et al",
        r"\be\.g\.": "eg",
        r"\bi\.e\.": "ie",
        r"\bcf\.": "cf",
        r"\bviz\.": "viz",
        r"\bvs\.": "vs",
        r"\bca\.": "ca",
        r"\bc\.": "c",
        r"\bibid\.": "ibid",
        r"\bop\. cit\.": "op cit",
        r"\bloc\. cit\.": "loc cit",
        r"\bq\.v\.": "qv",
        r"\b[Nn]\.?[Bb]\.": "NB",
        r"\b[Pp]\.?[Ss]\.": "PS",
        r"\bff\.": "ff",
        r"\bpp\.": "pp",
        r"\bvols?\.": lambda m: m.group().replace(".", ""),
        r"\beds?\.": lambda m: m.group().replace(".", ""),
        r"\btrans\.": "trans",
        r"\brev\.": "rev",
        r"\brepr\.": "repr"
    }

    # Fill NaN values with empty strings
    excerpt: pd.Series = df[column].fillna("")

    # Apply abbreviation replacements
    for pattern, replacement in abbrev_dict.items():
        excerpt = excerpt.str.replace(pattern, replacement, regex=True)

    return (
       excerpt
            .str.split(r"\.\s+", regex=True)
            .str[:num_paragraphs]
            .str.join(". ")
            .add(".")
            .str.replace(r"^\.$", "", regex=True)
    )


def make_text_to_embed(df: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """Prepare text for embedding by filling NaN values.

    Args:
        df (pd.DataFrame): DataFrame containing the abstracts.
        columns (list[str]): List of column names to be combined for embedding.

    Returns:
        pd.Series: Series containing the text ready for embedding.

    Raises:
        ValueError: If any of the specified columns are not present in the DataFrame.

    """
    # If no columns specified
    if not columns:
        # Default columns to title and abstract if none specified
        columns = ["title", "excerpt"]

    # Raise error if columns are not present in df (use pd.index intersection)
    check_columns_(df, columns)

    # Fill NaN values with empty strings
    text_to_embed: pd.DataFrame = df[columns].fillna("")

    # Iterate over columns
    for col in columns:
        # Wrap the text in tags
        text_to_embed[col] = f"<{col}>" + text_to_embed[col] + f"</{col}>"
        # Remove empty tags
        text_to_embed[col] = text_to_embed[col].str.replace(rf"<{col}>\.?</{col}>", "", regex=True)

    return text_to_embed.agg(" ".join, axis=1).str.strip()


def get_psychology_sections_list() -> list[str]:
    """Get a list of psychology-related sections.

    Returns:
        list[str]: List of psychology-related section names.

    Notes:
        see https://www.apa.org/pubs/databases/training/class-codes

    """
    return [
        "general_psychology_history_&_systems",
        "psychometrics_&_statistics_&_methodology_tests_&_testing",
        "psychometrics_&_statistics_&_methodology_sensory_&_motor_testing",
        "psychometrics_&_statistics_&_methodology_developmental_scales_&_schedules",
        "psychometrics_&_statistics_&_methodology_personality_scales_&_inventories",
        "psychometrics_&_statistics_&_methodology_clinical_psychological_testing",
        "psychometrics_&_statistics_&_methodology_neuropsychological_assessment",
        "psychometrics_&_statistics_&_methodology_health_psychology_testing",
        "psychometrics_&_statistics_&_methodology_educational_measurement",
        "psychometrics_&_statistics_&_methodology_occupational_&_employment_testing",
        "psychometrics_&_statistics_&_methodology_consumer_opinion_&_attitude_testing",
        "psychometrics_&_statistics_&_methodology_statistics_&_mathematics",
        "psychometrics_&_statistics_&_methodology_research_methods_&_experimental_design",
        "human_experimental_psychology_sensory_perception",
        "human_experimental_psychology_visual_perception",
        "human_experimental_psychology_auditory_&_speech_perception",
        "human_experimental_psychology_motor_processes",
        "human_experimental_psychology_cognitive_processes",
        "human_experimental_psychology_learning_&_memory",
        "human_experimental_psychology_attention",
        "human_experimental_psychology_motivation_&_emotion",
        "human_experimental_psychology_consciousness_states",
        "human_experimental_psychology_psychic_phenomenon_&_paranormal_activities",
        "animal_experimental_&_comparative_psychology_learning_&_motivation",
        "animal_experimental_&_comparative_psychology_social_&_instinctive_behavior",
        "physiological_psychology_&_neuroscience_genetics",
        "physiological_psychology_&_neuroscience_neuropsychology_&_neurology",
        "physiological_psychology_&_neuroscience_electrophysiology",
        "physiological_psychology_&_neuroscience_physiological_processes",
        "physiological_psychology_&_neuroscience_psychophysiology",
        "physiological_psychology_&_neuroscience_psychopharmacology",
        "psychology_&_the_humanities_literature_&_fine_arts",
        "psychology_&_the_humanities_philosophy",
        "communication_systems_linguistics_&_language_&_speech",
        "communication_systems_mass_media_communications",
        "developmental_psychology_cognitive_&_perceptual_development",
        "developmental_psychology_psychosocial_&_personality_development",
        "developmental_psychology_aging_&_older_adult_development",
        "social_processes_&_social_issues_social_structure_&_organization",
        "social_processes_&_social_issues_religion",
        "social_processes_&_social_issues_culture_&_ethnology",
        "social_processes_&_social_issues_marriage_&_family",
        "social_processes_&_social_issues_divorce_&_remarriage",
        "social_processes_&_social_issues_childrearing_&_child_care",
        "social_processes_&_social_issues_political_processes_&_political_issues",
        "social_processes_&_social_issues_sex_&_gender_roles",
        "social_processes_&_social_issues_sexual_behavior_&_sexual_orientation",
        "social_processes_&_social_issues_drug_&_alcohol_usage_(legal)",
        "social_psychology_group_&_interpersonal_processes",
        "social_psychology_social_perception_&_cognition",
        "personality_psychology_personality_traits_&_processes",
        "personality_psychology_personality_theory",
        "personality_psychology_psychoanalytic_theory",
        "psychological_&_physical_disorders_psychological_disorders",
        "psychological_&_physical_disorders_affective_disorders",
        "psychological_&_physical_disorders_schizophrenia_&_psychotic_states",
        "psychological_&_physical_disorders_anxiety_disorders",
        "psychological_&_physical_disorders_personality_disorders",
        "psychological_&_physical_disorders_behavior_disorders_&_antisocial_behavior",
        "psychological_&_physical_disorders_substance_abuse_&_addiction",
        "psychological_&_physical_disorders_criminal_behavior_&_juvenile_delinquency",
        "psychological_&_physical_disorders_neurodevelopmental_&_autism_spectrum_disorders",
        "psychological_&_physical_disorders_learning_disorders",
        "psychological_&_physical_disorders_intellectual_developmental_disorders",
        "psychological_&_physical_disorders_eating_disorders",
        "psychological_&_physical_disorders_communication_disorders",
        "psychological_&_physical_disorders_environmental_toxins_&_health",
        "psychological_&_physical_disorders_physical_&_somatic_disorders",
        "psychological_&_physical_disorders_immunological_disorders",
        "psychological_&_physical_disorders_cancer",
        "psychological_&_physical_disorders_cardiovascular_disorders",
        "psychological_&_physical_disorders_neurological_disorders_&_brain_damage",
        "psychological_&_physical_disorders_vision_&_hearing_&_sensory_disorders",
        "health_&_mental_health_treatment_&_prevention_psychotherapy_&_psychotherapeutic_counseling",
        "health_&_mental_health_treatment_&_prevention_cognitive_therapy",
        "health_&_mental_health_treatment_&_prevention_behavior_therapy_&_behavior_modification",
        "health_&_mental_health_treatment_&_prevention_group_&_family_therapy",
        "health_&_mental_health_treatment_&_prevention_interpersonal_&_client_centered_&_humanistic_therapy",
        "health_&_mental_health_treatment_&_prevention_psychoanalytic_therapy",
        "health_&_mental_health_treatment_&_prevention_clinical_psychopharmacology",
        "health_&_mental_health_treatment_&_prevention_specialized_interventions",
        "health_&_mental_health_treatment_&_prevention_clinical_hypnosis",
        "health_&_mental_health_treatment_&_prevention_self_help_groups",
        "health_&_mental_health_treatment_&_prevention_lay_&_paraprofessional_&_pastoral_counseling",
        "health_&_mental_health_treatment_&_prevention_art_&_music_&_movement_therapy",
        "health_&_mental_health_treatment_&_prevention_health_psychology_&_medicine",
        "health_&_mental_health_treatment_&_prevention_behavioral_&_psychological_treatment_of_physical_illness",
        "health_&_mental_health_treatment_&_prevention_medical_treatment_of_physical_illness",
        "health_&_mental_health_treatment_&_prevention_promotion_&_maintenance_of_health_&_wellness",
        "health_&_mental_health_treatment_&_prevention_health_&_mental_health_services",
        "health_&_mental_health_treatment_&_prevention_outpatient_services",
        "health_&_mental_health_treatment_&_prevention_community_&_social_services",
        "health_&_mental_health_treatment_&_prevention_home_care_&_hospice",
        "health_&_mental_health_treatment_&_prevention_nursing_homes_&_residential_care",
        "health_&_mental_health_treatment_&_prevention_inpatient_&_hospital_services",
        "health_&_mental_health_treatment_&_prevention_rehabilitation",
        "health_&_mental_health_treatment_&_prevention_drug_&_alcohol_rehabilitation",
        "health_&_mental_health_treatment_&_prevention_occupational_&_vocational_rehabilitation",
        "health_&_mental_health_treatment_&_prevention_speech_&_language_therapy",
        "health_&_mental_health_treatment_&_prevention_criminal_rehabilitation_&_penology",
        "health_&_mental_health_personnel_issues_professional_education_&_training",
        "health_&_mental_health_personnel_issues_professional_personnel_attitudes_&_characteristics",
        "health_&_mental_health_personnel_issues_professional_ethics_&_standards_&_liability",
        "health_&_mental_health_personnel_issues_professional_impairment",
        "educational_&_school_psychology_educational_administration_&_personnel",
        "educational_&_school_psychology_curriculum_&_programs_&_teaching_methods",
        "educational_&_school_psychology_academic_learning_&_achievement",
        "educational_&_school_psychology_classroom_dynamics_&_student_adjustment_&_attitudes",
        "educational_&_school_psychology_special_&_compensatory_education",
        "educational_&_school_psychology_gifted_&_talented",
        "educational_&_school_psychology_educational/vocational_counseling_&_student_services",
        "organizational_psychology_&_human_resources_occupational_interests_&_guidance",
        "organizational_psychology_&_human_resources_personnel_management_&_selection_&_training",
        "organizational_psychology_&_human_resources_personnel_evaluation_&_job_performance",
        "organizational_psychology_&_human_resources_management_&_management_training",
        "organizational_psychology_&_human_resources_personnel_attitudes_&_job_satisfaction",
        "organizational_psychology_&_human_resources_organizational_behavior",
        "organizational_psychology_&_human_resources_working_conditions_&_industrial_safety",
        "sport_psychology_&_leisure_sports_&_exercise",
        "sport_psychology_&_leisure_recreation_&_leisure",
        "consumer_psychology_consumer_attitudes_&_behavior",
        "consumer_psychology_marketing_&_advertising",
        "engineering_&_environmental_psychology_human_factors_engineering",
        "engineering_&_environmental_psychology_lifespace_&_institutional_design",
        "engineering_&_environmental_psychology_community_&_environmental_planning",
        "engineering_&_environmental_psychology_environmental_issues_&_attitudes",
        "engineering_&_environmental_psychology_transportation",
        "cognitive_psychology_&_intelligent_systems_artificial_intelligence_&_expert_systems",
        "cognitive_psychology_&_intelligent_systems_robotics",
        "cognitive_psychology_&_intelligent_systems_neural_networks",
        "forensic_psychology_&_legal_issues_civil_rights_&_civil_law",
        "forensic_psychology_&_legal_issues_criminal_law_&_criminal_adjudication",
        "forensic_psychology_&_legal_issues_mediation_&_conflict_resolution",
        "forensic_psychology_&_legal_issues_crime_prevention",
        "forensic_psychology_&_legal_issues_police_&_legal_personnel"
    ]
