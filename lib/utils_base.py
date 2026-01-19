
import contextlib

import pycountry
import spacy


def get_psychology_sections_list() -> list[str]:
    """Get a list of psychology-related sections.

    Returns:
        list[str]: List of psychology-related section names.

    Notes:
        see https://www.apa.org/pubs/databases/training/class-codes

    """
    return [
        "general psychology history & systems",
        "psychometrics & statistics & methodology tests & testing",
        "psychometrics & statistics & methodology sensory & motor testing",
        "psychometrics & statistics & methodology developmental scales & schedules",
        "psychometrics & statistics & methodology personality scales & inventories",
        "psychometrics & statistics & methodology clinical psychological testing",
        "psychometrics & statistics & methodology neuropsychological assessment",
        "psychometrics & statistics & methodology health psychology testing",
        "psychometrics & statistics & methodology educational measurement",
        "psychometrics & statistics & methodology occupational & employment testing",
        "psychometrics & statistics & methodology consumer opinion & attitude testing",
        "psychometrics & statistics & methodology statistics & mathematics",
        "psychometrics & statistics & methodology research methods & experimental design",
        "human experimental psychology sensory perception",
        "human experimental psychology visual perception",
        "human experimental psychology auditory & speech perception",
        "human experimental psychology motor processes",
        "human experimental psychology cognitive processes",
        "human experimental psychology learning & memory",
        "human experimental psychology attention",
        "human experimental psychology motivation & emotion",
        "human experimental psychology consciousness states",
        "human experimental psychology psychic phenomenon & paranormal activities",
        "animal experimental & comparative psychology learning & motivation",
        "animal experimental & comparative psychology social & instinctive behavior",
        "physiological psychology & neuroscience genetics",
        "physiological psychology & neuroscience neuropsychology & neurology",
        "physiological psychology & neuroscience electrophysiology",
        "physiological psychology & neuroscience physiological processes",
        "physiological psychology & neuroscience psychophysiology",
        "physiological psychology & neuroscience psychopharmacology",
        "psychology & the humanities literature & fine arts",
        "psychology & the humanities philosophy",
        "communication systems linguistics & language & speech",
        "communication systems mass media communications",
        "developmental psychology cognitive & perceptual development",
        "developmental psychology psychosocial & personality development",
        "developmental psychology aging & older adult development",
        "social processes & social issues social structure & organization",
        "social processes & social issues religion",
        "social processes & social issues culture & ethnology",
        "social processes & social issues marriage & family",
        "social processes & social issues divorce & remarriage",
        "social processes & social issues childrearing & child care",
        "social processes & social issues political processes & political issues",
        "social processes & social issues sex & gender roles",
        "social processes & social issues sexual behavior & sexual orientation",
        "social processes & social issues drug & alcohol usage (legal)",
        "social psychology group & interpersonal processes",
        "social psychology social perception & cognition",
        "personality psychology personality traits & processes",
        "personality psychology personality theory",
        "personality psychology psychoanalytic theory",
        "psychological & physical disorders psychological disorders",
        "psychological & physical disorders affective disorders",
        "psychological & physical disorders schizophrenia & psychotic states",
        "psychological & physical disorders anxiety disorders",
        "psychological & physical disorders personality disorders",
        "psychological & physical disorders behavior disorders & antisocial behavior",
        "psychological & physical disorders substance abuse & addiction",
        "psychological & physical disorders criminal behavior & juvenile delinquency",
        "psychological & physical disorders neurodevelopmental & autism spectrum disorders",
        "psychological & physical disorders learning disorders",
        "psychological & physical disorders intellectual developmental disorders",
        "psychological & physical disorders eating disorders",
        "psychological & physical disorders communication disorders",
        "psychological & physical disorders environmental toxins & health",
        "psychological & physical disorders physical & somatic disorders",
        "psychological & physical disorders immunological disorders",
        "psychological & physical disorders cancer",
        "psychological & physical disorders cardiovascular disorders",
        "psychological & physical disorders neurological disorders & brain damage",
        "psychological & physical disorders vision & hearing & sensory disorders",
        "health & mental health treatment & prevention psychotherapy & psychotherapeutic counseling",
        "health & mental health treatment & prevention cognitive therapy",
        "health & mental health treatment & prevention behavior therapy & behavior modification",
        "health & mental health treatment & prevention group & family therapy",
        "health & mental health treatment & prevention interpersonal & client centered & humanistic therapy",
        "health & mental health treatment & prevention psychoanalytic therapy",
        "health & mental health treatment & prevention clinical psychopharmacology",
        "health & mental health treatment & prevention specialized interventions",
        "health & mental health treatment & prevention clinical hypnosis",
        "health & mental health treatment & prevention self help groups",
        "health & mental health treatment & prevention lay & paraprofessional & pastoral counseling",
        "health & mental health treatment & prevention art & music & movement therapy",
        "health & mental health treatment & prevention health psychology & medicine",
        "health & mental health treatment & prevention behavioral & psychological treatment of physical illness",
        "health & mental health treatment & prevention medical treatment of physical illness",
        "health & mental health treatment & prevention promotion & maintenance of health & wellness",
        "health & mental health treatment & prevention health & mental health services",
        "health & mental health treatment & prevention outpatient services",
        "health & mental health treatment & prevention community & social services",
        "health & mental health treatment & prevention home care & hospice",
        "health & mental health treatment & prevention nursing homes & residential care",
        "health & mental health treatment & prevention inpatient & hospital services",
        "health & mental health treatment & prevention rehabilitation",
        "health & mental health treatment & prevention drug & alcohol rehabilitation",
        "health & mental health treatment & prevention occupational & vocational rehabilitation",
        "health & mental health treatment & prevention speech & language therapy",
        "health & mental health treatment & prevention criminal rehabilitation & penology",
        "health & mental health personnel issues professional education & training",
        "health & mental health personnel issues professional personnel attitudes & characteristics",
        "health & mental health personnel issues professional ethics & standards & liability",
        "health & mental health personnel issues professional impairment",
        "educational & school psychology educational administration & personnel",
        "educational & school psychology curriculum & programs & teaching methods",
        "educational & school psychology academic learning & achievement",
        "educational & school psychology classroom dynamics & student adjustment & attitudes",
        "educational & school psychology special & compensatory education",
        "educational & school psychology gifted & talented",
        "educational & school psychology educational/vocational counseling & student services",
        "organizational psychology & human resources occupational interests & guidance",
        "organizational psychology & human resources personnel management & selection & training",
        "organizational psychology & human resources personnel evaluation & job performance",
        "organizational psychology & human resources management & management training",
        "organizational psychology & human resources personnel attitudes & job satisfaction",
        "organizational psychology & human resources organizational behavior",
        "organizational psychology & human resources working conditions & industrial safety",
        "sport psychology & leisure sports & exercise",
        "sport psychology & leisure recreation & leisure",
        "consumer psychology consumer attitudes & behavior",
        "consumer psychology marketing & advertising",
        "engineering & environmental psychology human factors engineering",
        "engineering & environmental psychology lifespace & institutional design",
        "engineering & environmental psychology community & environmental planning",
        "engineering & environmental psychology environmental issues & attitudes",
        "engineering & environmental psychology transportation",
        "cognitive psychology & intelligent systems artificial intelligence & expert systems",
        "cognitive psychology & intelligent systems robotics",
        "cognitive psychology & intelligent systems neural networks",
        "forensic psychology & legal issues civil rights & civil law",
        "forensic psychology & legal issues criminal law & criminal adjudication",
        "forensic psychology & legal issues mediation & conflict resolution",
        "forensic psychology & legal issues crime prevention",
        "forensic psychology & legal issues police & legal personnel"
    ]


def extract_countries(text: str, nlp_model: spacy.language.Language) -> list[str]:
    """Extract country names from text using spaCy NER and pycountry validation.

    Args:
        text: Input text to extract countries from
        nlp_model: Pre-loaded spaCy mode

    Returns:
        List of unique country names found in the text

    """
    # Validate input
    if not isinstance(text, str):
        return []

    # Process text with spaCy
    doc = nlp_model(text)

    countries = set()

    # Extract entities labeled as GPE (geopolitical entity) or LOC (location)
    for ent in doc.ents:
        # Check if entity is a GPE or LOC
        if ent.label_ in {"GPE", "LOC"}:
            # Try to match with pycountry
            country = None

            # Try exact name match
            with contextlib.suppress(KeyError, LookupError):
                country = pycountry.countries.get(name=ent.text)

            # Try fuzzy search if exact match fails
            if not country:
                try:
                    results = pycountry.countries.search_fuzzy(ent.text)
                    if results:
                        country = results[0]
                except (KeyError, LookupError):
                    pass

            if country:
                countries.add(country.name)

    return list(countries)
