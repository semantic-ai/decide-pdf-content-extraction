"""
NER Configuration and Constants

This module contains all configuration settings, model names, and constants
used by the NER extraction system.
"""

# Model Configuration
NER_MODELS = {
    'spacy': {
        'nl': 'nl_core_news_sm',
        'de': 'de_core_news_sm',
        'en': 'en_core_web_sm'
    },
    'flair': {
        'de': 'flair/ner-german-legal',
        'en': 'flair/ner-english',
        'nl': 'flair/ner-dutch'
    },
    'huggingface': {
        'nl': 'PedroDKE/multilingual-ner-abb',
        'en': 'PedroDKE/multilingual-ner-abb',
        'aggregation_strategy': 'simple'
    },
    'title_extraction': {
        'model': 'javdrher/decide-gemma3-270m',
        'max_new_tokens': 4000
    },
    'refinement': {
        'model': 'svercoutere/longformer-classifier-refinement-abb',
        'max_length': 2048,
        # Labels that can be refined by the model
        'refinable_labels': ['DATE', 'LOCATION', 'LOC', 'GPE'],
        # Mapping from model output indices to refined labels
        'label_classes': [
            'context_date',
            'context_location',
            'context_period',
            'entry_date',
            'expiry_date',
            'impact_location',
            'legal_date',
            'publication_date',
            'session_date',
            'validity_period'
        ]
    }
}

# Title extraction instruction for Gemma model
# Works for both Dutch and German legal documents
TITLE_EXTRACTION_INSTRUCTION = """
    Your task is to generate responses in JSON format.
    Ensure that your output strictly follows the provided JSON structure.
    Each key in the JSON should be correctly populated according to the instructions given.
    Pay attention to details and ensure the JSON is well-formed and valid.
    Only use phrases present in the given text.
    Extract the title from the following text and return it in JSON format with the key \"title\".
    The title should be the main heading or subject of the document.
"""

LABEL_MAPPINGS = {
    'spacy': {
        'PER': 'PERSON',
        'ORG': 'ORGANIZATION',
        'LOC': 'LOCATION',
        'GPE': 'LOCATION',
        'MISC': 'MISCELLANEOUS',
    },
    'flair': {
        'PER': 'PERSON',
        'ORG': 'ORGANIZATION',
        'LOC': 'LOCATION',
    },
    'huggingface': {},
    'regex': {},
}
