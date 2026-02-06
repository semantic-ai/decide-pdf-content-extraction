"""
NER Entity Extractors

This module contains different NER extraction methods organized by approach.

Classes:
- SpacyGeoAnalyzer: Belgian location extraction (returns spaCy Doc)
- BaseExtractor: Base class for factory pattern extractors (returns dicts)
- SpacyExtractor, FlairExtractor, etc.: Factory pattern implementations
"""
import json
import logging
from typing import List, Dict, Any
from .ner_models import model_manager
from .ner_config import TITLE_EXTRACTION_INSTRUCTION, NER_MODELS, LABEL_MAPPINGS
from .config import get_config


# ============================================================================
# FACTORY PATTERN EXTRACTORS (Return dicts for flexible NER)
# ============================================================================


class BaseExtractor:
    """Base class for all NER extractors."""

    def __init__(self, language: str = 'en', extractor_type: str = None):
        self.language = language
        self.config = get_config()
        self.extractor_type = extractor_type

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text. Must be implemented by subclasses.

        Args:
            text: Input text to process

        Returns:
            List of entity dictionaries with keys: text, label, start, end, confidence
        """
        raise NotImplementedError

    def _normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize entity: apply label mapping and normalize text/label format.

        Args:
            entity: Entity dictionary with 'text', 'label', 'start', 'end', optionally 'confidence'

        Returns:
            Normalized entity dictionary (new copy, doesn't mutate input)
        """
        entity = dict(entity)

        entity['text'] = entity['text'].strip()

        label = entity['label'].upper()

        if self.extractor_type and self.extractor_type in LABEL_MAPPINGS:
            mapping = LABEL_MAPPINGS[self.extractor_type]
            entity['label'] = mapping.get(label, label).upper()
        else:
            entity['label'] = label

        return entity

    def _resolve_overlaps(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve overlapping entities by keeping the one with highest confidence.
        When confidence ties, prefer longer spans.
        Only removes overlapping entities if they have the same label (entity type).

        Args:
            entities: List of entities

        Returns:
            List of entities (overlapping entities with same label resolved)
        """
        if not entities:
            return []

        # Sort by (confidence desc, length desc) - prefer higher confidence, then longer spans
        sorted_entities = sorted(
            entities,
            key=lambda x: (x.get('confidence', 1.0), x['end'] - x['start']),
            reverse=True
        )

        resolved = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already resolved entity of the SAME label
            # Using half-open intervals [start, end): ranges overlap if not (end1 <= start2 or end2 <= start1)
            overlaps_same_label = any(
                not (entity['end'] <= existing['start']
                     or existing['end'] <= entity['start'])
                and entity['label'] == existing['label']
                for existing in resolved
            )
            if not overlaps_same_label:
                resolved.append(entity)

        return resolved

    def post_process_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process entities: normalize and resolve overlaps.

        Normalizes entity text/labels, applies label mappings, and resolves overlapping
        entities with the same label by keeping the highest confidence one.

        Args:
            entities: List of raw entity dictionaries

        Returns:
            List of normalized and processed entities
        """
        if not self.config.ner.post_process:
            # Still normalize entities even if post-processing is disabled
            return [self._normalize_entity(e) for e in entities]

        # Normalize all entities first
        normalized = [self._normalize_entity(e) for e in entities]

        # Resolve overlaps (this also handles exact duplicates since they overlap)
        return self._resolve_overlaps(normalized)


class TitleExtractor(BaseExtractor):
    """Extract document title using Hugging Face Gemma model."""

    def __init__(self, language: str = 'nl'):
        super().__init__(language, extractor_type='title')

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract title from text using Gemma model.

        Returns a single entity with label 'TITLE'.
        """
        try:
            # Load the title extraction pipeline
            generator = model_manager.get_title_extraction_model()

            # Prepare the prompt combining instruction and text
            prompt = f"{TITLE_EXTRACTION_INSTRUCTION}\n\nText:\n{text}"

            # Create conversation format matching HuggingFace example
            conversation = [{"role": "user", "content": prompt}]

            # Generate the title (matching HF example format)
            max_tokens = NER_MODELS['title_extraction']['max_new_tokens']
            output = generator(
                conversation,
                max_new_tokens=max_tokens,
                return_full_text=False
            )[0]

            # Parse the generated text to extract JSON
            generated_text = output['generated_text']

            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                if '{' in generated_text and '}' in generated_text:
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    json_str = generated_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    title = result.get('title', '').strip()
                else:
                    # Fallback: treat whole response as title
                    title = generated_text.strip()
            except json.JSONDecodeError:
                # If JSON parsing fails, use the whole response
                title = generated_text.strip()

            if title:
                # Try to find the title in the original text
                start_pos = text.find(title)
                if start_pos != -1:
                    # Title found in original text
                    entities = [{
                        'text': title,
                        'label': 'TITLE',
                        'start': start_pos,
                        'end': start_pos + len(title),
                        'confidence': 1.0  # LLM doesn't provide confidence, default to 1.0
                    }]
                else:
                    # Title generated/extracted but not exact match in text
                    # Set start=0, end=0 to indicate it's a generated/inferred title
                    entities = [{
                        'text': title,
                        'label': 'TITLE',
                        'start': 0,
                        'end': 0,
                        'confidence': 0.8  # Lower confidence for generated titles
                    }]
                return entities

            return []

        except Exception as e:
            logging.warning(f"Error in title extraction: {e}")
            return []
