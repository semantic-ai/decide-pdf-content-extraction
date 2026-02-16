"""
NER Model Management

This module handles loading and caching of NER models with lazy initialization.
"""

import logging
from typing import Dict, Any
from transformers import pipeline
from .ner_config import NER_MODELS
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class to manage NER model loading and caching.

    This class ensures models are loaded only once and cached for reuse,
    improving performance and memory usage.
    """

    _instance = None
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_title_extraction_model(self):
        """
        Load and cache the title extraction model (Gemma).

        Returns:
            Loaded Hugging Face pipeline for text generation

        Raises:
            Exception: If model cannot be loaded
        """
        model_key = "title_extraction_pipeline"

        if model_key not in self._models:
            try:
                model_name = NER_MODELS['title_extraction']['model']

                # Explicitly load tokenizer and model first
                print(f"Loading title extraction model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="cpu"
                )

                # Create pipeline from the loaded model and tokenizer
                self._models[model_key] = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    # device="cpu"
                )
                print(f"Successfully loaded title extraction model")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                raise Exception(
                    f"Title extraction model could not be loaded. "
                    f"Error: {str(e)}\n{error_details}"
                )

        return self._models[model_key]


# Global model manager instance
model_manager = ModelManager()
