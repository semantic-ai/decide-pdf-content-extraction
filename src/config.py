"""
Configuration Management with Pydantic Validation

This module provides Pydantic models for validating application configuration
loaded from config.json. The configuration is designed to be extensible for
future settings beyond NER.
"""

import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from decide_ai_service_base.config import load_config

class NerConfig(BaseModel):
    """NER (Named Entity Recognition) configuration settings."""

    language: Literal["nl", "de", "en"] = Field(
        default="nl",
        description="Default language for NER extraction"
    )
    method: Literal["composite", "spacy", "huggingface", "flair", "regex", "title"] = Field(
        default="composite",
        description="Default NER extraction method"
    )
    post_process: bool = Field(
        default=True,
        description="Whether to apply post-processing to extracted entities"
    )
    labels: list[str] = Field(
        default_factory=lambda: ["CITY", "DOMAIN", "HOUSENUMBERS",
                                 "INTERSECTION", "POSTCODE", "PROVINCE", "ROAD", "STREET"],
        description="List of NER labels to extract"
    )
    enable_refinement: bool = Field(
        default=True,
        description="Whether to apply entity refinement to classify generic labels (DATE, LOCATION) into specific types"
    )


class SegmentationConfig(BaseModel):
    """Segmentation model configuration for document structure extraction."""

    model_name: str = Field(
        default="gpt-4.1",
        description="LLM deployment / model name"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the LLM endpoint"
    )
    endpoint: str | None = Field(
        default=None,
        description="API endpoint URL for the LLM service"
    )
    max_new_tokens: int = Field(
        default=26000,
        ge=100,
        description="Maximum tokens to generate"
    )
    text_limit_chars: int = Field(
        default=100000,
        ge=1000,
        description="Maximum characters of document text sent to the LLM; longer documents are silently truncated"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature (lower = more deterministic)"
    )

    @model_validator(mode="after")
    def max_new_tokens_exceeds_text_limit(self) -> "SegmentationConfig":
        # A rough upper bound: 1 char ≈ 0.25 tokens, so output tokens must cover at least the input chars
        if self.max_new_tokens < self.text_limit_chars // 4:
            raise ValueError(
                f"max_new_tokens ({self.max_new_tokens}) is too low for text_limit_chars "
                f"({self.text_limit_chars}); set max_new_tokens >= {self.text_limit_chars // 4}"
            )
        return self


class AppSettingsConfig(BaseModel):
    """Application-level settings."""

    mode: Literal["development", "production", "staging", "test"] = Field(
        default="development",
        description="Application mode (development, production, etc.)"
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="debug",
        description="Logging level (debug, info, warning, error)"
    )

    @field_validator('log_level', mode='before')
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize log level to lowercase and strip whitespace."""
        return v.strip().lower() if isinstance(v, str) else v


class AppConfig(BaseSettings):
    """Root application configuration model."""

    model_config = SettingsConfigDict(
        extra="forbid", # Reject extra fields not defined in the model
        env_nested_delimiter="__",  # allows SEGMENTATION__LLM__API_KEY etc.
        env_ignore_empty=True,      # treat empty string env vars as unset
    )

    app: AppSettingsConfig = Field(
        default_factory=AppSettingsConfig,
        description="Application-level settings"
    )
    ner: NerConfig = Field(
        default_factory=NerConfig,
        description="NER configuration settings"
    )
    segmentation: SegmentationConfig = Field(
        default_factory=SegmentationConfig,
        description="Segmentation model configuration"
    )


# Global config instance (lazy-loaded)
_config: AppConfig | None = None

def get_config() -> AppConfig:
    """
    Get the current configuration instance.

    If not yet loaded, attempts to load from default location.

    Returns:
        AppConfig instance
    """
    if _config is None:
        return load_config()
    return _config


def reset_config():
    """Reset the global config cache (useful for testing)."""
    global _config
    _config = None
