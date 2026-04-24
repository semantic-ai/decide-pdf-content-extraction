"""
Configuration Management with Pydantic Validation

This module provides Pydantic models for validating application configuration
loaded from config.json. The configuration is designed to be extensible for
future settings beyond NER.
"""

import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator, ConfigDict, ValidationError


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
    model_output_cap_tokens: int | None = Field(
        default=None,
        ge=1000,
        description=(
            "Hard output-token ceiling of the model (e.g. 8192 for Mistral Large, "
            "32768 for gpt-4.1). When set, text_limit_chars is validated against it "
            "at startup. When omitted, no cap check is performed."
        )
    )
    text_limit_chars: int = Field(
        default=12000,
        ge=1000,
        description=(
            "Maximum characters of PDF text sent to the LLM per call. "
            "The output budget is auto-derived as int(text_limit_chars / 4 * 1.2) "
            "to account for ~4 chars/token and 20 % tag-insertion overhead. "
            "When model_output_cap_tokens is set, validated at startup so the "
            "derived output budget never exceeds the model cap."
        )
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature (lower = more deterministic)"
    )

    @model_validator(mode='after')
    def text_limit_within_model_output_cap(self) -> 'SegmentationConfig':
        if self.model_output_cap_tokens is None:
            return self
        max_safe_chars = int(self.model_output_cap_tokens * 4 / 1.2)
        if self.text_limit_chars > max_safe_chars:
            raise ValueError(
                f"text_limit_chars={self.text_limit_chars} would require "
                f"~{int(self.text_limit_chars / 4 * 1.2)} output tokens but "
                f"model_output_cap_tokens={self.model_output_cap_tokens} "
                f"(safe limit: {max_safe_chars} chars). "
                f"Reduce text_limit_chars or increase model_output_cap_tokens."
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


class AppConfig(BaseModel):
    """Root application configuration model."""

    # Reject extra fields not defined in the model
    model_config = ConfigDict(extra="forbid")

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


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load and validate configuration from config.json.

    Args:
        config_path: Path to config.json file. If None, searches for config.json
                    in the project root (parent of src/ directory).

    Returns:
        Validated AppConfig instance

    Raises:
        FileNotFoundError: If config.json is not found
        json.JSONDecodeError: If config.json contains invalid JSON
        ValidationError: If configuration doesn't match the Pydantic model
    """
    global _config

    # Return cached config if already loaded
    if _config is not None:
        return _config

    # Determine config file path
    if config_path is None:
        src_dir = Path(__file__).resolve().parent
        project_root = src_dir.parent
        config_path = project_root / "config.json"
    else:
        config_path = Path(config_path).resolve()

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Please create config.json at the project root."
        )

    # Read and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in config file {config_path}: {e}"
        ) from e

    # Validate with Pydantic
    try:
        _config = AppConfig.model_validate(config_data)
    except ValidationError as e:
        raise ValueError(
            f"Configuration validation failed for {config_path}:\n{e}"
        ) from e

    return _config


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
