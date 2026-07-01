import pytest
from pydantic import ValidationError

from src.config import SegmentationConfig, LlmConfig


def test_nested_llm_defaults():
    cfg = SegmentationConfig()
    assert isinstance(cfg.llm, LlmConfig)
    assert cfg.llm.provider == "mistralai"
    assert cfg.llm.max_retries == 3
    assert cfg.llm.retry_delay == 15.0
    assert cfg.text_limit_chars == 100000


def test_nested_llm_from_dict():
    cfg = SegmentationConfig(
        llm={
            "provider": "mistralai",
            "model_name": "mistral-large-latest",
            "base_url": "https://api.mistral.ai/v1",
            "temperature": 0.1,
        },
        max_new_tokens=250000,
        text_limit_chars=1000000,
    )
    assert cfg.llm.model_name == "mistral-large-latest"
    assert cfg.llm.base_url == "https://api.mistral.ai/v1"
    assert cfg.llm.temperature == 0.1


def test_validator_rejects_low_max_new_tokens():
    with pytest.raises(ValidationError):
        SegmentationConfig(max_new_tokens=1000, text_limit_chars=100000)
