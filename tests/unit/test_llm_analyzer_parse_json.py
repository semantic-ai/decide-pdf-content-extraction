import pytest

from src.LLMAnalyzer import LLMAnalyzer


@pytest.fixture
def parser():
    # Bypass __init__ so we don't construct a real chat model / need credentials.
    return LLMAnalyzer.__new__(LLMAnalyzer)


def test_parses_clean_json(parser):
    assert parser._parse_json('{"a": 1}') == {"a": 1}


def test_parses_fenced_json(parser):
    text = "```json\n{\"a\": 1}\n```"
    assert parser._parse_json(text) == {"a": 1}


def test_parses_embedded_json(parser):
    text = 'Here you go: {"a": 1} thanks'
    assert parser._parse_json(text) == {"a": 1}


def test_repairs_truncated_json(parser):
    # Missing closing brace — json_repair should recover it.
    text = '{"a": 1, "b": 2'
    assert parser._parse_json(text) == {"a": 1, "b": 2}


def test_raises_on_non_json(parser):
    with pytest.raises(ValueError):
        parser._parse_json("no json here at all")


def test_raises_on_empty(parser):
    with pytest.raises(ValueError):
        parser._parse_json("   ")
