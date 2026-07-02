import inspect
from unittest.mock import patch, MagicMock

import src.segmentors as segmentors
from src.segmentors import LLMSegmentor


def _make_segmentor(mock_analyzer_cls):
    seg = LLMSegmentor(
        task_uri="http://task/1",
        endpoint="https://api.mistral.ai/v1",
        model_name="mistral-large-latest",
        provider="mistralai",
    )
    seg.analyzer.analyze_single_entry.return_value = {
        "tagged_text": "TAGGED",
        "document_classification": "Minute",
    }
    return seg


def test_segment_is_synchronous_and_async_segment_removed():
    assert not inspect.iscoroutinefunction(LLMSegmentor.segment)
    assert not hasattr(LLMSegmentor, "async_segment")


@patch("src.segmentors.SpanAligner")
@patch("src.segmentors.log_date")
@patch("src.segmentors.LLMAnalyzer")
def test_segment_returns_spans_and_logs_progress(mock_analyzer_cls, mock_log_date, mock_aligner):
    seg = _make_segmentor(mock_analyzer_cls)
    mock_aligner.map_tags_to_original.return_value = "MAPPED"
    mock_aligner.get_annotations_from_tagged_text.return_value = {
        "spans": [{"labels": ["TITLE"], "start": 0, "end": 3, "text": "Foo"}]
    }

    result = seg.segment("Foo bar")

    assert result == [{"label": "TITLE", "start": 0, "end": 3, "text": "Foo"}]
    # three progress markers preserved
    assert mock_log_date.call_count == 3
    seg.analyzer.analyze_single_entry.assert_called_once()


@patch("src.segmentors.SpanAligner")
@patch("src.segmentors.log_date")
@patch("src.segmentors.LLMAnalyzer")
def test_segment_empty_tagged_text_returns_empty(mock_analyzer_cls, mock_log_date, mock_aligner):
    seg = LLMSegmentor(task_uri="http://task/1", endpoint="x", model_name="m", provider="mistralai")
    seg.analyzer.analyze_single_entry.return_value = {"tagged_text": "", "document_classification": "Non-Decision"}

    result = seg.segment("whatever")

    assert result == []
