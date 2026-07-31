from unittest.mock import MagicMock, patch

import src.segmentors as segmentors
from src.segmentors import GemmaSegmentor, LLMSegmentor


def test_gemma_segmentor_logs_ml_call():
    task = MagicMock()
    seg = GemmaSegmentor(task_uri="http://task/1", endpoint="http://local-model/", task=task)

    generator_mock = MagicMock()
    generator_mock.return_value = [{"generated_text": "<TITLE>Foo</TITLE>"}]

    with patch.object(seg, "get_generator", return_value=generator_mock), \
         patch("decide_ai_service_base.ai_logging.record_ml_call") as mock_log:
        seg.segment("Foo bar")

    mock_log.assert_called_once()
    call_args = mock_log.call_args[0]
    assert call_args[0] is task
    assert call_args[1] == "http://local-model/"
    assert call_args[2] == seg.model_name       # model_uri (added by consolidation)
    assert isinstance(call_args[3], float)


def test_gemma_segmentor_skips_logging_without_task():
    seg = GemmaSegmentor(task_uri="http://task/1", endpoint="http://local-model/")

    generator_mock = MagicMock()
    generator_mock.return_value = [{"generated_text": "<TITLE>Foo</TITLE>"}]

    with patch.object(seg, "get_generator", return_value=generator_mock), \
         patch("decide_ai_service_base.ai_logging.record_ml_call") as mock_log:
        seg.segment("Foo bar")

    mock_log.assert_not_called()


@patch("src.segmentors.SpanAligner")
@patch("src.segmentors.log_date")
def test_llm_segmentor_propagates_task_to_analyzer(mock_log_date, mock_aligner):
    task = MagicMock()

    response = MagicMock()
    response.content = '{"document_classification": "Minute", "spans": [{"tag": "title", "start_line": 1, "end_line": 1}]}'
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    mock_chat = MagicMock()
    mock_chat.invoke.return_value = response

    with patch("src.LLMAnalyzer.init_chat_model", return_value=mock_chat), \
         patch("decide_ai_service_base.ai_logging.record_llm_call") as mock_log:
        seg = LLMSegmentor(
            task_uri="http://task/1",
            endpoint="https://api.mistral.ai/v1",
            model_name="mistral-large-latest",
            provider="mistralai",
            task=task,
        )

        mock_aligner.map_tags_to_original.return_value = "<title>Foo</title> bar"
        mock_aligner.get_annotations_from_tagged_text.return_value = {
            "spans": [{"labels": ["TITLE"], "start": 0, "end": 3, "text": "Foo"}]
        }

        seg.segment("Foo bar")

    mock_log.assert_called_once()
