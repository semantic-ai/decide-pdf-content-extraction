from unittest.mock import MagicMock, patch

import src.segmentors as segmentors
from src.segmentors import LLMSegmentor


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
            model_name="mistral-large-2512",
            provider="mistralai",
            task=task,
        )

        mock_aligner.map_tags_to_original.return_value = "<title>Foo</title> bar"
        mock_aligner.get_annotations_from_tagged_text.return_value = {
            "spans": [{"labels": ["TITLE"], "start": 0, "end": 3, "text": "Foo"}]
        }

        seg.segment("Foo bar")

    mock_log.assert_called_once()
