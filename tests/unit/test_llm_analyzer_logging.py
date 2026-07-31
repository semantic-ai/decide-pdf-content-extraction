from unittest.mock import MagicMock, patch

from src.LLMAnalyzer import LLMAnalyzer


def _make_analyzer(task=None, endpoint="http://endpoint/"):
    analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
    analyzer._task = task
    analyzer._endpoint = endpoint
    analyzer._provider = "test"
    analyzer.model_name = "test-model"
    analyzer._max_retries = 1
    analyzer._retry_delay = 0.0
    return analyzer


def test_analyze_logs_llm_call_when_task_provided():
    task = MagicMock()
    analyzer = _make_analyzer(task=task)

    response = MagicMock()
    response.content = '{"key": "value"}'
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    analyzer._chat_model = MagicMock()
    analyzer._chat_model.invoke.return_value = response

    with patch("decide_ai_service_base.ai_logging.record_llm_call") as mock_log:
        analyzer.analyze_single_entry(
            text="hello",
            system_prompt="sys",
            user_prompt_template="user: {text}",
            expected_schema={"key": {"default": "", "type": str}},
        )

    mock_log.assert_called_once()
    call_args = mock_log.call_args[0]
    assert call_args[0] is task
    assert call_args[1] == "http://endpoint/"
    assert call_args[2] == "test-model"        # model_uri (added by consolidation)
    assert call_args[3] is response
    assert isinstance(call_args[4], float)


def test_analyze_skips_logging_when_task_none():
    analyzer = _make_analyzer(task=None)

    response = MagicMock()
    response.content = '{"key": "value"}'
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    analyzer._chat_model = MagicMock()
    analyzer._chat_model.invoke.return_value = response

    with patch("decide_ai_service_base.ai_logging.record_llm_call") as mock_log:
        analyzer.analyze_single_entry(
            text="hello",
            system_prompt="sys",
            user_prompt_template="user: {text}",
            expected_schema={"key": {"default": "", "type": str}},
        )

    mock_log.assert_not_called()
