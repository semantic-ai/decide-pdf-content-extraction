from unittest.mock import MagicMock, patch

from src.ai_logging import record_ml_call, record_llm_call


def test_record_ml_call_invokes_task():
    task = MagicMock()
    with patch("src.ai_logging.get_agent_uri", return_value="http://agent/uri"):
        record_ml_call(task, "http://endpoint/", 1.5)

    task.record_ai_call.assert_called_once_with(
        endpoint="http://endpoint/",
        model_uri="http://agent/uri",
        tokens_in=0,
        tokens_out=0,
        duration=1.5,
    )


def test_record_llm_call_with_usage_metadata():
    task = MagicMock()
    response = MagicMock()
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    with patch("src.ai_logging.get_agent_uri", return_value="http://agent/uri"):
        record_llm_call(task, "http://endpoint/", response, 2.0)

    task.record_ai_call.assert_called_once_with(
        endpoint="http://endpoint/",
        model_uri="http://agent/uri",
        tokens_in=100,
        tokens_out=50,
        duration=2.0,
    )


def test_record_llm_call_without_usage_metadata():
    task = MagicMock()
    response = MagicMock(spec=[])

    with patch("src.ai_logging.get_agent_uri", return_value="http://agent/uri"):
        record_llm_call(task, "http://endpoint/", response, 2.0)

    task.record_ai_call.assert_called_once_with(
        endpoint="http://endpoint/",
        model_uri="http://agent/uri",
        tokens_in=0,
        tokens_out=0,
        duration=2.0,
    )


def test_agent_uri_used_as_model_uri():
    task = MagicMock()
    with patch("src.ai_logging.get_agent_uri", return_value="http://custom/agent"):
        record_ml_call(task, "http://endpoint/", 0.5)

    kwargs = task.record_ai_call.call_args[1]
    assert kwargs["model_uri"] == "http://custom/agent"
