from unittest.mock import MagicMock, patch
import pytest

from src.task import PdfContentExtractionTask


class ConcretePdfTask(PdfContentExtractionTask):
    pass


@pytest.fixture
def task():
    with patch("src.task.PdfContentExtractionTask.__init__", return_value=None):
        t = ConcretePdfTask.__new__(ConcretePdfTask)
        t.logger = MagicMock()
        t.task_uri = "http://task/1"
        t.results_container_uris = []
        return t


def test_skipped_pdf_creates_skipped_manifestation_only(task):
    skipped_result = {
        "skipped": True,
        "pdf_url": "http://example.com/long.pdf",
        "byte_size": 999,
        "filename": "/tmp/long.pdf",
    }

    with patch.object(task, "fetch_data_from_input_container", return_value={"filenames": [], "download_urls": []}), \
         patch.object(task, "extract_content_from_pdf", return_value=[skipped_result]), \
         patch.object(task, "should_split_decisions", return_value=True), \
         patch.object(task, "create_manifestation", return_value="http://manif/skipped-1") as mock_manif, \
         patch.object(task, "create_eli_expression") as mock_expr, \
         patch.object(task, "create_eli_work") as mock_work, \
         patch.object(task, "create_output_container", return_value="http://container/1") as mock_container, \
         patch("src.task.get_segmentor"):

        task.process()

    mock_manif.assert_called_once_with(999, "http://example.com/long.pdf", skipped=True)
    mock_container.assert_called_once_with("http://manif/skipped-1")
    mock_expr.assert_not_called()
    mock_work.assert_not_called()
    assert task.results_container_uris == ["http://container/1"]


def test_normal_pdf_runs_full_pipeline(task):
    normal_result = {
        "content": "decision text",
        "pdf_url": "http://example.com/short.pdf",
        "byte_size": 512,
        "filename": "/tmp/short.pdf",
    }

    with patch.object(task, "fetch_data_from_input_container", return_value={"filenames": [], "download_urls": []}), \
         patch.object(task, "extract_content_from_pdf", return_value=[normal_result]), \
         patch.object(task, "should_split_decisions", return_value=True), \
         patch.object(task, "create_manifestation", return_value="http://manif/1") as mock_manif, \
         patch.object(task, "create_eli_expression", return_value="http://expr/1"), \
         patch.object(task, "create_eli_work", return_value="http://work/1"), \
         patch.object(task, "create_title_annotation", return_value="http://title/1"), \
         patch.object(task, "create_output_container", return_value="http://container/1"), \
         patch.object(task, "split_decisions", return_value=[{"text": "decision text", "title": "T", "title_start": 0, "title_end": 1}]), \
         patch("src.task.get_segmentor"), \
         patch("src.task.langdetect.detect", return_value="nl"):

        task.process()

    # normal path calls create_manifestation WITHOUT skipped=True
    mock_manif.assert_called_once_with(512, "http://example.com/short.pdf")
