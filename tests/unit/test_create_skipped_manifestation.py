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
        return t


@patch("src.task.update")
def test_create_skipped_manifestation_inserts_correct_triples(mock_update, task):
    uri = task.create_manifestation(2048, "http://example.com/doc.pdf", skipped=True)

    assert uri.startswith("http://data.lblod.info/id/manifestations/")
    assert mock_update.called

    sparql = mock_update.call_args[0][0]
    assert "eli:Manifestation" in sparql
    assert "ext:skippedDueToPageLimit" in sparql
    assert "true" in sparql
    assert "http://example.com/doc.pdf" in sparql
    assert "2048" in sparql


@patch("src.task.update")
def test_create_skipped_manifestation_returns_unique_uris(mock_update, task):
    uri1 = task.create_manifestation(100, "http://a.com/a.pdf", skipped=True)
    uri2 = task.create_manifestation(200, "http://b.com/b.pdf", skipped=True)

    assert uri1 != uri2
