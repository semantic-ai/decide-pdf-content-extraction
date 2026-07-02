import os
from unittest.mock import MagicMock, patch, mock_open
import pytest

from src.task import PdfContentExtractionTask


class ConcretePdfTask(PdfContentExtractionTask):
    pass


@pytest.fixture
def task(monkeypatch):
    monkeypatch.setenv("MOUNTED_SHARE_FOLDER", "/mnt/share")
    monkeypatch.setenv("APACHE_TIKA_URL", "http://apache-tika:9998/tika")
    with patch("src.task.PdfContentExtractionTask.__init__", return_value=None):
        t = ConcretePdfTask.__new__(ConcretePdfTask)
        t.logger = MagicMock()
        return t


def _make_tika_meta_response(n_pages: int):
    resp = MagicMock()
    resp.json.return_value = {"xmpTPg:NPages": str(n_pages)}
    resp.raise_for_status = MagicMock()
    return resp


def _make_tika_content_response(content: str):
    resp = MagicMock()
    resp.content = content.encode("utf-8")
    resp.raise_for_status = MagicMock()
    return resp


@patch("src.task.os.makedirs")
@patch("src.task.requests.get")
@patch("src.task.os.path.isfile", return_value=True)
@patch("src.task.os.path.getsize", return_value=1024)
@patch("builtins.open", mock_open(read_data=b"pdf bytes"))
def test_pdf_over_limit_is_skipped(mock_getsize, mock_isfile, mock_get, mock_makedirs, task):
    meta_resp = _make_tika_meta_response(11)

    with patch("src.task.requests.put", return_value=meta_resp) as mock_put:
        results = task.extract_content_from_pdf({
            "filenames": ["doc.pdf"],
            "download_urls": ["http://example.com/doc.pdf"],
        })

    assert len(results) == 1
    assert results[0]["skipped"] is True
    assert "content" not in results[0]
    # Only one Tika call (meta), not two
    assert mock_put.call_count == 1
    called_url = mock_put.call_args[0][0]
    assert called_url.endswith("/meta")


@patch("src.task.os.makedirs")
@patch("src.task.requests.get")
@patch("src.task.os.path.isfile", return_value=True)
@patch("src.task.os.path.getsize", return_value=1024)
@patch("builtins.open", mock_open(read_data=b"pdf bytes"))
def test_pdf_at_limit_is_processed(mock_getsize, mock_isfile, mock_get, mock_makedirs, task):
    meta_resp = _make_tika_meta_response(10)
    content_resp = _make_tika_content_response("decision text")

    with patch("src.task.requests.put", side_effect=[meta_resp, content_resp]):
        results = task.extract_content_from_pdf({
            "filenames": ["doc.pdf"],
            "download_urls": ["http://example.com/doc.pdf"],
        })

    assert len(results) == 1
    assert results[0].get("skipped") is not True
    assert results[0]["content"] == "decision text"


@patch("src.task.os.makedirs")
@patch("src.task.requests.get")
@patch("src.task.os.path.isfile", return_value=True)
@patch("src.task.os.path.getsize", return_value=1024)
@patch("builtins.open", mock_open(read_data=b"pdf bytes"))
def test_pdf_under_limit_is_processed(mock_getsize, mock_isfile, mock_get, mock_makedirs, task):
    meta_resp = _make_tika_meta_response(5)
    content_resp = _make_tika_content_response("some content")

    with patch("src.task.requests.put", side_effect=[meta_resp, content_resp]):
        results = task.extract_content_from_pdf({
            "filenames": ["doc.pdf"],
            "download_urls": ["http://example.com/doc.pdf"],
        })

    assert len(results) == 1
    assert results[0]["content"] == "some content"
