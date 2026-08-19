"""Unit tests for FetchStatus, FetchError, DefaultWebContentFetcher, JinaReaderFetcher,
and the populate_content fallback logic."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch, PropertyMock
import httpx
import pytest

from search_orchestrator import (
    DefaultWebContentFetcher,
    FetchError,
    FetchStatus,
    SearchResult,
    populate_content,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(url: str = "http://example.com") -> SearchResult:
    return SearchResult(query="test", title="Test", url=url, snippet="snippet")


def _mock_httpx_response(status_code: int = 200, text: str = "<html><body>hello</body></html>",
                          content_type: str = "text/html") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# DefaultWebContentFetcher
# ---------------------------------------------------------------------------

class TestDefaultWebContentFetcher:
    def test_fetch_success(self):
        fetcher = DefaultWebContentFetcher()
        resp = _mock_httpx_response(200, "<html><body><p>Hello World</p></body></html>")
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = resp
            result = fetcher.fetch("http://example.com")
        assert "Hello World" in result

    def test_fetch_empty_url(self):
        fetcher = DefaultWebContentFetcher()
        assert fetcher.fetch("") == ""

    def test_fetch_non_http_url(self):
        fetcher = DefaultWebContentFetcher()
        assert fetcher.fetch("ftp://example.com") == ""

    def test_fetch_http_521_raises_fetch_error(self):
        fetcher = DefaultWebContentFetcher()
        http_response = MagicMock()
        http_response.status_code = 521
        err = httpx.HTTPStatusError("521", request=MagicMock(), response=http_response)
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value.raise_for_status.side_effect = err
            with pytest.raises(FetchError) as exc_info:
                fetcher.fetch("http://csdn.net/article")
        assert exc_info.value.status == FetchStatus.HTTP_521

    def test_fetch_http_403_raises_fetch_error(self):
        fetcher = DefaultWebContentFetcher()
        http_response = MagicMock()
        http_response.status_code = 403
        err = httpx.HTTPStatusError("403", request=MagicMock(), response=http_response)
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value.raise_for_status.side_effect = err
            with pytest.raises(FetchError) as exc_info:
                fetcher.fetch("http://example.com")
        assert exc_info.value.status == FetchStatus.HTTP_403

    def test_fetch_timeout_raises_fetch_error(self):
        fetcher = DefaultWebContentFetcher()
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(FetchError) as exc_info:
                fetcher.fetch("http://example.com")
        assert exc_info.value.status == FetchStatus.TIMEOUT

    def test_fetch_network_error_raises_connection_error(self):
        fetcher = DefaultWebContentFetcher()
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.side_effect = httpx.NetworkError("network error")
            with pytest.raises(FetchError) as exc_info:
                fetcher.fetch("http://example.com")
        assert exc_info.value.status == FetchStatus.CONNECTION_ERROR


# ---------------------------------------------------------------------------
# JinaReaderFetcher
# ---------------------------------------------------------------------------

class TestJinaReaderFetcher:
    def _fetcher(self):
        from fallback_fetcher import JinaReaderFetcher
        return JinaReaderFetcher(api_url="https://r.jina.ai/")

    def test_fetch_success(self):
        fetcher = self._fetcher()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"content": "# Hello from Jina"}
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = resp
            result = fetcher.fetch("http://example.com")
        assert "Hello from Jina" in result

    def test_fetch_truncates_to_max_length(self):
        fetcher = self._fetcher()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"content": "x" * 10000}
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = resp
            result = fetcher.fetch("http://example.com", max_length=100)
        assert len(result) == 100

    def test_fetch_empty_url(self):
        fetcher = self._fetcher()
        assert fetcher.fetch("") == ""

    def test_fetch_http_403_raises_fetch_error(self):
        fetcher = self._fetcher()
        http_response = MagicMock()
        http_response.status_code = 403
        err = httpx.HTTPStatusError("403", request=MagicMock(), response=http_response)
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value.raise_for_status.side_effect = err
            with pytest.raises(FetchError) as exc_info:
                fetcher.fetch("http://example.com")
        assert exc_info.value.status == FetchStatus.HTTP_403

    def test_fetch_timeout_raises_fetch_error(self):
        fetcher = self._fetcher()
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(FetchError) as exc_info:
                fetcher.fetch("http://example.com")
        assert exc_info.value.status == FetchStatus.TIMEOUT


# ---------------------------------------------------------------------------
# populate_content
# ---------------------------------------------------------------------------

class TestPopulateContent:
    def test_primary_success(self):
        fetcher = MagicMock()
        fetcher.fetch.return_value = "primary content"
        results = [_make_result()]
        out = populate_content(results, fetcher)
        assert out[0].content == "primary content"
        assert out[0].fetch_status == FetchStatus.SUCCESS.value

    def test_primary_fails_fallback_succeeds(self):
        primary = MagicMock()
        primary.fetch.side_effect = FetchError("521", status=FetchStatus.HTTP_521)
        fallback = MagicMock()
        fallback.fetch.return_value = "fallback content"
        results = [_make_result()]
        out = populate_content(results, primary, fallback_fetcher=fallback)
        assert out[0].content == "fallback content"
        assert out[0].fetch_status == FetchStatus.FALLBACK_SUCCESS.value

    def test_both_fail(self):
        primary = MagicMock()
        primary.fetch.side_effect = FetchError("521", status=FetchStatus.HTTP_521)
        fallback = MagicMock()
        fallback.fetch.side_effect = FetchError("timeout", status=FetchStatus.TIMEOUT)
        results = [_make_result()]
        out = populate_content(results, primary, fallback_fetcher=fallback)
        assert out[0].content is None
        assert out[0].fetch_status == FetchStatus.EMPTY_OR_FAILED.value

    def test_no_fallback_fetcher_primary_fails(self):
        primary = MagicMock()
        primary.fetch.side_effect = FetchError("521", status=FetchStatus.HTTP_521)
        results = [_make_result()]
        out = populate_content(results, primary, fallback_fetcher=None)
        assert out[0].content is None
        assert out[0].fetch_status == FetchStatus.EMPTY_OR_FAILED.value

    def test_use_fallback_fetcher_false_skips_fallback(self):
        """When fallback_fetcher=None (simulating use_fallback_fetcher=False),
        the fallback should never be called."""
        primary = MagicMock()
        primary.fetch.side_effect = FetchError("521", status=FetchStatus.HTTP_521)
        fallback = MagicMock()
        results = [_make_result()]
        out = populate_content(results, primary, fallback_fetcher=None)
        fallback.fetch.assert_not_called()
        assert out[0].fetch_status == FetchStatus.EMPTY_OR_FAILED.value

    def test_primary_returns_empty_string_tries_fallback(self):
        """If primary returns '' (no raise), fallback should be attempted."""
        primary = MagicMock()
        primary.fetch.return_value = ""
        fallback = MagicMock()
        fallback.fetch.return_value = "fallback content"
        results = [_make_result()]
        out = populate_content(results, primary, fallback_fetcher=fallback)
        assert out[0].content == "fallback content"
        assert out[0].fetch_status == FetchStatus.FALLBACK_SUCCESS.value

    def test_skip_already_populated(self):
        """Results with existing content should not be re-fetched."""
        fetcher = MagicMock()
        result = _make_result()
        result = result.model_copy(update={"content": "existing"})
        out = populate_content([result], fetcher)
        fetcher.fetch.assert_not_called()
        assert out[0].content == "existing"

    def test_no_fetcher_returns_results_unchanged(self):
        results = [_make_result()]
        out = populate_content(results, None)
        assert out == results