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
    FirecrawlFetcher,
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
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
            200, text="<html><body>Hello World</body></html>", headers={"content-type": "text/html"}
        ))) as client:
            assert DefaultWebContentFetcher(client=client).fetch("https://example.com") == "Hello World"

    @pytest.mark.parametrize("url", ["", "ftp://example.com", "http://127.0.0.1/admin", "http://169.254.169.254/", "http://localhost:8080/", "http://10.0.0.1/"])
    def test_fetch_rejects_ssrf_urls(self, url):
        assert DefaultWebContentFetcher().fetch(url) == ""

    @pytest.mark.parametrize("status", [403, 404, 429, 500, 502, 503, 504, 521])
    def test_http_error_status(self, status):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status))) as client:
            with pytest.raises(FetchError) as exc:
                DefaultWebContentFetcher(client=client).fetch("https://example.com")
        assert exc.value.status.value == f"http_{status}"

    @pytest.mark.parametrize("error, status", [(httpx.ReadTimeout("timeout"), FetchStatus.TIMEOUT), (httpx.ConnectError("failed"), FetchStatus.CONNECTION_ERROR)])
    def test_network_errors(self, error, status):
        def handler(request):
            raise error
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(FetchError) as exc:
                DefaultWebContentFetcher(client=client).fetch("https://example.com")
        assert exc.value.status == status


class TestFirecrawlFetcher:
    def test_fetch_posts_markdown_request_and_truncates_result(self, monkeypatch):
        import json
        monkeypatch.setattr("web_fetchers.resolve_public_addresses", lambda *a: ["93.184.216.34"])
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"success": True, "data": {"markdown": "# Firecrawl result"}})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = FirecrawlFetcher("http://localhost:3002", client=client, network_isolated=True).fetch("https://example.com/page", max_length=8)
        assert result == "# Firecr"
        assert str(requests[0].url) == "http://localhost:3002/v2/scrape"
        assert json.loads(requests[0].content) == {"url": "https://example.com/page", "formats": ["markdown"], "onlyMainContent": True}

    @pytest.mark.parametrize("payload", [{"success": False, "error": "failed"}, {"success": True, "data": {"html": "x"}}])
    def test_invalid_payload(self, payload, monkeypatch):
        monkeypatch.setattr("web_fetchers.resolve_public_addresses", lambda *a: ["93.184.216.34"])
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as client:
            with pytest.raises(FetchError) as exc:
                FirecrawlFetcher(client=client, network_isolated=True).fetch("https://example.com")
        assert exc.value.status == FetchStatus.PARSE_ERROR

    def test_rejects_invalid_api_url(self):
        with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
            FirecrawlFetcher("localhost:3002")


class TestJinaReaderFetcher:
    def test_fetch_success(self, monkeypatch):
        from fallback_fetcher import JinaReaderFetcher
        monkeypatch.setattr("web_fetchers.resolve_public_addresses", lambda *a: ["93.184.216.34"])
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"content": "# Hello from Jina"}))) as client:
            fetcher = JinaReaderFetcher(client=client, network_isolated=True)
            assert fetcher.fetch("https://example.com", max_length=7) == "# Hello"


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
        assert out[0].fetch_error == "primary=http_521; fallback=timeout"

    def test_no_fallback_fetcher_primary_fails(self):
        primary = MagicMock()
        primary.fetch.side_effect = FetchError("521", status=FetchStatus.HTTP_521)
        results = [_make_result()]
        out = populate_content(results, primary, fallback_fetcher=None)
        assert out[0].content is None
        assert out[0].fetch_status == FetchStatus.EMPTY_OR_FAILED.value
        assert out[0].fetch_error == "primary=http_521"

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
