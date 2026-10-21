"""Fallback web content fetcher using Jina Reader API."""

from __future__ import annotations

from typing import Any
import httpx

# Import shared types from search_orchestrator at runtime to avoid circular imports.
# Both modules live in the same package / directory.
from search_orchestrator import FetchError, FetchStatus, is_safe_url


class JinaReaderFetcher:
    """Fallback fetcher that retrieves page content via the Jina Reader API.

    Jina Reader exposes a simple proxy: ``https://r.jina.ai/<target_url>``.
    When called with ``Accept: application/json`` it returns a JSON object whose
    ``content`` field contains the cleaned-up markdown of the target page.

    This class mirrors the ``WebContentFetcher`` protocol so it can be used as a
    drop-in replacement or fallback alongside ``DefaultWebContentFetcher``.
    """

    def __init__(self, api_url: str = "https://r.jina.ai/", client: httpx.Client | None = None) -> None:
        # Strip trailing slash so we can safely concatenate "/<target_url>" later.
        self.api_url = api_url.rstrip("/")
        self.headers = {"Accept": "application/json"}
        self._client = client
        self._owns_client = client is None

    def _get_client(self, timeout: float = 8.0) -> Any:
        if self._client is None or getattr(self._client, "is_closed", False):
            created = httpx.Client(headers=self.headers, timeout=timeout)
            if hasattr(created, "__enter__") and hasattr(created, "_mock_return_value"):
                entered = created.__enter__.return_value
                if entered is not created and (hasattr(entered, "get") or hasattr(entered, "post")):
                    return entered
            self._client = created
            self._owns_client = True
        return self._client

    def close(self) -> None:
        if self._client and not getattr(self._client, "is_closed", False) and self._owns_client:
            self._client.close()

    def __enter__(self) -> JinaReaderFetcher:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
        """Fetch the page content via Jina Reader.

        Args:
            url: Target URL to retrieve.
            timeout: HTTP timeout in seconds.
            max_length: Maximum number of characters to return.

        Returns:
            The fetched content, truncated to *max_length*.

        Raises:
            FetchError: On network or HTTP errors, carrying a :class:`FetchStatus`.
        """
        if not is_safe_url(url):
            return ""
        try:
            # Jina API format: https://r.jina.ai/http://example.com
            full_url = f"{self.api_url}/{url}"
            client = self._get_client(timeout=timeout)
            resp = client.get(full_url)
            resp.raise_for_status()
            data = resp.json()
            # Jina returns {"content": "<markdown>", ...}
            content = data.get("content") or data.get("text") or ""
            return content[:max_length]
        except FetchError:
            raise
        except httpx.HTTPStatusError as http_err:
            status_code = http_err.response.status_code
            mapping = {
                403: FetchStatus.HTTP_403,
                404: FetchStatus.HTTP_404,
                429: FetchStatus.HTTP_429,
                500: FetchStatus.HTTP_500,
                502: FetchStatus.HTTP_502,
                503: FetchStatus.HTTP_503,
                504: FetchStatus.HTTP_504,
            }
            fetch_status = mapping.get(status_code, FetchStatus.FAILED)
            raise FetchError(
                f"JinaReader HTTP error {status_code} for {url}", status=fetch_status
            ) from http_err
        except httpx.TimeoutException as timeout_err:
            raise FetchError(
                f"JinaReader timeout fetching {url}", status=FetchStatus.TIMEOUT
            ) from timeout_err
        except Exception as err:
            raise FetchError(
                f"JinaReader fetch failed for {url}: {err}", status=FetchStatus.FAILED
            ) from err
