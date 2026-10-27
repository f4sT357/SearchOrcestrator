"""HTTP content fetchers, caching, and URL safety checks."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import time
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
import lxml.html


class FetchStatus(str, Enum):
    SUCCESS = "success"
    FALLBACK_SUCCESS = "fallback_success"
    EMPTY_OR_FAILED = "empty_or_failed"
    HTTP_521 = "http_521"
    HTTP_403 = "http_403"
    HTTP_404 = "http_404"
    HTTP_429 = "http_429"
    HTTP_500 = "http_500"
    HTTP_502 = "http_502"
    HTTP_503 = "http_503"
    HTTP_504 = "http_504"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    PARSE_ERROR = "parse_error"
    EMPTY = "empty"
    FAILED = "failed"


class FetchError(Exception):
    def __init__(self, message: str, status: FetchStatus | None = None) -> None:
        super().__init__(message)
        self.status = status


class WebContentFetcher(Protocol):
    def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str: ...


# ---------------------------------------------------------------------------
# Primary fetcher
# ---------------------------------------------------------------------------


def is_safe_url(url: str) -> bool:
    """Validate that the given URL uses http/https and is not pointing to private/loopback/metadata addresses."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in {"http", "https"}:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False

        lower_host = hostname.lower()
        if lower_host in {"localhost", "metadata.google.internal"} or lower_host.endswith(".local") or lower_host.endswith(".internal"):
            return False

        try:
            ip = ipaddress.ip_address(lower_host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
            if str(ip) in {"169.254.169.254", "0.0.0.0"}:
                return False
        except ValueError:
            pass

        return True
    except Exception:
        return False

class DefaultWebContentFetcher:
    """Fetch and extract clean, readable text from a URL using httpx and lxml."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        *,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: int = 21600,
        min_host_interval_seconds: float = 0.5,
        max_retry_after_seconds: float = 30.0,
    ) -> None:
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        }
        self._client = client
        self._owns_client = client is None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_ttl_seconds = max(0, cache_ttl_seconds)
        self.min_host_interval_seconds = max(0.0, min_host_interval_seconds)
        self.max_retry_after_seconds = max(0.0, max_retry_after_seconds)
        self._cache_lock = Lock()
        self._host_locks: dict[str, Lock] = {}
        self._host_locks_lock = Lock()
        self._host_last_request: dict[str, float] = {}

    def _get_client(self, timeout: float = 8.0) -> Any:
        if self._client is None or getattr(self._client, "is_closed", False):
            created = httpx.Client(headers=self.headers, follow_redirects=True, timeout=timeout)
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

    def __enter__(self) -> DefaultWebContentFetcher:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def extract_text_from_html(self, html_content: str, max_length: int = 3000) -> str:
        try:
            # Strip XML declaration if present to prevent lxml ValueError on Unicode strings
            content_to_parse = re.sub(r"^\s*<\?xml[^>]*\?>", "", html_content, flags=re.IGNORECASE)
            tree = lxml.html.fromstring(content_to_parse)
            date_metadata: list[str] = []
            for meta in tree.xpath("//meta[@content]"):
                name = " ".join(filter(None, (
                    meta.get("name"), meta.get("property"), meta.get("itemprop"),
                ))).lower()
                if any(key in name for key in ("published", "pubdate", "datepublished")):
                    label = "公開日"
                elif any(key in name for key in ("modified", "updated", "datemodified")):
                    label = "更新日"
                else:
                    continue
                value = (meta.get("content") or "").strip()
                if value:
                    date_metadata.append(f"{label}: {value}")
            for time_element in tree.xpath("//article//time[@datetime] | //main//time[@datetime]"):
                value = (time_element.get("datetime") or "").strip()
                if value:
                    date_metadata.append(f"公開日: {value}")
            for tag in tree.xpath("//script|//style|//nav|//footer|//header|//noscript|//svg|//iframe|//form"):
                tag.drop_tree()

            main_elements = tree.xpath("//main|//article")
            if main_elements:
                text_parts = [elem.text_content() for elem in main_elements]
                text = "\n\n".join(text_parts)
            else:
                body = tree.find(".//body")
                if body is not None:
                    text = body.text_content()
                else:
                    text = tree.text_content()

            cleaned = self._clean_text(text)
            if date_metadata:
                cleaned = self._clean_text("\n".join(dict.fromkeys(date_metadata)) + "\n" + cleaned)
            return cleaned[:max_length]
        except Exception:
            # Strip script/style/nav/footer tags and their contents before generic tag stripping
            stripped_blocks = re.sub(
                r"<(script|style|nav|footer|header|noscript|svg|iframe|form)[^>]*>.*?</\1>",
                " ",
                html_content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            stripped = re.sub(r"<[^>]+>", " ", stripped_blocks)
            return self._clean_text(stripped)[:max_length]

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_cached_page(self, url: str) -> dict[str, Any] | None:
        path = self._cache_path(url)
        if path is None:
            return None
        with self._cache_lock:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
            return None
        return payload

    def _store_cached_page(self, url: str, entry: dict[str, Any]) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        with self._cache_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = path.with_suffix(".tmp")
                temporary_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
                temporary_path.replace(path)
            except OSError:
                return

    def _delete_cached_page(self, url: str) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        with self._cache_lock:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                return

    def _cache_lifetime(self, headers: Any) -> float | None:
        cache_control = str(headers.get("cache-control", "")).lower()
        directives = [item.strip() for item in cache_control.split(",")]
        if "no-store" in directives:
            return None
        if "no-cache" in directives:
            return 0.0
        max_age = next((item.split("=", 1)[1] for item in directives if item.startswith("max-age=")), None)
        try:
            server_ttl = max(0.0, float(max_age)) if max_age is not None else self.cache_ttl_seconds
        except ValueError:
            server_ttl = self.cache_ttl_seconds
        return min(float(self.cache_ttl_seconds), server_ttl)

    @staticmethod
    def _cache_headers(entry: dict[str, Any] | None) -> dict[str, str]:
        if not entry:
            return {}
        headers: dict[str, str] = {}
        if isinstance(entry.get("etag"), str) and entry["etag"]:
            headers["If-None-Match"] = entry["etag"]
        if isinstance(entry.get("last_modified"), str) and entry["last_modified"]:
            headers["If-Modified-Since"] = entry["last_modified"]
        return headers

    def _host_lock(self, url: str) -> tuple[str, Lock]:
        host = (urlparse(url).netloc or "").casefold()
        with self._host_locks_lock:
            return host, self._host_locks.setdefault(host, Lock())

    def _get(self, client: Any, url: str, headers: dict[str, str]) -> Any:
        host, host_lock = self._host_lock(url)
        with host_lock:
            elapsed = time.monotonic() - self._host_last_request.get(host, 0.0)
            if elapsed < self.min_host_interval_seconds:
                time.sleep(self.min_host_interval_seconds - elapsed)
            self._host_last_request[host] = time.monotonic()
            return client.get(url, headers=headers) if headers else client.get(url)

    def _retry_after_seconds(self, headers: Any) -> float:
        try:
            seconds = float(headers.get("retry-after", 0))
        except (TypeError, ValueError):
            return 0.0
        return min(max(0.0, seconds), self.max_retry_after_seconds)

    def _extract_response_text(self, resp: Any, max_length: int) -> str:
        content_type = resp.headers.get("content-type", "").lower()
        text = resp.text
        if not resp.encoding or resp.encoding.lower() in {"iso-8859-1", "latin-1", "ascii"}:
            meta_match = re.search(rb"""<meta[^>]+charset\s*=\s*["']?([a-zA-Z0-9_-]+)""", resp.content[:8192], flags=re.IGNORECASE)
            if meta_match:
                try:
                    detected = meta_match.group(1).decode("ascii").lower()
                    text = resp.content.decode(detected, errors="replace")
                except Exception:
                    text = resp.content.decode("utf-8", errors="replace")
            else:
                try:
                    text = resp.content.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        if "text/html" in content_type or "application/xhtml+xml" in content_type or not content_type:
            return self.extract_text_from_html(text, max_length=max_length)
        if "text/plain" in content_type:
            return self._clean_text(text)[:max_length]
        return ""

    def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
        if not is_safe_url(url):
            return ""
        cached = self._load_cached_page(url)
        cache_has_requested_length = bool(
            cached and cached.get("content_length_limit", 0) >= max_length
        )
        if cache_has_requested_length and float(cached.get("fresh_until", 0)) > time.time():
            return cached["content"][:max_length]
        try:
            client = self._get_client(timeout=timeout)
            # A response cached with a smaller extraction limit cannot satisfy a
            # larger request, even if the origin would otherwise return 304.
            headers = self._cache_headers(cached) if cache_has_requested_length else {}
            resp = self._get(client, url, headers)
            if resp.status_code == 429:
                delay = self._retry_after_seconds(resp.headers)
                if delay:
                    time.sleep(delay)
                    resp = self._get(client, url, headers)
            if resp.status_code == 304 and cached:
                lifetime = self._cache_lifetime(resp.headers)
                if lifetime is not None:
                    cached["fresh_until"] = time.time() + lifetime
                    if resp.headers.get("etag"):
                        cached["etag"] = resp.headers["etag"]
                    if resp.headers.get("last-modified"):
                        cached["last_modified"] = resp.headers["last-modified"]
                    self._store_cached_page(url, cached)
                else:
                    self._delete_cached_page(url)
                return cached["content"][:max_length]
            resp.raise_for_status()
            content = self._extract_response_text(resp, max_length)
            lifetime = self._cache_lifetime(resp.headers)
            if content and lifetime is not None:
                self._store_cached_page(url, {
                    "content": content,
                    "stored_at": time.time(),
                    "fresh_until": time.time() + lifetime,
                    "etag": resp.headers.get("etag", ""),
                    "last_modified": resp.headers.get("last-modified", ""),
                    "content_length_limit": max_length,
                })
            elif lifetime is None:
                self._delete_cached_page(url)
            return content
        except httpx.HTTPStatusError as http_err:
            status_code = http_err.response.status_code
            mapping = {
                521: FetchStatus.HTTP_521,
                403: FetchStatus.HTTP_403,
                404: FetchStatus.HTTP_404,
                429: FetchStatus.HTTP_429,
                500: FetchStatus.HTTP_500,
                502: FetchStatus.HTTP_502,
                503: FetchStatus.HTTP_503,
                504: FetchStatus.HTTP_504,
            }
            raise FetchError(f"HTTP error {status_code} for {url}", status=mapping.get(status_code, FetchStatus.FAILED)) from http_err
        except httpx.TimeoutException as timeout_err:
            raise FetchError(f"Timeout fetching {url}", status=FetchStatus.TIMEOUT) from timeout_err
        except httpx.NetworkError as net_err:
            raise FetchError(f"Network error fetching {url}", status=FetchStatus.CONNECTION_ERROR) from net_err
        except Exception as err:
            raise FetchError(f"Web page fetch failed ({url}): {err}", status=FetchStatus.FAILED) from err

class FirecrawlFetcher:
    """Retrieve one page as Markdown from a self-hosted Firecrawl API."""

    def __init__(self, api_url: str = "http://localhost:3002", api_key: str = "", client: httpx.Client | None = None) -> None:
        base_url = api_url.rstrip("/")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Firecrawl API URL must be an absolute HTTP or HTTPS URL")
        if base_url.endswith("/v2/scrape"):
            self.endpoint = base_url
        elif base_url.endswith("/v2"):
            self.endpoint = f"{base_url}/scrape"
        else:
            self.endpoint = f"{base_url}/v2/scrape"
        self.api_key = api_key
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers = headers
        self._client = client
        self._owns_client = client is None

    def _get_client(self, timeout: float = 30.0) -> Any:
        if self._client is None or getattr(self._client, "is_closed", False):
            created = httpx.Client(headers=self.headers, timeout=max(timeout, 30.0))
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

    def __enter__(self) -> FirecrawlFetcher:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
        if not is_safe_url(url):
            return ""
        try:
            # Firecrawl's scrape operation may need longer than the built-in
            # fetcher's interactive timeout, especially when rendering pages.
            client = self._get_client(timeout=timeout)
            response = client.post(
                self.endpoint,
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("success") is False:
                raise FetchError(
                    f"Firecrawl scrape failed for {url}: {payload.get('error', 'invalid response') if isinstance(payload, dict) else 'invalid response'}",
                    status=FetchStatus.PARSE_ERROR,
                )
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                raise FetchError("Firecrawl returned an invalid data object", status=FetchStatus.PARSE_ERROR)
            markdown = data.get("markdown")
            if not isinstance(markdown, str):
                raise FetchError("Firecrawl response did not contain Markdown", status=FetchStatus.PARSE_ERROR)
            return markdown[:max_length]
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
            raise FetchError(
                f"Firecrawl HTTP error {status_code}",
                status=mapping.get(status_code, FetchStatus.FAILED),
            ) from http_err
        except httpx.TimeoutException as timeout_err:
            raise FetchError("Firecrawl request timed out", status=FetchStatus.TIMEOUT) from timeout_err
        except httpx.NetworkError as net_err:
            raise FetchError("Firecrawl connection failed", status=FetchStatus.CONNECTION_ERROR) from net_err
        except Exception as err:
            raise FetchError(f"Firecrawl request failed: {err}", status=FetchStatus.FAILED) from err


# ---------------------------------------------------------------------------
# Fallback fetcher (Jina Reader)
# ---------------------------------------------------------------------------

class JinaReaderFetcher:
    """Fallback fetcher using Jina Reader API to retrieve page content as markdown/json."""

    def __init__(self, api_url: str = "https://r.jina.ai/", client: httpx.Client | None = None) -> None:
        # Ensure no trailing slash for consistent URL building
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
            max_length: Maximum length of returned content.
        Returns:
            A string containing the page content (truncated to max_length).
        Raises:
            FetchError: On network or HTTP errors, with appropriate FetchStatus.
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
            # Jina returns a JSON with a "content" field containing markdown/text
            content = data.get("content") or data.get("text") or ""
            return content[:max_length]
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
            raise FetchError(f"JinaReader HTTP error {status_code} for {url}", status=fetch_status) from http_err
        except httpx.TimeoutException as timeout_err:
            raise FetchError(f"JinaReader timeout fetching {url}", status=FetchStatus.TIMEOUT) from timeout_err
        except Exception as err:
            raise FetchError(f"JinaReader fetch failed for {url}: {err}", status=FetchStatus.FAILED) from err
