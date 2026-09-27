"""A bounded, evidence-oriented web research workflow with web page fetching."""

from __future__ import annotations

import json
import math
import operator
import os
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Annotated, Any, Callable, Protocol, TypedDict
from urllib.parse import urlparse

import httpx
import lxml.html
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from source_scoring import (
    composite_score,
    domain_trust_score,
    extract_content_date,
    freshness_score,
    source_tier,
)


# ---------------------------------------------------------------------------
# Fetch status / error
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    model: str = "lfm2.5-8b-a1b-heretic-imatrix"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # The multilingual reranker is CPU-intensive. These conservative defaults
    # keep an interactive run responsive while retaining multiple sources.
    max_search_queries: int = 6
    max_search_rounds: int = 2
    results_per_query: int = 5
    reranked_results_per_query: int = 2
    content_candidate_results_per_query: int = 3
    relevance_weight: float = 0.50
    trust_weight: float = 0.30
    freshness_weight: float = 0.20
    fetch_web_content: bool = True
    max_content_length: int = 3000
    fetch_timeout: float = 8.0
    # Fallback fetcher configuration
    use_fallback_fetcher: bool = True
    jina_api_url: str = "https://r.jina.ai/"

    def __post_init__(self) -> None:
        weights = (self.relevance_weight, self.trust_weight, self.freshness_weight)
        if any(not math.isfinite(weight) or weight < 0 or weight > 1 for weight in weights):
            raise ValueError("Scoring weights must be between 0 and 1")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Scoring weights must sum to 1")

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            model=os.getenv("SEARCH_MODEL", cls.model),
            base_url=os.getenv("SEARCH_BASE_URL", cls.base_url),
            api_key=os.getenv("SEARCH_API_KEY", cls.api_key),
            relevance_weight=float(os.getenv("SEARCH_WEIGHT_RELEVANCE", "0.50")),
            trust_weight=float(os.getenv("SEARCH_WEIGHT_TRUST", "0.30")),
            freshness_weight=float(os.getenv("SEARCH_WEIGHT_FRESHNESS", "0.20")),
        )


# ---------------------------------------------------------------------------
# Model / API helpers
# ---------------------------------------------------------------------------

def fetch_available_models(base_url: str, api_key: str = "", timeout: float = 3.0) -> list[str]:
    """Fetch available model IDs from an OpenAI-compatible /v1/models endpoint."""
    if not base_url:
        return []
    url = base_url.rstrip("/")
    if not url.endswith("/models"):
        url = f"{url}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        # Use default SSL verification for security
        with httpx.Client(headers=headers, timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models_data = data.get("data", [])
            models = [m.get("id") for m in models_data if isinstance(m, dict) and m.get("id")]
            return sorted(models)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class SearchTask(BaseModel):
    aspect: str = Field(description="調査する観点")
    query: str = Field(description="実際に検索する検索クエリ")
    reason: str = Field(description="検索する理由")


class SearchPlan(BaseModel):
    tasks: list[SearchTask]


class Evaluation(BaseModel):
    sufficient: bool
    missing_information: list[str]
    weak_evidence: list[str]
    reason: str
    additional_queries: list[str]


class SearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str
    content: str | None = None
    relevance_score: float | None = None
    fetch_status: str | None = None
    # Source quality metadata (populated by rerank_results)
    source_reliability: str | None = None   # "primary" | "secondary" | "other"
    source_trust_score: float | None = None  # 0-1 domain trust
    source_freshness_score: float | None = None  # 0-1 freshness estimate
    combined_score: float | None = None      # final ranking score
    page_date: str | None = None

    def to_evidence_text(self) -> str:
        score_parts = []
        if self.relevance_score is not None:
            score_parts.append(f"関連度スコア: {self.relevance_score:.3f}")
        if self.source_reliability is not None:
            score_parts.append(f"ソース区分: {self.source_reliability}")
        if self.source_trust_score is not None:
            score_parts.append(f"信頼度: {self.source_trust_score:.2f}")
        if self.source_freshness_score is not None:
            score_parts.append(f"新鮮度: {self.source_freshness_score:.2f}")
        if self.page_date:
            score_parts.append(f"ページ日付: {self.page_date}")
        score_info = f" [{', '.join(score_parts)}]" if score_parts else ""
        text = f"【タイトル】: {self.title}{score_info}\n【URL】: {self.url}\n【概要/スニペット】: {self.snippet}"
        if self.content:
            text += f"\n【Webページ本文抜粋】:\n{self.content}"
        return text



class State(TypedDict):
    query: str
    plan: SearchPlan | None
    task: SearchTask | None
    results: Annotated[list[SearchResult], operator.add]
    evaluation: Evaluation | None
    summary: str
    search_query_count: Annotated[int, operator.add]
    search_round: Annotated[int, operator.add]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class SearchClient(Protocol):
    def text(self, query: str, *, max_results: int): ...


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs): ...


class WebContentFetcher(Protocol):
    def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str: ...


# ---------------------------------------------------------------------------
# Primary fetcher
# ---------------------------------------------------------------------------

class DefaultWebContentFetcher:
    """Fetch and extract clean, readable text from a URL using httpx and lxml."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def extract_text_from_html(self, html_content: str, max_length: int = 3000) -> str:
        try:
            tree = lxml.html.fromstring(html_content)
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
            stripped = re.sub(r"<[^>]+>", " ", html_content)
            return self._clean_text(stripped)[:max_length]

    def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        try:
            # Default SSL verification (verify=True) for security
            with httpx.Client(headers=self.headers, follow_redirects=True, timeout=timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "").lower()
                if "text/html" in content_type or "application/xhtml+xml" in content_type or not content_type:
                    return self.extract_text_from_html(resp.text, max_length=max_length)
                elif "text/plain" in content_type:
                    return self._clean_text(resp.text)[:max_length]
                else:
                    return ""
        except httpx.HTTPStatusError as http_err:
            status_code = http_err.response.status_code
            # Map known status codes to FetchStatus enum
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
            fetch_status = mapping.get(status_code, FetchStatus.FAILED)
            raise FetchError(f"HTTP error {status_code} for {url}", status=fetch_status) from http_err
        except httpx.TimeoutException as timeout_err:
            raise FetchError(f"Timeout fetching {url}", status=FetchStatus.TIMEOUT) from timeout_err
        except httpx.NetworkError as net_err:
            raise FetchError(f"Network error fetching {url}", status=FetchStatus.CONNECTION_ERROR) from net_err
        except Exception as err:
            raise FetchError(f"Webページ取得失敗 ({url}): {err}", status=FetchStatus.FAILED) from err


# ---------------------------------------------------------------------------
# Fallback fetcher (Jina Reader)
# ---------------------------------------------------------------------------

class JinaReaderFetcher:
    """Fallback fetcher using Jina Reader API to retrieve page content as markdown/json."""

    def __init__(self, api_url: str = "https://r.jina.ai/") -> None:
        # Ensure no trailing slash for consistent URL building
        self.api_url = api_url.rstrip("/")
        self.headers = {"Accept": "application/json"}

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
        if not url:
            return ""
        try:
            # Jina API format: https://r.jina.ai/http://example.com
            full_url = f"{self.api_url}/{url}"
            with httpx.Client(headers=self.headers, timeout=timeout) as client:
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def add_search_results(results: list[SearchResult], response, query: str) -> int:
    """Append only URL-unique results and return the number added."""
    existing_urls = {item.url for item in results if item.url}
    added_count = 0
    for item in response:
        url = item.get("href", "")
        if not url or url in existing_urls:
            continue
        results.append(SearchResult(
            query=query, title=item.get("title", ""), url=url,
            snippet=item.get("body", ""),
        ))
        existing_urls.add(url)
        added_count += 1
    return added_count


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in results:
        if item.url and item.url not in seen:
            seen.add(item.url)
            unique.append(item)
    return unique


def rerank_results(
    results: list[SearchResult],
    reranker: Reranker,
    top_k: int,
    *,
    weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> list[SearchResult]:
    valid_results = [item for item in results if item.url]
    if not valid_results:
        return []
    pairs = [(item.query, f"{item.title}\n{item.snippet}") for item in valid_results]
    raw_scores = reranker.predict(pairs, batch_size=32)

    enriched: list[SearchResult] = []
    for item, rel_score in zip(valid_results, raw_scores, strict=True):
        rel = float(rel_score)
        tier = source_tier(item.url)
        trust = domain_trust_score(item.url)
        fresh = freshness_score(item.url)
        comb = composite_score(
            rel, item.url,
            w_relevance=weights[0], w_trust=weights[1], w_freshness=weights[2],
        )
        enriched.append(item.model_copy(update={
            "relevance_score": rel,
            "source_reliability": tier,
            "source_trust_score": trust,
            "source_freshness_score": fresh,
            "combined_score": comb,
        }))

    return sorted(enriched, key=lambda r: r.combined_score or 0.0, reverse=True)[:top_k]


def select_diverse_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Prefer distinct publisher hosts in the final shortlist, then fill by rank."""
    ranked = sorted(results, key=lambda r: r.combined_score or 0.0, reverse=True)
    diverse: list[SearchResult] = []
    remaining: list[SearchResult] = []
    seen_hosts: set[str] = set()
    for item in ranked:
        host = (urlparse(item.url).hostname or "").lower().removeprefix("www.")
        if host and host not in seen_hosts:
            diverse.append(item)
            seen_hosts.add(host)
        else:
            remaining.append(item)
    return (diverse + remaining)[:max(0, top_k)]



def populate_content(
    results: list[SearchResult],
    fetcher: WebContentFetcher,
    *,
    timeout: float = 8.0,
    max_length: int = 3000,
    fallback_fetcher: WebContentFetcher | None = None,
    weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> list[SearchResult]:
    """Fetch web content for each result, falling back to Jina Reader on failure."""
    if not fetcher:
        return results
    updated: list[SearchResult] = []
    for item in results:
        if item.content is not None or not item.url:
            updated.append(item)
            continue
        # Try primary fetcher
        primary_status: FetchStatus = FetchStatus.FAILED
        try:
            content = fetcher.fetch(item.url, timeout=timeout, max_length=max_length)
            if content:
                updated.append(item.model_copy(update={
                    "content": content,
                    "fetch_status": FetchStatus.SUCCESS.value,
                }))
                continue
        except FetchError as exc:
            primary_status = exc.status or FetchStatus.FAILED
        except Exception:
            primary_status = FetchStatus.FAILED

        # Fallback attempt
        if fallback_fetcher:
            try:
                fallback_content = fallback_fetcher.fetch(item.url, timeout=timeout, max_length=max_length)
                if fallback_content:
                    updated.append(item.model_copy(update={
                        "content": fallback_content,
                        "fetch_status": FetchStatus.FALLBACK_SUCCESS.value,
                    }))
                    continue
            except Exception:
                pass

        # Both failed
        updated.append(item.model_copy(update={"fetch_status": FetchStatus.EMPTY_OR_FAILED.value}))
    rescored: list[SearchResult] = []
    for item in updated:
        page_date = extract_content_date(item.content)
        freshness = freshness_score(item.url, content=item.content)
        combined = composite_score(
            item.relevance_score, item.url, content=item.content,
            w_relevance=weights[0], w_trust=weights[1], w_freshness=weights[2],
        )
        rescored.append(item.model_copy(update={
            "page_date": page_date.isoformat() if page_date else None,
            "source_freshness_score": freshness,
            "combined_score": combined,
        }))
    return rescored


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Format search results with snippets and fetched page contents for LLM prompting."""
    parts = []
    for i, item in enumerate(results, start=1):
        parts.append(
            f"--- 検索結果 #{i} / 出典番号 [S{i}] / 検索クエリ: {item.query} ---\n"
            f"{item.to_evidence_text()}"
        )
    return "\n\n".join(parts)


_SOURCE_REF_RE = re.compile(r"\[S(\d+)\]")
_SOURCE_REF_CANDIDATE_RE = re.compile(r"\[S[^\]\r\n]*\d[^\]\r\n]*\]", re.IGNORECASE)
_INLINE_URL_RE = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)


def citation_issues(text: str, source_count: int) -> list[str]:
    """Return citation references or URLs that cannot be checked against evidence."""
    issues = []
    for token in _SOURCE_REF_CANDIDATE_RE.findall(text):
        match = _SOURCE_REF_RE.fullmatch(token)
        if not match or not 1 <= int(match.group(1)) <= source_count:
            issues.append(token)
    if _INLINE_URL_RE.search(text):
        issues.append("本文中のURL")
    return issues


def render_verified_sources(text: str, results: list[SearchResult]) -> str:
    """Keep only known source references and generate links from retrieved evidence."""
    source_count = len(results)
    parsed_refs = [
        (token, _SOURCE_REF_RE.fullmatch(token))
        for token in _SOURCE_REF_CANDIDATE_RE.findall(text)
    ]
    used_numbers = sorted({
        int(match.group(1)) for _, match in parsed_refs
        if match and 1 <= int(match.group(1)) <= source_count
    })
    cleaned = _SOURCE_REF_CANDIDATE_RE.sub(
        lambda found: (
            found.group(0)
            if (match := _SOURCE_REF_RE.fullmatch(found.group(0)))
            and 1 <= int(match.group(1)) <= source_count
            else ""
        ),
        text,
    )
    # URLs are emitted only by this renderer, from the retrieved source list.
    cleaned = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = _INLINE_URL_RE.sub("", cleaned)
    cleaned = cleaned.rstrip()

    if not used_numbers:
        cleaned += "\n\n> 注: 出典番号を確認できなかったため、本文の主張と取得ソースの対応を検証できていません。"
    if citation_issues(text, source_count):
        cleaned += "\n\n> 注: 取得済みソースと照合できない出典番号またはURLを除去しました。"

    if used_numbers:
        cleaned += "\n\n## 参照ソース\n"
        for number in used_numbers:
            item = results[number - 1]
            title = (item.title.replace("\r", " ").replace("\n", " ")
                     .replace("[", "\\[").replace("]", "\\]")
                     .replace("(", "\\(").replace(")", "\\)")) or item.url
            if item.url.startswith(("http://", "https://")):
                safe_url = item.url.replace("<", "%3C").replace(">", "%3E").replace(" ", "%20")
                cleaned += f"- [S{number}] [{title}](<{safe_url}>)\n"
            else:
                cleaned += f"- [S{number}] {title}（URL形式を確認できません）\n"
    return cleaned


def parse_json_from_response(text: str) -> dict:
    """Extract JSON dictionary from model response text robustly."""
    text = text.strip()
    # 1. Try markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # 2. Try outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 3. Direct parse
    return json.loads(text)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def create_workflow(
    settings: Settings,
    search: SearchClient | None = None,
    reranker: Reranker | None = None,
    llm: ChatOpenAI | None = None,
    fetcher: WebContentFetcher | None = None,
    fallback_fetcher: WebContentFetcher | None = None,
    log: Callable[[str], None] | None = None,
) -> Any:
    """Build the graph and initialize its dependencies."""
    if llm is None:
        llm = ChatOpenAI(model=settings.model, base_url=settings.base_url, api_key=settings.api_key)
    if search is None:
        search = DDGS()
    if reranker is None:
        reranker = CrossEncoder(settings.reranker_model)
    if fetcher is None and settings.fetch_web_content:
        fetcher = DefaultWebContentFetcher()
    elif not settings.fetch_web_content:
        fetcher = None
    # Instantiate fallback fetcher if enabled and not provided
    if fallback_fetcher is None and settings.use_fallback_fetcher:
        fallback_fetcher = JinaReaderFetcher(api_url=settings.jina_api_url)

    # Keep all model stages on the same as-of date for a single research run.
    current_date = date.today().isoformat()

    def log_detail(message: str) -> None:
        if log is not None:
            log(message)

    def log_selected_results(results: list[SearchResult], label: str = "採用候補") -> None:
        for index, item in enumerate(results, start=1):
            scores = []
            if item.relevance_score is not None:
                scores.append(f"関連度 {item.relevance_score:.3f}")
            if item.source_trust_score is not None:
                scores.append(f"信頼度 {item.source_trust_score:.2f}")
            if item.source_freshness_score is not None:
                scores.append(f"新鮮度 {item.source_freshness_score:.2f}")
            if item.combined_score is not None:
                scores.append(f"総合 {item.combined_score:.3f}")
            details = " / ".join(scores) or "スコアなし"
            tier = item.source_reliability or "区分不明"
            status = item.fetch_status or ("取得済み" if item.content else "本文未取得")
            page_date = f" / ページ日付 {item.page_date}" if item.page_date else ""
            log_detail(
                f"  {label} {index}: {item.title or item.url} / {tier} / {details}"
                f" / 本文 {status}{page_date}"
            )

    def planner(state: State):
        prompt = f"""あなたはリサーチプランナーです。現在日: {current_date}
質問: {state['query']}

重複しない体系的な調査計画を作成し、必ず以下のJSON形式のみを出力してください（説明文や前置きは不要です）。

```json
{{
  "tasks": [
    {{
      "aspect": "調査観点1",
      "query": "検索クエリ1",
      "reason": "検索する理由1"
    }},
    {{
      "aspect": "調査観点2",
      "query": "検索クエリ2",
      "reason": "検索する理由2"
    }}
  ]
}}
```"""
        try:
            response = llm.invoke(prompt)
            data = parse_json_from_response(response.content)
            plan = SearchPlan.model_validate(data)
            if not plan.tasks:
                plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
            log_detail(f"調査計画: {len(plan.tasks)}件の観点を作成")
        except Exception as err:
            print(f"調査計画のパース失敗 (フォールバック適用): {err}")
            plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
            log_detail(f"調査計画の解析に失敗したため、基本検索に切り替えました: {err}")
        for index, task in enumerate(plan.tasks, start=1):
            log_detail(f"  計画 {index}: {task.aspect} / クエリ: {task.query} / 理由: {task.reason}")
        if len(plan.tasks) > settings.max_search_queries:
            log_detail(f"検索クエリ上限により、計画の先頭 {settings.max_search_queries} 件を実行します")
        return {"plan": plan}

    def dispatch_searches(state: State):
        if state["plan"] is None:
            return []
        return [Send("search", {"task": task}) for task in state["plan"].tasks[:settings.max_search_queries]]

    def search_task(state: State):
        task = state.get("task")
        if task is None:
            return {"results": [], "search_query_count": 0}
        fresh_results: list[SearchResult] = []
        log_detail(f"検索開始: {task.query}（観点: {task.aspect}）")
        try:
            response = search.text(task.query, max_results=settings.results_per_query)
            add_search_results(fresh_results, response, task.query)
        except Exception as error:
            print(f"検索失敗 ({task.query}): {error}")
            log_detail(f"検索失敗: {task.query} / {error}")
            return {"results": [], "search_query_count": 1}

        log_detail(f"検索候補: {len(fresh_results)}件（取得上限: {settings.results_per_query}）")

        candidate_count = (
            settings.content_candidate_results_per_query
            if fetcher else settings.reranked_results_per_query
        )
        scoring_weights = (settings.relevance_weight, settings.trust_weight, settings.freshness_weight)
        top_results = rerank_results(fresh_results, reranker, candidate_count, weights=scoring_weights)
        log_detail(f"再ランキング: {len(fresh_results)}件から上位候補 {len(top_results)}件を選択")
        if fetcher:
            top_results = populate_content(
                top_results,
                fetcher,
                timeout=settings.fetch_timeout,
                max_length=settings.max_content_length,
                fallback_fetcher=fallback_fetcher,
                weights=scoring_weights,
            )
            top_results = select_diverse_results(top_results, settings.reranked_results_per_query)

        log_selected_results(top_results)

        return {
            "results": top_results,
            "search_query_count": 1,
        }

    def evaluate(state: State):
        unique_results = deduplicate_results(state.get("results", []))
        formatted_evidence = format_results_for_llm(unique_results)
        prompt = f"""あなたはリサーチ品質評価担当です。現在日: {current_date}
質問: {state['query']}
調査計画: {state['plan']}
収集した検索結果およびWebページ本文:
{formatted_evidence}

## 評価指針
各検索結果には「ソース区分」（primary / secondary / other）、「信頼度」（0-1）、「新鮮度」（0-1）が付与されています。
一次情報を優先しますが、二次情報であることだけを理由に証拠を弱いと判定してはいけません。次の基準で主張ごとに厳密に評価してください:
1. **まず公式文書、研究論文、規制・標準機関などの一次情報を探し、重要な事実・数値を照合する。**
2. **一次情報が見つからない、存在しない、アクセスできない、または当該主張を扱っていない場合は、その事情を区別する。検索結果がないだけで「一次情報が存在しない」と断定しない。**
3. **一次情報を確認できない場合、編集責任のある専門媒体・業界紙・調査機関など、独自取材や方法を示す高品質な二次情報を優先する。**
4. **高品質な二次情報を使う場合、同じ主張を裏付ける独立した情報源が複数あるかを確認する。目安は異なる発行元ドメインの2件以上。転載、プレスリリースの再掲、同一通信社記事の配信先違いは独立した裏付けとして数えない。独立性を確認できない場合はその不確かさを明記する。**
5. **各主張についてソース間の一致・矛盾を確認する。重要な数値・仕様（性能、容量、価格など）は、可能な限り一次情報または複数の独立した根拠で照合する。**
6. **「other」や信頼性を判断できないソース1件だけに依拠する主張は weak_evidence とする。高品質で独立した複数の二次情報が一致していれば、二次情報のみでも十分と判定してよい。**
7. **新鮮度スコアが低い情報や、ページ間の矛盾・未確認事項を指摘する。**
8. **追加検索が有効な不足だけ additional_queries に入れる。一次情報の所在を確かめる検索に加え、裏付けが不足する場合は独立した専門二次情報も探すクエリを提案する。**

検索クエリとURLを見て、ソースが独立しているか慎重に判断してください。異なるURLだけでは独立した根拠とは言えません。一次情報がなくても、独立した高品質な二次情報が十分に裏付けるなら sufficient を true にできます。その場合、reason に一次情報を確認できなかった事情と二次情報を採用した理由を簡潔に記してください。

Webページ本文やスニペットを確認し、必ず以下のJSON形式のみを出力してください（説明文や前置きは不要です）。

```json
{{
  "sufficient": false,
  "missing_information": ["不足している情報1"],
  "weak_evidence": ["根拠が弱い点1（独立確認のない情報源、矛盾など）"],
  "reason": "評価理由",
  "additional_queries": ["追加検索クエリ1（一次情報を対象）", "追加検索クエリ2"]
}}
```

※ 十分な情報が揃っている場合は sufficient を true、missing_information / weak_evidence / additional_queries を空配列 [] にしてください。独立した高品質な二次情報で十分に裏付けられる場合も十分と判定できます。"""
        raw_evaluation = ""
        try:
            response = llm.invoke(prompt)
            raw_evaluation = response.content
            data = parse_json_from_response(raw_evaluation)
            eval_obj = Evaluation.model_validate(data)
        except Exception as err:
            log_detail(f"品質評価の応答を解析できませんでした: {err}")
            log_detail("判断: 応答形式を修正して品質評価を1回だけ再試行します")
            try:
                if not raw_evaluation:
                    raise RuntimeError("品質評価モデルから応答を取得できませんでした")
                repair_prompt = f'''次の品質評価案を、指定形式の有効なJSONに修正してください。
評価案の内容を保ち、値を推測で追加しないでください。JSON以外は出力しないでください。
必須キー: sufficient (boolean), missing_information (string[]), weak_evidence (string[]), reason (string), additional_queries (string[])

評価案:
{raw_evaluation}

JSON例:
{{"sufficient":false,"missing_information":[],"weak_evidence":[],"reason":"評価理由","additional_queries":[]}}'''
                repaired = llm.invoke(repair_prompt)
                eval_obj = Evaluation.model_validate(parse_json_from_response(repaired.content))
                log_detail("品質評価の再試行に成功しました")
            except Exception as retry_error:
                print(f"品質評価の再試行にも失敗しました: {retry_error}")
                log_detail(f"品質評価の再試行にも失敗しました: {retry_error}")
                fallback_queries = []
                if (
                    state.get("search_round", 0) == 0
                    and state.get("search_query_count", 0) < settings.max_search_queries
                    and settings.max_search_rounds > 0
                ):
                    fallback_queries = [f"{state['query']} 公式 一次資料"]
                    log_detail("判断: 評価結果を得られなかったため、一次資料を対象に追加検索します")
                eval_obj = Evaluation(
                    sufficient=False,
                    missing_information=["品質評価の応答を解析できず、情報の十分性を確認できませんでした。"],
                    weak_evidence=[],
                    reason=f"品質評価の応答を解析できませんでした: {retry_error}",
                    additional_queries=fallback_queries,
                )

        eval_obj.sufficient = eval_obj.sufficient and not (
            eval_obj.additional_queries or eval_obj.missing_information or eval_obj.weak_evidence
        )
        log_detail(f"品質評価: {'十分' if eval_obj.sufficient else '不足'} / 理由: {eval_obj.reason}")
        for missing in eval_obj.missing_information:
            log_detail(f"  不足情報: {missing}")
        for weak in eval_obj.weak_evidence:
            log_detail(f"  根拠が弱い点: {weak}")
        for query in eval_obj.additional_queries:
            log_detail(f"  追加検索案: {query}")
        return {"evaluation": eval_obj}

    def should_continue(state: State):
        evaluation = state.get("evaluation")
        if evaluation is None or evaluation.sufficient:
            log_detail("次の判断: 品質評価が十分（または未設定）のためレポート作成へ進みます")
            return "analyze"
        if state["search_round"] >= settings.max_search_rounds:
            log_detail(f"次の判断: 検索ラウンド上限 {settings.max_search_rounds} に達したためレポート作成へ進みます")
            return "analyze"
        if state["search_query_count"] >= settings.max_search_queries:
            log_detail(f"次の判断: 検索クエリ上限 {settings.max_search_queries} に達したためレポート作成へ進みます")
            return "analyze"
        if not evaluation.additional_queries:
            log_detail("次の判断: 追加検索案がないためレポート作成へ進みます")
            return "analyze"
        log_detail("次の判断: 情報不足が残っているため、追加検索を実行します")
        return "additional_search"

    def additional_search(state: State):
        evaluation = state.get("evaluation")
        if evaluation is None:
            return {"results": [], "search_query_count": 0, "search_round": 1}
        remaining = settings.max_search_queries - state["search_query_count"]
        queries = evaluation.additional_queries[:max(remaining, 0)]
        log_detail(f"追加検索ラウンド: {len(queries)}件を実行（残りクエリ枠: {remaining}）")
        if len(queries) < len(evaluation.additional_queries):
            log_detail(
                f"検索クエリ上限により追加検索案 {len(evaluation.additional_queries) - len(queries)}件を見送ります"
            )
        existing_urls = {item.url for item in state.get("results", []) if item.url}
        delta_results: list[SearchResult] = []
        for query in queries:
            log_detail(f"追加検索開始: {query}")
            query_fresh: list[SearchResult] = []
            try:
                response = search.text(query, max_results=settings.results_per_query)
                for item in response:
                    url = item.get("href", "")
                    if not url or url in existing_urls:
                        continue
                    query_fresh.append(SearchResult(
                        query=query, title=item.get("title", ""), url=url,
                        snippet=item.get("body", ""),
                    ))
                    existing_urls.add(url)
                log_detail(
                    f"追加検索候補: {len(query_fresh)}件（既出URLを除外、取得上限: {settings.results_per_query}）"
                )
                candidate_count = (
                    settings.content_candidate_results_per_query
                    if fetcher else settings.reranked_results_per_query
                )
                scoring_weights = (settings.relevance_weight, settings.trust_weight, settings.freshness_weight)
                top_query_results = rerank_results(
                    query_fresh, reranker, candidate_count, weights=scoring_weights,
                )
                if fetcher:
                    top_query_results = populate_content(
                        top_query_results,
                        fetcher,
                        timeout=settings.fetch_timeout,
                        max_length=settings.max_content_length,
                        fallback_fetcher=fallback_fetcher,
                        weights=scoring_weights,
                    )
                    top_query_results = select_diverse_results(
                        top_query_results, settings.reranked_results_per_query,
                    )
                log_selected_results(top_query_results, label="追加検索採用候補")
                delta_results.extend(top_query_results)
            except Exception as error:
                print(f"追加検索失敗 ({query}): {error}")
                log_detail(f"追加検索失敗: {query} / {error}")
        return {
            "results": delta_results,
            "search_query_count": len(queries),
            "search_round": 1,
        }

    def analyze(state: State):
        unique_results = deduplicate_results(state.get("results", []))
        log_detail(f"レポート作成: 重複を除いた根拠ソース {len(unique_results)}件を使用")
        formatted_evidence = format_results_for_llm(unique_results)
        evaluation = state.get("evaluation")
        quality_note = ""
        if evaluation is not None and not evaluation.sufficient:
            quality_note = f"""

調査品質の注意:
この調査は十分性を確認できていません。
評価理由: {evaluation.reason}
未確認事項: {", ".join(evaluation.missing_information) or "なし"}
レポートでは冒頭に調査が不完全であることを明記し、根拠から確認できる範囲と未確認事項を区別してください。"""
        answer_prompt = f"""以下のWeb検索結果および取得したWebページ本文だけを根拠に質問へ回答してください。
現在日（この調査の基準日）: {current_date}
質問: {state['query']}

収集された情報源:
{formatted_evidence}
{quality_note}

指示:
1. {current_date} 時点での情報として回答し、更新時期が重要な情報はページ日付と根拠を確認してください。
2. Webページの本文に記載されている具体的な事実、数値、詳細を最大限に活用して、質の高い体系的なレポートを作成してください。
3. 事実を述べる段落には、根拠にした出典番号 [S1] の形式を付けてください。複数なら [S1][S2] とします。
4. 検索結果にない出典番号やURLを本文に書かないでください。参照ソース一覧はプログラムが付けます。
5. 取得情報から確認できないことは断定せず、調査品質が不十分な場合はその制約を明記してください。"""
        answer = llm.invoke(answer_prompt).content
        issues = citation_issues(answer, len(unique_results))
        has_citations = bool(_SOURCE_REF_RE.search(answer))
        if unique_results and (issues or not has_citations):
            log_detail(
                "出典検証: 不明な出典番号またはURLを検出したため、取得済みソースに限定して回答を再生成します"
                if issues else "出典検証: 出典番号がないため、根拠ソース番号を付けて回答を再生成します"
            )
            allowed_refs = ", ".join(f"[S{i}]" for i in range(1, len(unique_results) + 1))
            retry_prompt = f"""以下の回答案を根拠ソースに照合して修正してください。
現在日（この調査の基準日）: {current_date}
質問: {state['query']}
有効な出典番号: {allowed_refs}
取得済みの根拠:
{formatted_evidence}
元の回答:
{answer}

本文中のURLは書かず、事実を含む段落には有効な出典番号を付けてください。
根拠がない主張は削除するか、確認できないと明示してください。
調査品質が不十分な場合の注意:
{quality_note or "特段の不足は検出されていません。"}"""
            answer = llm.invoke(retry_prompt).content
        else:
            log_detail("出典検証: 回答中の出典番号は取得済みソースと照合できました")

        return {"summary": render_verified_sources(answer, unique_results)}

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("search", search_task)
    builder.add_node("evaluate", evaluate)
    builder.add_node("additional_search", additional_search)
    builder.add_node("analyze", analyze)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_searches)
    builder.add_edge("search", "evaluate")
    builder.add_conditional_edges("evaluate", should_continue, {
        "analyze": "analyze", "additional_search": "additional_search",
    })
    builder.add_edge("additional_search", "evaluate")
    builder.add_edge("analyze", END)
    return builder.compile()
