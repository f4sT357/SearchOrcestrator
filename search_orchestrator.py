"""A bounded, evidence-oriented web research workflow with web page fetching."""

from __future__ import annotations

import json
import operator
import os
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Annotated, Any, Protocol, TypedDict

import httpx
import lxml.html
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


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
    fetch_web_content: bool = True
    max_content_length: int = 3000
    fetch_timeout: float = 8.0
    # Fallback fetcher configuration
    use_fallback_fetcher: bool = True
    jina_api_url: str = "https://r.jina.ai/"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            model=os.getenv("SEARCH_MODEL", cls.model),
            base_url=os.getenv("SEARCH_BASE_URL", cls.base_url),
            api_key=os.getenv("SEARCH_API_KEY", cls.api_key),
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

    def to_evidence_text(self) -> str:
        score_info = f" [関連度スコア: {self.relevance_score:.3f}]" if self.relevance_score is not None else ""
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


def rerank_results(results: list[SearchResult], reranker: Reranker, top_k: int) -> list[SearchResult]:
    valid_results = [item for item in results if item.url]
    if not valid_results:
        return []
    pairs = [(item.query, f"{item.title}\n{item.snippet}") for item in valid_results]
    scored = [
        item.model_copy(update={"relevance_score": float(score)})
        for item, score in zip(
            valid_results, reranker.predict(pairs, batch_size=32), strict=True
        )
    ]
    return sorted(scored, key=lambda item: item.relevance_score or 0.0, reverse=True)[:top_k]


def populate_content(
    results: list[SearchResult],
    fetcher: WebContentFetcher,
    *,
    timeout: float = 8.0,
    max_length: int = 3000,
    fallback_fetcher: WebContentFetcher | None = None,
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
    return updated


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Format search results with snippets and fetched page contents for LLM prompting."""
    parts = []
    for i, item in enumerate(results, start=1):
        parts.append(f"--- 検索結果 #{i} ---\n{item.to_evidence_text()}")
    return "\n\n".join(parts)


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

    def planner(state: State):
        prompt = f"""あなたはリサーチプランナーです。現在日: {date.today().isoformat()}
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
        except Exception as err:
            print(f"調査計画のパース失敗 (フォールバック適用): {err}")
            plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
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
        try:
            response = search.text(task.query, max_results=settings.results_per_query)
            add_search_results(fresh_results, response, task.query)
        except Exception as error:
            print(f"検索失敗 ({task.query}): {error}")
            return {"results": [], "search_query_count": 1}

        top_results = rerank_results(fresh_results, reranker, settings.reranked_results_per_query)
        if fetcher:
            top_results = populate_content(
                top_results,
                fetcher,
                timeout=settings.fetch_timeout,
                max_length=settings.max_content_length,
                fallback_fetcher=fallback_fetcher,
            )

        return {
            "results": top_results,
            "search_query_count": 1,
        }

    def evaluate(state: State):
        unique_results = deduplicate_results(state.get("results", []))
        formatted_evidence = format_results_for_llm(unique_results)
        prompt = f"""あなたはリサーチ品質評価担当です。現在日: {date.today().isoformat()}
質問: {state['query']}
調査計画: {state['plan']}
収集した検索結果およびWebページ本文:
{formatted_evidence}

計画の主要観点、根拠の強さ、情報源の信頼性と新しさを厳密に評価してください。
Webページ本文やスニペットを確認し、必ず以下のJSON形式のみを出力してください（説明文や前置きは不要です）。

```json
{{
  "sufficient": false,
  "missing_information": ["不足している情報1"],
  "weak_evidence": ["根拠が弱い点1"],
  "reason": "評価理由",
  "additional_queries": ["追加検索クエリ1", "追加検索クエリ2"]
}}
```

※ 十分な情報が揃っている場合は sufficient を true、missing_information / weak_evidence / additional_queries を空配列 [] にしてください。"""
        try:
            response = llm.invoke(prompt)
            data = parse_json_from_response(response.content)
            eval_obj = Evaluation.model_validate(data)
        except Exception as err:
            print(f"品質評価のパース失敗 (完了と判定): {err}")
            eval_obj = Evaluation(
                sufficient=True,
                missing_information=[],
                weak_evidence=[],
                reason="パース失敗による安全フォールバック",
                additional_queries=[],
            )

        eval_obj.sufficient = eval_obj.sufficient and not (
            eval_obj.additional_queries or eval_obj.missing_information or eval_obj.weak_evidence
        )
        return {"evaluation": eval_obj}

    def should_continue(state: State):
        evaluation = state.get("evaluation")
        if evaluation is None or evaluation.sufficient:
            return "analyze"
        if state["search_round"] >= settings.max_search_rounds:
            return "analyze"
        if state["search_query_count"] >= settings.max_search_queries:
            return "analyze"
        return "additional_search"

    def additional_search(state: State):
        evaluation = state.get("evaluation")
        if evaluation is None:
            return {"results": [], "search_query_count": 0, "search_round": 1}
        remaining = settings.max_search_queries - state["search_query_count"]
        queries = evaluation.additional_queries[:max(remaining, 0)]
        existing_urls = {item.url for item in state.get("results", []) if item.url}
        delta_results: list[SearchResult] = []
        for query in queries:
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
                top_query_results = rerank_results(query_fresh, reranker, settings.reranked_results_per_query)
                if fetcher:
                    top_query_results = populate_content(
                        top_query_results,
                        fetcher,
                        timeout=settings.fetch_timeout,
                        max_length=settings.max_content_length,
                        fallback_fetcher=fallback_fetcher,
                    )
                delta_results.extend(top_query_results)
            except Exception as error:
                print(f"追加検索失敗 ({query}): {error}")
        return {
            "results": delta_results,
            "search_query_count": len(queries),
            "search_round": 1,
        }

    def analyze(state: State):
        unique_results = deduplicate_results(state.get("results", []))
        formatted_evidence = format_results_for_llm(unique_results)
        response = llm.invoke(f"""以下のWeb検索結果および取得したWebページ本文だけを根拠に質問へ回答してください。
質問: {state['query']}

収集された情報源:
{formatted_evidence}

指示:
1. Webページの本文に記載されている具体的な事実、数値、詳細を最大限に活用して、質の高い体系的なレポートを作成してください。
2. 重要な主張には、根拠となる検索結果に含まれる URL を示してください。
3. 取得情報から確認できないことは断定せず、URLを捏造しないでください。""")
        return {"summary": response.content}

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
