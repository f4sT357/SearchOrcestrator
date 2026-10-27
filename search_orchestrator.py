"""A bounded, evidence-oriented web research workflow with web page fetching."""

from __future__ import annotations

import json
import math
import operator
import re
import time
from functools import lru_cache
from threading import Lock
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Callable, Protocol, TypedDict
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app_settings import Settings
from web_fetchers import (
    DefaultWebContentFetcher,
    FetchError,
    FetchStatus,
    FirecrawlFetcher,
    JinaReaderFetcher,
    WebContentFetcher,
    is_safe_url,
)

from source_scoring import (
    composite_score,
    domain_trust_score,
    extract_content_date,
    freshness_score,
    normalize_official_domains,
    source_tier,
)
from source_assessment import (
    SourceAssessment,
    SourceAssessmentCache,
    assessment_by_url,
    merge_assessments,
)


# ---------------------------------------------------------------------------
# Workflow errors
# ---------------------------------------------------------------------------

class EmptyModelResponseError(RuntimeError):
    """Raised when a local reasoning model consumes its budget without an answer."""


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
    official_queries: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    sufficient: bool
    missing_information: list[str]
    weak_evidence: list[str]
    reason: str
    additional_queries: list[str]
    official_additional_queries: list[str] = Field(default_factory=list)
    source_assessments: list[SourceAssessment] = Field(default_factory=list)



class SearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str
    content: str | None = None
    relevance_score: float | None = None
    fetch_status: str | None = None
    fetch_error: str | None = None
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
    understood_request: str
    plan: SearchPlan | None
    task: SearchTask | None
    results: Annotated[list[SearchResult], operator.add]
    evidence_results: Annotated[list[SearchResult], operator.add]
    official_results: Annotated[list[SearchResult], operator.add]
    evaluation: Evaluation | None
    run_memo: str
    summary: str
    search_query_count: Annotated[int, operator.add]
    search_round: Annotated[int, operator.add]
    searched_queries: Annotated[list[str], operator.add]
    normal_search_query_count: Annotated[int, operator.add]
    official_search_query_count: Annotated[int, operator.add]
    pending_normal_queries: list[str]
    pending_official_queries: list[str]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class SearchClient(Protocol):
    def text(self, query: str, *, max_results: int): ...


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs): ...


class _SigmoidReranker:
    """Convert CrossEncoder logits into comparable 0-1 relevance scores."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, pairs: list[tuple[str, str]], **kwargs) -> list[float]:
        logits = self.model.predict(pairs, **kwargs)
        scores = []
        for value in logits:
            score = float(value)
            if score >= 0:
                scores.append(1.0 / (1.0 + math.exp(-score)))
            else:
                exp_score = math.exp(score)
                scores.append(exp_score / (1.0 + exp_score))
        return scores


@lru_cache(maxsize=1)
def _load_reranker_cached(model_name: str) -> Reranker:
    from sentence_transformers import CrossEncoder

    return _SigmoidReranker(CrossEncoder(model_name))


_RERANKER_LOAD_LOCK = Lock()


def preload_reranker(model_name: str) -> None:
    """Load and cache the large reranker, for example during GUI startup."""
    with _RERANKER_LOAD_LOCK:
        _load_reranker_cached(model_name)


def _load_reranker(model_name: str) -> Reranker:
    # Serialize preload and research startup so they never create two models.
    with _RERANKER_LOAD_LOCK:
        return _load_reranker_cached(model_name)


# Sources below this calibrated relevance score cannot steer quality evaluation
# or the follow-up search queries it generates.
MIN_RELEVANCE_FOR_EVALUATION = 0.20
MIN_RELEVANCE_FOR_FOLLOWUP_QUERY = 0.60
MAX_EVIDENCE_SOURCES = 6
MAX_EVIDENCE_CHARS = 9000
_PLACEHOLDER_QUERY_RE = re.compile(
    r"(?:リサーチプランナー|詳細調査クエリ\s*\d+|検索クエリ\s*\d+|search query\s*#?\d+)",
    re.IGNORECASE,
)
_QUERY_STOP_TERMS = {
    "について", "キャラクター", "キャラ", "調査", "比較", "情報", "教えて", "ください",
    "ゲーム", "公式", "一次資料", "能力", "詳細", "性能", "評価", "最新", "確認",
}


def _evaluation_sources(
    results: list[SearchResult],
    *,
    official_candidates: list[SearchResult] | None = None,
) -> list[SearchResult]:
    relevant = [
        item for item in deduplicate_results(results)
        if item.relevance_score is not None
        and item.relevance_score >= MIN_RELEVANCE_FOR_EVALUATION
    ]
    ranked = sorted(relevant, key=lambda item: item.combined_score or 0.0, reverse=True)
    # Keep room for relevant primary evidence and results from the dedicated
    # official-search lane so a large pile of secondary results cannot crowd
    # every official candidate out of the evaluator context.
    official_urls = {
        item.url for item in (official_candidates or [])
        if item.url and item.relevance_score is not None and item.relevance_score >= 0.45
    }
    primary = [
        item for item in ranked
        if item.source_reliability == "primary" or item.url in official_urls
    ]
    primary_slots = min(3, MAX_EVIDENCE_SOURCES)
    selected_primary = select_diverse_results(primary, primary_slots)
    selected_urls = {item.url for item in selected_primary}
    remaining = [item for item in ranked if item.url not in selected_urls]
    selected_secondary = select_diverse_results(
        remaining, max(0, MAX_EVIDENCE_SOURCES - len(selected_primary)),
    )
    return selected_primary + selected_secondary


def _query_anchor_terms(query: str) -> set[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9-]*|[\u30A0-\u30FFー]{2,}|[\u3400-\u9FFF々]{2,}", query)
    return {
        term.casefold() for term in terms
        if term.casefold() not in _QUERY_STOP_TERMS and len(term) > 1
    }


def _compact_query_terms(text: str) -> str:
    """Turn a question or missing-information note into concise search terms."""
    candidate = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = re.sub(r"\[[^\]]*\]", " ", candidate)
    request_match = re.search(
        r"(?:について|に関して|に関する|を|が|は)?(?:調査|調べ|検索|教えて|説明|解説|まとめ|知りたい|確認)",
        candidate,
    )
    if request_match:
        candidate = candidate[:request_match.start()]
    candidate = re.sub(r"という|として|について|に関して|に関する", " ", candidate)
    candidate = re.sub(r"(?<=[\u3400-\u9fff])の(?=[\u3040-\u30ff\u3400-\u9fff「『\"'])", " ", candidate)
    candidate = re.sub(r"(?:の|を|が|は|と|に|で|へ|も|や)\s*$", " ", candidate)
    candidate = re.sub(r"[、。,.!?！？:：;；/／()（）「」『』\"'`]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def _keyword_followup_query(
    original_query: str,
    missing_information: list[str],
    searched_queries: list[str],
) -> str | None:
    """Build a short, non-repeating fallback from the topic and one missing facet."""
    subject = _compact_query_terms(original_query)
    if not subject:
        subject = original_query.strip()

    facets = [_compact_query_terms(item) for item in missing_information]
    facets = [facet for facet in facets if facet]
    is_character_topic = any(
        kw in original_query.casefold()
        for kw in ("キャラ", "character", "登場人物", "主人公", "キャスト", "声優")
    )
    if is_character_topic:
        facets.extend(("公式 キャラクター紹介", "公式 キャラクタープロフィール", "公式 告知"))
    else:
        facets.extend(("公式 発表", "公式 お知らせ", "公式サイト 最新情報", "公式 更新履歴", "公式 文書"))
    seen = {re.sub(r"\s+", " ", query).strip().casefold() for query in searched_queries}
    for facet in facets:
        query = re.sub(r"\s+", " ", f"{subject} {facet}").strip()
        if query.casefold() not in seen:
            return query
    return None


def _validate_followup_queries(
    original_query: str,
    proposed_queries: list[str],
    reranker: Reranker,
) -> list[str]:
    """Reject model-generated follow-ups that drift away from the user topic."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for proposed in proposed_queries:
        candidate = re.sub(r"\[\s*S?\d+\s*\]", " ", proposed, flags=re.IGNORECASE)
        candidate = candidate.strip().strip("[] \t\r\n:：-•")
        candidate = re.sub(r"^\d+[.)、:]\s*", "", candidate).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        normalized = candidate.casefold()
        if (
            not candidate
            or len(candidate) > 300
            or normalized in seen
            or _PLACEHOLDER_QUERY_RE.search(candidate)
        ):
            continue
        seen.add(normalized)
        cleaned.append(candidate)

    if not cleaned:
        return []

    scores = reranker.predict(
        [(original_query, candidate) for candidate in cleaned],
        batch_size=32,
    )
    anchors = _query_anchor_terms(original_query)
    accepted = []
    for candidate, raw_score in zip(cleaned, scores, strict=True):
        score = float(raw_score)
        candidate_folded = candidate.casefold()
        contains_anchor = any(term in candidate_folded for term in anchors)
        if score < MIN_RELEVANCE_FOR_FOLLOWUP_QUERY or (
            anchors and not contains_anchor and score < 0.75
        ):
            continue
        accepted.append(candidate)
    return accepted


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
    official_domains: tuple[str, ...] = (),
    trusted_domains: tuple[str, ...] = (),
) -> list[SearchResult]:
    valid_results = [item for item in results if item.url]
    if not valid_results:
        return []
    pairs = [(item.query, f"{item.title}\n{item.snippet}") for item in valid_results]
    raw_scores = reranker.predict(pairs, batch_size=32)

    enriched: list[SearchResult] = []
    for item, rel_score in zip(valid_results, raw_scores, strict=True):
        rel = float(rel_score)
        tier = source_tier(item.url, official_domains, trusted_domains)
        trust = domain_trust_score(item.url, official_domains, trusted_domains)
        fresh = freshness_score(item.url)
        comb = composite_score(
            rel, item.url,
            w_relevance=weights[0], w_trust=weights[1], w_freshness=weights[2],
            official_domains=official_domains,
            trusted_domains=trusted_domains,
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
    official_domains: tuple[str, ...] = (),
    trusted_domains: tuple[str, ...] = (),
) -> list[SearchResult]:
    """Fetch content while retaining a compact, source-safe failure trail."""
    if not fetcher:
        return results
    updated: list[SearchResult] = []
    for item in results:
        if item.content is not None or not item.url:
            updated.append(item)
            continue

        failure_reasons: list[str] = []
        try:
            content = fetcher.fetch(item.url, timeout=timeout, max_length=max_length)
            if content:
                updated.append(item.model_copy(update={
                    "content": content,
                    "fetch_status": FetchStatus.SUCCESS.value,
                    "fetch_error": None,
                }))
                continue
            failure_reasons.append("primary=empty")
        except FetchError as exc:
            status = (exc.status or FetchStatus.FAILED).value
            failure_reasons.append(f"primary={status}")
        except Exception:
            failure_reasons.append("primary=failed")

        if fallback_fetcher:
            try:
                fallback_content = fallback_fetcher.fetch(item.url, timeout=timeout, max_length=max_length)
                if fallback_content:
                    updated.append(item.model_copy(update={
                        "content": fallback_content,
                        "fetch_status": FetchStatus.FALLBACK_SUCCESS.value,
                        "fetch_error": None,
                    }))
                    continue
                failure_reasons.append("fallback=empty")
            except FetchError as exc:
                status = (exc.status or FetchStatus.FAILED).value
                failure_reasons.append(f"fallback={status}")
            except Exception:
                failure_reasons.append("fallback=failed")

        updated.append(item.model_copy(update={
            "fetch_status": FetchStatus.EMPTY_OR_FAILED.value,
            "fetch_error": "; ".join(failure_reasons)[:180] or "fetch=failed",
        }))

    rescored: list[SearchResult] = []
    for item in updated:
        page_date = extract_content_date(item.content)
        freshness = freshness_score(item.url, content=item.content)
        relevance = item.relevance_score if item.relevance_score is not None else 0.0
        combined = composite_score(
            relevance, item.url, content=item.content,
            w_relevance=weights[0], w_trust=weights[1], w_freshness=weights[2],
            official_domains=official_domains,
            trusted_domains=trusted_domains,
        )
        rescored.append(item.model_copy(update={
            "page_date": page_date.isoformat() if page_date else None,
            "source_freshness_score": freshness,
            "combined_score": combined,
        }))
    return rescored


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Format multiple sources while keeping total evidence within a prompt budget."""
    entries = []
    fixed_chars = 0
    for i, item in enumerate(results, start=1):
        heading = f"--- 検索結果 #{i} / 出典番号 [S{i}] / 検索クエリ: {item.query} ---\n"
        metadata = item.model_copy(update={"content": None}).to_evidence_text()
        fixed = f"{heading}{metadata}"
        entries.append((fixed, item.content or ""))
        fixed_chars += len(fixed) + 2

    remaining_body_chars = max(0, MAX_EVIDENCE_CHARS - fixed_chars)
    parts = []
    remaining_sources = len(entries)
    for fixed, body in entries:
        body_limit = remaining_body_chars // max(remaining_sources, 1)
        excerpt = body[:body_limit]
        if len(body) > len(excerpt):
            excerpt += "\n（本文は全体の入力上限に合わせて省略）"
        part = fixed + (f"\n【Webページ本文抜粋】:\n{excerpt}" if excerpt else "")
        parts.append(part)
        remaining_body_chars = max(0, remaining_body_chars - min(len(body), body_limit))
        remaining_sources -= 1
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


def normalize_evaluation_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize common local-LLM variants before validating the evaluation schema."""
    normalized = dict(data)
    def normalize_queries(key: str) -> None:
        queries = normalized.get(key, [])
        if isinstance(queries, list):
            normalized_queries = []
            for item in queries:
                query = item if isinstance(item, str) else (
                    item.get("query") or item.get("search_query") or item.get("text")
                    if isinstance(item, dict) else None
                )
                if isinstance(query, str) and query.strip():
                    normalized_queries.append(query.strip())
            normalized[key] = normalized_queries
        elif isinstance(queries, dict):
            query = queries.get("query") or queries.get("search_query") or queries.get("text")
            normalized[key] = [query.strip()] if isinstance(query, str) and query.strip() else []
        else:
            normalized[key] = []
    normalize_queries("additional_queries")
    normalize_queries("official_additional_queries")
    assessments = normalized.get("source_assessments", [])
    if isinstance(assessments, dict):
        assessments = [assessments]
    normalized["source_assessments"] = [
        item for item in assessments if isinstance(item, dict)
    ] if isinstance(assessments, list) else []
    return normalized


def append_run_memo_posts(
    board: str,
    updates: list[str],
    *,
    max_chars: int = 5000,
) -> tuple[str, list[str], int]:
    """Append distinct bulletin-board posts, evicting oldest whole posts at capacity."""
    entries = [line.strip() for line in board.splitlines() if line.strip()]
    known = {" ".join(entry.lstrip("-• ").split()).casefold() for entry in entries}
    added: list[str] = []
    for update in updates:
        body = " ".join(update.split()).strip()
        if not body:
            continue
        normalized = body.lstrip("-• ").casefold()
        if normalized in known:
            continue
        entry = body if body.startswith(("- ", "• ")) else f"- {body}"
        if len(entry) > max_chars:
            entry = entry[:max_chars - 1] + "…"
        entries.append(entry)
        added.append(entry)
        known.add(normalized)

    evicted = 0
    while entries and len("\n".join(entries)) > max_chars:
        entries.pop(0)
        evicted += 1
    return "\n".join(entries), added, evicted



def research_brief_context(raw_memo: str, known_urls: set[str], *, max_chars: int = 600) -> str:
    """Return only source-linked facts and open questions from a compact research brief."""
    try:
        memo = json.loads(raw_memo)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(memo, dict):
        return ""

    lines: list[str] = []
    unresolved = memo.get("unresolved", [])
    if isinstance(unresolved, list):
        questions = [str(item).strip()[:140] for item in unresolved if str(item).strip()][:3]
        if questions:
            lines.append("Open questions: " + " | ".join(questions))

    confirmed = memo.get("confirmed", [])
    if isinstance(confirmed, list):
        facts: list[str] = []
        for item in confirmed:
            if not isinstance(item, dict):
                continue
            url = str(item.get("source_url", "")).strip()
            note = str(item.get("note", "")).strip()[:140]
            if url in known_urls and note:
                facts.append(f"{note} ({url})")
        if facts:
            lines.append("Source-linked facts: " + " | ".join(facts[:3]))

    decisions = memo.get("source_decisions", [])
    if isinstance(decisions, list):
        decision_lines: list[str] = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            decision = str(item.get("decision", "")).strip()[:48]
            if url in known_urls and decision:
                decision_lines.append(f"{url}: {decision}")
        if decision_lines:
            lines.append("Source decisions: " + " | ".join(decision_lines[:3]))

    return "\n".join(lines)[:max_chars]
def strip_model_reasoning(text: str) -> str:
    """Remove visible reasoning sections before model text reaches user output."""
    if not isinstance(text, str):
        return ""

    # Some local models place their entire scratchpad before a closing tag but
    # omit the matching opening tag. In that case, only the text after the last
    # closing tag is the user-facing answer.
    closing_tags = list(re.finditer(r"</(?:think|analysis|reasoning)>", text, re.IGNORECASE))
    if closing_tags:
        text = text[closing_tags[-1].end():]

    text = re.sub(
        r"<(think|analysis|reasoning)>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<(?:think|analysis|reasoning)>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"</?(?:think|analysis|reasoning)>", "", text, flags=re.IGNORECASE)

    final_marker = re.search(r"(?:FINAL ANSWER|最終回答)\s*[:：]\s*", text, re.IGNORECASE)
    if final_marker:
        text = text[final_marker.end():]
    return text.strip()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class ResearchWorkflowNodes:
    """Encapsulates LangGraph node handlers and dependencies for testability and separation of concerns."""

    def __init__(
        self,
        settings: Settings,
        search: SearchClient | None = None,
        reranker: Reranker | None = None,
        llm: ChatOpenAI | None = None,
        fetcher: WebContentFetcher | None = None,
        fallback_fetcher: WebContentFetcher | None = None,
        log: Callable[[str], None] | None = None,
        board_update: Callable[[str], None] | None = None,
        progress: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.settings = settings
        assessment_cache_dir = Path(settings.source_assessment_cache_dir)
        if not assessment_cache_dir.is_absolute():
            assessment_cache_dir = Path(__file__).resolve().parent / assessment_cache_dir
        self.source_assessment_cache = SourceAssessmentCache(
            assessment_cache_dir,
            enabled=settings.source_assessment_cache_enabled,
        )
        self.log = log
        self.board_update = board_update
        self.progress = progress

        if llm is None:
            llm = ChatOpenAI(
                model=settings.model,
                base_url=settings.base_url,
                api_key=settings.api_key,
                temperature=0,
                extra_body=settings.llm_extra_body or None,
            )
        self.llm = llm
        self.planner_llm = self.llm.bind(max_tokens=settings.planner_max_output_tokens)
        self.evaluator_llm = self.llm.bind(max_tokens=settings.evaluation_max_output_tokens)
        self.answer_llm = self.llm.bind(max_tokens=settings.final_answer_max_output_tokens)
        self.memo_keeper_llm = self.llm.bind(max_tokens=settings.memo_keeper_max_output_tokens)
        self.board_writer_llm = self.llm.bind_tools([
            {
                "type": "function",
                "function": {
                    "name": "write_research_board",
                    "description": "必ず1回呼ぶ。新しい投稿が不要ならcontentを英小文字のnothingにする。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "掲示板に残す短い投稿。投稿不要の場合は正確にnothing。",
                            }
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                },
            }
        ], tool_choice="required")
        if search is None:
            search = DDGS()
        self.search = search
        if reranker is None:
            reranker = _load_reranker(settings.reranker_model)
        self.reranker = reranker
        if fetcher is None and settings.fetch_web_content:
            if settings.content_fetcher == "firecrawl":
                fetcher = FirecrawlFetcher(
                    api_url=settings.firecrawl_api_url,
                    api_key=settings.firecrawl_api_key,
                )
            else:
                cache_dir = Path(settings.web_cache_dir)
                if not cache_dir.is_absolute():
                    cache_dir = Path(__file__).resolve().parent / cache_dir
                fetcher = DefaultWebContentFetcher(
                    cache_dir=cache_dir if settings.web_cache_enabled else None,
                    cache_ttl_seconds=settings.web_cache_ttl_seconds,
                    min_host_interval_seconds=settings.web_host_min_interval_seconds,
                    max_retry_after_seconds=settings.web_max_retry_after_seconds,
                )
        elif not settings.fetch_web_content:
            fetcher = None
        self.fetcher = fetcher
        if fallback_fetcher is None and settings.use_fallback_fetcher:
            fallback_fetcher = JinaReaderFetcher(api_url=settings.jina_api_url)
        self.fallback_fetcher = fallback_fetcher
        self.current_date = date.today().isoformat()

    def log_detail(self, message: str) -> None:
        if self.log is not None:
            self.log(message)

    def log_source_trace(
        self,
        stage: str,
        item: SearchResult,
        decision: str,
        reason: str,
        *,
        official_candidate: bool = False,
        page_assessment: SourceAssessment | None = None,
    ) -> None:
        payload = {
            "stage": stage,
            "url": item.url,
            "decision": decision,
            "reason": reason,
            "official_candidate": official_candidate,
            "registered_official": source_tier(item.url, self.settings.official_domains) == "primary",
            "registered_trusted": source_tier(
                item.url, self.settings.official_domains, self.settings.trusted_domains,
            ) == "secondary",
            "relevance_score": item.relevance_score,
            "fetch_status": item.fetch_status,
            "fetch_error": item.fetch_error,
        }
        if page_assessment is not None:
            payload["page_assessment"] = {
                "credibility": page_assessment.credibility,
                "use_as_evidence": page_assessment.use_as_evidence,
                "support_scope": page_assessment.support_scope,
            }
        self.log_detail(f"[SOURCE_TRACE] {json.dumps(payload, ensure_ascii=False)}")
    def invoke_llm(
        self,
        role: str,
        prompt: str,
        *,
        llm: Any | None = None,
        source_count: int | None = None,
    ) -> Any:
        """Invoke an LLM and emit safe, structured performance metrics."""
        client = llm or self.llm
        if self.settings.llm_prompt_prefix.strip():
            prompt = f"{self.settings.llm_prompt_prefix.strip()}\n{prompt.lstrip()}"
        if self.settings.llm_prompt_suffix.strip():
            prompt = f"{prompt.rstrip()}\n{self.settings.llm_prompt_suffix.strip()}"
        started_at = time.perf_counter()
        metric: dict[str, Any] = {
            "role": role,
            "input_chars": len(prompt),
            "source_count": source_count,
        }
        try:
            response = client.invoke(prompt)
        except Exception as error:
            metric.update({
                "duration_ms": round((time.perf_counter() - started_at) * 1000),
                "status": "error",
                "error_type": type(error).__name__,
            })
            self.log_detail(f"[LLM_METRIC] {json.dumps(metric, ensure_ascii=False)}")
            raise

        content = getattr(response, "content", "")
        content_text = content if isinstance(content, str) else str(content)
        response_extra = getattr(response, "additional_kwargs", {})
        reasoning_content = (
            response_extra.get("reasoning_content", "")
            if isinstance(response_extra, dict) else ""
        )
        metric.update({
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
            "output_chars": len(content_text),
            "status": "completed" if content_text.strip() else "empty",
        })
        if reasoning_content:
            metric["reasoning_chars"] = len(str(reasoning_content))
        for metadata in (
            getattr(response, "usage_metadata", None),
            getattr(response, "response_metadata", None),
        ):
            if not isinstance(metadata, dict):
                continue
            usage = metadata.get("token_usage", metadata.get("usage", metadata))
            if not isinstance(usage, dict):
                continue
            for metric_name, keys in {
                "input_tokens": ("input_tokens", "prompt_tokens"),
                "output_tokens": ("output_tokens", "completion_tokens"),
                "total_tokens": ("total_tokens",),
            }.items():
                if metric_name in metric:
                    continue
                for key in keys:
                    value = usage.get(key)
                    if isinstance(value, int):
                        metric[metric_name] = value
                        break
        self.log_detail(f"[LLM_METRIC] {json.dumps(metric, ensure_ascii=False)}")
        if not content_text.strip():
            raise EmptyModelResponseError(
                "The model returned no final content. Disable thinking in LM Studio or increase its output budget."
            )
        return response
    def report_progress(self, stage: str, status: str, detail: str = "") -> None:
        if self.progress is not None:
            try:
                self.progress(stage, status, detail)
            except Exception as error:
                self.log_detail(f"進行状況の通知に失敗しました: {error}")

    def track_node(self, stage: str, node: Callable[[State], dict[str, Any]]):
        def tracked(state: State):
            detail = ""
            if stage == "search":
                task = state.get("task")
                detail = task.query if task else "通常検索"
            elif stage in ("official_search", "official_additional_search"):
                detail = f"{len(state.get('official_queries', []))}件の公式検索"
            elif stage == "additional_search":
                evaluation = state.get("evaluation")
                detail = f"検索ラウンド {state.get('search_round', 0) + 1}"
                if evaluation and evaluation.additional_queries:
                    detail += f" / {evaluation.additional_queries[0]}"
            elif stage == "evaluate":
                detail = f"検索 {state.get('search_round', 0) + 1}回目の結果"
            elif stage == "understand_input":
                detail = "ユーザーの依頼内容を整理"
            self.report_progress(stage, "started", detail)
            try:
                result = node(state)
            except Exception as error:
                self.report_progress(stage, "failed", str(error))
                raise
            self.report_progress(stage, "completed", detail)
            return result
        return tracked

    def log_selected_results(self, results: list[SearchResult], label: str = "採用候補") -> None:
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
            self.log_detail(
                f"  {label} {index}: {item.title or item.url} / {tier} / {details}"
                f" / 本文 {status}{page_date}"
            )

    def understand_input(self, state: State):
        query = state["query"].strip()
        understood = query[:1200]
        memo = json.dumps({
            "request": understood,
            "confirmed": [],
            "unresolved": [],
            "source_decisions": [],
            "next_action": "Create a research plan.",
        }, ensure_ascii=False, separators=(",", ":"))
        self.log_detail("入力整理: 元の質問をメモスナップショットにプログラムで自動記録しました")
        if self.board_update is not None:
            self.board_update(memo)
        return {"understood_request": understood, "run_memo": memo}

    def planner(self, state: State):
        prompt = f"""あなたはリサーチプランナーです。現在日: {self.current_date}
ユーザーの入力: {state['query']}
入力内容の整理は依頼の意図を理解するための補助です。元の入力と食い違う場合は元の入力を優先し、整理に書かれていない要望を追加しないでください。重複しない体系的な調査計画を作成してください。各 query は検索エンジン向けの短いキーワード列にしてください。質問文や依頼文をそのまま含めず、「対象名 + 調べる観点 + 必要なら情報源」の形で、1クエリにつき1観点、3〜8語程度にします。質問が「鳴潮 カルロッタ」のようなキーワード列なら、その語を対象名としてそのまま保持して観点だけを補ってください。

通常の調査計画とは別に、運営元・公式サイト・公式告知・公式文書を探すための official_queries を最大5件作ってください。検索語や検索先を少しずつ変え、公式発表・お知らせ・更新履歴・公式SNSや配信ページなど、質問に合う一次情報の入口を複数探します。検索語には「公式」「告知」「お知らせ」「公式サイト」などを含めます。対象が特定できない site: ドメインを推測して付けないでください。このレーンは一次情報を探す専用であり、見つかったページが本当に公式かは後で本文と発行元を確認します。

内部の思考過程は出力せず、以下のJSON形式のみを出力してください。<think>タグ、分析文、JSON以外の前置きは禁止です。

```json
{{
  "tasks": [
    {{
      "aspect": "調査観点1",
      "query": "対象名 観点 キーワード",
      "reason": "検索する理由1"
    }},
    {{
      "aspect": "調査観点2",
      "query": "対象名 別の観点 キーワード",
      "reason": "検索する理由2"
    }}
  ],
  "official_queries": ["対象名 公式 発表", "対象名 公式 お知らせ", "対象名 公式サイト 最新情報", "対象名 公式 更新履歴", "対象名 公式 配信"]
}}
```"""
        try:
            response = self.invoke_llm("planner", prompt, llm=self.planner_llm)
            data = parse_json_from_response(response.content)
            plan = SearchPlan.model_validate(data)
            if not plan.tasks:
                plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
            self.log_detail(f"調査計画: {len(plan.tasks)}件の観点を作成")
            self.log_detail("生成された調査計画の内容:")
            for index, task in enumerate(plan.tasks, start=1):
                self.log_detail(
                    f"  計画候補 {index}: {task.aspect} / クエリ: {task.query} / 理由: {task.reason}"
                )
            for index, query in enumerate(plan.official_queries, start=1):
                self.log_detail(f"  公式検索案 {index}: {query}")
        except Exception as err:
            print(f"調査計画のパース失敗 (フォールバック適用): {err}")
            plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
            self.log_detail(f"調査計画の解析に失敗したため、基本検索に切り替えました: {err}")

        try:
            validated_queries = _validate_followup_queries(
                state["query"], [task.query for task in plan.tasks], self.reranker,
            )
        except Exception as error:
            validated_queries = []
            self.log_detail(f"調査計画の関連度検査に失敗しました。元の質問から検索します: {error}")
        self.log_detail(
            f"関連度検査を通過した調査計画クエリ: {len(validated_queries)}/{len(plan.tasks)}件"
        )
        for index, query in enumerate(validated_queries, start=1):
            self.log_detail(f"  採用クエリ {index}: {query}")
        if len(validated_queries) != len(plan.tasks):
            self.log_detail(
                f"元の質問との関連が弱い調査計画を {len(plan.tasks) - len(validated_queries)}件除外しました"
            )
        official_queries = [
            re.sub(r"\s+", " ", item).strip()
            for item in plan.official_queries
            if item.strip() and len(item.strip()) <= 300
        ][:5]
        official_queries = list(dict.fromkeys(official_queries))
        try:
            official_queries = _validate_followup_queries(
                state["query"], official_queries, self.reranker,
            )
        except Exception as error:
            self.log_detail(f"公式検索案の関連度検査に失敗しました。基本クエリを使います: {error}")
            official_queries = []
        subject = _compact_query_terms(state["query"]) or state["query"].strip()
        official_fallbacks = [
            f"{subject} 公式 発表",
            f"{subject} 公式 お知らせ",
            f"{subject} 公式サイト 最新情報",
            f"{subject} 公式 更新履歴",
            f"{subject} 公式 配信",
        ]
        for fallback_query in official_fallbacks:
            if len(official_queries) >= 5:
                break
            if fallback_query.casefold() not in {query.casefold() for query in official_queries}:
                official_queries.append(fallback_query)
        plan = SearchPlan(tasks=[
            SearchTask(
                aspect="質問に関する調査",
                query=query,
                reason="元の質問に関連する情報を確認するため",
            )
            for query in validated_queries
        ], official_queries=official_queries)
        if not plan.tasks:
            plan = SearchPlan(tasks=[
                SearchTask(
                    aspect="総合調査",
                    query=state["query"],
                    reason="元の質問に直接答えるための基本情報収集",
                )
            ], official_queries=official_queries)
            self.log_detail("有効な調査計画が残らなかったため、元の質問だけで検索します")
        for index, task in enumerate(plan.tasks, start=1):
            self.log_detail(f"  計画 {index}: {task.aspect} / クエリ: {task.query} / 理由: {task.reason}")
        official_limit = min(5, max(self.settings.max_search_queries - 1, 0))
        regular_limit = self.settings.max_search_queries - official_limit if official_limit else self.settings.max_search_queries
        if len(plan.tasks) > regular_limit:
            self.log_detail(
                f"通常検索 {regular_limit}件と公式検索 {official_limit}件の枠を確保します"
            )
        return {"plan": plan}

    def dispatch_searches(self, state: State):
        if state["plan"] is None:
            return []
        # Reserve up to five initial query slots for the independent official-source lane.
        official_limit = min(5, max(self.settings.max_search_queries - 1, 0))
        selected_official_queries = state["plan"].official_queries[:official_limit]
        regular_limit = self.settings.max_search_queries - len(selected_official_queries)
        if not selected_official_queries:
            regular_limit = self.settings.max_search_queries
        sends = [
            Send("search", {"task": task})
            for task in state["plan"].tasks[:regular_limit]
        ]
        if selected_official_queries:
            sends.append(Send("official_search", {
                "query": state["query"],
                "official_queries": selected_official_queries,
            }))
        else:
            self.log_detail("検索クエリ上限のため、公式ソース専用検索を見送りました")
        return sends

    def official_search(self, state: State):
        """Search independent queries dedicated to finding first-party sources."""
        progress_stage = state.get("progress_stage", "official_search")
        queries = state.get("official_queries", [])[:5]
        if not queries:
            return {
                "results": [], "evidence_results": [], "official_results": [],
                "search_query_count": 0, "searched_queries": [],
                "official_search_query_count": 0,
            }
        self.log_detail(f"公式ソース専用検索を開始します（最大 {len(queries)} クエリ）")
        evidence_results: list[SearchResult] = []
        selected_results: list[SearchResult] = []
        searched_queries: list[str] = []
        weights = (self.settings.relevance_weight, self.settings.trust_weight, self.settings.freshness_weight)
        candidate_count = (
            self.settings.content_candidate_results_per_query
            if self.fetcher else self.settings.reranked_results_per_query
        )
        for query_index, query in enumerate(queries, start=1):
            self.report_progress(
                progress_stage, "progress",
                f"{query_index}/{len(queries)}件目 · 検索中: {query}",
            )
            self.log_detail(f"公式ソース専用検索: {query}")
            searched_queries.append(query)
            candidates: list[SearchResult] = []
            try:
                response = self.search.text(query, max_results=self.settings.results_per_query)
                add_search_results(candidates, response, query)
                self.report_progress(
                    progress_stage, "progress",
                    f"{query_index}/{len(queries)}件目 · 候補 {len(candidates)}件",
                )
                self.log_detail(f"公式ソース検索候補: {len(candidates)}件")
            except Exception as error:
                if "no results found" in str(error).casefold():
                    self.log_detail(f"公式ソース検索結果なし: {query}")
                else:
                    self.log_detail(f"公式ソース検索に失敗しました: {query} / {error}")
                continue
            query_candidates = rerank_results(
                candidates, self.reranker, candidate_count, weights=weights,
                official_domains=self.settings.official_domains,
                trusted_domains=self.settings.trusted_domains,
            )
            if self.fetcher:
                self.report_progress(
                    progress_stage, "progress",
                    f"{query_index}/{len(queries)}件目 · 本文を取得中（候補 {len(query_candidates)}件）",
                )
                query_candidates = populate_content(
                    query_candidates, self.fetcher, timeout=self.settings.fetch_timeout,
                    max_length=self.settings.max_content_length,
                    fallback_fetcher=self.fallback_fetcher, weights=weights,
                    official_domains=self.settings.official_domains,
                    trusted_domains=self.settings.trusted_domains,
                )
            evidence_results.extend(query_candidates)
            selected_results.extend(
                select_diverse_results(query_candidates, self.settings.reranked_results_per_query)
            )
        evidence_results = deduplicate_results(evidence_results)
        selected = select_diverse_results(
            deduplicate_results(selected_results),
            self.settings.reranked_results_per_query * len(queries),
        )
        self.log_selected_results(selected, label="公式ソース専用検索候補")
        if not selected:
            self.log_detail("公式ソース専用検索で採用できる候補はありませんでした")
        self.log_detail(f"公式ソース本文候補 {len(evidence_results)}件を品質評価へ渡します")
        return {
            "results": selected,
            "evidence_results": evidence_results,
            "official_results": evidence_results,
            "search_query_count": len(searched_queries),
            "searched_queries": searched_queries,
            "official_search_query_count": len(searched_queries),
        }

    def search_task(self, state: State):
        task = state.get("task")
        if task is None:
            return {"results": [], "evidence_results": [], "search_query_count": 0, "searched_queries": [], "normal_search_query_count": 0}
        fresh_results: list[SearchResult] = []
        self.log_detail(f"検索開始: {task.query}（観点: {task.aspect}）")
        try:
            response = self.search.text(task.query, max_results=self.settings.results_per_query)
            add_search_results(fresh_results, response, task.query)
        except Exception as error:
            if "no results found" in str(error).casefold():
                self.log_detail(f"検索結果なし: {task.query}")
            else:
                print(f"検索失敗 ({task.query}): {error}")
                self.log_detail(f"検索失敗: {task.query} / {error}")
            return {
                "results": [], "evidence_results": [], "search_query_count": 1,
                "searched_queries": [task.query], "normal_search_query_count": 1,
            }

        self.report_progress("search", "progress", f"候補 {len(fresh_results)}件 · {task.query}")
        self.log_detail(f"検索候補: {len(fresh_results)}件（取得上限: {self.settings.results_per_query}）")

        candidate_count = (
            self.settings.content_candidate_results_per_query
            if self.fetcher else self.settings.reranked_results_per_query
        )
        scoring_weights = (self.settings.relevance_weight, self.settings.trust_weight, self.settings.freshness_weight)
        top_results = rerank_results(
            fresh_results, self.reranker, candidate_count, weights=scoring_weights,
            official_domains=self.settings.official_domains,
            trusted_domains=self.settings.trusted_domains,
        )
        self.log_detail(f"再ランキング: {len(fresh_results)}件から上位候補 {len(top_results)}件を選択")
        if self.fetcher:
            self.report_progress("search", "progress", f"本文を取得中（候補 {len(top_results)}件） · {task.query}")
            top_results = populate_content(
                top_results,
                self.fetcher,
                timeout=self.settings.fetch_timeout,
                max_length=self.settings.max_content_length,
                fallback_fetcher=self.fallback_fetcher,
                weights=scoring_weights,
                official_domains=self.settings.official_domains,
                trusted_domains=self.settings.trusted_domains,
            )
        evidence_results = top_results
        self.log_detail(f"品質評価用に本文候補 {len(evidence_results)}件を保持します")
        top_results = select_diverse_results(top_results, self.settings.reranked_results_per_query)

        self.log_selected_results(top_results)

        return {
            "results": top_results,
            "evidence_results": evidence_results,
            "search_query_count": 1,
            "searched_queries": [task.query],
            "normal_search_query_count": 1,
        }

    def evaluate(self, state: State):
        all_results = deduplicate_results(
            (state.get("evidence_results", []) or state.get("results", []))
            + state.get("official_results", [])
        )
        unique_results = _evaluation_sources(
            all_results, official_candidates=state.get("official_results", []),
        )
        cached_source_assessments: list[SourceAssessment] = []
        assessment_targets: list[SearchResult] = []
        for item in unique_results:
            cached_assessment = self.source_assessment_cache.get(item.url, item.content)
            if cached_assessment is None:
                assessment_targets.append(item)
            else:
                cached_source_assessments.append(cached_assessment)
        assessment_target_urls = {item.url for item in assessment_targets if item.url}
        official_urls = {item.url for item in state.get("official_results", []) if item.url}
        evaluation_urls = {item.url for item in unique_results if item.url}
        for item in all_results:
            if item.url in evaluation_urls:
                decision = "included"
                reason = "Included in quality evaluation evidence."
            elif item.relevance_score is None:
                decision = "excluded"
                reason = "Missing relevance score."
            elif item.relevance_score < MIN_RELEVANCE_FOR_EVALUATION:
                decision = "excluded"
                reason = f"Relevance score below {MIN_RELEVANCE_FOR_EVALUATION:.2f}."
            else:
                decision = "excluded"
                reason = "Excluded by the maximum evidence-source limit."
            self.log_source_trace(
                "quality_evaluation", item, decision, reason,
                official_candidate=item.url in official_urls,
            )
        relevance_eligible = [
            item for item in all_results
            if item.relevance_score is not None
            and item.relevance_score >= MIN_RELEVANCE_FOR_EVALUATION
        ]
        excluded_count = len(all_results) - len(relevance_eligible)
        if excluded_count:
            self.log_detail(
                f"品質評価から低関連ソース {excluded_count}件を除外しました "
                f"（関連度 {MIN_RELEVANCE_FOR_EVALUATION:.2f} 未満）"
            )
        capped_count = len(relevance_eligible) - len(unique_results)
        if capped_count:
            self.log_detail(
                f"品質評価の入力上限と出典分散により {capped_count}件を今回の評価対象から外しました "
                f"（評価対象上限: {MAX_EVIDENCE_SOURCES}件）"
            )
        formatted_evidence = format_results_for_llm(unique_results)
        if not formatted_evidence:
            formatted_evidence = "No eligible evidence."
        brief_context = research_brief_context(state.get("run_memo", ""), evaluation_urls)
        assessment_targets_text = "\n".join(
            f"- {item.url} | {item.title[:120]} | fetch={item.fetch_status or 'unknown'}"
            for item in assessment_targets
        ) or "none"
        cached_assessments_text = "\n".join(
            f"- {assessment.url} | credibility={assessment.credibility:.2f} | use={assessment.use_as_evidence} | {assessment.reason[:140]}"
            for assessment in cached_source_assessments
        ) or "none"
        source_assessment_instruction = f"""
Page-level source assessment:
- Assess every URL in `URLs requiring assessment` and no other URL.
- Judge this fetched page, not the reputation of its domain. Consider publisher/author attribution, editorial accountability, traceable primary evidence, independence, recency, and relevance to the user's question.
- Set `use_as_evidence` to true only if the page directly supports a bounded claim in the final answer. Reject pages that are off-topic, lack a usable body, are unattributed reposts/aggregations, or make material claims without traceable support.
- A registered official or trusted domain is a ranking signal, not automatic evidence acceptance.
- Return concise reasons and a short `support_scope`; do not invent facts.
URLs requiring assessment:
{assessment_targets_text}
Previously cached page assessments (reuse unless the current evidence contradicts them):
{cached_assessments_text}
"""
        prompt = f"""You are a web-research quality evaluator (リサーチ品質評価担当). Return exactly one valid JSON object, with no prose or reasoning tags.
Date: {self.current_date}
Question: {state['query']}
Plan: {state['plan'].model_dump_json() if hasattr(state.get('plan'), 'model_dump_json') else str(state.get('plan', ''))}
Evidence (each item has a URL and retrieved page text when available):
{formatted_evidence}
{source_assessment_instruction}
Rules:
- Decide whether the evidence answers the question without unsupported claims.
- Prefer direct primary evidence for material facts. Secondary evidence must be relevant and independently traceable; a domain label alone is never enough. 二次情報であることだけを理由に証拠を弱いと判定してはいけません。
- For material secondary claims, seek independent support from at least two different publisher domains; syndicated or reposted copies do not count. 異なる発行元ドメインの2件以上を目安にし、転載・再掲は独立根拠として数えません。
- Mark missing, stale, contradictory, inaccessible, or unsupported material as insufficient. Do not claim that a source exists or is absent unless the evidence shows it.
- Add short search-engine queries only for information that would materially improve the answer. Put first-party or official-document searches in `official_additional_queries`.
Return this JSON schema (keep strings concise):
{{"sufficient":false,"missing_information":[],"weak_evidence":[],"reason":"","additional_queries":[],"official_additional_queries":[],"source_assessments":[{{"url":"","credibility":0.0,"use_as_evidence":false,"reason":"","support_scope":""}}]}}
"""
        raw_evaluation = ""
        evaluation_failed = False
        try:
            response = self.invoke_llm(
                "evaluate", prompt, llm=self.evaluator_llm, source_count=len(unique_results),
            )
            raw_evaluation = response.content
            data = normalize_evaluation_payload(parse_json_from_response(raw_evaluation))
            eval_obj = Evaluation.model_validate(data)
        except Exception as err:
            self.log_detail(f"品質評価の応答を解析できませんでした: {err}")
            self.log_detail("判断: 応答形式を修正して品質評価を1回だけ再試行します")
            try:
                if not raw_evaluation:
                    raise RuntimeError("品質評価モデルから応答を取得できませんでした")
                repair_prompt = f'''次の品質評価案を、指定形式の有効なJSONに修正してください。
評価案の内容を保ち、値を推測で追加しないでください。JSON以外は出力しないでください。
必須キー: sufficient (boolean), missing_information (string[]), weak_evidence (string[]), reason (string), additional_queries (string[]), official_additional_queries (string[])
両方のクエリ配列の各要素は検索文そのものの文字列です。{{"query": "..."}} のようなオブジェクトを配列に入れないでください。

評価案:
{raw_evaluation}

JSON例:
{{"sufficient":false,"missing_information":[],"weak_evidence":[],"reason":"評価理由","additional_queries":[],"official_additional_queries":[]}}'''
                repaired = self.invoke_llm(
                    "evaluate_repair", repair_prompt, llm=self.evaluator_llm, source_count=len(unique_results),
                )
                repaired_data = normalize_evaluation_payload(parse_json_from_response(repaired.content))
                eval_obj = Evaluation.model_validate(repaired_data)
                self.log_detail("品質評価の再試行に成功しました")
            except Exception as retry_error:
                evaluation_failed = True
                print(f"品質評価の再試行にも失敗しました: {retry_error}")
                self.log_detail(f"品質評価の再試行にも失敗しました: {retry_error}")
                fallback_queries = []
                if (
                    state.get("search_round", 0) == 0
                    and state.get("search_query_count", 0) < self.settings.max_search_queries
                    and self.settings.max_search_rounds > 0
                ):
                    fallback_query = _keyword_followup_query(
                        state["query"], [], state.get("searched_queries", []),
                    )
                    if fallback_query:
                        fallback_queries = [fallback_query]
                        self.log_detail("判断: 評価結果を得られなかったため、対象名と公式情報を使った検索に切り替えます")
                eval_obj = Evaluation(
                    sufficient=False,
                    missing_information=["品質評価の応答を解析できず、情報の十分性を確認できませんでした。"],
                    weak_evidence=[],
                    reason=f"品質評価の応答を解析できませんでした: {retry_error}",
                    additional_queries=fallback_queries,
                    official_additional_queries=[],
                )

        fresh_source_assessments = [
            assessment for assessment in eval_obj.source_assessments
            if assessment.url in assessment_target_urls
        ]
        eval_obj.source_assessments = merge_assessments(
            cached_source_assessments,
            fresh_source_assessments,
            allowed_urls=(item.url for item in unique_results),
        )
        result_by_url = {item.url: item for item in unique_results if item.url}
        for assessment in fresh_source_assessments:
            result = result_by_url.get(assessment.url)
            if result is not None:
                self.source_assessment_cache.put(assessment, result.content)

        for assessment in eval_obj.source_assessments:
            result = result_by_url.get(assessment.url)
            if result is None:
                continue
            accepted = (
                assessment.use_as_evidence
                and assessment.credibility >= self.settings.source_assessment_min_credibility
            )
            self.log_source_trace(
                "page_assessment",
                result,
                "accepted" if accepted else "rejected",
                assessment.reason,
                official_candidate=result.url in official_urls,
                page_assessment=assessment,
            )

        proposed_queries = eval_obj.additional_queries
        proposed_official_queries = eval_obj.official_additional_queries
        try:
            eval_obj.additional_queries = _validate_followup_queries(
                state["query"], proposed_queries, self.reranker,
            )
        except Exception as error:
            eval_obj.additional_queries = []
            self.log_detail(f"追加検索案の関連度検査に失敗したため、モデル生成クエリを破棄しました: {error}")
        try:
            eval_obj.official_additional_queries = _validate_followup_queries(
                state["query"], proposed_official_queries, self.reranker,
            )
        except Exception as error:
            eval_obj.official_additional_queries = []
            self.log_detail(f"公式レーンの追加案の関連度検査に失敗したため、クエリを破棄しました: {error}")
        rejected_count = len(proposed_queries) - len(eval_obj.additional_queries)
        if rejected_count:
            self.log_detail(
                f"元の質問との関連が弱い追加検索案 {rejected_count}件を破棄しました"
            )

        can_search_more = (
            state.get("search_round", 0) < self.settings.max_search_rounds
            and state.get("search_query_count", 0) < self.settings.max_search_queries
        )
        if not can_search_more:
            eval_obj.official_additional_queries = []
        if (
            not eval_obj.sufficient
            and not eval_obj.additional_queries
            and not eval_obj.official_additional_queries
            and can_search_more
            and (not evaluation_failed or state.get("search_round", 0) == 0)
        ):
            fallback_query = _keyword_followup_query(
                state["query"], eval_obj.missing_information,
                state.get("searched_queries", []),
            )
            eval_obj.additional_queries = [fallback_query] if fallback_query else []
            if fallback_query:
                self.log_detail("有効な追加検索案がないため、不足項目から短いキーワード検索を作成します")
            else:
                self.log_detail("不足項目に対応する未検索のキーワードがないため、追加検索を終了します")

        eval_obj.sufficient = eval_obj.sufficient and not (
            eval_obj.additional_queries or eval_obj.official_additional_queries
            or eval_obj.missing_information or eval_obj.weak_evidence
        )
        self.log_detail(f"品質評価: {'十分' if eval_obj.sufficient else '不足'} / 理由: {eval_obj.reason}")
        for missing in eval_obj.missing_information:
            self.log_detail(f"  不足情報: {missing}")
        for weak in eval_obj.weak_evidence:
            self.log_detail(f"  根拠が弱い点: {weak}")
        for query in eval_obj.additional_queries:
            self.log_detail(f"  通常レーン追加案: {query}")
        for query in eval_obj.official_additional_queries:
            self.log_detail(f"  公式レーン追加案: {query}")
        for assessment in eval_obj.source_assessments:
            decision = "accepted" if (
                assessment.use_as_evidence
                and assessment.credibility >= self.settings.source_assessment_min_credibility
            ) else "rejected"
            self.log_detail(
                f"Source assessment: {decision} | credibility={assessment.credibility:.2f} | "
                f"{assessment.url} | {assessment.reason}"
            )
        return {"evaluation": eval_obj}

    def update_board(self, state: State):
        evaluation = state.get("evaluation")
        all_results = deduplicate_results(
            (state.get("evidence_results", []) or state.get("results", []))
            + state.get("official_results", [])
        )
        board_sources = _evaluation_sources(
            all_results, official_candidates=state.get("official_results", []),
        )
        assessment_map = assessment_by_url(evaluation.source_assessments if evaluation else [])
        source_items = []
        for item in board_sources[:3]:
            assessment = assessment_map.get(item.url)
            if assessment is None:
                decision = "unassessed"
                reason = "No completed page-level assessment; retained as a candidate only."
                if item.fetch_error:
                    reason = f"{reason} Fetch: {item.fetch_error}"
                credibility = None
            elif (
                assessment.use_as_evidence
                and assessment.credibility >= self.settings.source_assessment_min_credibility
            ):
                decision = "accepted_as_evidence"
                reason = assessment.reason
                credibility = assessment.credibility
            else:
                decision = "rejected_as_evidence"
                reason = assessment.reason
                credibility = assessment.credibility
            source_items.append({
                "url": item.url[:220],
                "title": item.title[:120],
                "fetch_status": item.fetch_status,
            "fetch_error": item.fetch_error,
                "decision": decision,
                "reason": reason[:180],
                "credibility": credibility,
            })
        known_source_urls = {item["url"] for item in source_items}
        evaluation_summary = {
            "sufficient": evaluation.sufficient if evaluation else False,
            "reason": (evaluation.reason if evaluation else "Not evaluated")[:240],
            "missing_information": (evaluation.missing_information if evaluation else [])[:3],
            "weak_evidence": (evaluation.weak_evidence if evaluation else [])[:3],
        }
        current_memo = (state.get("run_memo", "") or "")[-self.settings.memo_keeper_max_chars:]
        fallback_payload = {
            "request": state["query"][:300],
            "confirmed": [],
            "unresolved": [str(item)[:120] for item in evaluation_summary["missing_information"]],
            "source_decisions": source_items,
            "next_action": evaluation_summary["reason"],
        }
        memo_payload: dict[str, Any] = fallback_payload
        if self.settings.memo_keeper_enabled:
            prompt = f"""You are a memo keeper for one research run. Update a compact JSON memo only; do not answer the user's question.
Keep only source-linked confirmed facts, unresolved questions, URL-level source decisions, and the next action. A confirmed fact must use a URL from Selected sources; do not invent URLs or facts. Do not include reasoning or prose outside JSON. Keep each list to at most 3 entries and the entire JSON concise.

User request: {state['query'][:600]}
Previous compact memo: {current_memo or '{}'}
New quality evaluation: {json.dumps(evaluation_summary, ensure_ascii=False)}
Selected sources: {json.dumps(source_items, ensure_ascii=False)}

Return exactly this JSON schema:
{{"request":"","confirmed":[{{"source_url":"","note":""}}],"unresolved":[""],"source_decisions":[{{"url":"","decision":"","reason":""}}],"next_action":""}}"""
            try:
                response = self.invoke_llm(
                    "memo_keeper", prompt, llm=self.memo_keeper_llm, source_count=len(board_sources),
                )
                parsed = parse_json_from_response(strip_model_reasoning(response.content))
                if not isinstance(parsed, dict):
                    raise ValueError("Memo keeper response is not a JSON object")
                memo_payload = {
                    "request": str(parsed.get("request") or state["query"])[:300],
                    "confirmed": [
                        {
                            "source_url": str(item.get("source_url", "")).strip()[:220],
                            "note": str(item.get("note", "")).strip()[:140],
                        }
                        for item in parsed.get("confirmed", [])
                        if isinstance(item, dict)
                        and str(item.get("source_url", "")).strip() in known_source_urls
                        and str(item.get("note", "")).strip()
                    ][:3],
                    "unresolved": [str(item)[:140] for item in parsed.get("unresolved", []) if str(item).strip()][:3],
                    "source_decisions": [
                        {
                            "url": str(item.get("url", "")).strip()[:220],
                            "decision": str(item.get("decision", "")).strip()[:48],
                            "reason": str(item.get("reason", "")).strip()[:120],
                        }
                        for item in parsed.get("source_decisions", [])
                        if isinstance(item, dict)
                        and str(item.get("url", "")).strip() in known_source_urls
                    ][:3],
                    "next_action": str(parsed.get("next_action", ""))[:180],
                }
                self.log_detail("メモキーパー: 独立した短文メモを更新しました")
            except Exception as error:
                self.log_detail(f"メモキーパーに失敗したため決定論的メモへフォールバックしました: {type(error).__name__}")
        memo = json.dumps(memo_payload, ensure_ascii=False, separators=(",", ":"))
        if len(memo) > self.settings.memo_keeper_max_chars:
            memo = json.dumps(fallback_payload, ensure_ascii=False, separators=(",", ":"))
        if self.board_update is not None:
            self.board_update(memo)
        return {"run_memo": memo}

    def prepare_followups(self, state: State):
        evaluation = state.get("evaluation")
        if evaluation is None or evaluation.sufficient:
            return {"pending_normal_queries": [], "pending_official_queries": []}
        if state.get("search_round", 0) >= self.settings.max_search_rounds:
            self.log_detail(f"次の判断: 検索ラウンド上限 {self.settings.max_search_rounds} に達しました")
            return {"pending_normal_queries": [], "pending_official_queries": []}
        remaining = self.settings.max_search_queries - state.get("search_query_count", 0)
        if remaining <= 0:
            self.log_detail(f"次の判断: 検索クエリ上限 {self.settings.max_search_queries} に達しました")
            return {"pending_normal_queries": [], "pending_official_queries": []}
        searched = {
            re.sub(r"\s+", " ", query).strip().casefold()
            for query in state.get("searched_queries", [])
        }
        def pending(queries):
            unique = []
            for query in queries:
                normalized = re.sub(r"\s+", " ", query).strip().casefold()
                if normalized and normalized not in searched:
                    unique.append(query)
                    searched.add(normalized)
                elif normalized:
                    self.log_detail(f"検索済みの追加案をスキップします: {query}")
            return unique
        normal = pending(evaluation.additional_queries)
        official = pending(evaluation.official_additional_queries)
        # Rotate a shared, finite budget toward the lane with fewer searches so far.
        lanes = [
            ["normal", normal, state.get("normal_search_query_count", 0)],
            ["official", official, state.get("official_search_query_count", 0)],
        ]
        lanes.sort(key=lambda lane: (lane[2], 0 if lane[0] == "official" else 1))
        selected = {"normal": [], "official": []}
        for _ in range(remaining):
            available = [lane for lane in lanes if lane[1]]
            if not available:
                break
            lane = min(available, key=lambda item: (item[2] + len(selected[item[0]]), item[0] != "official"))
            if lane[0] == "official" and len(selected["official"]) >= 5:
                lane[1].clear()
                continue
            selected[lane[0]].append(lane[1].pop(0))
        if len(selected["normal"]) < len(normal) or len(selected["official"]) < len(official):
            self.log_detail("共有クエリ上限により、一部の追加検索案を次回以降に回せず見送りました")
        if selected["normal"] or selected["official"]:
            self.log_detail(
                f"独立レーンで追加検索します（通常 {len(selected['normal'])}件 / 公式 {len(selected['official'])}件）"
            )
            return {
                "pending_normal_queries": selected["normal"],
                "pending_official_queries": selected["official"],
                "search_round": 1,
            }
        self.log_detail("実行できる未検索の追加案がないため、レポート作成へ進みます")
        return {"pending_normal_queries": [], "pending_official_queries": []}

    def dispatch_followups(self, state: State):
        sends = []
        normal = state.get("pending_normal_queries", [])
        official = state.get("pending_official_queries", [])
        if normal:
            sends.append(Send("additional_search", {
                "queries": normal,
                "existing_urls": [item.url for item in
                    state.get("results", []) + state.get("evidence_results", [])
                    + state.get("official_results", []) if item.url],
                "search_round": state.get("search_round", 0),
            }))
        if official:
            sends.append(Send("official_additional_search", {
                "official_queries": official,
                "progress_stage": "official_additional_search",
            }))
        if not sends:
            return "analyze"
        return sends

    def additional_search(self, state: State):
        queries = state.get("queries", [])
        self.log_detail(f"通常レーンの追加検索: {len(queries)}件")
        existing_urls = set(state.get("existing_urls", [])) | {
            item.url for item in state.get("results", []) + state.get("evidence_results", [])
            if item.url
        }
        delta_results: list[SearchResult] = []
        delta_evidence_results: list[SearchResult] = []
        for query_index, query in enumerate(queries, start=1):
            self.report_progress(
                "additional_search", "progress",
                f"{query_index}/{len(queries)}件目 · 検索中: {query}",
            )
            self.log_detail(f"追加検索開始: {query}")
            query_fresh: list[SearchResult] = []
            try:
                response = self.search.text(query, max_results=self.settings.results_per_query)
                for item in response:
                    url = item.get("href", "")
                    if not url or url in existing_urls:
                        continue
                    query_fresh.append(SearchResult(
                        query=query, title=item.get("title", ""), url=url,
                        snippet=item.get("body", ""),
                    ))
                    existing_urls.add(url)
                self.log_detail(
                    f"追加検索候補: {len(query_fresh)}件（既出URLを除外、取得上限: {self.settings.results_per_query}）"
                )
                self.report_progress(
                    "additional_search", "progress",
                    f"{query_index}/{len(queries)}件目 · 新規候補 {len(query_fresh)}件",
                )
                candidate_count = (
                    self.settings.content_candidate_results_per_query
                    if self.fetcher else self.settings.reranked_results_per_query
                )
                scoring_weights = (self.settings.relevance_weight, self.settings.trust_weight, self.settings.freshness_weight)
                top_query_results = rerank_results(
                    query_fresh, self.reranker, candidate_count, weights=scoring_weights,
                    official_domains=self.settings.official_domains,
                    trusted_domains=self.settings.trusted_domains,
                )
                if self.fetcher:
                    self.report_progress(
                        "additional_search", "progress",
                        f"{query_index}/{len(queries)}件目 · 本文を取得中（候補 {len(top_query_results)}件）",
                    )
                    top_query_results = populate_content(
                        top_query_results,
                        self.fetcher,
                        timeout=self.settings.fetch_timeout,
                        max_length=self.settings.max_content_length,
                        fallback_fetcher=self.fallback_fetcher,
                        weights=scoring_weights,
                        official_domains=self.settings.official_domains,
                        trusted_domains=self.settings.trusted_domains,
                    )
                delta_evidence_results.extend(top_query_results)
                self.log_detail(f"品質評価用に追加検索の本文候補 {len(top_query_results)}件を保持します")
                top_query_results = select_diverse_results(
                    top_query_results, self.settings.reranked_results_per_query,
                )
                self.log_selected_results(top_query_results, label="追加検索採用候補")
                delta_results.extend(top_query_results)
            except Exception as error:
                if "no results found" in str(error).casefold():
                    self.log_detail(f"追加検索結果なし: {query}")
                else:
                    print(f"追加検索失敗 ({query}): {error}")
                    self.log_detail(f"追加検索失敗: {query} / {error}")
        return {
            "results": delta_results,
            "evidence_results": delta_evidence_results,
            "search_query_count": len(queries),
            "searched_queries": queries,
            "normal_search_query_count": len(queries),
        }

    def analyze(self, state: State):
        candidate_results = deduplicate_results(
            (state.get("evidence_results", []) or state.get("results", []))
            + state.get("official_results", [])
        )
        evaluation = state.get("evaluation")
        assessment_map = assessment_by_url(evaluation.source_assessments if evaluation else [])
        eligible_candidates = []
        for item in candidate_results:
            assessment = assessment_map.get(item.url)
            if assessment is None or (
                assessment.use_as_evidence
                and assessment.credibility >= self.settings.source_assessment_min_credibility
            ):
                eligible_candidates.append(item)
        unique_results = _evaluation_sources(
            eligible_candidates, official_candidates=state.get("official_results", []),
        )
        self.log_detail(
            f"Final evidence: {len(unique_results)} source(s) after relevance, diversity, and page assessment."
        )
        official_urls = {item.url for item in state.get("official_results", []) if item.url}
        final_urls = {item.url for item in unique_results if item.url}
        for item in candidate_results:
            assessment = assessment_map.get(item.url)
            if assessment is not None and not (
                assessment.use_as_evidence
                and assessment.credibility >= self.settings.source_assessment_min_credibility
            ):
                decision = "rejected"
                reason = f"Rejected by page assessment: {assessment.reason}"
            elif item.url in final_urls:
                decision = "included"
                reason = "Included in final-answer evidence."
            else:
                decision = "excluded"
                reason = "Excluded by final evidence selection."
            self.log_source_trace(
                "final_answer", item, decision, reason,
                official_candidate=item.url in official_urls,
                page_assessment=assessment,
            )
        if not unique_results:
            self.log_detail("Final answer skipped: no eligible evidence remained after assessment.")
            return {
                "summary": (
                    "## \u8abf\u67fb\u7d50\u679c\n\n"
                    "\u6700\u7d42\u56de\u7b54\u306e\u6839\u62e0\u3068\u3057\u3066\u63a1\u7528\u3067\u304d\u308b\u60c5\u5831\u6e90\u304c\u6b8b\u3089\u306a\u304b\u3063\u305f\u305f\u3081\u3001"
                    "\u6839\u62e0\u306e\u306a\u3044\u56de\u7b54\u306f\u751f\u6210\u3057\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u691c\u7d22\u6761\u4ef6\u3084\u767b\u9332\u30c9\u30e1\u30a4\u30f3\u3092\u898b\u76f4\u3057\u3001"
                    "\u518d\u8abf\u67fb\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
                )
            }
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
        brief_context = research_brief_context(state.get("run_memo", ""), final_urls)
        answer_prompt = f"""以下のWeb検索結果および取得したWebページ本文だけを根拠に質問へ回答してください。
現在日（この調査の基準日）: {self.current_date}
質問: {state['query']}

この調査回だけの共有ボード（過去の投稿を含む索引用。事実の根拠として引用せず、必ず下の取得済みソースで確認すること）:
{brief_context or "none"}

収集された情報源:
{formatted_evidence}
{quality_note}

指示:
1. {self.current_date} 時点での情報として回答し、更新時期が重要な情報はページ日付と根拠を確認してください。
2. Webページの本文に記載されている具体的な事実、数値、詳細を最大限に活用して、質の高い体系的なレポートを作成してください。
3. 事実を述べる段落には、根拠にした出典番号 [S1] の形式を付けてください。複数なら [S1][S2] とします。
4. 検索結果にない出典番号やURLを本文に書かないでください。参照ソース一覧はプログラムが付けます。
5. 取得情報から確認できないことは断定せず、調査品質が不十分な場合はその制約を明記してください。
6. 元の質問で求められていない性能・育成・ガチャ詳細を必須情報のように付け加えず、公式確定情報とリーク・予想を明確に分けてください。
7. 思考過程や自己対話は出力せず、完成したレポート本文だけを出力してください。<think>等のタグも含めないでください。"""
        try:
            answer = strip_model_reasoning(self.invoke_llm(
                "final_answer", answer_prompt, llm=self.answer_llm, source_count=len(unique_results),
            ).content)
        except EmptyModelResponseError as error:
            self.log_detail(f"Final answer generation produced no content: {error}")
            return {"summary": (
                "調査結果を生成できませんでした。LM Studioでこのモデルの思考モードを無効化するか、"
                "役割別の出力上限を増やしてから再実行してください。"
            )}
        issues = citation_issues(answer, len(unique_results))
        has_citations = bool(_SOURCE_REF_RE.search(answer))
        if unique_results and (issues or not has_citations):
            self.log_detail(
                "出典検証: 不明な出典番号またはURLを検出したため、取得済みソースに限定して回答を再生成します"
                if issues else "出典検証: 出典番号がないため、根拠ソース番号を付けて回答を再生成します"
            )
            allowed_refs = ", ".join(f"[S{i}]" for i in range(1, len(unique_results) + 1))
            retry_prompt = f"""以下の回答案を根拠ソースに照合して修正してください。
現在日（この調査の基準日）: {self.current_date}
質問: {state['query']}
有効な出典番号: {allowed_refs}
取得済みの根拠:
{formatted_evidence}
元の回答:
{answer}

本文中のURLは書かず、事実を含む段落には有効な出典番号を付けてください。
根拠がない主張は削除するか、確認できないと明示してください。
思考過程や自己対話は出力せず、修正後の回答本文だけを出力してください。<think>等のタグも含めないでください。
調査品質が不十分な場合の注意:
{quality_note or "特段の不足は検出されていません。"}"""
            try:
                answer = strip_model_reasoning(self.invoke_llm(
                    "final_answer_citation_retry", retry_prompt,
                    llm=self.answer_llm, source_count=len(unique_results),
                ).content)
            except EmptyModelResponseError as error:
                self.log_detail(f"Citation retry produced no content; keeping the first answer: {error}")
        else:
            self.log_detail("出典検証: 出典番号の形式と範囲を確認しました（主張の裏付けは未検証）")

        return {"summary": render_verified_sources(answer, unique_results)}


def create_workflow(
    settings: Settings,
    search: SearchClient | None = None,
    reranker: Reranker | None = None,
    llm: ChatOpenAI | None = None,
    fetcher: WebContentFetcher | None = None,
    fallback_fetcher: WebContentFetcher | None = None,
    log: Callable[[str], None] | None = None,
    board_update: Callable[[str], None] | None = None,
    progress: Callable[[str, str, str], None] | None = None,
) -> Any:
    """Build the graph and initialize its dependencies."""
    nodes = ResearchWorkflowNodes(
        settings=settings,
        search=search,
        reranker=reranker,
        llm=llm,
        fetcher=fetcher,
        fallback_fetcher=fallback_fetcher,
        log=log,
        board_update=board_update,
        progress=progress,
    )

    builder = StateGraph(State)
    builder.add_node("understand_input", nodes.track_node("understand_input", nodes.understand_input))
    builder.add_node("planner", nodes.track_node("planner", nodes.planner))
    builder.add_node("search", nodes.track_node("search", nodes.search_task))
    builder.add_node("official_search", nodes.track_node("official_search", nodes.official_search))
    builder.add_node("evaluate", nodes.track_node("evaluate", nodes.evaluate))
    builder.add_node("board_update", nodes.track_node("board_update", nodes.update_board))
    builder.add_node("additional_search", nodes.track_node("additional_search", nodes.additional_search))
    builder.add_node("prepare_followups", nodes.track_node("prepare_followups", nodes.prepare_followups))
    builder.add_node("official_additional_search", nodes.track_node("official_additional_search", nodes.official_search))
    builder.add_node("analyze", nodes.track_node("analyze", nodes.analyze))
    builder.add_edge(START, "understand_input")
    builder.add_edge("understand_input", "planner")
    builder.add_conditional_edges("planner", nodes.dispatch_searches)
    builder.add_edge("search", "evaluate")
    builder.add_edge("official_search", "evaluate")
    builder.add_edge("evaluate", "board_update")
    builder.add_edge("board_update", "prepare_followups")
    builder.add_conditional_edges("prepare_followups", nodes.dispatch_followups)
    builder.add_edge("additional_search", "evaluate")
    builder.add_edge("official_additional_search", "evaluate")
    builder.add_edge("analyze", END)
    return builder.compile()
