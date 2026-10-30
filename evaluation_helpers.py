"""Pure helpers for evidence selection and evaluation response handling."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Protocol
from urllib.parse import urlparse

from research_models import SearchResult
from security import is_safe_url, safe_search_query, untrusted_data


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs): ...


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
    # A few adversarial, very long URLs cannot consume the entire context.
    selected = []
    url_chars = 0
    for item in selected_primary + selected_secondary:
        cost = len(_encoded_evidence(item.url))
        if url_chars + cost <= MAX_EVIDENCE_CHARS // 2:
            selected.append(item)
            url_chars += cost
    return selected


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

    facets = [_compact_query_terms(item) for item in missing_information if safe_search_query(item, original_query)]
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
        if query.casefold() not in seen and safe_search_query(query, original_query):
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
        if not safe_search_query(proposed, original_query):
            continue
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
        if not math.isfinite(score) or score < MIN_RELEVANCE_FOR_FOLLOWUP_QUERY or (
            anchors and not contains_anchor and score < 0.75
        ):
            continue
        accepted.append(candidate)
    return accepted


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in results:
        if is_safe_url(item.url) and item.url not in seen:
            seen.add(item.url)
            unique.append(item)
    return unique


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


def _encoded_evidence(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Bound serialized metadata AND bodies, preserving every selected URL."""
    if not results:
        return ""
    envelope = len(untrusted_data("web_evidence", [])) - 2
    limit = MAX_EVIDENCE_CHARS - envelope
    scale = 1.0
    while True:
        fixed = []
        for i, item in enumerate(results, start=1):
            heading = f"--- 検索結果 #{i} / 出典番号 [S{i}] / 検索クエリ: {item.query[:int(300 * scale)]} ---\n"
            metadata = item.model_copy(update={
                "content": None, "title": item.title[:int(200 * scale)],
                "snippet": item.snippet[:int(500 * scale)],
            }).to_evidence_text()
            fixed.append(heading + metadata)
        fixed_size = len(_encoded_evidence(fixed))
        if fixed_size <= limit:
            break
        if scale == 0:
            # Internal selection reserves URL space first. Public callers with
            # an impossible URL-only budget receive no partial source mapping.
            return untrusted_data("web_evidence", "Evidence metadata exceeds input budget.")
        scale = scale / 2 if scale > 0.01 else 0

    remaining = limit - fixed_size
    parts = []
    for index, (metadata, item) in enumerate(zip(fixed, results, strict=True)):
        body = (item.content or "")[:MAX_EVIDENCE_CHARS]
        allowance = remaining // (len(results) - index)
        prefix = "\n【Webページ本文抜粋】:\n"
        base_size = len(_encoded_evidence(metadata))
        low, high = 0, len(body)
        while low < high:
            middle = (low + high + 1) // 2
            cost = len(_encoded_evidence(metadata + prefix + body[:middle])) - base_size
            if cost <= allowance:
                low = middle
            else:
                high = middle - 1
        part = metadata + prefix + body[:low] if low else metadata
        remaining -= len(_encoded_evidence(part)) - base_size
        parts.append(part)
    return untrusted_data("web_evidence", parts)


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
