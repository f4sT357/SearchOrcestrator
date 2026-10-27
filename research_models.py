"""Domain models and LangGraph state shared by research components."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from source_assessment import SourceAssessment


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



@dataclass(frozen=True)
class BoardContext:
    board_sources: list[SearchResult]
    source_items: list[dict[str, Any]]
    evaluation_summary: dict[str, Any]
    fallback_payload: dict[str, Any]
    current_memo: str


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
