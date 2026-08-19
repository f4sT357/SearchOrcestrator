"""A bounded, evidence-oriented web research workflow."""

from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Protocol, TypedDict

from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


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

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            model=os.getenv("SEARCH_MODEL", cls.model),
            base_url=os.getenv("SEARCH_BASE_URL", cls.base_url),
            api_key=os.getenv("SEARCH_API_KEY", cls.api_key),
        )


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
    relevance_score: float | None = None


class State(TypedDict):
    query: str
    plan: SearchPlan | None
    task: SearchTask | None
    results: Annotated[list[SearchResult], operator.add]
    evaluation: Evaluation | None
    summary: str
    search_query_count: Annotated[int, operator.add]
    search_round: Annotated[int, operator.add]


class SearchClient(Protocol):
    def text(self, query: str, *, max_results: int): ...


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs): ...


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


def create_workflow(
    settings: Settings,
    search: SearchClient | None = None,
    reranker: Reranker | None = None,
    llm: ChatOpenAI | None = None,
):
    """Build the graph and initialize its dependencies."""
    if llm is None:
        llm = ChatOpenAI(model=settings.model, base_url=settings.base_url, api_key=settings.api_key)
    planner_llm = llm.with_structured_output(SearchPlan)
    evaluation_llm = llm.with_structured_output(Evaluation)
    if search is None:
        search = DDGS()
    if reranker is None:
        reranker = CrossEncoder(settings.reranker_model)

    def planner(state: State):
        response = planner_llm.invoke(f"""あなたはリサーチプランナーです。現在日: {date.today().isoformat()}
質問: {state['query']}
重複しない体系的な調査計画を作成してください。各タスクに aspect、query、reason を含めてください。""")
        return {"plan": response}

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
        return {
            "results": rerank_results(fresh_results, reranker, settings.reranked_results_per_query),
            "search_query_count": 1,
        }

    def evaluate(state: State):
        unique_results = deduplicate_results(state.get("results", []))
        response = evaluation_llm.invoke(f"""あなたはリサーチ品質評価担当です。現在日: {date.today().isoformat()}
質問: {state['query']}
調査計画: {state['plan']}
検索結果: {unique_results}
計画の主要観点、根拠の強さ、情報源の信頼性と新しさを厳密に評価してください。
不足時は missing_information と weak_evidence を具体化し、追加検索クエリを最大3件指定してください。
追加検索、不足情報、または弱い根拠があれば sufficient は false にしてください。""")
        response.sufficient = response.sufficient and not (
            response.additional_queries or response.missing_information or response.weak_evidence
        )
        return {"evaluation": response}

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
                delta_results.extend(
                    rerank_results(query_fresh, reranker, settings.reranked_results_per_query)
                )
            except Exception as error:
                print(f"追加検索失敗 ({query}): {error}")
        return {
            "results": delta_results,
            "search_query_count": len(queries),
            "search_round": 1,
        }

    def analyze(state: State):
        unique_results = deduplicate_results(state.get("results", []))
        response = llm.invoke(f"""以下の検索結果だけを根拠に質問へ回答してください。
質問: {state['query']}
検索結果: {unique_results}
重要な主張には、根拠となる検索結果に含まれる URL を示してください。確認できないことは断定せず、URLを捏造しないでください。""")
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
