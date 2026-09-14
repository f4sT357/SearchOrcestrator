"""Improved workflow controller for SearchOrcestrator."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import Field

from search_orchestrator import (
    Evaluation,
    SearchPlan,
    SearchResult,
    SearchTask,
    Settings,
    SearchClient,
    Reranker,
    WebContentFetcher,
    JinaReaderFetcher,
    DefaultWebContentFetcher,
    ChatOpenAI,
    DDGS,
    CrossEncoder,
    date,
    deduplicate_results,
    populate_content,
    parse_json_from_response,
    rerank_results,
)


class PlannedSearchResult(SearchResult):
    """Search result carrying the plan aspect through reranking and fetching."""

    aspect: str = Field(default="追加調査")


class State(TypedDict):
    query: str
    plan: SearchPlan | None
    task: SearchTask | None
    results: Annotated[list[SearchResult], operator.add]
    evaluation: Evaluation | None
    summary: str
    search_query_count: Annotated[int, operator.add]
    search_round: int
    searched_queries: Annotated[list[str], operator.add]


def _canonical_url(url: str) -> str:
    """Return a stable URL key for deduplication without changing displayed URLs."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return url.strip()
        hostname = (parts.hostname or "").lower()
        if not hostname:
            return url.strip()
        port = parts.port
        netloc = hostname
        if port is not None and not ((parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))
    except ValueError:
        return url.strip()


def _unique_queries(queries: list[str]) -> list[str]:
    """Normalize and deduplicate search queries while preserving planner order."""
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            unique.append(normalized)
            seen.add(key)
    return unique


def _make_result(task: SearchTask, item: dict) -> PlannedSearchResult | None:
    url = item.get("href", "")
    if not url:
        return None
    return PlannedSearchResult(
        aspect=task.aspect,
        query=task.query,
        title=item.get("title", ""),
        url=url,
        snippet=item.get("body", ""),
    )


def _format_evidence(results: list[SearchResult], plan: SearchPlan | None) -> str:
    parts = []
    for i, item in enumerate(results, 1):
        aspect = getattr(item, "aspect", None)
        task = next((t for t in plan.tasks if t.query == item.query), None) if plan else None
        aspect = aspect or (task.aspect if task else "追加調査")
        reason = task.reason if task else "評価で不足した情報を補完"
        score = f" [関連度スコア: {item.relevance_score:.3f}]" if item.relevance_score is not None else ""
        fetch_status = f" [本文取得: {item.fetch_status}]" if item.fetch_status else ""
        text = (
            f"--- 検索結果 #{i} ---\n"
            f"【調査観点】: {aspect}\n"
            f"【検索理由】: {reason}\n"
            f"【タイトル】: {item.title}{score}{fetch_status}\n"
            f"【URL】: {item.url}\n"
            f"【概要/スニペット】: {item.snippet}"
        )
        if item.content:
            text += f"\n【Webページ本文抜粋】:\n{item.content}"
        parts.append(text)
    return "\n\n".join(parts)


def create_workflow(
    settings: Settings,
    search: SearchClient | None = None,
    reranker: Reranker | None = None,
    llm: ChatOpenAI | None = None,
    fetcher: WebContentFetcher | None = None,
    fallback_fetcher: WebContentFetcher | None = None,
) -> Any:
    """Build a bounded workflow with deduplicated queries and a total search budget."""
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
    if fallback_fetcher is None and settings.use_fallback_fetcher:
        fallback_fetcher = JinaReaderFetcher(api_url=settings.jina_api_url)

    def planner(state: State):
        prompt = f"""あなたはリサーチプランナーです。現在日: {date.today().isoformat()}
質問: {state['query']}

重複しない体系的な調査計画を作成し、以下のJSON形式のみを出力してください。
{{"tasks":[{{"aspect":"調査観点","query":"検索クエリ","reason":"検索理由"}}]}}
"""
        try:
            data = parse_json_from_response(llm.invoke(prompt).content)
            plan = SearchPlan.model_validate(data)
        except Exception:
            plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
        if not plan.tasks:
            plan = SearchPlan(tasks=[SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")])
        unique_tasks: list[SearchTask] = []
        seen: set[str] = set()
        for task in plan.tasks:
            query = task.query.strip()
            key = query.casefold()
            if query and key not in seen:
                unique_tasks.append(task.model_copy(update={"query": query}))
                seen.add(key)
        if not unique_tasks:
            unique_tasks = [SearchTask(aspect="総合調査", query=state["query"], reason="基本情報収集")]
        return {"plan": SearchPlan(tasks=unique_tasks)}

    def dispatch_searches(state: State):
        tasks = state["plan"].tasks if state["plan"] else []
        initial_limit = max(settings.max_search_queries, 0)
        return [Send("search", {"task": task}) for task in tasks[:initial_limit]]

    def search_task(state: State):
        task = state.get("task")
        if task is None:
            return {"results": [], "search_query_count": 0, "searched_queries": []}
        try:
            response = search.text(task.query, max_results=settings.results_per_query)
            fresh: list[SearchResult] = []
            seen: set[str] = set()
            for item in response:
                result = _make_result(task, item)
                if result:
                    key = _canonical_url(result.url)
                    if key not in seen:
                        fresh.append(result)
                        seen.add(key)
            top = rerank_results(fresh, reranker, settings.reranked_results_per_query)
            if fetcher:
                top = populate_content(top, fetcher, timeout=settings.fetch_timeout,
                                       max_length=settings.max_content_length,
                                       fallback_fetcher=fallback_fetcher)
            return {"results": top, "search_query_count": 1, "searched_queries": [task.query]}
        except Exception as error:
            print(f"検索失敗 ({task.query}): {error}")
            return {"results": [], "search_query_count": 1, "searched_queries": [task.query]}

    def evaluate(state: State):
        evidence = _format_evidence(deduplicate_results(state.get("results", [])), state.get("plan"))
        prompt = f"""あなたはリサーチ品質評価担当です。現在日: {date.today().isoformat()}
質問: {state['query']}
調査計画: {state['plan']}
収集した検索結果およびWebページ本文:
{evidence}

主要観点、根拠の強さ、情報源の信頼性と新しさを評価してください。
不足がある場合だけ、まだ検索していない追加検索クエリを提示してください。JSONのみ出力してください。
{{"sufficient":false,"missing_information":[],"weak_evidence":[],"reason":"","additional_queries":[]}}
"""
        try:
            evaluation = Evaluation.model_validate(parse_json_from_response(llm.invoke(prompt).content))
        except Exception:
            evaluation = Evaluation(sufficient=True, missing_information=[], weak_evidence=[],
                                    reason="評価パース失敗による安全フォールバック", additional_queries=[])
        evaluation.sufficient = evaluation.sufficient and not (
            evaluation.missing_information or evaluation.weak_evidence or evaluation.additional_queries
        )
        return {"evaluation": evaluation}

    def should_continue(state: State):
        evaluation = state.get("evaluation")
        if evaluation is None or evaluation.sufficient:
            return "analyze"
        if state.get("search_round", 0) >= settings.max_search_rounds:
            return "analyze"
        if state.get("search_query_count", 0) >= settings.max_search_queries:
            return "analyze"
        return "additional_search"

    def additional_search(state: State):
        evaluation = state.get("evaluation")
        next_round = state.get("search_round", 0) + 1
        if evaluation is None:
            return {"results": [], "search_query_count": 0, "search_round": next_round, "searched_queries": []}
        remaining = max(settings.max_search_queries - state.get("search_query_count", 0), 0)
        searched = {q.casefold() for q in state.get("searched_queries", [])}
        queries = [q for q in _unique_queries(evaluation.additional_queries) if q.casefold() not in searched][:remaining]
        existing_urls = {_canonical_url(item.url) for item in state.get("results", []) if item.url}
        delta: list[SearchResult] = []
        for query in queries:
            try:
                response = search.text(query, max_results=settings.results_per_query)
                fresh: list[SearchResult] = []
                for item in response:
                    result = PlannedSearchResult(
                        aspect="追加調査",
                        query=query,
                        title=item.get("title", ""),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
                    )
                    key = _canonical_url(result.url)
                    if key and key not in existing_urls:
                        fresh.append(result)
                        existing_urls.add(key)
                top = rerank_results(fresh, reranker, settings.reranked_results_per_query)
                if fetcher:
                    top = populate_content(top, fetcher, timeout=settings.fetch_timeout,
                                           max_length=settings.max_content_length,
                                           fallback_fetcher=fallback_fetcher)
                delta.extend(top)
            except Exception as error:
                print(f"追加検索失敗 ({query}): {error}")
        return {"results": delta, "search_query_count": len(queries),
                "search_round": next_round, "searched_queries": queries}

    def analyze(state: State):
        evidence = _format_evidence(deduplicate_results(state.get("results", [])), state.get("plan"))
        response = llm.invoke(f"""以下のWeb検索結果および取得したWebページ本文だけを根拠に質問へ回答してください。
質問: {state['query']}

収集された情報源:
{evidence}

重要な主張には、根拠となる情報源のURLを示してください。URLは上記の情報源に存在するものだけを使用してください。確認できないことは断定せず、URLを捏造しないでください。
""")
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
    builder.add_conditional_edges("evaluate", should_continue,
                                  {"analyze": "analyze", "additional_search": "additional_search"})
    builder.add_edge("additional_search", "evaluate")
    builder.add_edge("analyze", END)
    return builder.compile()
