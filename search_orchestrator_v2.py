"""Improved workflow controller for SearchOrcestrator."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

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


class State(TypedDict):
    query: str
    plan: SearchPlan | None
    task: SearchTask | None
    results: Annotated[list[SearchResult], operator.add]
    evaluation: Evaluation | None
    summary: str
    search_query_count: Annotated[int, operator.add]
    search_round: int


def _make_result(task: SearchTask, item: dict) -> SearchResult | None:
    url = item.get("href", "")
    if not url:
        return None
    return SearchResult(
        query=task.query,
        title=item.get("title", ""),
        url=url,
        snippet=item.get("body", ""),
    )


def _format_evidence(results: list[SearchResult], plan: SearchPlan | None) -> str:
    task_by_query = {task.query: task for task in plan.tasks} if plan else {}
    parts = []
    for i, item in enumerate(results, 1):
        task = task_by_query.get(item.query)
        aspect = task.aspect if task else "追加調査"
        reason = task.reason if task else "評価で不足した情報を補完"
        score = f" [関連度スコア: {item.relevance_score:.3f}]" if item.relevance_score is not None else ""
        text = (
            f"--- 検索結果 #{i} ---\n"
            f"【調査観点】: {aspect}\n"
            f"【検索理由】: {reason}\n"
            f"【タイトル】: {item.title}{score}\n"
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
    """Build a bounded workflow with separate initial and total query budgets."""
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
        return {"plan": plan}

    def dispatch_searches(state: State):
        tasks = state["plan"].tasks if state["plan"] else []
        # max_search_queries is the total budget; cap the initial fan-out so an
        # evaluation round can still perform additional searches.
        initial_limit = min(settings.max_search_queries, 4)
        return [Send("search", {"task": task}) for task in tasks[:initial_limit]]

    def search_task(state: State):
        task = state.get("task")
        if task is None:
            return {"results": [], "search_query_count": 0}
        try:
            response = search.text(task.query, max_results=settings.results_per_query)
            fresh: list[SearchResult] = []
            seen: set[str] = set()
            for item in response:
                result = _make_result(task, item)
                if result and result.url not in seen:
                    fresh.append(result)
                    seen.add(result.url)
            top = rerank_results(fresh, reranker, settings.reranked_results_per_query)
            if fetcher:
                top = populate_content(
                    top,
                    fetcher,
                    timeout=settings.fetch_timeout,
                    max_length=settings.max_content_length,
                    fallback_fetcher=fallback_fetcher,
                )
            return {"results": top, "search_query_count": 1}
        except Exception as error:
            print(f"検索失敗 ({task.query}): {error}")
            return {"results": [], "search_query_count": 1}

    def evaluate(state: State):
        evidence = _format_evidence(deduplicate_results(state.get("results", [])), state.get("plan"))
        prompt = f"""あなたはリサーチ品質評価担当です。現在日: {date.today().isoformat()}
質問: {state['query']}
調査計画: {state['plan']}
収集した検索結果およびWebページ本文:
{evidence}

主要観点、根拠の強さ、情報源の信頼性と新しさを評価してください。
不足がある場合だけ追加検索クエリを提示してください。JSONのみ出力してください。
{{"sufficient":false,"missing_information":[],"weak_evidence":[],"reason":"","additional_queries":[]}}
"""
        try:
            evaluation = Evaluation.model_validate(parse_json_from_response(llm.invoke(prompt).content))
        except Exception:
            evaluation = Evaluation(
                sufficient=True,
                missing_information=[],
                weak_evidence=[],
                reason="評価パース失敗による安全フォールバック",
                additional_queries=[],
            )
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
            return {"results": [], "search_query_count": 0, "search_round": next_round}
        remaining = max(settings.max_search_queries - state.get("search_query_count", 0), 0)
        queries = [q.strip() for q in evaluation.additional_queries if q.strip()][:remaining]
        existing_urls = {item.url for item in state.get("results", []) if item.url}
        delta: list[SearchResult] = []
        for query in queries:
            try:
                response = search.text(query, max_results=settings.results_per_query)
                fresh: list[SearchResult] = []
                for item in response:
                    result = SearchResult(
                        query=query,
                        title=item.get("title", ""),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
                    )
                    if result.url and result.url not in existing_urls:
                        fresh.append(result)
                        existing_urls.add(result.url)
                top = rerank_results(fresh, reranker, settings.reranked_results_per_query)
                if fetcher:
                    top = populate_content(
                        top,
                        fetcher,
                        timeout=settings.fetch_timeout,
                        max_length=settings.max_content_length,
                        fallback_fetcher=fallback_fetcher,
                    )
                delta.extend(top)
            except Exception as error:
                print(f"追加検索失敗 ({query}): {error}")
        return {
            "results": delta,
            "search_query_count": len(queries),
            "search_round": next_round,
        }

    def analyze(state: State):
        evidence = _format_evidence(deduplicate_results(state.get("results", [])), state.get("plan"))
        response = llm.invoke(f"""以下のWeb検索結果および取得したWebページ本文だけを根拠に質問へ回答してください。
質問: {state['query']}

収集された情報源:
{evidence}

重要な主張には、根拠となる情報源のURLを示してください。確認できないことは断定せず、URLを捏造しないでください。
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
    builder.add_conditional_edges(
        "evaluate",
        should_continue,
        {"analyze": "analyze", "additional_search": "additional_search"},
    )
    builder.add_edge("additional_search", "evaluate")
    builder.add_edge("analyze", END)
    return builder.compile()
