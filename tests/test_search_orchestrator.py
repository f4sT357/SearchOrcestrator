from unittest.mock import MagicMock

from search_orchestrator import (
    Evaluation,
    SearchPlan,
    SearchResult,
    SearchTask,
    Settings,
    add_search_results,
    create_workflow,
    deduplicate_results,
    rerank_results,
)


def test_add_search_results_ignores_duplicate_urls() -> None:
    results = [SearchResult(query="old", title="Old", url="https://example.com", snippet="")]
    added = add_search_results(results, [
        {"href": "https://example.com", "title": "Duplicate", "body": ""},
        {"href": "https://other.example", "title": "New", "body": "text"},
    ], "new query")
    assert added == 1
    assert [result.url for result in results] == ["https://example.com", "https://other.example"]


def test_deduplicate_results_removes_empty_and_duplicate_urls() -> None:
    results = [
        SearchResult(query="q", title="A", url="https://example.com", snippet=""),
        SearchResult(query="q", title="B", url="https://example.com", snippet=""),
        SearchResult(query="q", title="C", url="", snippet=""),
    ]
    assert [result.title for result in deduplicate_results(results)] == ["A"]


def test_rerank_results_orders_and_limits() -> None:
    class DummyReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs):
            return [0.1, 0.9, 0.5]

    results = [
        SearchResult(query="q", title="Low", url="https://example.com/1", snippet="s1"),
        SearchResult(query="q", title="High", url="https://example.com/2", snippet="s2"),
        SearchResult(query="q", title="Mid", url="https://example.com/3", snippet="s3"),
    ]
    reranked = rerank_results(results, DummyReranker(), top_k=2)
    assert len(reranked) == 2
    assert reranked[0].title == "High"
    assert reranked[0].relevance_score == 0.9
    assert reranked[1].title == "Mid"
    assert reranked[1].relevance_score == 0.5


def test_workflow_end_to_end_with_mocks() -> None:
    class MockSearch:
        def text(self, query: str, *, max_results: int):
            return [
                {"href": f"https://example.com/{query}/1", "title": f"{query} 1", "body": "Body 1"},
                {"href": f"https://example.com/{query}/2", "title": f"{query} 2", "body": "Body 2"},
            ]

    class MockReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs):
            return [0.8 for _ in pairs]

    # Mock LLM
    mock_llm = MagicMock()
    mock_planner = MagicMock()
    mock_planner.invoke.return_value = SearchPlan(
        tasks=[
            SearchTask(aspect="Aspect1", query="query1", reason="reason1"),
            SearchTask(aspect="Aspect2", query="query2", reason="reason2"),
        ]
    )
    mock_evaluator = MagicMock()
    mock_evaluator.invoke.return_value = Evaluation(
        sufficient=True,
        missing_information=[],
        weak_evidence=[],
        reason="Sufficient info found",
        additional_queries=[],
    )

    def with_structured_output_side_effect(schema):
        if schema == SearchPlan:
            return mock_planner
        if schema == Evaluation:
            return mock_evaluator
        return MagicMock()

    mock_llm.with_structured_output.side_effect = with_structured_output_side_effect
    mock_llm.invoke.return_value = MagicMock(content="Final summary report.")

    settings = Settings(max_search_queries=2, max_search_rounds=1)
    workflow = create_workflow(
        settings=settings,
        search=MockSearch(),
        reranker=MockReranker(),
        llm=mock_llm,
    )

    result = workflow.invoke({
        "query": "Test Topic",
        "plan": None, "task": None, "results": [], "evaluation": None,
        "summary": "", "search_query_count": 0, "search_round": 0,
    })

    assert result["summary"] == "Final summary report."
    assert len(result["results"]) > 0
    assert result["evaluation"] is not None
    assert result["evaluation"].sufficient is True
