from unittest.mock import MagicMock

from search_orchestrator import Settings, SearchResult
from search_orchestrator_v2 import (
    _deduplicate_results,
    _parse_model_json,
    create_workflow,
)


def test_workflow_uses_total_query_budget_and_allows_additional_round() -> None:
    class MockSearch:
        def __init__(self):
            self.queries = []

        def text(self, query: str, *, max_results: int):
            self.queries.append(query)
            return [{"href": f"https://example.com/{query}", "title": query, "body": f"evidence for {query}"}]

    class MockReranker:
        def predict(self, pairs, **kwargs):
            return [0.9 for _ in pairs]

    class MockFetcher:
        def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
            return f"content for {url}"

    llm = MagicMock()
    evaluation_calls = 0

    def invoke(prompt: str):
        nonlocal evaluation_calls
        if "リサーチプランナー" in prompt:
            return MagicMock(content='''{"tasks":[{"aspect":"A","query":"q1","reason":"r1"},{"aspect":"B","query":"q2","reason":"r2"}]}''')
        if "リサーチ品質評価担当" in prompt:
            evaluation_calls += 1
            if evaluation_calls == 1:
                return MagicMock(content='''{"sufficient":false,"missing_information":["more"],"weak_evidence":[],"reason":"need more","additional_queries":["q3","q4","q5"]}''')
            return MagicMock(content='''{"sufficient":true,"missing_information":[],"weak_evidence":[],"reason":"enough","additional_queries":[]}''')
        return MagicMock(content="final")

    llm.invoke.side_effect = invoke
    search = MockSearch()
    workflow = create_workflow(Settings(max_search_queries=4, max_search_rounds=2, fetch_web_content=True), search=search, reranker=MockReranker(), llm=llm, fetcher=MockFetcher())
    result = workflow.invoke({"query": "topic", "plan": None, "task": None, "results": [], "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0})

    assert search.queries == ["q1", "q2", "q3", "q4"]
    assert result["search_query_count"] == 4
    assert result["search_round"] == 1
    assert evaluation_calls == 2
    assert result["summary"] == "final"


def test_workflow_stops_at_round_limit() -> None:
    class MockSearch:
        def __init__(self):
            self.queries = []

        def text(self, query: str, *, max_results: int):
            self.queries.append(query)
            return [{"href": f"https://example.com/{len(self.queries)}", "title": query, "body": "body"}]

    class MockReranker:
        def predict(self, pairs, **kwargs):
            return [0.9 for _ in pairs]

    llm = MagicMock()
    eval_calls = 0

    def invoke(prompt: str):
        nonlocal eval_calls
        if "リサーチプランナー" in prompt:
            return MagicMock(content='{"tasks":[{"aspect":"A","query":"q1","reason":"r"}]}')
        if "リサーチ品質評価担当" in prompt:
            eval_calls += 1
            return MagicMock(content='''{"sufficient":false,"missing_information":["x"],"weak_evidence":[],"reason":"need more","additional_queries":["q2"]}''')
        return MagicMock(content="final")

    llm.invoke.side_effect = invoke
    search = MockSearch()
    workflow = create_workflow(Settings(max_search_queries=5, max_search_rounds=1, fetch_web_content=False), search=search, reranker=MockReranker(), llm=llm)
    result = workflow.invoke({"query": "topic", "plan": None, "task": None, "results": [], "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0})

    assert search.queries == ["q1", "q2"]
    assert result["search_round"] == 1
    assert eval_calls == 2
    assert result["summary"] == "final"


def test_workflow_suppresses_duplicate_planned_and_additional_queries() -> None:
    class MockSearch:
        def __init__(self):
            self.queries = []

        def text(self, query: str, *, max_results: int):
            self.queries.append(query)
            return [{"href": f"https://example.com/{query}", "title": query, "body": "body"}]

    class MockReranker:
        def predict(self, pairs, **kwargs):
            return [0.9 for _ in pairs]

    llm = MagicMock()
    eval_calls = 0

    def invoke(prompt: str):
        nonlocal eval_calls
        if "リサーチプランナー" in prompt:
            return MagicMock(content='''{"tasks":[{"aspect":"A","query":"q1","reason":"r1"},{"aspect":"A2","query":" Q1 ","reason":"duplicate"},{"aspect":"B","query":"q2","reason":"r2"}]}''')
        if "リサーチ品質評価担当" in prompt:
            eval_calls += 1
            if eval_calls == 1:
                return MagicMock(content='''{"sufficient":false,"missing_information":["more"],"weak_evidence":[],"reason":"need more","additional_queries":["q2"," q3 ","Q3"," ","q4"]}''')
            return MagicMock(content='''{"sufficient":true,"missing_information":[],"weak_evidence":[],"reason":"enough","additional_queries":[]}''')
        return MagicMock(content="final")

    llm.invoke.side_effect = invoke
    search = MockSearch()
    workflow = create_workflow(Settings(max_search_queries=4, max_search_rounds=2, fetch_web_content=False), search=search, reranker=MockReranker(), llm=llm)
    result = workflow.invoke({"query": "topic", "plan": None, "task": None, "results": [], "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0})

    assert search.queries == ["q1", "q2", "q3", "q4"]
    assert result["search_query_count"] == 4
    assert result["search_round"] == 1
    assert result["summary"] == "final"


def test_workflow_does_not_repeat_evaluation_when_only_duplicate_queries_are_suggested() -> None:
    class MockSearch:
        def __init__(self):
            self.queries = []

        def text(self, query: str, *, max_results: int):
            self.queries.append(query)
            return [{"href": f"https://example.com/{query}", "title": query, "body": "body"}]

    class MockReranker:
        def predict(self, pairs, **kwargs):
            return [0.9 for _ in pairs]

    llm = MagicMock()
    eval_calls = 0

    def invoke(prompt: str):
        nonlocal eval_calls
        if "リサーチプランナー" in prompt:
            return MagicMock(content='{"tasks":[{"aspect":"A","query":"q1","reason":"r1"}]}')
        if "リサーチ品質評価担当" in prompt:
            eval_calls += 1
            return MagicMock(content='''{"sufficient":false,"missing_information":["more"],"weak_evidence":[],"reason":"need more","additional_queries":["q1"," Q1 "]}''')
        return MagicMock(content="final")

    llm.invoke.side_effect = invoke
    search = MockSearch()
    workflow = create_workflow(Settings(max_search_queries=3, max_search_rounds=3, fetch_web_content=False), search=search, reranker=MockReranker(), llm=llm)
    result = workflow.invoke({"query": "topic", "plan": None, "task": None, "results": [], "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0})

    assert search.queries == ["q1"]
    assert eval_calls == 1
    assert result["summary"] == "final"


def test_evidence_dedup_uses_canonical_urls() -> None:
    results = [
        SearchResult(query="q1", title="First", url="HTTPS://Example.com/path#section", snippet="a"),
        SearchResult(query="q2", title="Duplicate", url="https://example.com/path", snippet="b"),
        SearchResult(query="q3", title="Other", url="https://example.com/other", snippet="c"),
    ]
    unique = _deduplicate_results(results)
    assert [item.title for item in unique] == ["First", "Other"]


def test_model_json_parser_handles_nested_json_after_prose() -> None:
    text = '''モデルの説明です。\n```json\n{"tasks":[{"aspect":"A","query":"q1","reason":"reason with {braces}"}]}\n```'''
    assert _parse_model_json(text)["tasks"][0]["query"] == "q1"


def test_workflow_keeps_results_when_reranker_fails() -> None:
    class MockSearch:
        def text(self, query: str, *, max_results: int):
            return [{"href": "https://example.com/result", "title": "Result", "body": "body"}]

    class FailingReranker:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("reranker unavailable")

    llm = MagicMock()

    def invoke(prompt: str):
        if "リサーチプランナー" in prompt:
            return MagicMock(content='{"tasks":[{"aspect":"A","query":"q1","reason":"r"}]}')
        if "リサーチ品質評価担当" in prompt:
            return MagicMock(content='{"sufficient":true,"missing_information":[],"weak_evidence":[],"reason":"enough","additional_queries":[]}')
        return MagicMock(content="final")

    llm.invoke.side_effect = invoke
    workflow = create_workflow(
        Settings(max_search_queries=1, max_search_rounds=1, fetch_web_content=False),
        search=MockSearch(),
        reranker=FailingReranker(),
        llm=llm,
    )
    result = workflow.invoke({"query": "topic", "plan": None, "task": None, "results": [], "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0})

    assert result["summary"] == "final"
    assert len(result["results"]) == 1
    assert result["results"][0].title == "Result"
