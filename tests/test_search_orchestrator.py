from unittest.mock import MagicMock

from search_orchestrator import (
    DefaultWebContentFetcher,
    Evaluation,
    SearchPlan,
    SearchResult,
    SearchTask,
    Settings,
    add_search_results,
    create_workflow,
    deduplicate_results,
    fetch_available_models,
    format_results_for_llm,
    populate_content,
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


def test_default_web_content_fetcher_extracts_html_cleanly() -> None:
    fetcher = DefaultWebContentFetcher()
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title><style>.ads { color: red; }</style></head>
    <body>
        <header><nav><a href="/">Home</a></nav></header>
        <main>
            <h1>Main Title</h1>
            <p>This is the first paragraph with important information.</p>
            <script>console.log("ignore me");</script>
        </main>
        <footer><p>Copyright 2026</p></footer>
    </body>
    </html>
    """
    text = fetcher.extract_text_from_html(html, max_length=1000)
    assert "Main Title" in text
    assert "This is the first paragraph with important information." in text
    assert "console.log" not in text
    assert "Copyright" not in text


def test_populate_content_updates_results() -> None:
    class MockFetcher:
        def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
            if "fail" in url:
                return ""
            return f"Full page content from {url}"

    results = [
        SearchResult(query="q", title="A", url="https://example.com/a", snippet="snippet a"),
        SearchResult(query="q", title="B", url="https://example.com/fail", snippet="snippet b"),
    ]
    updated = populate_content(results, MockFetcher())
    assert len(updated) == 2
    assert updated[0].content == "Full page content from https://example.com/a"
    assert updated[0].fetch_status == "success"
    assert updated[1].content is None
    assert updated[1].fetch_status == "empty_or_failed"


def test_format_results_for_llm_includes_content() -> None:
    results = [
        SearchResult(
            query="q", title="A", url="https://example.com/a", snippet="snip",
            content="Detailed body content here", relevance_score=0.95
        )
    ]
    formatted = format_results_for_llm(results)
    assert "【タイトル】: A [関連度スコア: 0.950]" in formatted
    assert "【URL】: https://example.com/a" in formatted
    assert "【概要/スニペット】: snip" in formatted
    assert "【Webページ本文抜粋】:\nDetailed body content here" in formatted


def test_parse_json_from_response() -> None:
    from search_orchestrator import parse_json_from_response

    # Markdown json block
    text1 = 'Some preamble\n```json\n{"key": "value"}\n```\nSome postamble'
    assert parse_json_from_response(text1) == {"key": "value"}

    # Raw braces with surrounding text
    text2 = 'Thought: let me output json: {"key": 123} Hope this helps.'
    assert parse_json_from_response(text2) == {"key": 123}

    # Direct JSON
    text3 = '{"sufficient": true, "missing_information": []}'
    assert parse_json_from_response(text3) == {"sufficient": True, "missing_information": []}


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

    class MockFetcher:
        def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
            return f"Fetched body text for {url}"

    # Mock LLM invoke return based on prompt content
    mock_llm = MagicMock()

    def llm_invoke_side_effect(prompt: str):
        if "リサーチプランナー" in prompt:
            return MagicMock(content='''```json
{
  "tasks": [
    {"aspect": "Aspect1", "query": "query1", "reason": "reason1"},
    {"aspect": "Aspect2", "query": "query2", "reason": "reason2"}
  ]
}
```''')
        elif "リサーチ品質評価担当" in prompt:
            return MagicMock(content='''```json
{
  "sufficient": true,
  "missing_information": [],
  "weak_evidence": [],
  "reason": "Sufficient info found",
  "additional_queries": []
}
```''')
        else:
            return MagicMock(content="Final summary report with detailed web evidence.")

    mock_llm.invoke.side_effect = llm_invoke_side_effect

    settings = Settings(max_search_queries=2, max_search_rounds=1, fetch_web_content=True)
    workflow = create_workflow(
        settings=settings,
        search=MockSearch(),
        reranker=MockReranker(),
        llm=mock_llm,
        fetcher=MockFetcher(),
    )

    result = workflow.invoke({
        "query": "Test Topic",
        "plan": None, "task": None, "results": [], "evaluation": None,
        "summary": "", "search_query_count": 0, "search_round": 0,
    })

    assert result["summary"] == "Final summary report with detailed web evidence."
    assert len(result["results"]) > 0
    assert any(r.content is not None for r in result["results"])
    assert result["evaluation"] is not None
    assert result["evaluation"].sufficient is True


def test_fetch_available_models_with_mock() -> None:
    from unittest.mock import patch

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "model-b"}, {"id": "model-a"}, {"id": "model-c"}]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client.get", return_value=mock_resp):
        models = fetch_available_models("http://localhost:1234/v1")
        assert models == ["model-a", "model-b", "model-c"]

    # Test error handling returns empty list
    with patch("httpx.Client.get", side_effect=Exception("Connection refused")):
        models_err = fetch_available_models("http://localhost:1234/v1")
        assert models_err == []
