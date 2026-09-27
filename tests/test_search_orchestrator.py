from unittest.mock import MagicMock
from datetime import date

from search_orchestrator import (
    DefaultWebContentFetcher,
    Evaluation,
    SearchPlan,
    SearchResult,
    SearchTask,
    Settings,
    add_search_results,
    create_workflow,
    citation_issues,
    deduplicate_results,
    fetch_available_models,
    format_results_for_llm,
    populate_content,
    rerank_results,
    render_verified_sources,
    select_diverse_results,
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


def test_rerank_results_uses_configurable_trust_weight() -> None:
    class DummyReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs):
            return [0.99, 0.10]

    results = [
        SearchResult(query="q", title="High relevance", url="https://example.com/a", snippet="s"),
        SearchResult(query="q", title="Primary", url="https://docs.nvidia.com/a", snippet="s"),
    ]
    ranked = rerank_results(
        results, DummyReranker(), 2, weights=(0.0, 1.0, 0.0),
    )
    assert ranked[0].title == "Primary"


def test_final_shortlist_prefers_distinct_publisher_hosts() -> None:
    sources = [
        SearchResult(query="q", title="Host A high", url="https://www.example.com/a", snippet="", combined_score=0.9),
        SearchResult(query="q", title="Host A next", url="https://example.com/b", snippet="", combined_score=0.8),
        SearchResult(query="q", title="Host B", url="https://specialist.example.org/c", snippet="", combined_score=0.7),
    ]
    selected = select_diverse_results(sources, 2)
    assert [item.title for item in selected] == ["Host A high", "Host B"]


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


def test_default_fetcher_preserves_published_date_metadata() -> None:
    fetcher = DefaultWebContentFetcher()
    html = '''
    <html><head><meta property="article:published_time" content="2024-03-15"></head>
    <body><article><h1>Article</h1><p>Body text.</p></article></body></html>
    '''
    text = fetcher.extract_text_from_html(html, max_length=1000)
    assert "公開日: 2024-03-15" in text
    assert "Body text." in text


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


def test_populate_content_uses_page_date_to_refresh_freshness() -> None:
    class MockFetcher:
        def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
            return "公開日: 2020年1月2日\nArticle body"

    item = SearchResult(
        query="q", title="A", url="https://example.com/article", snippet="s",
        relevance_score=0.8,
    )
    updated = populate_content([item], MockFetcher())
    assert updated[0].page_date == "2020-01-02"
    assert updated[0].source_freshness_score < 0.5
    assert "ページ日付: 2020-01-02" in updated[0].to_evidence_text()


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
    assert "出典番号 [S1]" in formatted


def test_citation_validation_and_source_list_use_only_retrieved_sources() -> None:
    sources = [SearchResult(
        query="q", title="Known source", url="https://example.com/source", snippet="evidence",
    )]
    draft = "Claim [S1]. Unsupported [S8]. https://fake.example/path"

    assert citation_issues(draft, len(sources)) == ["[S8]", "本文中のURL"]
    rendered = render_verified_sources(draft, sources)
    assert "[S1]" in rendered
    assert "[S8]" not in rendered
    assert "fake.example" not in rendered
    assert "[Known source](<https://example.com/source>)" in rendered
    assert "照合できない出典番号またはURLを除去" in rendered


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

    assert result["summary"].startswith("Final summary report with detailed web evidence.")
    assert "出典番号を確認できなかった" in result["summary"]
    assert len(result["results"]) > 0
    assert any(r.content is not None for r in result["results"])
    assert result["evaluation"] is not None
    assert result["evaluation"].sufficient is True
    current_date = date.today().isoformat()
    prompts = [call.args[0] for call in mock_llm.invoke.call_args_list]
    assert len(prompts) >= 3
    assert all(current_date in prompt for prompt in prompts)
    evaluation_prompt = next(prompt for prompt in prompts if "リサーチ品質評価担当" in prompt)
    assert "二次情報であることだけを理由に証拠を弱いと判定してはいけません" in evaluation_prompt
    assert "異なる発行元ドメインの2件以上" in evaluation_prompt
    assert "転載" in evaluation_prompt
    assert "検索クエリ: query1" in evaluation_prompt


def test_settings_validate_scoring_weight_sum() -> None:
    try:
        Settings(relevance_weight=0.5, trust_weight=0.4, freshness_weight=0.2)
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("invalid scoring weights should be rejected")


def test_page_date_can_promote_a_candidate_before_final_selection() -> None:
    class MockSearch:
        def text(self, query: str, *, max_results: int):
            return [
                {"href": f"https://example.com/page-{i}", "title": f"Page {i}", "body": "same snippet"}
                for i in range(1, 4)
            ]

    class MockReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs):
            return [0.8 for _ in pairs]

    class MockFetcher:
        def fetch(self, url: str, *, timeout: float = 8.0, max_length: int = 3000) -> str:
            year = date.today().year if url.endswith("page-3") else 2020
            return f"公開日: {year}-01-01\nEvidence body"

    mock_llm = MagicMock()

    def invoke(prompt: str):
        if "リサーチプランナー" in prompt:
            return MagicMock(content='{"tasks": [{"aspect": "A", "query": "q", "reason": "r"}]}')
        if "リサーチ品質評価担当" in prompt:
            return MagicMock(content='{"sufficient": true, "missing_information": [], "weak_evidence": [], "reason": "ok", "additional_queries": []}')
        return MagicMock(content="Recent evidence supports the answer. [S1]")

    mock_llm.invoke.side_effect = invoke
    workflow = create_workflow(
        settings=Settings(max_search_queries=1, max_search_rounds=1, use_fallback_fetcher=False),
        search=MockSearch(),
        reranker=MockReranker(),
        llm=mock_llm,
        fetcher=MockFetcher(),
    )
    result = workflow.invoke({
        "query": "Test", "plan": None, "task": None, "results": [],
        "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0,
    })

    assert [item.url for item in result["results"]] == [
        "https://example.com/page-3", "https://example.com/page-1",
    ]
    assert "[S1] [Page 3](<https://example.com/page-3>)" in result["summary"]


def test_invalid_evaluation_response_is_not_treated_as_sufficient() -> None:
    class MockSearch:
        def text(self, query: str, *, max_results: int):
            return [{"href": "https://example.com/a", "title": "A", "body": "snippet"}]

    class MockReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs):
            return [0.8 for _ in pairs]

    mock_llm = MagicMock()

    def invoke(prompt: str):
        if "リサーチプランナー" in prompt:
            return MagicMock(content='{"tasks": [{"aspect": "A", "query": "q", "reason": "r"}]}')
        if "リサーチ品質評価担当" in prompt:
            return MagicMock(content="not valid JSON")
        assert "十分性を確認できていません" in prompt
        assert "現在日（この調査の基準日）" in prompt
        return MagicMock(content="調査は不完全です。")

    mock_llm.invoke.side_effect = invoke
    settings = Settings(
        max_search_queries=1,
        max_search_rounds=1,
        fetch_web_content=False,
        use_fallback_fetcher=False,
    )
    workflow = create_workflow(
        settings=settings,
        search=MockSearch(),
        reranker=MockReranker(),
        llm=mock_llm,
    )
    result = workflow.invoke({
        "query": "Test", "plan": None, "task": None, "results": [],
        "evaluation": None, "summary": "", "search_query_count": 0, "search_round": 0,
    })

    assert result["evaluation"].sufficient is False
    assert result["evaluation"].missing_information
    assert "解析できませんでした" in result["evaluation"].reason
    assert result["summary"].startswith("調査は不完全です。")


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
