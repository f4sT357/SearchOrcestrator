from unittest.mock import MagicMock
from datetime import date
import json
import pytest
import httpx
import app_settings
import search_orchestrator as orchestrator

from search_orchestrator import (
    DefaultWebContentFetcher,
    Evaluation,
    SearchPlan,
    SearchResult,
    SearchTask,
    ResearchWorkflowNodes,
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


def test_followup_dispatch_preserves_prior_urls() -> None:
    search = MagicMock()
    search.text.return_value = [
        {"href": "https://example.com/old", "title": "old", "body": ""},
        {"href": "https://example.com/new", "title": "new", "body": ""},
    ]
    reranker = MagicMock()
    reranker.predict.side_effect = lambda pairs, **kwargs: [0.8] * len(pairs)
    nodes = ResearchWorkflowNodes(Settings(fetch_web_content=False), search=search,
                                  reranker=reranker, llm=MagicMock())
    old = SearchResult(query="first", title="old", url="https://example.com/old", snippet="")
    send = nodes.dispatch_followups({"pending_normal_queries": ["followup"],
                                     "results": [old], "search_round": 1})[0]
    result = nodes.additional_search(send.arg)
    assert [item.url for item in result["results"]] == ["https://example.com/new"]


def test_final_answer_uses_evaluation_evidence_candidates() -> None:
    llm = MagicMock()
    llm.bind.return_value = llm
    llm.invoke.return_value.content = "Supported statement [S1]"
    nodes = ResearchWorkflowNodes(Settings(fetch_web_content=False), search=MagicMock(),
                                  reranker=MagicMock(), llm=llm)
    evidence = SearchResult(query="q", title="Evidence", url="https://example.com/evidence",
                            snippet="important evidence", relevance_score=0.8, combined_score=0.8)
    result = nodes.analyze({"query": "q", "results": [], "evidence_results": [evidence],
                            "official_results": [], "evaluation": None, "run_memo": "private memo"})
    assert "important evidence" in llm.invoke.call_args.args[0]
    assert "private memo" not in llm.invoke.call_args.args[0]
    assert "https://example.com/evidence" in result["summary"]



def test_final_answer_receives_only_source_linked_research_brief() -> None:
    llm = MagicMock()
    llm.bind.return_value = llm
    llm.invoke.return_value.content = "Supported statement [S1]"
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(), llm=llm,
    )
    evidence = SearchResult(
        query="q", title="Evidence", url="https://example.com/evidence",
        snippet="important evidence", relevance_score=0.8, combined_score=0.8,
    )
    memo = json.dumps({
        "confirmed": [
            {"source_url": evidence.url, "note": "Verified release detail"},
            {"source_url": "https://untrusted.example/fake", "note": "Invented detail"},
        ],
        "unresolved": ["Confirm the release date"],
        "source_decisions": [
            {"url": evidence.url, "decision": "included", "reason": "Relevant"},
            {"url": "https://untrusted.example/fake", "decision": "included", "reason": "Fake"},
        ],
    })

    nodes.analyze({
        "query": "q", "results": [], "evidence_results": [evidence],
        "official_results": [], "evaluation": None, "run_memo": memo,
    })

    prompt = llm.invoke.call_args.args[0]
    assert "Verified release detail" in prompt
    assert "Confirm the release date" in prompt
    assert "Invented detail" not in prompt
    assert "untrusted.example" not in prompt

def test_llm_metrics_report_usage_without_prompt_content() -> None:
    response = MagicMock(content="safe response")
    response.usage_metadata = {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
    }
    llm = MagicMock()
    llm.invoke.return_value = response
    logs: list[str] = []
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(),
        llm=llm, log=logs.append,
    )

    result = nodes.invoke_llm("test_role", "private prompt text", source_count=3)

    assert result is response
    metric = json.loads(next(item.removeprefix("[LLM_METRIC] ") for item in logs))
    assert metric["role"] == "test_role"
    assert metric["input_chars"] == 19
    assert metric["source_count"] == 3
    assert metric["output_chars"] == 13
    assert metric["input_tokens"] == 12
    assert metric["output_tokens"] == 4
    assert metric["total_tokens"] == 16
    assert metric["status"] == "completed"
    assert metric["duration_ms"] >= 0
    assert all("private prompt text" not in item for item in logs)

def test_llm_metrics_reject_empty_final_content_and_records_reasoning() -> None:
    response = MagicMock(content="")
    response.additional_kwargs = {"reasoning_content": "internal reasoning"}
    response.usage_metadata = {"output_tokens": 128}
    llm = MagicMock()
    llm.invoke.return_value = response
    logs: list[str] = []
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(),
        llm=llm, log=logs.append,
    )

    with pytest.raises(orchestrator.EmptyModelResponseError, match="no final content"):
        nodes.invoke_llm("test_role", "private prompt text")

    metric = json.loads(next(item.removeprefix("[LLM_METRIC] ") for item in logs))
    assert metric["status"] == "empty"
    assert metric["output_chars"] == 0
    assert metric["reasoning_chars"] == len("internal reasoning")
    assert metric["output_tokens"] == 128
    assert all("private prompt text" not in item for item in logs)
def test_llm_metrics_record_failures_without_prompt_content() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("provider failure")
    logs: list[str] = []
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(),
        llm=llm, log=logs.append,
    )

    with pytest.raises(RuntimeError, match="provider failure"):
        nodes.invoke_llm("failure_role", "private prompt text")

    metric = json.loads(next(item.removeprefix("[LLM_METRIC] ") for item in logs))
    assert metric["role"] == "failure_role"
    assert metric["status"] == "error"
    assert metric["error_type"] == "RuntimeError"
    assert metric["duration_ms"] >= 0
    assert all("private prompt text" not in item for item in logs)

def test_add_search_results_ignores_duplicate_urls() -> None:
    results = [SearchResult(query="old", title="Old", url="https://example.com", snippet="")]
    added = add_search_results(results, [
        {"href": "https://example.com", "title": "Duplicate", "body": ""},
        {"href": "https://other.example", "title": "New", "body": "text"},
    ], "new query")
    assert added == 1
    assert [result.url for result in results] == ["https://example.com", "https://other.example"]


def test_settings_load_config_then_environment_overrides(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model": "config-model",
        "max_concurrency": 4,
        "max_search_queries": 12,
        "max_search_rounds": 3,
        "fetch_web_content": False,
        "use_fallback_fetcher": True,
        "max_content_length": 4500,
        "content_fetcher": "firecrawl",
        "firecrawl_api_url": "http://firecrawl.local:3002",
        "firecrawl_api_key": "must-not-load-from-file",
    }), encoding="utf-8")
    for name in (
        "SEARCH_MODEL", "SEARCH_BASE_URL", "SEARCH_API_KEY",
        "SEARCH_WEIGHT_RELEVANCE", "SEARCH_WEIGHT_TRUST", "SEARCH_WEIGHT_FRESHNESS",
        "FIRECRAWL_API_URL", "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SEARCH_MODEL", "environment-model")
    monkeypatch.setenv("SEARCH_TRUSTED_DOMAINS", "env-trusted.example")
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "environment-firecrawl-key")

    settings = Settings.from_environment(config_path)

    assert settings.model == "environment-model"
    assert settings.max_concurrency == 4
    assert settings.max_search_queries == 12
    assert settings.max_search_rounds == 3
    assert settings.fetch_web_content is False
    assert settings.use_fallback_fetcher is True
    assert settings.max_content_length == 4500
    assert settings.content_fetcher == "firecrawl"
    assert settings.firecrawl_api_url == "http://localhost:3002"
    assert settings.firecrawl_api_key == "environment-firecrawl-key"



def test_settings_loads_local_config_after_shared_config_and_before_environment(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps({
        "model": "shared-model", "max_search_queries": 4,
    }), encoding="utf-8")
    (tmp_path / "config.local.json").write_text(json.dumps({
        "model": "local-model", "max_search_queries": 6, "trusted_domains": ["specialist.example"],
    }), encoding="utf-8")
    monkeypatch.setattr(app_settings, "__file__", str(tmp_path / "app_settings.py"))
    monkeypatch.delenv("SEARCH_MODEL", raising=False)

    local_settings = Settings.from_environment()

    assert local_settings.model == "local-model"
    assert local_settings.max_search_queries == 6
    assert local_settings.trusted_domains == ("specialist.example",)

    monkeypatch.setenv("SEARCH_MODEL", "environment-model")
    monkeypatch.setenv("SEARCH_TRUSTED_DOMAINS", "env-trusted.example")
    environment_settings = Settings.from_environment()

    assert environment_settings.model == "environment-model"
    assert environment_settings.trusted_domains == ("env-trusted.example",)

def test_settings_default_external_fallback_is_disabled(tmp_path, monkeypatch) -> None:
    for name in (
        "SEARCH_MODEL", "SEARCH_BASE_URL", "SEARCH_API_KEY",
        "SEARCH_WEIGHT_RELEVANCE", "SEARCH_WEIGHT_TRUST", "SEARCH_WEIGHT_FRESHNESS",
        "FIRECRAWL_API_URL", "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_environment(tmp_path / "missing.json")
    assert settings.use_fallback_fetcher is False
    assert settings.firecrawl_api_key == ""


def test_workflow_uses_configured_firecrawl_fetcher(monkeypatch) -> None:
    from unittest.mock import patch

    settings = Settings(
        content_fetcher="firecrawl",
        firecrawl_api_url="http://localhost:3300",
        firecrawl_api_key="test-key",
    )
    with patch("search_orchestrator.FirecrawlFetcher") as fetcher_class:
        create_workflow(
            settings=settings,
            search=MagicMock(),
            reranker=MagicMock(),
            llm=MagicMock(),
        )
    fetcher_class.assert_called_once_with(
        api_url="http://localhost:3300",
        api_key="test-key",
    )


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
        SearchResult(query="q", title="Registered official", url="https://news.official.example/a", snippet="s"),
    ]
    ranked = rerank_results(
        results, DummyReranker(), 2, weights=(0.0, 1.0, 0.0),
        official_domains=("official.example",),
    )
    assert ranked[0].title == "Registered official"

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
    mock_llm.bind.return_value = mock_llm

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
        elif "調査依頼の受付担当" in prompt:
            return MagicMock(content="対象: Test Topic\n目的: 調査\n観点: 概要\n条件: なし\n曖昧な点: なし")
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
        "understood_request": "",
        "plan": None,
        "task": None,
        "results": [],
        "evidence_results": [],
        "official_results": [],
        "evaluation": None,
        "run_memo": "",
        "summary": "",
        "search_query_count": 0,
        "search_round": 0,
        "searched_queries": [],
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
    assert sum(current_date in prompt for prompt in prompts) >= 3
    evaluation_prompt = next(prompt for prompt in prompts if "リサーチ品質評価担当" in prompt)
    assert "二次情報であることだけを理由に証拠を弱いと判定してはいけません" in evaluation_prompt
    assert "異なる発行元ドメインの2件以上" in evaluation_prompt
    assert "転載" in evaluation_prompt
    assert "query1" in evaluation_prompt


def test_settings_validate_scoring_weight_sum() -> None:
    try:
        Settings(relevance_weight=0.5, trust_weight=0.4, freshness_weight=0.2)
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("invalid scoring weights should be rejected")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("results_per_query", 0),
        ("reranked_results_per_query", 0),
        ("content_candidate_results_per_query", -1),
        ("fetch_timeout", 0),
        ("fetch_timeout", float("inf")),
    ],
)
def test_settings_reject_invalid_fetch_limits(field_name: str, value: float) -> None:
    with pytest.raises(ValueError, match=field_name):
        Settings(**{field_name: value})


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
    mock_llm.bind.return_value = mock_llm

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
        "query": "Test",
        "understood_request": "",
        "plan": None,
        "task": None,
        "results": [],
        "evidence_results": [],
        "official_results": [],
        "evaluation": None,
        "run_memo": "",
        "summary": "",
        "search_query_count": 0,
        "search_round": 0,
        "searched_queries": [],
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
    mock_llm.bind.return_value = mock_llm

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
        "query": "Test",
        "understood_request": "",
        "plan": None,
        "task": None,
        "results": [],
        "evidence_results": [],
        "official_results": [],
        "evaluation": None,
        "run_memo": "",
        "summary": "",
        "search_query_count": 0,
        "search_round": 0,
        "searched_queries": [],
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



def test_understand_input_uses_original_query_without_llm() -> None:
    llm = MagicMock()
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(), llm=llm,
    )

    result = nodes.understand_input({"query": "Keep this request", "run_memo": ""})

    assert result["understood_request"] == "Keep this request"
    assert "Keep this request" in result["run_memo"]
    llm.invoke.assert_not_called()


def test_update_board_uses_deterministic_summary_when_memo_keeper_is_disabled() -> None:
    llm = MagicMock()
    updates: list[str] = []
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False, memo_keeper_enabled=False), search=MagicMock(), reranker=MagicMock(),
        llm=llm, board_update=updates.append,
    )
    source = SearchResult(
        query="q", title="Source", url="https://example.com/source", snippet="evidence",
        relevance_score=0.9, combined_score=0.9,
    )
    evaluation = Evaluation(
        sufficient=True, missing_information=[], weak_evidence=[], reason="Enough evidence",
        additional_queries=[], official_additional_queries=[],
    )

    result = nodes.update_board({
        "query": "question", "results": [source], "official_results": [],
        "evaluation": evaluation, "run_memo": "",
    })

    assert "https://example.com/source" in result["run_memo"]
    assert updates == [result["run_memo"]]
    llm.invoke.assert_not_called()


def test_content_fetcher_reuses_fresh_disk_cache(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "cache-control": "max-age=120", "etag": '"v1"'},
            text="<html><body><main>cached body</main></body></html>",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = DefaultWebContentFetcher(
        client=client, cache_dir=tmp_path, cache_ttl_seconds=60, min_host_interval_seconds=0,
    )

    assert fetcher.fetch("https://example.com/page") == "cached body"
    assert fetcher.fetch("https://example.com/page") == "cached body"
    assert len(requests) == 1
    assert [path.name for path in tmp_path.iterdir()] != ["page.json"]


def test_content_fetcher_refetches_when_requested_length_grows(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "cache-control": "max-age=120", "etag": '"v1"'},
            text="0123456789",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = DefaultWebContentFetcher(
        client=client, cache_dir=tmp_path, cache_ttl_seconds=60, min_host_interval_seconds=0,
    )

    assert fetcher.fetch("https://example.com/page", max_length=5) == "01234"
    assert fetcher.fetch("https://example.com/page", max_length=10) == "0123456789"
    assert len(requests) == 2
    assert "if-none-match" not in requests[1].headers


def test_content_fetcher_revalidates_stale_cache_with_etag(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/plain", "cache-control": "max-age=0", "etag": '"v1"'},
                text="cached body",
            )
        return httpx.Response(304, headers={"cache-control": "max-age=120", "etag": '"v1"'})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = DefaultWebContentFetcher(
        client=client, cache_dir=tmp_path, cache_ttl_seconds=60, min_host_interval_seconds=0,
    )

    assert fetcher.fetch("https://example.com/page") == "cached body"
    assert fetcher.fetch("https://example.com/page") == "cached body"
    assert len(requests) == 2
    assert requests[1].headers["if-none-match"] == '"v1"'


def test_content_fetcher_does_not_store_no_store_responses(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "cache-control": "no-store"},
            text=f"response {calls}",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = DefaultWebContentFetcher(
        client=client, cache_dir=tmp_path, cache_ttl_seconds=60, min_host_interval_seconds=0,
    )

    assert fetcher.fetch("https://example.com/no-store") == "response 1"
    assert fetcher.fetch("https://example.com/no-store") == "response 2"
    assert calls == 2
    assert list(tmp_path.iterdir()) == []


def test_content_fetcher_retries_429_after_retry_after(tmp_path, monkeypatch) -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "cache-control": "no-store"},
            text="available",
        )

    monkeypatch.setattr("search_orchestrator.time.sleep", delays.append)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = DefaultWebContentFetcher(
        client=client, cache_dir=tmp_path, min_host_interval_seconds=0, max_retry_after_seconds=5,
    )

    assert fetcher.fetch("https://example.com/retry") == "available"
    assert calls == 2
    assert delays == [2.0]

def test_source_trace_records_url_decision_and_official_status() -> None:
    logs: list[str] = []
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False, official_domains=("python.org",)),
        search=MagicMock(), reranker=MagicMock(), llm=MagicMock(), log=logs.append,
    )
    item = SearchResult(
        query="q", title="Official", url="https://www.python.org/downloads/", snippet="",
        relevance_score=0.8, fetch_status="success", fetch_error="primary=http_403",
    )

    nodes.log_source_trace(
        "quality_evaluation", item, "included", "Included in quality evaluation evidence.",
        official_candidate=True,
    )

    trace = json.loads(next(line.removeprefix("[SOURCE_TRACE] ") for line in logs))
    assert trace["url"] == item.url
    assert trace["decision"] == "included"
    assert trace["official_candidate"] is True
    assert trace["registered_official"] is True
    assert trace["fetch_status"] == "success"
    assert trace["fetch_error"] == "primary=http_403"

def test_memo_keeper_updates_bounded_json_without_main_prompt_leakage() -> None:
    llm = MagicMock()
    llm.bind.return_value.invoke.return_value.content = json.dumps({
        "request": "question",
        "confirmed": [{"source_url": "https://example.com/source", "note": "Confirmed fact"}],
        "unresolved": ["Open point"],
        "source_decisions": [{
            "url": "https://example.com/source", "decision": "included", "reason": "Relevant",
        }],
        "next_action": "Answer from the selected evidence.",
    })
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False, memo_keeper_max_chars=600),
        search=MagicMock(), reranker=MagicMock(), llm=llm,
    )
    source = SearchResult(
        query="q", title="Source", url="https://example.com/source", snippet="evidence",
        relevance_score=0.9, combined_score=0.9,
    )
    evaluation = Evaluation(
        sufficient=True, missing_information=[], weak_evidence=[], reason="Enough evidence",
        additional_queries=[], official_additional_queries=[],
    )

    result = nodes.update_board({
        "query": "question", "results": [source], "official_results": [],
        "evaluation": evaluation, "run_memo": "private old memo",
    })

    memo = json.loads(result["run_memo"])
    assert memo["confirmed"] == [{"source_url": source.url, "note": "Confirmed fact"}]
    assert memo["source_decisions"][0]["url"] == source.url
    assert len(result["run_memo"]) <= 600
    memo_prompt = llm.bind.return_value.invoke.call_args.args[0]
    assert "private old memo" in memo_prompt
    assert nodes.llm.invoke.call_count == 0


def test_settings_accepts_utf8_bom_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_bytes(b"\xef\xbb\xbf{\"model\": \"bom-model\"}")

    settings = Settings.from_environment(config_path=config_path)

    assert settings.model == "bom-model"

def test_final_answer_prompt_keeps_all_selected_source_urls() -> None:
    response = MagicMock(content="Supported statement [S1]")
    llm = MagicMock()
    llm.bind.return_value = llm
    llm.invoke.return_value = response
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(), llm=llm,
    )
    sources = [
        SearchResult(
            query="q", title=f"Source {index}", url=f"https://source{index}.example/page",
            snippet="summary", content="evidence body " * 200,
            relevance_score=0.9, combined_score=1.0 - index / 100,
        )
        for index in range(1, 7)
    ]

    nodes.analyze({
        "query": "q", "results": [], "evidence_results": sources,
        "official_results": [], "evaluation": None, "run_memo": "",
    })

    prompt = llm.invoke.call_args.args[0]
