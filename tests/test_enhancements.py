"""Unit tests for enhancements:
- XML declaration handling & script tag stripping in DefaultWebContentFetcher
- Meta charset detection in DefaultWebContentFetcher
- Context-aware _keyword_followup_query
- Fallback parsing in update_board for local LLMs
- CLI progress and verbosity flags in main.py
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import pytest

import main as cli
from search_orchestrator import (
    DefaultWebContentFetcher,
    Settings,
    _keyword_followup_query,
    create_workflow,
)


def test_default_fetcher_handles_xml_declaration() -> None:
    fetcher = DefaultWebContentFetcher()
    html_with_xml = '<?xml version="1.0" encoding="utf-8"?><html><body><main><p>Valid content with XML header</p></main></body></html>'
    extracted = fetcher.extract_text_from_html(html_with_xml)
    assert "Valid content with XML header" in extracted


def test_default_fetcher_strips_scripts_and_styles_in_fallback() -> None:
    fetcher = DefaultWebContentFetcher()
    malformed_html = """
    <html>
      <head>
        <script>console.log("SECRET SCRIPT CODE");</script>
        <style>body { color: red; }</style>
      </head>
      <body>
        <main>
          <p>Main visible content</p>
        </main>
      </body>
    </html>
    """
    # Mock lxml.html.fromstring to simulate a parsing error and trigger regex fallback
    with patch("lxml.html.fromstring", side_effect=Exception("Simulated parser error")):
        extracted = fetcher.extract_text_from_html(malformed_html)
    assert "Main visible content" in extracted
    assert "SECRET SCRIPT CODE" not in extracted
    assert "color: red" not in extracted


def test_default_fetcher_detects_meta_charset() -> None:
    fetcher = DefaultWebContentFetcher()
    japanese_text = "これは日本語のテストページです。"
    html_bytes = f'<html><head><meta charset="utf-8"></head><body><p>{japanese_text}</p></body></html>'.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.encoding = "iso-8859-1"  # simulated incorrect HTTP charset header
    mock_resp.content = html_bytes
    mock_resp.text = html_bytes.decode("iso-8859-1")
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_client:
        instance = mock_client.return_value.__enter__.return_value
        instance.get.return_value = mock_resp
        result = fetcher.fetch("https://example.com/jp")

    assert japanese_text in result


def test_keyword_followup_query_is_context_aware() -> None:
    # 1. Non-character generic topic: should generate generic official announcement fallbacks
    generic_query = "AIモデル 最新動向"
    followup = _keyword_followup_query(generic_query, [], [])
    assert followup is not None
    assert any(term in followup for term in ("公式 発表", "公式 お知らせ", "公式サイト", "公式 文書"))
    assert "キャラクター" not in followup

    # 2. Character-related topic: should generate character profile fallbacks
    char_query = "ゲーム 登場人物 キャラクタースキル"
    char_followup = _keyword_followup_query(char_query, [], [])
    assert char_followup is not None
    assert any(term in char_followup for term in ("キャラクター", "告知"))


def test_update_board_fallback_when_tool_call_is_text_json() -> None:
    # Test update_board node fallback when LLM emits text JSON instead of OpenAI tool_calls
    mock_llm = MagicMock()
    # Response without tool_calls attribute / empty tool_calls, but content contains JSON
    mock_resp = MagicMock()
    mock_resp.tool_calls = []
    mock_resp.content = '{"request":"A社 新製品","confirmed":[],"unresolved":[],"source_decisions":[],"next_action":"A社の新製品発表について回答作成"}'
    mock_llm.bind.return_value.invoke.return_value = mock_resp

    board_updates = []
    settings = Settings(max_search_rounds=1, memo_keeper_enabled=True)
    workflow = create_workflow(
        settings=settings,
        llm=mock_llm,
        reranker=MagicMock(),
        board_update=lambda memo: board_updates.append(memo),
    )

    # Invoke board_update node directly or through minimal graph state
    result = workflow.nodes["board_update"].invoke({
        "query": "A社 新製品",
        "plan": None,
        "run_memo": "",
        "results": [],
        "evidence_results": [],
        "official_results": [],
        "evaluation": None,
    })

    assert "A社の新製品発表" in result.get("run_memo", "")
    assert len(board_updates) > 0
    assert "A社の新製品発表" in board_updates[-1]

    # Verify that run_memo is also returned when board_update callback is None
    workflow_no_cb = create_workflow(
        settings=settings,
        llm=mock_llm,
        reranker=MagicMock(),
        board_update=None,
    )
    result_no_cb = workflow_no_cb.nodes["board_update"].invoke({
        "query": "A社 新製品",
        "plan": None,
        "run_memo": "",
        "results": [],
        "evidence_results": [],
        "official_results": [],
        "evaluation": None,
    })
    assert "A社の新製品発表" in result_no_cb.get("run_memo", "")


def test_cli_verbosity_and_quiet_options(monkeypatch) -> None:
    loaded = Settings(model="test-model")
    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda cls: loaded))
    workflow = MagicMock()
    workflow.invoke.return_value = {"summary": "Completed report"}
    create_workflow_mock = MagicMock(return_value=workflow)
    monkeypatch.setattr(cli, "create_workflow", create_workflow_mock)

    captured_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)

    # 1. Run with --quiet: should suppress progress banners
    cli.main(["--quiet", "テストテーマ"])
    out_quiet = captured_out.getvalue()
    assert "調査を開始します" not in out_quiet
    assert "Completed report" in out_quiet

    # 2. Run without --quiet: should output start message and invoke with progress callback
    captured_out.seek(0)
    captured_out.truncate(0)
    cli.main(["テストテーマ"])
    out_normal = captured_out.getvalue()
    assert "調査を開始します: 「テストテーマ」" in out_normal
    assert "Completed report" in out_normal

    # Verify callbacks were provided to create_workflow
    kwargs = create_workflow_mock.call_args.kwargs
    progress_fn = kwargs.get("progress")
    assert progress_fn is not None
    assert kwargs.get("log") is not None
    assert kwargs.get("board_update") is not None

    # Test that cli_progress only outputs on started, not on completed
    captured_out.seek(0)
    captured_out.truncate(0)
    progress_fn("understand_input", "started", "依頼整理中")
    assert "[依頼整理] 進行中 (依頼整理中)" in captured_out.getvalue()

    captured_out.seek(0)
    captured_out.truncate(0)
    progress_fn("understand_input", "completed", "依頼整理完了")
    assert captured_out.getvalue() == ""
