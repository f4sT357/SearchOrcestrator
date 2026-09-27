from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import main as cli
from search_orchestrator import Settings


def test_cli_options_override_loaded_settings(monkeypatch) -> None:
    loaded = Settings(
        max_concurrency=4,
        max_search_queries=11,
        max_search_rounds=4,
        max_content_length=4000,
        use_fallback_fetcher=False,
    )
    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda cls: loaded))
    workflow = MagicMock()
    workflow.invoke.return_value = {"summary": "report"}
    create_workflow = MagicMock(return_value=workflow)
    monkeypatch.setattr(cli, "create_workflow", create_workflow)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    cli.main([
        "--max-concurrency", "3",
        "--max-search-queries", "8",
        "--max-search-rounds", "0",
        "--max-content-length", "7000",
        "--content-fetcher", "firecrawl",
        "--firecrawl-url", "http://localhost:3002",
        "--no-fetch",
        "--jina-fallback",
        "調査テーマ",
    ])

    settings = create_workflow.call_args.args[0]
    assert settings.max_concurrency == 3
    assert settings.max_search_queries == 8
    assert settings.max_search_rounds == 0
    assert settings.max_content_length == 7000
    assert settings.fetch_web_content is False
    assert settings.use_fallback_fetcher is True
    assert settings.content_fetcher == "firecrawl"
    assert settings.firecrawl_api_url == "http://localhost:3002"
    workflow.invoke.assert_called_once()
    assert workflow.invoke.call_args.args[0]["query"] == "調査テーマ"
    assert workflow.invoke.call_args.args[1] == {"max_concurrency": 3}
