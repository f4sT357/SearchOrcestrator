"""Keep the attacker/victim probes in the normal offline regression suite."""

import base64
import json
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app_settings import Settings
from search_orchestrator import InvalidModelResponseError, ResearchWorkflowNodes
from security import redact_sensitive_text, safe_search_query, security_event
from tests.redteam_scenarios import CASES


@pytest.mark.parametrize("name", CASES)
def test_adversarial_scenario(name, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Red-team regressions cannot perform real network I/O")
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    assert CASES[name](), name


@pytest.mark.parametrize("query", [
    "LM Studio 4B model", "Python 3.12 security", "プロンプトインジェクション 攻撃例", "Base64 encoding examples",
])
def test_normal_research_with_placeholder_key_is_preserved(query):
    search = MagicMock()
    nodes = ResearchWorkflowNodes(Settings(fetch_web_content=False), llm=MagicMock(), search=search, reranker=MagicMock())
    nodes._search_text(query, max_results=1)
    search.text.assert_called_once_with(query, max_results=1)


@pytest.mark.parametrize("secret", ["A7!k", "RtKey7Q4", "abc-def_+123", "短い秘密"])
def test_known_secrets_and_common_transforms_are_masked_and_blocked(secret):
    transforms = [secret, secret.lower(), base64.b64encode(secret.encode()).decode(), secret.encode().hex(), " ".join(secret)]
    for payload in transforms:
        assert not safe_search_query("topic " + payload, secrets=(secret,))
        masked = redact_sensitive_text("Provider diagnostic: " + payload, secrets=(secret,))
        assert payload not in masked
        assert "[REDACTED]" in masked


def test_overlong_model_output_is_rejected_with_safe_metrics():
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="x" * 65537, additional_kwargs={})
    logs = []
    nodes = ResearchWorkflowNodes(Settings(fetch_web_content=False), llm=llm, search=MagicMock(), reranker=MagicMock(), log=logs.append)
    with pytest.raises(InvalidModelResponseError):
        nodes.invoke_llm("final_answer", "data")
    metric = json.loads(logs[0].removeprefix("[LLM_METRIC] "))
    assert metric["status"] == "invalid_output"
    assert "x" * 100 not in logs[0]


def test_malformed_unicode_cannot_crash_security_logging(caplog):
    security_event("malformed_unicode_probe", "\ud800")
    assert "malformed_unicode_probe" in caplog.text


@pytest.mark.parametrize("query", ["Python\ufeffsecurity", "Python\u2028security", "Python\u0085security"])
def test_format_and_line_separator_characters_cannot_hide_a_payload(query):
    assert not safe_search_query(query)


def test_non_object_assessment_cache_is_ignored(tmp_path):
    from source_assessment import SourceAssessmentCache
    cache = SourceAssessmentCache(tmp_path)
    (tmp_path / cache._filename("https://example.com")).write_text("[]", encoding="utf-8")
    assert cache.get("https://example.com", "Python security") is None


def test_punctuation_only_configured_key_is_still_protected():
    assert not safe_search_query("topic ====", secrets=("====",))
    assert "====" not in redact_sensitive_text("diagnostic ====", secrets=("====",))


def test_gui_error_signals_mask_short_secrets(monkeypatch):
    from gui.worker import ResearchWorker
    secret = "A7!k"
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic provider error: " + secret)
    monkeypatch.setattr("gui.worker.create_workflow", fail)
    worker = ResearchWorker(query="Python", settings=Settings(api_key=secret))
    logs, errors, progress = [], [], []
    worker.log_signal.connect(logs.append)
    worker.error_signal.connect(errors.append)
    worker.progress_signal.connect(lambda *args: progress.append(args))
    worker.run()
    assert errors and "[REDACTED]" in errors[0]
    assert secret not in repr((logs, errors, progress))
