import json

import httpx
import pytest

from lmstudio_client import (
    LMStudioModelLoadError,
    ensure_lmstudio_model_loaded,
    native_api_base_url,
)


def test_native_api_base_url_removes_openai_v1_path() -> None:
    assert native_api_base_url("http://localhost:1234/v1") == "http://localhost:1234"


def test_ensure_model_reuses_matching_loaded_instance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json={"models": [{
            "key": "official/qwen3.5-4b",
            "type": "llm",
            "loaded_instances": [{
                "id": "qwen-instance",
                "config": {"context_length": 65536},
            }],
        }]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        loaded = ensure_lmstudio_model_loaded(
            "http://localhost:1234/v1", "official/qwen3.5-4b", 65536, client=client,
        )

    assert loaded.instance_id == "qwen-instance"
    assert loaded.reused_existing is True


def test_ensure_model_reloads_only_selected_stale_instance() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{
                "key": "official/qwen3.5-4b",
                "type": "llm",
                "loaded_instances": [{
                    "id": "stale-instance",
                    "config": {"context_length": 13056},
                }],
            }]})
        if request.url.path.endswith("/unload"):
            assert json.loads(request.content) == {"instance_id": "stale-instance"}
            return httpx.Response(200, json={"instance_id": "stale-instance"})
        assert request.url.path.endswith("/load")
        assert json.loads(request.content)["context_length"] == 65536
        return httpx.Response(200, json={
            "instance_id": "new-instance",
            "load_config": {"context_length": 65536},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        loaded = ensure_lmstudio_model_loaded(
            "http://localhost:1234/v1", "official/qwen3.5-4b", 65536, client=client,
        )

    assert loaded.instance_id == "new-instance"
    assert loaded.reused_existing is False
    assert calls == [
        ("GET", "/api/v1/models"),
        ("POST", "/api/v1/models/unload"),
        ("POST", "/api/v1/models/load"),
    ]


def test_ensure_model_rejects_unverified_context_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{
                "key": "official/qwen3.5-4b", "type": "llm", "loaded_instances": [],
            }]})
        return httpx.Response(200, json={
            "instance_id": "new-instance",
            "load_config": {"context_length": 8192},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LMStudioModelLoadError, match="expected 65536"):
            ensure_lmstudio_model_loaded(
                "http://localhost:1234/v1", "official/qwen3.5-4b", 65536, client=client,
            )
