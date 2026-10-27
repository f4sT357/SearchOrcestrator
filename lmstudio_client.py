"""Small client for LM Studio's native model-management API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class LMStudioModelLoadError(RuntimeError):
    """Raised when LM Studio cannot provide the requested loaded model."""


@dataclass(frozen=True)
class LoadedModel:
    """Verified LM Studio model instance used by the research workflow."""

    model_key: str
    instance_id: str
    context_length: int
    reused_existing: bool


def native_api_base_url(openai_base_url: str) -> str:
    """Translate an OpenAI-compatible base URL to LM Studio's native API root."""
    parsed = urlsplit(openai_base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LM Studio base URL must be an absolute HTTP or HTTPS URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _models_url(openai_base_url: str) -> str:
    return f"{native_api_base_url(openai_base_url)}/api/v1/models"


def _read_models(client: httpx.Client, openai_base_url: str) -> list[dict[str, Any]]:
    response = client.get(_models_url(openai_base_url))
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(models, list):
        raise LMStudioModelLoadError("LM Studio returned an invalid model list")
    return [model for model in models if isinstance(model, dict)]


def fetch_lmstudio_model_keys(
    openai_base_url: str,
    api_key: str = "",
    *,
    timeout: float = 3.0,
) -> list[str]:
    """Return downloaded LLM keys, including models not currently loaded."""
    try:
        with httpx.Client(headers=_headers(api_key), timeout=timeout) as client:
            models = _read_models(client, openai_base_url)
    except (httpx.HTTPError, ValueError, LMStudioModelLoadError):
        return []
    return sorted({
        str(model["key"])
        for model in models
        if model.get("type") == "llm" and isinstance(model.get("key"), str)
    })


def _selected_model(models: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    for model in models:
        if model.get("key") == model_key:
            return model
    raise LMStudioModelLoadError(f"LM Studio does not know the selected model: {model_key}")


def _loaded_instance_with_context(model: dict[str, Any], context_length: int) -> str | None:
    instances = model.get("loaded_instances", [])
    if not isinstance(instances, list):
        return None
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        config = instance.get("config", {})
        if isinstance(config, dict) and config.get("context_length") == context_length:
            instance_id = instance.get("id")
            if isinstance(instance_id, str) and instance_id:
                return instance_id
    return None


def ensure_lmstudio_model_loaded(
    openai_base_url: str,
    model_key: str,
    context_length: int,
    api_key: str = "",
    *,
    timeout: float = 90.0,
    client: httpx.Client | None = None,
) -> LoadedModel:
    """Load one selected model with a verified context window.

    Existing instances of other models are intentionally preserved. Only stale
    instances of the selected model are unloaded before the replacement load.
    """
    if not model_key.strip():
        raise ValueError("LM Studio model key is required")
    if context_length < 1024:
        raise ValueError("LM Studio context length must be at least 1024")

    owns_client = client is None
    active_client = client or httpx.Client(headers=_headers(api_key), timeout=timeout)
    try:
        model = _selected_model(_read_models(active_client, openai_base_url), model_key)
        existing_instance = _loaded_instance_with_context(model, context_length)
        if existing_instance:
            return LoadedModel(model_key, existing_instance, context_length, reused_existing=True)

        instances = model.get("loaded_instances", [])
        if isinstance(instances, list):
            unload_url = f"{native_api_base_url(openai_base_url)}/api/v1/models/unload"
            for instance in instances:
                if not isinstance(instance, dict) or not isinstance(instance.get("id"), str):
                    continue
                response = active_client.post(unload_url, json={"instance_id": instance["id"]})
                response.raise_for_status()

        load_url = f"{native_api_base_url(openai_base_url)}/api/v1/models/load"
        response = active_client.post(load_url, json={
            "model": model_key,
            "context_length": context_length,
            "echo_load_config": True,
        })
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise LMStudioModelLoadError("LM Studio returned an invalid load response")
        instance_id = payload.get("instance_id")
        load_config = payload.get("load_config", {})
        actual_context = load_config.get("context_length") if isinstance(load_config, dict) else None
        if not isinstance(instance_id, str) or not instance_id:
            raise LMStudioModelLoadError("LM Studio did not return a model instance ID")
        if actual_context != context_length:
            raise LMStudioModelLoadError(
                f"LM Studio loaded context {actual_context!r}, expected {context_length}"
            )
        return LoadedModel(model_key, instance_id, context_length, reused_existing=False)
    except httpx.HTTPError as error:
        raise LMStudioModelLoadError(f"LM Studio model operation failed: {error}") from error
    finally:
        if owns_client:
            active_client.close()
