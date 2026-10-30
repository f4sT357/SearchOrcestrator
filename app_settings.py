"""Application configuration loading and validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from source_assessment import DEFAULT_MIN_CREDIBILITY_FOR_EVIDENCE
from source_scoring import normalize_official_domains


@dataclass(frozen=True)
class Settings:
    """Validated configuration shared by CLI, GUI, and workflow code."""

    model: str = ""
    lmstudio_context_length: int = 65536
    lmstudio_auto_load_model: bool = True
    base_url: str = "http://localhost:1234/v1"
    api_key: str = field(default="lm-studio", repr=False)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    max_concurrency: int = 2
    max_search_queries: int = 13
    max_search_rounds: int = 2
    results_per_query: int = 8
    reranked_results_per_query: int = 2
    content_candidate_results_per_query: int = 3
    relevance_weight: float = 0.50
    trust_weight: float = 0.30
    freshness_weight: float = 0.20
    fetch_web_content: bool = True
    max_content_length: int = 3000
    fetch_timeout: float = 8.0
    content_fetcher: str = "builtin"
    firecrawl_api_url: str = "http://localhost:3002"
    firecrawl_api_key: str = field(default="", repr=False)
    remote_fetchers_network_isolated: bool = False
    use_fallback_fetcher: bool = False
    jina_api_url: str = "https://r.jina.ai/"
    official_domains: tuple[str, ...] = ()
    trusted_domains: tuple[str, ...] = ()
    use_llm_board_summary: bool = False
    memo_keeper_enabled: bool = True
    memo_keeper_max_chars: int = 1200
    memo_keeper_max_output_tokens: int = 300
    planner_max_output_tokens: int = 700
    evaluation_max_output_tokens: int = 900
    final_answer_max_output_tokens: int = 1200
    llm_prompt_prefix: str = ""
    llm_prompt_suffix: str = ""
    llm_extra_body: dict[str, Any] = field(default_factory=dict)
    web_cache_enabled: bool = True
    web_cache_ttl_seconds: int = 21600
    web_cache_dir: str = ".cache/web-content"
    web_host_min_interval_seconds: float = 0.5
    web_max_retry_after_seconds: float = 30.0
    source_assessment_cache_enabled: bool = True
    source_assessment_cache_dir: str = ".cache/source-assessments"
    source_assessment_min_credibility: float = DEFAULT_MIN_CREDIBILITY_FOR_EVIDENCE

    @property
    def protected_secrets(self) -> tuple[str, ...]:
        # LM Studio's documented placeholder is not an authentication secret.
        return (self.api_key if self.api_key != "lm-studio" else "", self.firecrawl_api_key)

    def __post_init__(self) -> None:
        object.__setattr__(self, "official_domains", normalize_official_domains(self.official_domains))
        object.__setattr__(self, "trusted_domains", normalize_official_domains(self.trusted_domains))
        if self.lmstudio_context_length < 1024:
            raise ValueError("lmstudio_context_length must be at least 1024")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.max_search_queries < 1 or self.max_search_rounds < 0:
            raise ValueError("max_search_queries must be at least 1 and max_search_rounds cannot be negative")
        result_limits = {
            "results_per_query": self.results_per_query,
            "reranked_results_per_query": self.reranked_results_per_query,
            "content_candidate_results_per_query": self.content_candidate_results_per_query,
        }
        for name, value in result_limits.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_content_length < 1:
            raise ValueError("max_content_length must be at least 1")
        if not math.isfinite(self.fetch_timeout) or self.fetch_timeout <= 0:
            raise ValueError("fetch_timeout must be greater than 0")
        if self.memo_keeper_max_chars < 400:
            raise ValueError("memo_keeper_max_chars must be at least 400")
        if self.memo_keeper_max_output_tokens < 32:
            raise ValueError("memo_keeper_max_output_tokens must be at least 32")
        if any(limit < 64 for limit in (
            self.planner_max_output_tokens,
            self.evaluation_max_output_tokens,
            self.final_answer_max_output_tokens,
        )):
            raise ValueError("LLM output-token limits must be at least 64")
        if not isinstance(self.llm_extra_body, dict):
            raise ValueError("llm_extra_body must be a JSON object")
        object.__setattr__(self, "llm_extra_body", dict(self.llm_extra_body))
        if self.web_cache_ttl_seconds < 0:
            raise ValueError("web_cache_ttl_seconds cannot be negative")
        if any(not math.isfinite(value) or value < 0 for value in (self.web_host_min_interval_seconds, self.web_max_retry_after_seconds)):
            raise ValueError("web fetch timing settings cannot be negative")
        if not math.isfinite(self.source_assessment_min_credibility) or not 0 <= self.source_assessment_min_credibility <= 1:
            raise ValueError("source_assessment_min_credibility must be between 0 and 1")
        if self.content_fetcher not in {"builtin", "firecrawl"}:
            raise ValueError("content_fetcher must be 'builtin' or 'firecrawl'")
        weights = (self.relevance_weight, self.trust_weight, self.freshness_weight)
        if any(not math.isfinite(weight) or weight < 0 or weight > 1 for weight in weights):
            raise ValueError("Scoring weights must be between 0 and 1")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Scoring weights must sum to 1")

    @classmethod
    def from_environment(cls, config_path: str | Path | None = None) -> "Settings":
        """Load shared defaults, local overrides, then supported environment overrides."""
        shared_path = Path(config_path) if config_path else Path(__file__).with_name("config.json")
        config_paths = [shared_path]
        if config_path is None:
            config_paths.append(shared_path.with_name("config.local.json"))

        values: dict[str, Any] = {}
        allowed = set(cls.__dataclass_fields__) - {"firecrawl_api_key"}
        for path in config_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8-sig") as config_file:
                    loaded = json.load(config_file)
                if not isinstance(loaded, dict):
                    raise ValueError("The top level of the settings file must be a JSON object")
                values.update({key: value for key, value in loaded.items() if key in allowed})
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unable to read settings file {path}: {exc}") from exc
        for key, env_name in (
            ("model", "SEARCH_MODEL"),
            ("base_url", "SEARCH_BASE_URL"),
            ("api_key", "SEARCH_API_KEY"),
            ("firecrawl_api_url", "FIRECRAWL_API_URL"),
            ("firecrawl_api_key", "FIRECRAWL_API_KEY"),
        ):
            if env_name in os.environ:
                values[key] = os.environ[env_name]
        for key, env_name in (
            ("relevance_weight", "SEARCH_WEIGHT_RELEVANCE"),
            ("trust_weight", "SEARCH_WEIGHT_TRUST"),
            ("freshness_weight", "SEARCH_WEIGHT_FRESHNESS"),
        ):
            if env_name in os.environ:
                values[key] = float(os.environ[env_name])
        if "SEARCH_OFFICIAL_DOMAINS" in os.environ:
            values["official_domains"] = tuple(
                domain.strip() for domain in os.environ["SEARCH_OFFICIAL_DOMAINS"].split(",") if domain.strip()
            )
        if "SEARCH_TRUSTED_DOMAINS" in os.environ:
            values["trusted_domains"] = tuple(
                domain.strip() for domain in os.environ["SEARCH_TRUSTED_DOMAINS"].split(",") if domain.strip()
            )
        return cls(**values)
