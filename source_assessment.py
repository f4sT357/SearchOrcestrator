"""Page-level evidence assessments independent of domain allow-lists.

The workflow owns model calls; this module owns the durable assessment shape,
content-aware cache, and deterministic filtering used by later workflow nodes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field


DEFAULT_MIN_CREDIBILITY_FOR_EVIDENCE = 0.55


class SourceAssessment(BaseModel):
    """Assessment of whether one fetched page may support a final answer."""

    url: str
    credibility: float = Field(ge=0.0, le=1.0)
    use_as_evidence: bool
    reason: str = ""
    support_scope: str = ""


def content_fingerprint(content: str | None) -> str | None:
    """Return a stable fingerprint only when full page content was retrieved."""
    if not content:
        return None
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SourceAssessmentCache:
    """Local cache keyed by URL and invalidated when fetched content changes."""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled

    @staticmethod
    def _filename(url: str) -> str:
        return f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"

    def get(self, url: str, content: str | None) -> SourceAssessment | None:
        fingerprint = content_fingerprint(content)
        if not self.enabled or fingerprint is None:
            return None
        path = self.directory / self._filename(url)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("content_fingerprint") != fingerprint:
                return None
            return SourceAssessment.model_validate(payload["assessment"])
        except (OSError, TypeError, ValueError, KeyError):
            return None

    def put(self, assessment: SourceAssessment, content: str | None) -> None:
        fingerprint = content_fingerprint(content)
        if not self.enabled or fingerprint is None:
            return
        path = self.directory / self._filename(assessment.url)
        payload = {
            "version": 1,
            "content_fingerprint": fingerprint,
            "assessment": assessment.model_dump(mode="json"),
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return


def assessment_by_url(assessments: Iterable[SourceAssessment]) -> dict[str, SourceAssessment]:
    """Keep the last assessment for each URL so fresh decisions override cached ones."""
    return {assessment.url: assessment for assessment in assessments if assessment.url}


def merge_assessments(
    cached: Iterable[SourceAssessment],
    fresh: Iterable[SourceAssessment],
    *,
    allowed_urls: Iterable[str],
) -> list[SourceAssessment]:
    """Merge cached and new assessments, ignoring model output for unknown URLs."""
    allowed = set(allowed_urls)
    merged = assessment_by_url(assessment for assessment in cached if assessment.url in allowed)
    merged.update(assessment_by_url(assessment for assessment in fresh if assessment.url in allowed))
    return list(merged.values())


def accepted_evidence_urls(
    assessments: Iterable[SourceAssessment],
    *,
    min_credibility: float = DEFAULT_MIN_CREDIBILITY_FOR_EVIDENCE,
) -> set[str]:
    """Return URLs explicitly accepted at or above the configured threshold."""
    return {
        assessment.url
        for assessment in assessments
        if assessment.use_as_evidence and assessment.credibility >= min_credibility
    }
