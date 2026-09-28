"""Source scoring utilities for SearchOrchestrator.

Three complementary dimensions are measured:
  1. Source type   – is this a primary (vendor/official) source?
  2. Domain trust  – how authoritative is the domain?
  3. Freshness     – how recent is the content (inferred from URL)?

The final composite score (0-1) is used to boost or penalise results
*after* the cross-encoder relevance score is produced, so that highly
relevant but low-authority sources cannot dominate the final ranking.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, datetime
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain databases
# ---------------------------------------------------------------------------

# Primary sources – official organization, product, research, and standards domains
PRIMARY_DOMAINS: frozenset[str] = frozenset({
    # NVIDIA
    "nvidia.com",
    "developer.nvidia.com",
    "docs.nvidia.com",
    "investor.nvidia.com",
    # AMD
    "amd.com",
    "developer.amd.com",
    "rocm.docs.amd.com",
    "investor.amd.com",
    # Intel
    "intel.com",
    "ark.intel.com",
    # Hyperscaler compute / AI
    "cloud.google.com",
    "ai.google.dev",
    "aws.amazon.com",
    "azure.microsoft.com",
    "learn.microsoft.com",
    # Academic
    "arxiv.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "nature.com",
    "science.org",
    # Standards bodies & consortia
    "opencompute.org",
    "isa-alliance.org",
    # Patent databases
    "patents.google.com",
    "j-platpat.inpit.go.jp",
})

# Secondary sources – credible industry news / analyst firms
SECONDARY_DOMAINS: frozenset[str] = frozenset({
    "techcrunch.com",
    "theverge.com",
    "venturebeat.com",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "businesswire.com",
    "prnewswire.com",
    "anandtech.com",
    "tomshardware.com",
    "servethehome.com",
    "semianalysis.com",
    "nextplatform.com",
    "hpcwire.com",
    "globaldata.com",
    "marketsandmarkets.com",
    "idc.com",
    "gartner.com",
})

# Trust score table (source_tier -> float 0-1)
TRUST_SCORES: dict[str, float] = {
    "primary": 1.0,
    "secondary": 0.65,
    # Unclassified is unknown, not intrinsically low quality. Let relevance
    # and the evidence evaluator decide whether an unfamiliar publisher is useful.
    "other": 0.50,
}

# URL-embedded year pattern for freshness inference
_YEAR_RE = re.compile(r"/(?:20(\d{2}))[/-]")
_DATE_RE = re.compile(
    r"(?P<year>20\d{2})\s*(?:年|[-/.])\s*"
    r"(?P<month>0?[1-9]|1[0-2])\s*(?:月|[-/.])\s*"
    r"(?P<day>0?[1-9]|[12]\d|3[01])(?!\d)\s*日?"
)
_DATE_LABEL_RE = re.compile(
    r"(?:最終更新日|更新日|公開日|投稿日|掲載日|published(?:\s+on)?|"
    r"modified(?:\s+on)?|updated(?:\s+on)?|date)\s*[:：]?\s*([^\n]{0,80})",
    re.IGNORECASE,
)
_UPDATED_DATE_LABEL_RE = re.compile(
    r"(?:最終更新日|更新日|modified(?:\s+on)?|updated(?:\s+on)?)\s*[:：]?\s*([^\n]{0,80})",
    re.IGNORECASE,
)
_ENGLISH_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise_domain(url: str) -> str:
    """Extract the *effective* domain from *url* (strips www. prefix)."""
    try:
        host = urlparse(url).hostname or ""
        return host.rstrip(".").lower().removeprefix("www.")
    except Exception:
        return ""


def normalize_official_domains(domains: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize user-registered hosts, accepting either hostnames or URLs.

    Registered domains match themselves and their subdomains. Only register
    domains independently confirmed by the user as official.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    if isinstance(domains, str):
        domains = re.split(r"[,;\r\n]+", domains)
    for entry in domains or ():
        if not isinstance(entry, str) or not entry.strip():
            continue
        candidate = entry.strip()
        parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
        host = (parsed.hostname or "").rstrip(".").lower().removeprefix("www.")
        if not host:
            raise ValueError(f"公式ドメインを解釈できません: {entry}")
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError(f"公式ドメインを解釈できません: {entry}") from exc
        labels = host.split(".")
        if len(labels) < 2 or any(
            not label or len(label) > 63
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels
        ):
            raise ValueError(f"公式ドメインの形式が正しくありません: {entry}")
        if host not in seen:
            normalized.append(host)
            seen.add(host)
    return tuple(normalized)


def source_tier(url: str, official_domains: Iterable[str] = ()) -> str:
    """Return ``"primary"``, ``"secondary"``, or ``"other"`` for *url*."""
    domain = _normalise_domain(url)
    if not domain:
        return "other"
    registered_domains = normalize_official_domains(official_domains)
    if any(domain == registered or domain.endswith("." + registered) for registered in registered_domains):
        return "primary"
    # Exact match
    if domain in PRIMARY_DOMAINS:
        return "primary"
    if domain in SECONDARY_DOMAINS:
        return "secondary"
    # Suffix match (sub-domains)
    for d in PRIMARY_DOMAINS:
        if domain.endswith("." + d):
            return "primary"
    for d in SECONDARY_DOMAINS:
        if domain.endswith("." + d):
            return "secondary"
    return "other"


def extract_content_date(content: str | None) -> date | None:
    """Extract a likely publication/update date from labeled page text.

    Labeled dates are preferred. As a fallback, a full date near the start of
    the page is accepted because article pages commonly place it below title.
    """
    if not content:
        return None

    candidates = [match.group(1) for match in _UPDATED_DATE_LABEL_RE.finditer(content)]
    candidates.extend(match.group(1) for match in _DATE_LABEL_RE.finditer(content))
    candidates.append(content[:500])

    for candidate in candidates:
        match = _DATE_RE.search(candidate)
        if match:
            try:
                return date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            except ValueError:
                pass
        english_match = _ENGLISH_DATE_RE.search(candidate)
        if english_match:
            for pattern in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
                try:
                    return datetime.strptime(english_match.group(0), pattern).date()
                except ValueError:
                    continue
    return None


def freshness_score(
    url: str,
    reference_year: int | None = None,
    *,
    content: str | None = None,
    reference_date: date | None = None,
) -> float:
    """Estimate content freshness (0-1) from page date, then URL year.

    A labeled publication/update date in page text takes precedence. If no
    page date is found, the year embedded in *url* is used. If neither exists,
    a neutral score of ``0.5`` is returned so the
    freshness dimension does not penalise undated content too heavily.

    Scoring curve:
      - Same year as *reference_year*  → 1.0
      - 1 year old                     → 0.9
      - 2 years old                    → 0.7
      - 3 years old                    → 0.5
      - 5+ years old                   → 0.2
    """
    try:
        page_date = extract_content_date(content)
        if page_date:
            today = reference_date or date.today()
            age = (today - page_date).days / 365.25
        else:
            match = _YEAR_RE.search(url)
            if not match:
                return 0.5
            ref = reference_year or datetime.now().year
            age = ref - (2000 + int(match.group(1)))
        if age <= 0:
            return 1.0
        if age <= 1:
            return 1.0 - 0.1 * age
        if age <= 2:
            return 0.9 - 0.2 * (age - 1)
        if age <= 3:
            return 0.7 - 0.2 * (age - 2)
        return max(0.2, 0.5 - (age - 3) * 0.05)
    except Exception:
        return 0.5


def domain_trust_score(url: str, official_domains: Iterable[str] = ()) -> float:
    """Return a trust score (0-1) based on the domain tier of *url*."""
    return TRUST_SCORES[source_tier(url, official_domains)]


def composite_score(
    relevance: float | None,
    url: str,
    *,
    content: str | None = None,
    w_relevance: float = 0.50,
    w_trust: float = 0.30,
    w_freshness: float = 0.20,
    official_domains: Iterable[str] = (),
) -> float:
    """Combine relevance, trust, and freshness into a single ranking score.

    Weights must be non-negative and sum to 1. Defaults: relevance 50 %, trust 30 %, freshness 20 %.
    When *relevance* is ``None`` the remaining weight is redistributed
    proportionally to trust and freshness.
    """
    weights = (w_relevance, w_trust, w_freshness)
    if any(not math.isfinite(weight) or weight < 0 or weight > 1 for weight in weights):
        raise ValueError("Composite score weights must be between 0 and 1")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Composite score weights must sum to 1")

    trust = domain_trust_score(url, official_domains)
    fresh = freshness_score(url, content=content)
    if relevance is None:
        total_w = w_trust + w_freshness
        return (trust * w_trust + fresh * w_freshness) / total_w if total_w else 0.5
    return relevance * w_relevance + trust * w_trust + fresh * w_freshness


__all__ = [
    "PRIMARY_DOMAINS",
    "SECONDARY_DOMAINS",
    "normalize_official_domains",
    "source_tier",
    "extract_content_date",
    "freshness_score",
    "domain_trust_score",
    "composite_score",
]
