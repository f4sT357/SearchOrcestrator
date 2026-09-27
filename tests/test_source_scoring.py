from datetime import date

import pytest

from source_scoring import composite_score, domain_trust_score, extract_content_date, freshness_score


def test_extract_content_date_supports_labeled_iso_and_japanese_dates() -> None:
    assert extract_content_date("Published: 2024-03-15\nText") == date(2024, 3, 15)
    assert extract_content_date("更新日：2025年7月9日\nText") == date(2025, 7, 9)
    assert extract_content_date("Published: 2024-03-15\nUpdated: September 9, 2025") == date(2025, 9, 9)


def test_extract_content_date_ignores_invalid_dates() -> None:
    assert extract_content_date("公開日: 2025年2月31日\nNo valid date") is None


def test_freshness_prefers_page_date_and_falls_back_to_url_year() -> None:
    reference = date(2026, 9, 28)
    page_dated = freshness_score(
        "https://example.com/2020/old", content="公開日: 2025-08-01", reference_date=reference,
    )
    url_only = freshness_score("https://example.com/2024/old", reference_year=2026)
    undated = freshness_score("https://example.com/article", reference_date=reference)

    assert 0.85 < page_dated < 0.9
    assert url_only == 0.7
    assert undated == 0.5

    within_year = freshness_score(
        "https://example.com/article", content="公開日: 2026-01-01", reference_date=reference,
    )
    same_day = freshness_score(
        "https://example.com/article", content="公開日: 2026-09-28", reference_date=reference,
    )
    assert 0.9 < within_year < 1.0
    assert same_day == 1.0


def test_composite_score_accepts_configurable_weights_and_validates_them() -> None:
    relevance_heavy = composite_score(
        1.0, "https://example.com/article",
        w_relevance=0.8, w_trust=0.1, w_freshness=0.1,
    )
    trust_heavy = composite_score(
        1.0, "https://example.com/article",
        w_relevance=0.1, w_trust=0.8, w_freshness=0.1,
    )
    assert relevance_heavy > trust_heavy
    with pytest.raises(ValueError, match="sum to 1"):
        composite_score(0.5, "https://example.com", w_relevance=0.5, w_trust=0.5, w_freshness=0.5)


def test_unclassified_domain_is_neutral_for_ranking() -> None:
    assert domain_trust_score("https://unlisted-specialist.example/article") == 0.5
