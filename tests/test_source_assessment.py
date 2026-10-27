from unittest.mock import MagicMock

from search_orchestrator import Evaluation, ResearchWorkflowNodes, SearchResult, Settings
from source_assessment import SourceAssessment, SourceAssessmentCache, merge_assessments


def test_source_assessment_cache_reuses_only_matching_content(tmp_path) -> None:
    cache = SourceAssessmentCache(tmp_path)
    assessment = SourceAssessment(
        url="https://example.com/page",
        credibility=0.78,
        use_as_evidence=True,
        reason="Named publisher and directly supported claim.",
        support_scope="Published release detail.",
    )

    cache.put(assessment, "first fetched page body")

    assert cache.get(assessment.url, "first fetched page body") == assessment
    assert cache.get(assessment.url, "changed fetched page body") is None


def test_merge_assessments_uses_fresh_result_and_ignores_unknown_url() -> None:
    url = "https://example.com/page"
    cached = SourceAssessment(
        url=url, credibility=0.60, use_as_evidence=True, reason="Cached", support_scope="Old",
    )
    fresh = SourceAssessment(
        url=url, credibility=0.85, use_as_evidence=True, reason="Fresh", support_scope="New",
    )
    unknown = SourceAssessment(
        url="https://unknown.example/page", credibility=1.0, use_as_evidence=True,
        reason="Must be ignored", support_scope="",
    )

    assert merge_assessments([cached], [fresh, unknown], allowed_urls=[url]) == [fresh]


def test_final_answer_excludes_page_rejected_by_assessment() -> None:
    response = MagicMock(content="Supported statement [S1]")
    llm = MagicMock()
    llm.bind.return_value = llm
    llm.invoke.return_value = response
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(), llm=llm,
    )
    accepted = SearchResult(
        query="q", title="Accepted page", url="https://example.com/accepted",
        snippet="approved evidence", relevance_score=0.9, combined_score=0.9,
    )
    rejected = SearchResult(
        query="q", title="Rejected page", url="https://example.com/rejected",
        snippet="unsupported claim", relevance_score=0.9, combined_score=0.8,
    )
    evaluation = Evaluation(
        sufficient=True, missing_information=[], weak_evidence=[], reason="Enough evidence",
        additional_queries=[], official_additional_queries=[],
        source_assessments=[SourceAssessment(
            url=rejected.url, credibility=0.20, use_as_evidence=False,
            reason="Unsupported aggregation.", support_scope="",
        )],
    )

    result = nodes.analyze({
        "query": "q", "results": [], "evidence_results": [accepted, rejected],
        "official_results": [], "evaluation": evaluation, "run_memo": "",
    })

    prompt = llm.invoke.call_args.args[0]
    assert "approved evidence" in prompt
    assert "unsupported claim" not in prompt
    assert "https://example.com/accepted" in result["summary"]
    assert "https://example.com/rejected" not in result["summary"]


def test_final_answer_is_not_generated_when_all_pages_are_rejected() -> None:
    llm = MagicMock()
    llm.bind.return_value = llm
    nodes = ResearchWorkflowNodes(
        Settings(fetch_web_content=False), search=MagicMock(), reranker=MagicMock(), llm=llm,
    )
    rejected = SearchResult(
        query="q", title="Rejected page", url="https://example.com/rejected",
        snippet="unsupported claim", relevance_score=0.9, combined_score=0.8,
    )
    evaluation = Evaluation(
        sufficient=False, missing_information=["No reliable evidence"], weak_evidence=[],
        reason="All candidates were rejected", additional_queries=[], official_additional_queries=[],
        source_assessments=[SourceAssessment(
            url=rejected.url, credibility=0.20, use_as_evidence=False,
            reason="Unsupported aggregation.", support_scope="",
        )],
    )

    result = nodes.analyze({
        "query": "q", "results": [], "evidence_results": [rejected],
        "official_results": [], "evaluation": evaluation, "run_memo": "",
    })

    assert "\u6839\u62e0\u306e\u306a\u3044\u56de\u7b54\u306f\u751f\u6210\u3057\u307e\u305b\u3093\u3067\u3057\u305f" in result["summary"]
    llm.invoke.assert_not_called()
