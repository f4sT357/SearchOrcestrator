"""Run a research workflow against a locally hosted OpenAI-compatible model."""

import argparse
import math
import sys
import warnings
from dataclasses import replace

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
    category=UserWarning,
    module="langchain_core",
)

from search_orchestrator import Settings, create_workflow


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options and run the configured research workflow."""
    # Local models can emit characters that the Windows CP932 console cannot
    # represent. Use UTF-8 so a successful run does not fail while printing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI搭載のWebリサーチオーケストレーター")
    parser.add_argument(
        "query",
        nargs="?",
        default="NVIDIAとAMDのAI GPU戦略について比較調査してください。",
        help="調査したい質問やテーマ",
    )
    parser.add_argument("--model", help="使用するLLMモデル名")
    parser.add_argument("--base-url", help="OpenAI互換APIのエンドポイントURL")
    parser.add_argument("--api-key", help="APIキー")
    parser.add_argument("--max-concurrency", type=int, help="並列実行数の上限")
    parser.add_argument("--max-search-queries", type=int, help="最大検索クエリ数")
    parser.add_argument("--max-search-rounds", type=int, help="最大検索ラウンド数")
    parser.add_argument("--results-per-query", type=int, help="検索ごとの取得件数")
    parser.add_argument("--content-candidate-results-per-query", type=int, help="本文取得前の候補件数")
    parser.add_argument("--reranked-results-per-query", type=int, help="検索ごとの最終採用件数")
    parser.add_argument("--content-fetcher", choices=("builtin", "firecrawl"), help="本文取得エンジン")
    parser.add_argument("--firecrawl-url", help="セルフホスト Firecrawl API の URL")
    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument("--fetch", dest="fetch_web_content", action="store_true", help="Web本文取得を有効化")
    fetch_group.add_argument("--no-fetch", dest="fetch_web_content", action="store_false", help="Web本文取得を無効化")
    parser.set_defaults(fetch_web_content=None)
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument("--jina-fallback", dest="use_fallback_fetcher", action="store_true", help="失敗時にJina ReaderへURLを送信して再取得")
    fallback_group.add_argument("--no-jina-fallback", dest="use_fallback_fetcher", action="store_false", help="Jina Readerへの外部送信を無効化")
    parser.set_defaults(use_fallback_fetcher=None)
    parser.add_argument("--max-content-length", type=int, help="Webページ本文の最大取得文字数")
    parser.add_argument("--weight-relevance", type=float, help="順位スコアの関連度重み (0-1)")
    parser.add_argument("--weight-trust", type=float, help="順位スコアの信頼度重み (0-1)")
    parser.add_argument("--weight-freshness", type=float, help="順位スコアの新鮮度重み (0-1)")
    args = parser.parse_args(argv)

    if args.max_concurrency is not None and args.max_concurrency < 1:
        parser.error("--max-concurrency は1以上を指定してください")
    if args.max_search_queries is not None and args.max_search_queries < 1:
        parser.error("--max-search-queries は1以上を指定してください")
    if args.max_search_rounds is not None and args.max_search_rounds < 0:
        parser.error("--max-search-rounds は0以上を指定してください")
    for option, value in (
        ("--results-per-query", args.results_per_query),
        ("--content-candidate-results-per-query", args.content_candidate_results_per_query),
        ("--reranked-results-per-query", args.reranked_results_per_query),
    ):
        if value is not None and value < 1:
            parser.error(f"{option} は1以上を指定してください")
    if args.max_content_length is not None and args.max_content_length < 1:
        parser.error("--max-content-length は1以上を指定してください")

    settings = Settings.from_environment()
    relevance_weight = args.weight_relevance if args.weight_relevance is not None else settings.relevance_weight
    trust_weight = args.weight_trust if args.weight_trust is not None else settings.trust_weight
    freshness_weight = args.weight_freshness if args.weight_freshness is not None else settings.freshness_weight
    if any(
        not math.isfinite(weight) or weight < 0 or weight > 1
        for weight in (relevance_weight, trust_weight, freshness_weight)
    ):
        parser.error("順位スコアの各重みは0から1の範囲で指定してください")
    if abs(relevance_weight + trust_weight + freshness_weight - 1.0) > 1e-6:
        parser.error("順位スコアの重みの合計は1にしてください")

    settings = replace(
        settings,
        model=args.model or settings.model,
        base_url=args.base_url or settings.base_url,
        api_key=args.api_key or settings.api_key,
        max_concurrency=(settings.max_concurrency if args.max_concurrency is None else args.max_concurrency),
        max_search_queries=(settings.max_search_queries if args.max_search_queries is None else args.max_search_queries),
        max_search_rounds=(settings.max_search_rounds if args.max_search_rounds is None else args.max_search_rounds),
        results_per_query=(settings.results_per_query if args.results_per_query is None else args.results_per_query),
        content_candidate_results_per_query=(
            settings.content_candidate_results_per_query
            if args.content_candidate_results_per_query is None
            else args.content_candidate_results_per_query
        ),
        reranked_results_per_query=(
            settings.reranked_results_per_query
            if args.reranked_results_per_query is None
            else args.reranked_results_per_query
        ),
        fetch_web_content=(settings.fetch_web_content if args.fetch_web_content is None else args.fetch_web_content),
        use_fallback_fetcher=(settings.use_fallback_fetcher if args.use_fallback_fetcher is None else args.use_fallback_fetcher),
        max_content_length=(settings.max_content_length if args.max_content_length is None else args.max_content_length),
        content_fetcher=args.content_fetcher or settings.content_fetcher,
        firecrawl_api_url=args.firecrawl_url or settings.firecrawl_api_url,
        relevance_weight=relevance_weight,
        trust_weight=trust_weight,
        freshness_weight=freshness_weight,
        official_domains=settings.official_domains,
    )

    workflow = create_workflow(settings)
    result = workflow.invoke(
        {
            "query": args.query,
            "understood_request": "",
            "plan": None,
            "task": None,
            "results": [],
            "evidence_results": [],
            "official_results": [],
            "evaluation": None,
            "run_memo": "",
            "summary": "",
            "search_query_count": 0,
            "search_round": 0,
            "searched_queries": [],
        },
        {"max_concurrency": settings.max_concurrency},
    )
    print("\n=========================")
    print(result["summary"])


if __name__ == "__main__":
    main()
