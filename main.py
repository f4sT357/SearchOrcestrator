"""Run a research workflow against a locally hosted OpenAI-compatible model."""

import argparse
import math
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
    category=UserWarning,
    module="langchain_core",
)

from search_orchestrator import Settings, create_workflow


def main() -> None:
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
    parser.add_argument("--max-concurrency", type=int, default=2, help="並列実行数の上限")
    parser.add_argument("--no-fetch", action="store_true", help="Webページの本文取得を無効化する")
    parser.add_argument("--max-content-length", type=int, default=3000, help="Webページ本文の最大取得文字数")
    parser.add_argument("--weight-relevance", type=float, help="順位スコアの関連度重み (0-1)")
    parser.add_argument("--weight-trust", type=float, help="順位スコアの信頼度重み (0-1)")
    parser.add_argument("--weight-freshness", type=float, help="順位スコアの新鮮度重み (0-1)")
    args = parser.parse_args()

    if args.max_concurrency < 1:
        parser.error("--max-concurrency は1以上を指定してください")
    if args.max_content_length < 1:
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

    settings = Settings(
        model=args.model or settings.model,
        base_url=args.base_url or settings.base_url,
        api_key=args.api_key or settings.api_key,
        fetch_web_content=not args.no_fetch,
        max_content_length=args.max_content_length,
        relevance_weight=relevance_weight,
        trust_weight=trust_weight,
        freshness_weight=freshness_weight,
    )

    workflow = create_workflow(settings)
    result = workflow.invoke(
        {
            "query": args.query,
            "plan": None,
            "task": None,
            "results": [],
            "evaluation": None,
            "summary": "",
            "search_query_count": 0,
            "search_round": 0,
        },
        {"max_concurrency": args.max_concurrency},
    )
    print("\n=========================")
    print(result["summary"])


if __name__ == "__main__":
    main()
