"""Run a research workflow against a locally hosted OpenAI-compatible model."""

import argparse
import sys

from search_orchestrator import Settings, create_workflow


def main() -> None:
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
    args = parser.parse_args()

    settings = Settings.from_environment()
    if args.model:
        settings = Settings(
            model=args.model,
            base_url=args.base_url or settings.base_url,
            api_key=args.api_key or settings.api_key,
        )
    elif args.base_url or args.api_key:
        settings = Settings(
            model=settings.model,
            base_url=args.base_url or settings.base_url,
            api_key=args.api_key or settings.api_key,
        )

    workflow = create_workflow(settings)
    result = workflow.invoke({
        "query": args.query,
        "plan": None, "task": None, "results": [], "evaluation": None,
        "summary": "", "search_query_count": 0, "search_round": 0,
    }, {"max_concurrency": args.max_concurrency})
    print("\n=========================")
    print(result["summary"])


if __name__ == "__main__":
    main()

