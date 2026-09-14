"""Run a research workflow against a locally hosted OpenAI-compatible model."""

import argparse
import sys
import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
    category=UserWarning,
    module="langchain_core",
)

from search_orchestrator import Settings
from search_orchestrator_v2 import create_workflow


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI搭載のWebリサーチオーケストレーター")
    parser.add_argument(
        "query", nargs="?",
        default="NVIDIAとAMDのAI GPU戦略について比較調査してください。",
        help="調査したい質問やテーマ",
    )
    parser.add_argument("--model", help="使用するLLMモデル名")
    parser.add_argument("--base-url", help="OpenAI互換APIのエンドポイントURL")
    parser.add_argument("--api-key", help="APIキー")
    parser.add_argument("--max-concurrency", type=int, default=2, help="並列実行数の上限")
    parser.add_argument("--no-fetch", action="store_true", help="Webページの本文取得を無効化する")
    parser.add_argument("--max-content-length", type=int, default=3000, help="Webページ本文の最大取得文字数")
    args = parser.parse_args()

    settings = Settings.from_environment()
    settings = Settings(
        model=args.model or settings.model,
        base_url=args.base_url or settings.base_url,
        api_key=args.api_key or settings.api_key,
        fetch_web_content=not args.no_fetch,
        max_content_length=args.max_content_length,
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
