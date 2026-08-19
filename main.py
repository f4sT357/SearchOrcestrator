"""Run a research workflow against a locally hosted OpenAI-compatible model."""

import sys

from search_orchestrator import Settings, create_workflow


def main() -> None:
    # Local models can emit characters that the Windows CP932 console cannot
    # represent. Use UTF-8 so a successful run does not fail while printing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    workflow = create_workflow(Settings.from_environment())
    result = workflow.invoke({
        "query": "NVIDIAとAMDのAI GPU戦略について比較調査してください。",
        "plan": None, "task": None, "results": [], "evaluation": None,
        "summary": "", "search_query_count": 0, "search_round": 0,
    }, {"max_concurrency": 2})
    print("\n=========================")
    print(result["summary"])


if __name__ == "__main__":
    main()
