"""Background worker thread and task runners for running the research workflow."""

from __future__ import annotations

import sys
import os
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal

# Ensure parent directory is in sys.path when running from gui/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from search_orchestrator import Settings, create_workflow, preload_reranker, fetch_available_models


class RerankerPreloadSignals(QObject):
    ready = Signal()
    failed = Signal(str)


class RerankerPreloadRunnable(QRunnable):
    """QRunnable task to warm the reranker cache via QThreadPool."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.signals = RerankerPreloadSignals()
        self.model_name = model_name

    def run(self) -> None:
        try:
            preload_reranker(self.model_name)
        except Exception as exc:
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            try:
                self.signals.ready.emit()
            except RuntimeError:
                pass


def preload_reranker_background(model_name: str, signals: RerankerPreloadSignals) -> None:
    """Warm the reranker cache using QThreadPool without blocking the GUI event loop."""
    runnable = RerankerPreloadRunnable(model_name)
    runnable.signals.ready.connect(signals.ready.emit)
    runnable.signals.failed.connect(signals.failed.emit)
    QThreadPool.globalInstance().start(runnable)


class ModelFetchSignals(QObject):
    models_fetched = Signal(list, bool)
    fetch_failed = Signal(str, bool)


class ModelFetchRunnable(QRunnable):
    """QRunnable task to query available models asynchronously via QThreadPool."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        show_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.signals = ModelFetchSignals()
        self.base_url = base_url
        self.api_key = api_key
        self.show_feedback = show_feedback

    def run(self) -> None:
        try:
            models = fetch_available_models(base_url=self.base_url, api_key=self.api_key, timeout=3.0)
            try:
                self.signals.models_fetched.emit(models, self.show_feedback)
            except RuntimeError:
                pass
        except Exception as exc:
            try:
                self.signals.fetch_failed.emit(str(exc), self.show_feedback)
            except RuntimeError:
                pass


def _fetch_settings_summary(settings: Settings) -> str:
    """Describe the effective body-fetch configuration without exposing secrets."""
    engine_names = {
        "builtin": "標準取得 (httpx / lxml)",
        "firecrawl": "Firecrawl セルフホスト",
    }
    engine_name = engine_names.get(settings.content_fetcher, settings.content_fetcher)
    fetch_status = "有効" if settings.fetch_web_content else "無効"
    summary = f"本文取得: {fetch_status} / エンジン: {engine_name}"
    if settings.content_fetcher == "firecrawl":
        summary += f" / API URL: {settings.firecrawl_api_url}"
    return summary


class ResearchWorker(QThread):
    """Executes the search orchestrator workflow asynchronously."""

    log_signal = Signal(str)
    board_signal = Signal(str)
    progress_signal = Signal(str, str, str)
    result_signal = Signal(dict)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(
        self,
        query: str,
        settings: Settings,
        max_concurrency: int = 2,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.query = query
        self.settings = settings
        self.max_concurrency = max_concurrency

    def run(self) -> None:
        try:
            fetch_settings = _fetch_settings_summary(self.settings)
            self.log_signal.emit(
                f"ワークフローを初期化中... (モデル: {self.settings.model}, {fetch_settings})"
            )
            workflow = create_workflow(
                self.settings,
                log=self.log_signal.emit,
                board_update=self.board_signal.emit,
                progress=self.progress_signal.emit,
            )

            self.log_signal.emit(f"調査を開始します: 「{self.query}」")
            initial_state = {
                "query": self.query,
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
            }

            self.log_signal.emit("依頼内容を整理し、調査計画の立案とWeb検索を実行中...")
            result = workflow.invoke(
                initial_state,
                {"max_concurrency": self.max_concurrency},
            )

            self.progress_signal.emit("workflow", "completed", "")
            self.log_signal.emit("調査と回答生成が正常に完了しました。")
            self.result_signal.emit(result)
        except Exception as exc:
            self.progress_signal.emit("workflow", "failed", str(exc))
            self.log_signal.emit(f"エラーが発生しました: {exc}")
            self.error_signal.emit(str(exc))
        finally:
            self.finished_signal.emit()
