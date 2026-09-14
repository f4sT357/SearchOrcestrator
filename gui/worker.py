"""Background worker thread for running the research workflow."""

from __future__ import annotations

import sys
import os
from typing import Any

from PySide6.QtCore import QThread, Signal

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from search_orchestrator import Settings
from search_orchestrator_v2 import create_workflow


class ResearchWorker(QThread):
    """Executes the search orchestrator workflow asynchronously."""

    log_signal = Signal(str)
    result_signal = Signal(dict)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, query: str, settings: Settings, max_concurrency: int = 2, parent: Any = None) -> None:
        super().__init__(parent)
        self.query = query
        self.settings = settings
        self.max_concurrency = max_concurrency

    def run(self) -> None:
        try:
            fetch_mode_str = "有効" if self.settings.fetch_web_content else "無効"
            self.log_signal.emit(
                f"ワークフローを初期化中... (モデル: {self.settings.model}, Webページ本文取得: {fetch_mode_str})"
            )
            workflow = create_workflow(self.settings)
            self.log_signal.emit(f"調査を開始します: 「{self.query}」")
            initial_state = {
                "query": self.query,
                "plan": None,
                "task": None,
                "results": [],
                "evaluation": None,
                "summary": "",
                "search_query_count": 0,
                "search_round": 0,
            }
            self.log_signal.emit("調査計画の立案およびWeb検索を実行中...")
            result = workflow.invoke(initial_state, {"max_concurrency": self.max_concurrency})
            self.log_signal.emit("調査と回答生成が正常に完了しました。")
            self.result_signal.emit(result)
        except Exception as exc:
            self.log_signal.emit(f"エラーが発生しました: {exc}")
            self.error_signal.emit(str(exc))
        finally:
            self.finished_signal.emit()
