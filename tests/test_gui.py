import os
import sys
import json

# Set offscreen platform for headless test environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.worker import ResearchWorker
from search_orchestrator import Settings


def test_gui_main_window_initialization(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("gui.main_window.Settings.from_environment", lambda: Settings())
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.config_path = str(tmp_path / "config.json")
    assert window.windowTitle() == "Search Orchestrator - AI Web Research Assistant"
    assert window.query_edit.toPlainText() != ""
    assert window.tabs.count() == 3
    assert window.start_btn.isEnabled()
    assert window.model_combo is not None
    assert window.model_combo.isEditable()
    assert window.refresh_models_btn is not None
    assert window.jina_fallback_cb is not None
    window.api_key_edit.setText("test-secret")
    window.firecrawl_api_key_edit.setText("firecrawl-secret")
    window._save_config()
    saved_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "api_key" not in saved_config
    assert "use_fallback_fetcher" in saved_config
    assert saved_config["content_fetcher"] == "builtin"
    assert saved_config["firecrawl_api_url"] == "http://localhost:3002"
    assert "firecrawl_api_key" not in saved_config
    window.close()


def test_gui_worker_signal_connections() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    settings = Settings(model="test-model")
    worker = ResearchWorker(query="test query", settings=settings)
    assert worker.query == "test query"
    assert worker.settings.model == "test-model"


def test_worker_logs_selected_fetcher_without_logging_api_key(monkeypatch) -> None:
    class FakeWorkflow:
        def invoke(self, state, options):
            return {"summary": "done", "results": []}

    monkeypatch.setattr("gui.worker.create_workflow", lambda settings, log: FakeWorkflow())
    settings = Settings(
        content_fetcher="firecrawl",
        firecrawl_api_url="http://localhost:3002",
        firecrawl_api_key="do-not-log-this-key",
    )
    worker = ResearchWorker(query="test query", settings=settings)
    messages: list[str] = []
    worker.log_signal.connect(messages.append)

    worker.run()

    initialization_log = next(message for message in messages if "ワークフローを初期化中" in message)
    assert "本文取得: 有効" in initialization_log
    assert "エンジン: Firecrawl セルフホスト" in initialization_log
    assert "API URL: http://localhost:3002" in initialization_log
    assert "do-not-log-this-key" not in initialization_log
