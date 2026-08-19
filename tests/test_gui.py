import os
import sys

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


def test_gui_main_window_initialization() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    assert window.windowTitle() == "Search Orchestrator - AI Web Research Assistant"
    assert window.query_edit.toPlainText() != ""
    assert window.tabs.count() == 3
    assert window.start_btn.isEnabled()
    assert window.model_combo is not None
    assert window.model_combo.isEditable()
    assert window.refresh_models_btn is not None


def test_gui_worker_signal_connections() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    settings = Settings(model="test-model")
    worker = ResearchWorker(query="test query", settings=settings)
    assert worker.query == "test query"
    assert worker.settings.model == "test-model"
