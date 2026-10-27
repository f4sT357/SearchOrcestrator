"""Entry point for launching the Search Orchestrator PySide6 GUI."""

import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def _configure_startup_logger() -> logging.Logger:
    """Write GUI startup diagnostics somewhere users can inspect after launch."""
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "SearchOrchestrator" / "logs")
    candidates.append(Path(tempfile.gettempdir()) / "SearchOrchestrator" / "logs")

    logger = logging.getLogger("search_orchestrator.gui.startup")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(directory / "gui-startup.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
            logger.info("Startup diagnostics initialized; log_path=%s", handler.baseFilename)
            return logger
        except OSError:
            continue
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.error("Could not create a startup log file; logging to stderr")
    return logger


def _start_single_instance_server(app_name: str, logger: logging.Logger):
    """Return a local server, or None when another GUI instance is running."""
    from PySide6.QtNetwork import QLocalServer, QLocalSocket

    server = QLocalServer()
    probe = QLocalSocket()
    probe.connectToServer(app_name)
    if probe.waitForConnected(250):
        probe.disconnectFromServer()
        logger.info("Another GUI instance is already running")
        return None

    QLocalServer.removeServer(app_name)
    if not server.listen(app_name):
        raise RuntimeError(f"Could not reserve GUI instance server: {server.errorString()}")
    return server


def _create_startup_window():
    from PySide6.QtWidgets import QLabel, QMainWindow, QProgressBar, QVBoxLayout, QWidget

    window = QMainWindow()
    window.setWindowTitle("Search Orchestrator を準備中")
    window.setFixedSize(500, 220)

    central_widget = QWidget(window)
    layout = QVBoxLayout(central_widget)
    layout.setContentsMargins(32, 28, 32, 28)
    layout.setSpacing(14)

    title = QLabel("Search Orchestrator")
    title.setStyleSheet("font-size: 22px; font-weight: 700;")
    detail = QLabel("起動画面を表示しました。調査コンポーネントを準備しています…")
    detail.setWordWrap(True)
    progress = QProgressBar()
    progress.setRange(0, 0)

    layout.addWidget(title)
    layout.addWidget(detail)
    layout.addWidget(progress)
    layout.addStretch()
    window.setCentralWidget(central_widget)
    return window, detail, progress


def _create_bootstrap_runnable():
    from PySide6.QtCore import QObject, QRunnable, Signal

    class BootstrapSignals(QObject):
        stage_changed = Signal(str)
        ready = Signal(object)
        failed = Signal(str)

    class BootstrapRunnable(QRunnable):
        def __init__(self) -> None:
            super().__init__()
            self.signals = BootstrapSignals()

        def run(self) -> None:
            try:
                self.signals.stage_changed.emit("検索・LLM連携のコンポーネントを読み込んでいます…")
                from gui.main_window import MainWindow

                self.signals.stage_changed.emit("調査画面を組み立てています…")
                self.signals.ready.emit(MainWindow)
            except Exception as exc:
                self.signals.failed.emit(str(exc))

    return BootstrapRunnable()


def main() -> None:
    from PySide6.QtCore import QThreadPool, QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QMessageBox

    logger = _configure_startup_logger()
    logger.info(
        "Starting GUI; python=%s cwd=%s QT_QPA_PLATFORM=%s",
        sys.executable,
        os.getcwd(),
        os.environ.get("QT_QPA_PLATFORM", "(default)"),
    )

    app = QApplication(sys.argv)
    logger.info("QApplication created; platform=%s", app.platformName())
    default_font = QFont()
    default_font.setPointSize(10)
    app.setFont(default_font)
    app.setApplicationName("Search Orchestrator")
    app.setApplicationDisplayName("Search Orchestrator")

    instance_server = _start_single_instance_server("SearchOrchestrator-GUI-v1", logger)
    if instance_server is None:
        QMessageBox.information(
            None,
            "Search Orchestrator",
            "Search Orchestrator はすでに起動しています。",
        )
        return

    startup_window, startup_detail, startup_progress = _create_startup_window()
    startup_window.show()
    startup_window.raise_()
    startup_window.activateWindow()
    logger.info("Startup shell shown")
    window_holder: dict[str, Any] = {"main": None}

    def show_main_window(main_window_class: type[Any]) -> None:
        startup_detail.setText("Reranker とモデル一覧をバックグラウンドで準備します…")
        window = main_window_class()
        window_holder["main"] = window
        logger.info("MainWindow constructed; title=%s", window.windowTitle())
        window.show()
        window.raise_()
        window.activateWindow()
        startup_window.close()

        def record_window_state() -> None:
            app.processEvents()
            handle = int(window.winId())
            logger.info(
                "MainWindow shown; visible=%s minimized=%s active=%s handle=%s geometry=%s",
                window.isVisible(),
                window.isMinimized(),
                window.isActiveWindow(),
                handle,
                window.geometry().getRect(),
            )

        QTimer.singleShot(500, record_window_state)

    def show_bootstrap_error(error: str) -> None:
        logger.error("GUI component bootstrap failed: %s", error)
        startup_progress.setRange(0, 100)
        startup_progress.setValue(0)
        startup_detail.setText("コンポーネントの準備に失敗しました。起動ログを確認してください。")
        QMessageBox.critical(startup_window, "起動エラー", error)

    runnable = _create_bootstrap_runnable()
    app.bootstrap_runnable = runnable
    app.instance_server = instance_server
    runnable.signals.stage_changed.connect(startup_detail.setText)
    runnable.signals.ready.connect(show_main_window)
    runnable.signals.failed.connect(show_bootstrap_error)
    QTimer.singleShot(100, lambda: QThreadPool.globalInstance().start(runnable))

    logger.info("Entering Qt event loop")
    try:
        exit_code = app.exec()
        logger.info("Qt event loop exited; exit_code=%s", exit_code)
        sys.exit(exit_code)
    except Exception:
        logger.exception("GUI startup failed")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()