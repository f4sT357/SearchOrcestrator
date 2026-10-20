"""Entry point for launching the Search Orchestrator PySide6 GUI."""

import os
import sys
import logging
import tempfile
import traceback
from pathlib import Path

# Ensure parent directory is in sys.path
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


def main() -> None:
    # High DPI scaling support
    from PySide6.QtGui import QFont

    logger = _configure_startup_logger()
    logger.info(
        "Starting GUI; python=%s cwd=%s QT_QPA_PLATFORM=%s",
        sys.executable,
        os.getcwd(),
        os.environ.get("QT_QPA_PLATFORM", "(default)"),
    )
    try:
        logger.info("Importing Qt and GUI modules")
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        logger.info("Qt and GUI modules imported")
    except Exception:
        logger.exception("GUI module import failed")
        traceback.print_exc()
        raise

    app = QApplication(sys.argv)
    logger.info("QApplication created; platform=%s", app.platformName())

    # Set a safe default font size to avoid QFont warnings
    default_font = QFont()
    default_font.setPointSize(10)
    app.setFont(default_font)
    app.setApplicationName("Search Orchestrator")
    app.setApplicationDisplayName("Search Orchestrator")

    try:
        window = MainWindow()
        logger.info("MainWindow constructed; title=%s", window.windowTitle())
        window.show()
        window.raise_()
        window.activateWindow()

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
        logger.info("Entering Qt event loop")
        exit_code = app.exec()
        logger.info("Qt event loop exited; exit_code=%s", exit_code)
        sys.exit(exit_code)
    except Exception:
        logger.exception("GUI startup failed")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
