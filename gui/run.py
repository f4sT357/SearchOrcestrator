"""Entry point for launching the Search Orchestrator PySide6 GUI."""

import os
import sys

# Ensure parent directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> None:
    # High DPI scaling support
    app = QApplication(sys.argv)
    app.setApplicationName("Search Orchestrator")
    app.setApplicationDisplayName("Search Orchestrator")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
