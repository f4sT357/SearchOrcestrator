"""Qt stylesheet used by the main application window."""

MAIN_WINDOW_STYLESHEET = """
            QWidget {
                font-family: 'Segoe UI', 'Meiryo', sans-serif;
                font-size: 13px;
                color: #2b2b2b;
            }
            QMainWindow {
                background-color: #f6f8fa;
            }
            #titleLabel {
                font-size: 20px;
                font-weight: bold;
                color: #1a1f2c;
            }
            #subtitleLabel {
                font-size: 12px;
                color: #57606a;
                margin-bottom: 4px;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #24292f;
            }
            #startBtn {
                background-color: #2da44e;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                min-height: 24px;
            }
            #startBtn:hover {
                background-color: #2c974b;
            }
            #startBtn:disabled {
                background-color: #94d3a2;
            }
            #secondaryBtn, QPushButton {
                background-color: #f6f8fa;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                padding: 6px 14px;
                color: #24292f;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                border-color: #1b1f2426;
            }
            QTextEdit, QPlainTextEdit, QLineEdit, QSpinBox {
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                padding: 6px;
            }
            QTextEdit:focus, QPlainTextEdit:focus, QLineEdit:focus, QSpinBox:focus {
                border: 2px solid #0969da;
            }
            QTabWidget::pane {
                border: 1px solid #d0d7de;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #eaeef2;
                border: 1px solid #d0d7de;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
            }
            QProgressBar {
                border: 1px solid #d0d7de;
                border-radius: 7px;
                background-color: #eaeef2;
            }
            QProgressBar::chunk {
                background-color: #0969da;
                border-radius: 6px;
            }
            #statusLabel {
                font-weight: bold;
                color: #24292f;
            }
            QTableWidget {
                gridline-color: #e1e4e8;
                border: 1px solid #d0d7de;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #f6f8fa;
                padding: 4px;
                border: 1px solid #d0d7de;
                font-weight: bold;
            }
        """
