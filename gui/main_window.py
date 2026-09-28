"""PySide6 Main Window for Search Orchestrator."""

from __future__ import annotations

import datetime
import os
import sys
from threading import Thread
from typing import Any

from PySide6.QtCore import QDateTime, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
from gui.worker import (
    RerankerPreloadSignals,
    ResearchWorker,
    preload_reranker_background,
)
from search_orchestrator import (
    SearchResult, Settings, fetch_available_models, normalize_official_domains,
)


class MainWindow(QMainWindow):
    """Main application window for the Search Orchestrator GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.worker: ResearchWorker | None = None
        self.last_results: list[SearchResult] = []
        self.default_settings = Settings.from_environment()
        self.config_path = os.path.join(parent_dir, "config.json")

        self.setWindowTitle("Search Orchestrator - AI Web Research Assistant")
        self.resize(1050, 780)
        self.setMinimumSize(800, 600)

        self._setup_ui()
        self._apply_stylesheet()

        self._reranker_preload_signals = RerankerPreloadSignals(self)
        self._reranker_preload_signals.ready.connect(self._on_reranker_preloaded)
        self._reranker_preload_signals.failed.connect(self._on_reranker_preload_failed)
        self._reranker_preload_thread = Thread(
            target=preload_reranker_background,
            args=(self.default_settings.reranker_model, self._reranker_preload_signals),
            name="cross-encoder-preload",
            daemon=True,
        )
        self.status_label.setText("CrossEncoderをバックグラウンドで準備中...")
        self._log("GUI起動後のバックグラウンドでCrossEncoderを読み込みます")
        self._reranker_preload_thread.start()

        # Fetch models once on application startup as requested
        self._refresh_model_list(show_feedback=False)

    def _on_reranker_preloaded(self) -> None:
        self._log("CrossEncoderの準備が完了しました")
        if self.worker is None:
            self.status_label.setText("待機中（CrossEncoder準備完了）")

    def _on_reranker_preload_failed(self, error: str) -> None:
        self._log(f"CrossEncoderの事前読み込みに失敗しました: {error}")
        if self.worker is None:
            self.status_label.setText("CrossEncoderを準備できませんでした。調査開始時に再試行します")

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)
        title_label = QLabel("Search Orchestrator", self)
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel(
            "ローカルLLM・DuckDuckGo検索・CrossEncoder再ランキングによる自律調査エージェント",
            self,
        )
        subtitle_label.setObjectName("subtitleLabel")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addLayout(header_layout)

        # Query Input Section
        input_group = QGroupBox("調査テーマ・質問", self)
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(8)

        self.query_edit = QTextEdit(self)
        self.query_edit.setPlaceholderText("調査したいテーマや質問を入力してください... (例: NVIDIAとAMDのAI GPU戦略について比較調査してください。)")
        self.query_edit.setText("NVIDIAとAMDのAI GPU戦略について比較調査してください。")
        self.query_edit.setFixedHeight(70)
        input_layout.addWidget(self.query_edit)

        # Button Row
        button_row = QHBoxLayout()
        self.start_btn = QPushButton("🚀 調査を開始する", self)
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._on_start_clicked)
        button_row.addWidget(self.start_btn)

        self.toggle_settings_btn = QPushButton("⚙️ 設定を表示/非表示", self)
        self.toggle_settings_btn.setObjectName("secondaryBtn")
        self.toggle_settings_btn.setCheckable(True)
        self.toggle_settings_btn.toggled.connect(self._toggle_settings)
        button_row.addWidget(self.toggle_settings_btn)

        button_row.addStretch()
        input_layout.addLayout(button_row)
        main_layout.addWidget(input_group)

        # Settings Panel (Collapsible)
        self.settings_group = QGroupBox("詳細設定", self)
        self.settings_group.setVisible(False)
        settings_layout = QFormLayout(self.settings_group)
        settings_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Model selection with Combobox and Refresh button
        model_row = QHBoxLayout()
        self.model_combo = QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.addItem(self.default_settings.model)
        self.model_combo.setCurrentText(self.default_settings.model)
        self.model_combo.setToolTip(
            "調査計画・品質評価・レポート作成に使うLLMです。"
            "下のBase URLで指定したAPIから利用できるモデル名を選ぶか入力してください。"
        )
        model_row.addWidget(self.model_combo, stretch=1)

        self.refresh_models_btn = QPushButton("🔄 AIモデル一覧を更新", self)
        self.refresh_models_btn.setObjectName("secondaryBtn")
        self.refresh_models_btn.clicked.connect(self._on_refresh_models_clicked)
        self.refresh_models_btn.setToolTip("Base URLに接続し、利用可能なモデル名を読み直します。")
        model_row.addWidget(self.refresh_models_btn)

        settings_layout.addRow("LLMモデル名:", model_row)

        self.base_url_edit = QLineEdit(self.default_settings.base_url, self)
        self.base_url_edit.setToolTip(
            "LLM APIの接続先です。OpenAI互換APIのURLを入力します。"
            "LM Studioでは通常 http://localhost:1234/v1 の形式です。"
        )
        settings_layout.addRow("Base URL (APIエンドポイント):", self.base_url_edit)

        self.api_key_edit = QLineEdit(self.default_settings.api_key, self)
        self.api_key_edit.setToolTip("LLM APIの認証キーです。認証不要のローカルAPIでは既定値のままで構いません。")
        settings_layout.addRow("APIキー:", self.api_key_edit)

        self.official_domains_edit = QLineEdit(
            ", ".join(self.default_settings.official_domains), self,
        )
        self.official_domains_edit.setPlaceholderText("例: publisher.example, game.example.jp")
        self.official_domains_edit.setToolTip(
            "公式と確認済みのドメインをカンマ区切りで登録します。URLも入力でき、"
            "登録したドメインとそのサブドメインを一次情報として順位付けします。"
            "検索で見つかっただけの候補は自動登録しません。"
        )
        settings_layout.addRow("登録済み公式ドメイン:", self.official_domains_edit)

        # Numeric Options
        num_row = QHBoxLayout()
        self.concurrency_spin = QSpinBox(self)
        self.concurrency_spin.setRange(1, 32)
        self.concurrency_spin.setValue(self.default_settings.max_concurrency)
        self.concurrency_spin.setToolTip(
            "同時に実行するWeb検索の数です。増やすと速くなる場合がありますが、"
            "検索サービスや本文取得先への負荷も増えます。"
        )
        concurrency_label = QLabel("並列実行数:")
        concurrency_label.setToolTip(self.concurrency_spin.toolTip())
        num_row.addWidget(concurrency_label)
        num_row.addWidget(self.concurrency_spin)

        self.max_queries_spin = QSpinBox(self)
        self.max_queries_spin.setRange(1, 100)
        self.max_queries_spin.setValue(self.default_settings.max_search_queries)
        self.max_queries_spin.setToolTip("初回検索と追加検索を合わせた、調査全体の検索語句数の上限です。")
        queries_label = QLabel("最大検索クエリ数:")
        queries_label.setToolTip(self.max_queries_spin.toolTip())
        num_row.addWidget(queries_label)
        num_row.addWidget(self.max_queries_spin)

        self.max_rounds_spin = QSpinBox(self)
        self.max_rounds_spin.setRange(1, 20)
        self.max_rounds_spin.setValue(self.default_settings.max_search_rounds)
        self.max_rounds_spin.setToolTip(
            "検索して本文を評価する追加調査の繰り返し回数の上限です。"
            "情報不足と判断された場合に次の検索語を作ります。"
        )
        rounds_label = QLabel("最大検索ラウンド数:")
        rounds_label.setToolTip(self.max_rounds_spin.toolTip())
        num_row.addWidget(rounds_label)
        num_row.addWidget(self.max_rounds_spin)
        num_row.addStretch()

        settings_layout.addRow("実行制御:", num_row)

        candidate_row = QHBoxLayout()
        self.results_per_query_spin = QSpinBox(self)
        self.results_per_query_spin.setRange(1, 100)
        self.results_per_query_spin.setValue(self.default_settings.results_per_query)
        self.results_per_query_spin.setToolTip(
            "検索サービスから、検索語句1つにつき何件の候補を受け取るかです。"
            "件数を増やすと、順位付け対象も増えます。"
        )
        results_label = QLabel("検索取得数:")
        results_label.setToolTip(self.results_per_query_spin.toolTip())
        candidate_row.addWidget(results_label)
        candidate_row.addWidget(self.results_per_query_spin)

        self.content_candidates_spin = QSpinBox(self)
        self.content_candidates_spin.setRange(1, 100)
        self.content_candidates_spin.setValue(self.default_settings.content_candidate_results_per_query)
        self.content_candidates_spin.setToolTip(
            "検索候補のうち、ページ本文を取得して品質評価にも読ませる件数です。"
            "最終採用数より多くすると、少し順位が低い候補も本文で比較できます。"
        )
        content_candidates_label = QLabel("本文取得候補数:")
        content_candidates_label.setToolTip(self.content_candidates_spin.toolTip())
        candidate_row.addWidget(content_candidates_label)
        candidate_row.addWidget(self.content_candidates_spin)

        self.final_results_spin = QSpinBox(self)
        self.final_results_spin.setRange(1, 100)
        self.final_results_spin.setValue(self.default_settings.reranked_results_per_query)
        self.final_results_spin.setToolTip(
            "検索語句1つにつき、最終レポートの出典として残す件数です。"
            "品質評価用に読む本文候補数とは別の設定です。"
        )
        final_results_label = QLabel("最終採用数:")
        final_results_label.setToolTip(self.final_results_spin.toolTip())
        candidate_row.addWidget(final_results_label)
        candidate_row.addWidget(self.final_results_spin)
        candidate_row.addStretch()
        settings_layout.addRow("検索候補数:", candidate_row)
        self.results_per_query_spin.valueChanged.connect(self.content_candidates_spin.setMaximum)
        self.content_candidates_spin.valueChanged.connect(self.final_results_spin.setMaximum)

        weights_row = QHBoxLayout()
        self.relevance_weight_spin = QDoubleSpinBox(self)
        self.trust_weight_spin = QDoubleSpinBox(self)
        self.freshness_weight_spin = QDoubleSpinBox(self)
        for spin, value, description in (
            (
                self.relevance_weight_spin,
                self.default_settings.relevance_weight * 100,
                "検索語句との関連度を順位に反映する割合です。",
            ),
            (
                self.trust_weight_spin,
                self.default_settings.trust_weight * 100,
                "情報源の信頼度を順位に反映する割合です。未登録のサイトは中立評価になります。",
            ),
            (
                self.freshness_weight_spin,
                self.default_settings.freshness_weight * 100,
                "ページの新しさを順位に反映する割合です。更新時期が重要な調査で上げます。",
            ),
        ):
            spin.setRange(0, 100)
            spin.setDecimals(0)
            spin.setSingleStep(5)
            spin.setSuffix("%")
            spin.setValue(value)
            spin.setToolTip(description)
        relevance_label = QLabel("関連度")
        relevance_label.setToolTip(self.relevance_weight_spin.toolTip())
        weights_row.addWidget(relevance_label)
        weights_row.addWidget(self.relevance_weight_spin)
        trust_label = QLabel("信頼度")
        trust_label.setToolTip(self.trust_weight_spin.toolTip())
        weights_row.addWidget(trust_label)
        weights_row.addWidget(self.trust_weight_spin)
        freshness_label = QLabel("新鮮度")
        freshness_label.setToolTip(self.freshness_weight_spin.toolTip())
        weights_row.addWidget(freshness_label)
        weights_row.addWidget(self.freshness_weight_spin)
        weights_row.addStretch()
        settings_layout.addRow("ソース順位の重み（合計100%）:", weights_row)

        # Web Fetch Options
        web_row = QHBoxLayout()
        self.fetch_content_cb = QCheckBox("Webページ本文を実際に取得して精読する", self)
        self.fetch_content_cb.setChecked(self.default_settings.fetch_web_content)
        self.fetch_content_cb.setToolTip(
            "検索結果の見出しや概要だけでなく、ページ本文も取得して品質評価とレポート作成に使います。"
        )
        web_row.addWidget(self.fetch_content_cb)

        self.jina_fallback_cb = QCheckBox(
            "失敗時にJina Readerを使う（URLを外部サービスへ送信）", self
        )
        self.jina_fallback_cb.setChecked(self.default_settings.use_fallback_fetcher)
        self.jina_fallback_cb.setToolTip(
            "通常の本文取得に失敗したURLをJina Readerへ送って再取得します。"
            "URLが外部サービスに送信されるため、必要な場合だけ有効にしてください。"
        )

        self.max_content_length_spin = QSpinBox(self)
        self.max_content_length_spin.setRange(500, 10000)
        self.max_content_length_spin.setSingleStep(500)
        self.max_content_length_spin.setValue(self.default_settings.max_content_length)
        self.max_content_length_spin.setToolTip(
            "1ページから取得する本文の最大文字数です。長くすると詳細が残りますが、"
            "LLMへ渡す文章量も増えます。全ページ合計には別途上限があります。"
        )
        content_length_label = QLabel("本文最大文字数:")
        content_length_label.setToolTip(self.max_content_length_spin.toolTip())
        web_row.addWidget(content_length_label)
        web_row.addWidget(self.max_content_length_spin)
        web_row.addStretch()

        settings_layout.addRow("Webアクセス:", web_row)
        settings_layout.addRow("取得失敗時:", self.jina_fallback_cb)

        self.fetcher_combo = QComboBox(self)
        self.fetcher_combo.addItem("標準取得 (httpx / lxml)", "builtin")
        self.fetcher_combo.addItem("Firecrawl セルフホスト", "firecrawl")
        self.fetcher_combo.setToolTip(
            "Web本文の取得方法です。標準取得はアプリから直接ページを読み込み、"
            "Firecrawlは指定したセルフホストAPIを使います。"
        )
        fetcher_index = self.fetcher_combo.findData(self.default_settings.content_fetcher)
        if fetcher_index >= 0:
            self.fetcher_combo.setCurrentIndex(fetcher_index)
        settings_layout.addRow("本文取得エンジン:", self.fetcher_combo)

        self.firecrawl_url_edit = QLineEdit(self.default_settings.firecrawl_api_url, self)
        self.firecrawl_url_edit.setPlaceholderText("http://localhost:3002")
        self.firecrawl_url_edit.setToolTip("Firecrawlを選んだ場合に接続するAPIのURLです。例: http://localhost:3002")
        settings_layout.addRow("Firecrawl API URL:", self.firecrawl_url_edit)

        self.firecrawl_api_key_edit = QLineEdit(self.default_settings.firecrawl_api_key, self)
        self.firecrawl_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.firecrawl_api_key_edit.setPlaceholderText("不要なセルフホスト構成では空欄")
        self.firecrawl_api_key_edit.setToolTip("APIキー認証を設定したFirecrawlサーバーで使います。不要なら空欄にします。")
        settings_layout.addRow("Firecrawl APIキー (任意):", self.firecrawl_api_key_edit)
        main_layout.addWidget(self.settings_group)

        # Progress / Status Bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("待機中", self)
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)

        # Tabs for Results, Sources, and Logs
        self.tabs = QTabWidget(self)

        # Tab 1: Summary Report
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(8, 8, 8, 8)

        self.summary_browser = QTextBrowser(self)
        self.summary_browser.setOpenExternalLinks(True)
        self.summary_browser.setPlaceholderText("調査が完了すると、ここにレポートが表示されます。")
        summary_layout.addWidget(self.summary_browser)

        summary_actions = QHBoxLayout()
        self.copy_summary_btn = QPushButton("📋 レポートをコピー", self)
        self.copy_summary_btn.clicked.connect(self._copy_summary_to_clipboard)
        summary_actions.addWidget(self.copy_summary_btn)

        self.save_summary_btn = QPushButton("💾 テキストとして保存", self)
        self.save_summary_btn.clicked.connect(self._save_summary_to_file)
        summary_actions.addWidget(self.save_summary_btn)
        summary_actions.addStretch()
        summary_layout.addLayout(summary_actions)

        self.tabs.addTab(summary_tab, "📄 調査レポート")

        # Tab 2: Sources Table
        sources_tab = QWidget()
        sources_layout = QVBoxLayout(sources_tab)
        sources_layout.setContentsMargins(8, 8, 8, 8)

        self.sources_table = QTableWidget(0, 6, self)
        self.sources_table.setHorizontalHeaderLabels(["#", "クエリ", "タイトル", "スコア", "本文取得", "URL (クリックで開く)"])
        self.sources_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.sources_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.sources_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.sources_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.sources_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.sources_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.sources_table.cellDoubleClicked.connect(self._on_source_cell_clicked)
        sources_layout.addWidget(self.sources_table)

        sources_actions = QHBoxLayout()
        self.copy_sources_btn = QPushButton("📋 ソース一覧をコピー", self)
        self.copy_sources_btn.clicked.connect(self._copy_sources_to_clipboard)
        sources_actions.addWidget(self.copy_sources_btn)
        sources_actions.addStretch()
        sources_layout.addLayout(sources_actions)

        self.tabs.addTab(sources_tab, "🔗 参照ソース一覧")

        # Tab 3: Execution Logs
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(8, 8, 8, 8)

        self.log_edit = QPlainTextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("計画・検索・ソース選定・品質評価・次の処理を選んだ理由がここに表示されます...")
        log_layout.addWidget(self.log_edit)

        log_actions = QHBoxLayout()
        self.clear_log_btn = QPushButton("🗑️ ログをクリア", self)
        self.clear_log_btn.clicked.connect(self.log_edit.clear)
        log_actions.addWidget(self.clear_log_btn)
        log_actions.addStretch()
        log_layout.addLayout(log_actions)

        self.tabs.addTab(log_tab, "🔎 詳細実行ログ")

        main_layout.addWidget(self.tabs, stretch=1)

    def _toggle_settings(self, checked: bool) -> None:
        self.settings_group.setVisible(checked)

    def _on_refresh_models_clicked(self) -> None:
        self._refresh_model_list(show_feedback=True)

    def _refresh_model_list(self, show_feedback: bool = False) -> None:
        """Fetch available models from the configured base_url (only on startup or button click)."""
        base_url = self.base_url_edit.text().strip() or self.default_settings.base_url
        api_key = self.api_key_edit.text().strip()
        current_selection = self.model_combo.currentText().strip()

        models = fetch_available_models(base_url=base_url, api_key=api_key, timeout=3.0)
        if models:
            self.model_combo.clear()
            for m in models:
                self.model_combo.addItem(m)
            if current_selection and current_selection in models:
                self.model_combo.setCurrentText(current_selection)
            elif current_selection:
                self.model_combo.addItem(current_selection)
                self.model_combo.setCurrentText(current_selection)
            else:
                self.model_combo.setCurrentIndex(0)
            if show_feedback:
                QMessageBox.information(
                    self, "モデル更新完了", f"LM Studio から {len(models)} 件のモデルを取得しました:\n\n" + "\n".join(models[:10])
                )
        else:
            if show_feedback:
                QMessageBox.warning(
                    self,
                    "モデル取得不可",
                    f"エンドポイント ({base_url}) からモデル一覧を取得できませんでした。\nLM Studio が起動しているか確認してください。",
                )

    def _save_config(self) -> None:
        data = {
            "model": self.model_combo.currentText().strip(),
            "base_url": self.base_url_edit.text().strip(),
            "max_concurrency": self.concurrency_spin.value(),
            "max_search_queries": self.max_queries_spin.value(),
            "max_search_rounds": self.max_rounds_spin.value(),
            "results_per_query": self.results_per_query_spin.value(),
            "content_candidate_results_per_query": self.content_candidates_spin.value(),
            "reranked_results_per_query": self.final_results_spin.value(),
            "fetch_web_content": self.fetch_content_cb.isChecked(),
            "use_fallback_fetcher": self.jina_fallback_cb.isChecked(),
            "content_fetcher": self.fetcher_combo.currentData(),
            "firecrawl_api_url": self.firecrawl_url_edit.text().strip(),
            "max_content_length": self.max_content_length_spin.value(),
            "relevance_weight": self.relevance_weight_spin.value() / 100,
            "trust_weight": self.trust_weight_spin.value() / 100,
            "freshness_weight": self.freshness_weight_spin.value() / 100,
            "official_domains": list(normalize_official_domains(self.official_domains_edit.text())),
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def closeEvent(self, event: Any) -> None:
        self._save_config()
        super().closeEvent(event)

    def _on_start_clicked(self) -> None:
        query = self.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "入力エラー", "調査テーマまたは質問を入力してください。")
            return

        weight_total = (
            self.relevance_weight_spin.value()
            + self.trust_weight_spin.value()
            + self.freshness_weight_spin.value()
        )
        if weight_total != 100:
            QMessageBox.warning(self, "設定エラー", "ソース順位の重みの合計を100%にしてください。")
            return

        try:
            official_domains = normalize_official_domains(self.official_domains_edit.text())
        except ValueError as error:
            QMessageBox.warning(self, "設定エラー", str(error))
            return

        self._save_config()

        max_concurrency = self.concurrency_spin.value()
        settings = Settings(
            model=self.model_combo.currentText().strip() or self.default_settings.model,
            base_url=self.base_url_edit.text().strip() or self.default_settings.base_url,
            api_key=self.api_key_edit.text().strip() or self.default_settings.api_key,
            max_concurrency=max_concurrency,
            max_search_queries=self.max_queries_spin.value(),
            max_search_rounds=self.max_rounds_spin.value(),
            results_per_query=self.results_per_query_spin.value(),
            content_candidate_results_per_query=self.content_candidates_spin.value(),
            reranked_results_per_query=self.final_results_spin.value(),
            fetch_web_content=self.fetch_content_cb.isChecked(),
            use_fallback_fetcher=self.jina_fallback_cb.isChecked(),
            content_fetcher=self.fetcher_combo.currentData(),
            firecrawl_api_url=self.firecrawl_url_edit.text().strip() or "http://localhost:3002",
            firecrawl_api_key=(
                self.firecrawl_api_key_edit.text().strip()
                or self.default_settings.firecrawl_api_key
            ),
            max_content_length=self.max_content_length_spin.value(),
            relevance_weight=self.relevance_weight_spin.value() / 100,
            trust_weight=self.trust_weight_spin.value() / 100,
            freshness_weight=self.freshness_weight_spin.value() / 100,
            official_domains=official_domains,
        )

        # UI state during run
        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳ 調査中...")
        self.status_label.setText("調査を実行しています...")
        self.progress_bar.setRange(0, 0)  # Marquee/busy mode
        self.tabs.setCurrentIndex(2)  # Switch to log tab during execution

        self._log(f"調査を開始します: 「{query}」")

        # Launch Worker Thread
        self.worker = ResearchWorker(
            query=query,
            settings=settings,
            max_concurrency=max_concurrency,
            parent=self,
        )
        self.worker.log_signal.connect(self._log)
        self.worker.result_signal.connect(self._on_result_received)
        self.worker.error_signal.connect(self._on_error_received)
        self.worker.finished_signal.connect(self._on_worker_finished)
        self.worker.start()

    def _on_result_received(self, result: dict[str, Any]) -> None:
        summary = result.get("summary", "")
        results = result.get("results", [])
        self.last_results = results

        # Display Summary
        self.summary_browser.setMarkdown(summary)

        # Display Sources Table
        self.sources_table.setRowCount(0)
        for i, item in enumerate(results, start=1):
            row = self.sources_table.rowCount()
            self.sources_table.insertRow(row)

            score_text = f"{item.relevance_score:.3f}" if item.relevance_score is not None else "-"
            content_status = "✅ 取得済" if item.content else "➖ スニペットのみ"
            self.sources_table.setItem(row, 0, QTableWidgetItem(str(i)))
            self.sources_table.setItem(row, 1, QTableWidgetItem(item.query))
            self.sources_table.setItem(row, 2, QTableWidgetItem(item.title))
            self.sources_table.setItem(row, 3, QTableWidgetItem(score_text))
            self.sources_table.setItem(row, 4, QTableWidgetItem(content_status))
            self.sources_table.setItem(row, 5, QTableWidgetItem(item.url))

        # Switch to summary tab
        self.tabs.setCurrentIndex(0)
        self.status_label.setText("✅ 調査完了")

    def _on_error_received(self, error_msg: str) -> None:
        self.status_label.setText("❌ エラーが発生しました")
        QMessageBox.critical(self, "エラー", f"調査の実行中にエラーが発生しました:\n\n{error_msg}")

    def _on_worker_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.start_btn.setText("🚀 調査を開始する")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.worker = None

    def _log(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{timestamp}] {message}")

    def _on_source_cell_clicked(self, row: int, column: int) -> None:
        url_item = self.sources_table.item(row, 5)
        if url_item and url_item.text():
            QDesktopServices.openUrl(QUrl(url_item.text()))

    def _copy_summary_to_clipboard(self) -> None:
        text = self.summary_browser.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "コピー完了", "レポートをクリップボードにコピーしました。")

    def _save_summary_to_file(self) -> None:
        text = self.summary_browser.toPlainText()
        if not text:
            QMessageBox.warning(self, "保存不可", "保存するレポートがありません。")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "レポートを保存", "research_report.md", "Markdown Files (*.md);;Text Files (*.txt)"
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                QMessageBox.information(self, "保存完了", f"レポートを保存しました:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "保存エラー", f"ファイルの保存に失敗しました:\n{e}")

    def _copy_sources_to_clipboard(self) -> None:
        if not self.last_results:
            QMessageBox.warning(self, "コピー不可", "コピーするソース情報がありません。")
            return
        lines = ["# 参照ソース一覧\n"]
        for i, item in enumerate(self.last_results, start=1):
            score_text = f" (スコア: {item.relevance_score:.3f})" if item.relevance_score is not None else ""
            lines.append(f"{i}. [{item.title}]({item.url}){score_text}\n   クエリ: {item.query}\n")
        QApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(self, "コピー完了", "ソース一覧をクリップボードにコピーしました。")

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet("""
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
        """)
