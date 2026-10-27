# Search Orchestrator 引き継ぎメモ

更新日: 2026-10-27  
対象ブランチ: `main`  
対象コミット: `3931e6b`（`origin/main` へ反映済み）

## 現在の状態

保守性レビューを行い、設定、Web取得、データモデル、評価用ヘルパー、GUIスタイルを既存の大規模モジュールから分離した。外部から見えるインポート経路と主要な動作は維持している。

- 最新テスト結果: `95 passed in 18.63s`
- 作業時の Python: `.venv` 内の Python
- ライブのLLM、検索サービス、Firecrawlを使う結合試験とGUIの目視確認は未実施
- `config.local.json` と `.cache/` はGit管理外

## 今回のリファクタリング

### `9975eef refactor configuration and web fetching modules`

- `Settings` と設定値検証を `app_settings.py` へ分離
- Web本文取得、URL安全性検証、キャッシュ処理を `web_fetchers.py` へ分離
- 検索件数やタイムアウトなどの設定値検証を追加
- 既存利用者向けに `search_orchestrator.py` からの再エクスポートを維持

### `3931e6b refactor workflow and GUI responsibilities`

- ワークフロー用モデルを `research_models.py` へ分離
- 評価、JSON解析、根拠選択用の純粋関数を `evaluation_helpers.py` へ分離
- planner、evaluate、update_board、analyze の処理を小さな補助メソッドへ分割
- GUIのスタイルを `gui/styles.py` へ分離
- `MainWindow._setup_ui` と設定パネル、結果タブの構築を小さなメソッドへ分割
- 未使用のインポートと古い属性を整理

リモートのREADME更新を取り込んでからリベースしたため、作業途中に使われていたコミットIDとは異なる。上記が現在の正式なID。

## 現在の責務分担

| ファイル | 主な責務 |
| --- | --- |
| `app_settings.py` | 設定の読み込み、既定値、環境変数、値の検証 |
| `research_models.py` | `SearchTask`、`SearchPlan`、`Evaluation`、`SearchResult`、`State`、`BoardContext` |
| `web_fetchers.py` | 通常HTTP、Jina Reader、Firecrawl、URL検証、本文キャッシュ |
| `evaluation_helpers.py` | 評価結果の正規化、JSON解析、根拠の整形・選択、追加検索語の検証 |
| `source_scoring.py` | 関連度、信頼度、鮮度と複合スコア |
| `source_assessment.py` | ソース評価と評価キャッシュ |
| `search_orchestrator.py` | LangGraphへの接続、各ノードのオーケストレーション、公開互換レイヤー |
| `gui/main_window.py` | メイン画面、設定入力、実行開始、結果表示 |
| `gui/worker.py` | GUIから調査を非同期実行するワーカー |
| `gui/styles.py` | Qtスタイルシート |
| `main.py` | CLIの引数処理と起動 |

## 互換性を壊しやすい箇所

- `Settings`、検索モデル、Fetcher群は現在も `search_orchestrator` からインポートできる。利用側とテストがあるため、移動時は再エクスポートを残す。
- GUIテストは `gui.main_window.preload_reranker_background` を差し替える。この名前は未使用に見えても削除しない。
- `app_settings.py` の設定ファイル探索はモジュール位置に依存する。テストでは `app_settings.__file__` を差し替えるケースがある。
- 新しいトップレベルモジュールを追加した場合は、配布に含めるため `pyproject.toml` の `tool.setuptools.py-modules` も更新する。
- APIキーは環境変数で渡す。`config.local.json` に秘密情報を追加したり、Gitへ登録したりしない。
- Webアクセス、LLM応答、再ランキングモデルのロードは遅く失敗し得る。単体テストでは実サービスを呼ばず、既存の差し替え方を維持する。

## 残っているスパゲッティ化リスク

### 1. `ResearchWorkflowNodes` の責務集中

`search_orchestrator.py` は1,762行、そのうち `ResearchWorkflowNodes` は1,283行ある。各メソッドは短くなったが、LLM呼び出し、検索、評価、ボード更新、追加検索、最終回答生成を1クラスが所有している。

次は、動作を変えずに以下の協調オブジェクトへ段階的に委譲するのが安全。

1. LLM呼び出し、再試行、空応答処理を担当するランタイム
2. 通常検索・公式検索・追加検索を担当する検索サービス
3. 品質評価とフォローアップ生成を担当する評価サービス
4. 根拠選択、引用検証、最終回答生成を担当する回答サービス

`ResearchWorkflowNodes` 自体はLangGraphのノード名と `State` の入出力を保つ薄いアダプターにする。最初から継承階層や多数のファイルへ分けず、1サービスずつ抽出して各段階でテストを通すこと。

### 2. `MainWindow` の責務集中

`gui/main_window.py` は856行、`MainWindow` は795行ある。UI構築メソッドは分割済みだが、設定フォーム、実行制御、結果表示、ファイル保存を1クラスが管理している。

次の候補は `SettingsPanel`、`ResultsTabs`、実行状態を扱うコントローラーの順。Qtのsignal/slot境界が変わるため、まずウィジェット生成だけを子クラスへ移し、処理ロジックの移動は別コミットにする。

### 3. 障害理由が失われる箇所

`fetch_available_models()` は例外をまとめて捕捉し、空の一覧を返す。GUI上では「モデルが0件」と「接続・認証・JSON解析の失敗」を区別できない。利用者向けのエラー表示とログを追加する場合は、既存の戻り値を保ちながら詳細結果を返す別APIを追加すると互換性を保ちやすい。

### 4. 文書と実装構成の同期

READMEの構成説明は今回追加した `app_settings.py`、`web_fetchers.py`、`research_models.py`、`evaluation_helpers.py`、`gui/styles.py` をまだ十分に説明していない。次の文書更新時に、この引き継ぎメモの責務表をREADMEへ反映する。

## 推奨する次の作業順

1. `ResearchWorkflowNodes.__init__` の依存関係を小さなデータクラスへまとめる。
2. LLMランタイムを抽出し、既存ノードから委譲する。
3. 検索サービス、評価サービス、回答サービスを1つずつ抽出する。
4. 各抽出後に全テストを実行し、1責務ごとにコミットする。
5. GUIを `SettingsPanel` と `ResultsTabs` に分ける。
6. モデル一覧取得のエラー可視化と構造化ログを追加する。
7. READMEの構成説明を更新し、GUIを目視確認する。

## 開発・確認コマンド

```powershell
# 依存関係
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

# 全テスト
.\.venv\Scripts\python.exe -m pytest -q

# GUI
.\.venv\Scripts\python.exe gui/run.py

# CLI
.\.venv\Scripts\python.exe main.py "調査したい内容"
```

ライブ確認では、OpenAI互換APIのURLとモデルを設定してから、短いクエリでCLIとGUIをそれぞれ1回実行する。Web本文取得を変更した場合は通常HTTP、Jina fallback、Firecrawlのうち変更対象だけを個別に確認する。

## 完了判定の目安

- `pytest -q` が全件成功する
- `search_orchestrator` からの既存インポートが成功する
- CLIとGUIが起動し、設定保存と結果表示が動く
- URL、引用、評価結果が既存形式のまま保持される
- `config.local.json`、APIキー、キャッシュがコミットに含まれない
- 1コミットに複数の責務変更を混ぜない
