# Search Orchestrator

ローカルの OpenAI 互換 API、DuckDuckGo 検索、および **Webサイト本文の自動取得（Web Fetching / Scraping）** を使い、調査計画、Web検索、再ランキング、Web本文抽出、品質評価、追加検索、高品質な回答生成を自律的に行う Python アプリケーションです。

## 主な特徴
- **自律的リサーチワークフロー (LangGraph)**: 質問から観点ごとの調査計画を立案・並列検索
- **CrossEncoder による再ランキング**: 検索結果を高精度にスコアリングして関連度の高いソースを選定
- **Webサイト本文の自動実アクセス (Web Fetching)**: 検索スニペットだけでなく、Webページ本文のテキストを安全に取得・クリーンアップし、詳細で具体的な根拠に基づいたレポートを生成
- **品質評価と自律追加検索**: 不足している観点や根拠の弱さを自動評価し、必要に応じて追加検索
- **デスクトップ GUI (PySide6) & CLI**: 直感的なGUIアプリと柔軟なコマンドライン実行に対応

## セットアップ

Python 3.11 から 3.13 を用意してから、次を実行します。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`sentence-transformers` は初回実行時に再ランキングモデルをダウンロードします。ローカル API の既定接続先は `http://localhost:1234/v1` です。必要に応じて次の環境変数を設定できます。

```powershell
$env:SEARCH_MODEL = "your-model-name"
$env:SEARCH_BASE_URL = "http://localhost:1234/v1"
$env:SEARCH_API_KEY = "your-api-key"
```

## 実行とテスト

### 1. GUI (デスクトップアプリ) での実行
```powershell
python gui/run.py
```
> PySide6 による GUI ウィンドウが起動します。クエリ入力、設定変更（モデル名、URL、並列数、Web本文取得のON/OFF、本文文字数など）、調査レポートの閲覧・コピー・ファイル保存、参照ソース一覧のテーブル表示（クリックでブラウザ表示、本文取得状況の確認）、リアルタイム実行ログの確認が可能です。

### 2. CLI (コマンドライン) での実行
```powershell
# デフォルトのクエリで実行
python main.py

# 任意のクエリを指定して実行
python main.py "最新の生成AIトレンドについて調査してください"

# オプション指定（モデル名、エンドポイント、Web取得無効化などの指定）
python main.py --model "my-model" --base-url "http://localhost:1234/v1" "調査テーマ"
python main.py --no-fetch "スニペットのみで高速実行したい場合"
```

### 3. テストの実行
```powershell
pytest -v
```

Windows では出力を UTF-8 に固定しているため、ローカルモデルが日本語以外の Unicode 文字を含む回答を返しても表示できます。

## 構成

- `search_orchestrator.py`: 内部コアロジック（LangGraph ワークフロー、WebContentFetcher、再ランキング、評価・分析）
- `gui/`: PySide6 を用いたデスクトップ GUI パッケージ
  - `main_window.py`: メインウィンドウ UI、設定パネル、参照ソーステーブル、スタイル定義
  - `worker.py`: QThread による非同期・非ブロッキング実行ワーカー
  - `run.py`: GUI アプリケーションの起動エントリポイント
- `main.py`: CLI 実行入口
- `main1.1.py`: 既存の実行コマンド向け互換入口
- `tests/`: 外部サービス不要のユニットテスト（Webフェッチ、コアロジックおよび GUI 初期化テスト）

