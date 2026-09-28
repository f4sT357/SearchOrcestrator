# Search Orchestrator

ローカルの OpenAI 互換 API、DuckDuckGo 検索、および **Webサイト本文の自動取得（Web Fetching / Scraping）** を使い、調査計画、Web検索、再ランキング、Web本文抽出、品質評価、追加検索、高品質な回答生成を自律的に行う Python アプリケーションです。

## 主な特徴
- **自律的リサーチワークフロー (LangGraph)**: 最初に依頼内容を整理して共有ボードへ記録し、その理解に基づいて調査計画を立案・並列検索
- **CrossEncoder による再ランキング**: 検索結果を高精度にスコアリングして関連度の高いソースを選定
- **Webサイト本文の自動実アクセス (Web Fetching)**: 検索スニペットだけでなく、Webページ本文のテキストを安全に取得・クリーンアップし、詳細で具体的な根拠に基づいたレポートを生成
- **品質評価と自律追加検索**: 不足している観点や根拠の弱さを自動評価し、必要に応じて追加検索
- **公式ソース専用レーン**: 通常検索とは独立したクエリで公式サイト・運営告知を探し、本文と発行元を品質評価へ渡す
- **調査回限りの共有ボード**: 調査開始時に依頼内容の理解を記録し、その後も検索ラウンド間の確認済み事項・未確認点を投稿形式で追記します。GUIでリアルタイムに確認でき、矛盾や訂正も過去の投稿を消さずに保持します（最大5000文字）。調査終了後は保存しません
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
# 合成スコアの重み（合計1.0）
$env:SEARCH_WEIGHT_RELEVANCE = "0.50"
$env:SEARCH_WEIGHT_TRUST = "0.30"
$env:SEARCH_WEIGHT_FRESHNESS = "0.20"
$env:SEARCH_OFFICIAL_DOMAINS = "publisher.example,game.example.jp"
$env:FIRECRAWL_API_URL = "http://localhost:3002"
# APIキーが必要な構成の場合のみ設定
$env:FIRECRAWL_API_KEY = "your-firecrawl-key"
```

設定は既定値、`config.json`、環境変数、CLI オプションの順で上書きされます。GUI と CLI は同じ `config.json` を読み込みます。API キーは `SEARCH_API_KEY` で渡す方法を推奨します。GUI は入力した API キーを設定ファイルへ保存しません。

## 実行とテスト

### 1. GUI (デスクトップアプリ) での実行
```powershell
python gui/run.py
```
> PySide6 による GUI ウィンドウが起動します。クエリ入力、設定変更（モデル名、URL、並列数、Web本文取得のON/OFF、本文文字数、関連度・信頼度・新鮮度の重みなど）、調査レポートの閲覧・コピー・ファイル保存、参照ソース一覧のテーブル表示（クリックでブラウザ表示、本文取得状況の確認）、リアルタイム実行ログの確認が可能です。スコアの重みは合計100%に設定してください。

### 2. CLI (コマンドライン) での実行
```powershell
# デフォルトのクエリで実行
python main.py

# 任意のクエリを指定して実行
python main.py "最新の生成AIトレンドについて調査してください"

# オプション指定（モデル名、エンドポイント、Web取得無効化などの指定）
python main.py --model "my-model" --base-url "http://localhost:1234/v1" "調査テーマ"
python main.py --no-fetch "スニペットのみで高速実行したい場合"
python main.py --weight-relevance 0.4 --weight-trust 0.4 --weight-freshness 0.2 "調査テーマ"
python main.py --max-search-queries 10 --max-search-rounds 2 "調査テーマ"
python main.py --results-per-query 8 --content-candidate-results-per-query 5 --reranked-results-per-query 3 "調査テーマ"
python main.py --content-fetcher firecrawl --firecrawl-url "http://localhost:3002" "調査テーマ"
```

本文取得に失敗した場合、Jina Reader へ URL を送信して再取得する機能は既定で無効です。利用する場合は GUI の「Webアクセス」設定、または CLI の `--jina-fallback` で明示的に有効化できます。

### Firecrawl セルフホストを本文取得に使う

Firecrawl は同梱の `firecrawl-start.bat` を実行すると起動できます。初回は公式リポジトリの固定バージョンを取得し、必要な `.env` を作って Docker Compose で起動します。停止は `firecrawl-stop.bat` です。どちらもプロジェクトのルートで実行してください。Docker Desktop が停止中なら起動を試みます。

API は `http://localhost:3002` で、この PC からだけ接続できます。GUI の詳細設定で本文取得エンジンを「Firecrawl セルフホスト」に変更し、API URL を指定します。API キーは `FIRECRAWL_API_KEY` 環境変数から読み込み、GUI の設定ファイルには保存しません。CLI では `--content-fetcher firecrawl` と `--firecrawl-url` を指定できます。

停止スクリプトは Firecrawl のコンテナを停止します。Docker Desktop 自体は他のコンテナに影響しないよう起動したままにします。

アプリは Firecrawl の `POST /v2/scrape` に Markdown 取得を依頼します。セルフホスト手順とサービス構成は [Firecrawl の公式ガイド](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md) を参照してください。

品質評価では一次情報を優先します。一次情報を確認できないテーマでは、二次情報という理由だけで根拠を退けず、独立した発行元による複数の高品質な情報源、記事間の転載関係、主張の一致や矛盾を照合します。最終候補では同一ホストの結果に偏らないよう異なるホストを優先します。登録リスト外のドメインは低品質とはみなさず、中立の信頼度スコアで順位付けして内容評価に委ねます。重み設定は関連度・信頼度・新鮮度の順で、各値は0から1、合計は1にしてください。

調査開始時にモデルが入力内容の対象・目的・条件・曖昧な点を整理し、調査ボードへ記録してから計画を立てます。計画では通常の検索と別に、質問に応じた最大5件の公式ソース専用クエリを作成し、各クエリで複数ページの本文を取得して品質評価へ渡します。通常の追加検索でも公式・告知系のクエリで見つけた候補を専用プールへ合流させ、評価と最終回答の出典選択で候補が埋もれにくいようにします。公式検索で見つかったドメインは候補として扱い、自動では信頼登録しません。ユーザーが確認したドメインはGUIの「登録済み公式ドメイン」、`config.json` の `official_domains` 配列、または `SEARCH_OFFICIAL_DOMAINS` で登録できます。登録したドメインとそのサブドメインは一次情報として順位付けします。品質評価が作る共有ボードは新しい投稿だけを追記し、現在の調査状態にだけ保持します。ファイルや次回の調査には保存しません。

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

## 今後の改善予定
- **userが決めた回数繰り返す　だけではなく目的が達成されるまで繰り返す　という方式への任意切り替え**
- **ログの詳細化(コンテキスト使用量可視化)**
- **詳細設定の初期設定をより汎用的な数値に変更**
- **調査ボードの情報を自動で整理する仕組み、もしくは簡易メモリエージェントの実装**
- **リサーチ終了後のセルフ評価とリトライ機能**
- **推奨モデルの選定**
- **用途別モデルルーティング実装(軽量化のため)**

