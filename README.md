# Search Orchestrator

ローカルの OpenAI 互換 API と DuckDuckGo 検索を使い、調査計画、検索、再ランキング、品質評価、追加検索、回答生成を行う Python アプリケーションです。

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

```powershell
python main.py
pytest -q
```

Windows では出力を UTF-8 に固定しているため、ローカルモデルが日本語以外の Unicode 文字を含む回答を返しても表示できます。

## 構成

- `search_orchestrator.py`: ワークフロー、状態、検索結果の整形、再ランキング
- `main.py`: 実行入口
- `main1.1.py`: 既存の実行コマンド向け互換入口
- `tests/`: 外部サービス不要のユニットテスト

検索結果は LangGraph の累積状態です。追加検索ノードは新規取得分だけを返すため、既存の結果を重複して累積しません。再ランキングは CPU 負荷が高いため、既定では最大6クエリ・各5件を取得し、並列実行数を2に制限しています。
