# Search Orchestrator

ローカルの OpenAI 互換 API、DuckDuckGo 検索、および **Webサイト本文の自動取得（Web Fetching / Scraping）** を使い、調査計画、Web検索、再ランキング、Web本文抽出、品質評価、追加検索、高品質な回答生成を自律的に行う Python アプリケーションです。

## ブランチについて

以下は2026年10月27日時点の一覧です。最新の状態は [GitHubのブランチ一覧](https://github.com/f4sT357/SearchOrcestrator/branches) を確認してください。

| ブランチ | 内容・状態 |
| --- | --- |
| [main](https://github.com/f4sT357/SearchOrcestrator/tree/main) | 基準となるブランチ。 |
| [fix/research-workflow-maintainability](https://github.com/f4sT357/SearchOrcestrator/tree/fix/research-workflow-maintainability) | 調査ワークフロー改善の作業ブランチ。コードレビュー、全90テスト、配布物のimport検証を完了し、mainへ統合済み。 |
| [auto/improve-search-workflow-20260914](https://github.com/f4sT357/SearchOrcestrator/tree/auto/improve-search-workflow-20260914) | 検索ワークフロー改善用のブランチ。mainとの差分と現在の利用状況は要確認。 |

変更ブランチをmainに取り込む前に、差分を確認し、依存関係を用意して `pytest -v` を実行してください。GUIや実際の検索に影響する変更は、該当機能の動作も確認します。

## 主な特徴
- **自律的リサーチワークフロー (LangGraph)**: 最初に依頼内容を整理して共有ボードへ記録し、その理解に基づいて調査計画を立案・並列検索
- **CrossEncoder による再ランキング**: 検索結果を高精度にスコアリングして関連度の高いソースを選定
- **Webサイト本文の自動実アクセス (Web Fetching)**: 検索スニペットだけでなく、Webページ本文のテキストを安全に取得・クリーンアップし、詳細で具体的な根拠に基づいたレポートを生成
- **品質評価と自律追加検索**: 不足している観点や根拠の弱さを自動評価し、必要に応じて追加検索
- **公式ソース専用レーン**: 通常検索とは独立したクエリで公式サイト・運営告知を探し、本文と発行元を品質評価へ渡す
- **調査回限りのメモキーパー**: 同じローカルLLMが、確認済み事項・未確認点・URL採否・次の行動だけを短いJSONメモに圧縮します。メインの計画・品質評価・最終回答とはコンテキストを分離し、GUIには更新済みメモを表示します。調査終了後は保存しません
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
$env:SEARCH_TRUSTED_DOMAINS = "specialist.example,news.example.jp"
$env:FIRECRAWL_API_URL = "http://localhost:3002"
# APIキーが必要な構成の場合のみ設定
$env:FIRECRAWL_API_KEY = "your-firecrawl-key"
```

設定は既定値、共有する `config.json`、git無視の `config.local.json`、環境変数、CLI オプションの順で上書きされます。モデル名など環境固有の値は `SEARCH_MODEL`、`config.local.json`、またはGUIで指定してください。GUIは `config.local.json` へ保存するため、ローカルのモデル選択を共有設定へ書き込みません。APIキーは `SEARCH_API_KEY` で渡す方法を推奨し、GUIは入力したAPIキーを設定ファイルへ保存しません。

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

内蔵の本文取得では、同一URLを実行内・実行間で再利用します。既定では `.cache/web-content` にURLをSHA-256でハッシュ化したキャッシュを保存し、6時間以内は通信せず再利用します。期限後は `ETag` / `Last-Modified` があれば条件付きGETで再検証し、`304 Not Modified` では本文を再利用します。`Cache-Control: no-store` は保存せず、サーバー指定の `max-age` を尊重します。ホストごとの取得は1件ずつにし、`429` の `Retry-After` は最大30秒待機して1回だけ再試行します。`config.json` または `config.local.json` では `web_cache_enabled`、`web_cache_ttl_seconds`、`web_cache_dir`、`web_host_min_interval_seconds`、`web_max_retry_after_seconds` を変更できます。
### Firecrawl セルフホストを本文取得に使う

Firecrawl は同梱の `firecrawl-start.bat` を実行すると起動できます。初回は公式リポジトリの固定バージョンを取得し、必要な `.env` を作って Docker Compose で起動します。停止は `firecrawl-stop.bat` です。どちらもプロジェクトのルートで実行してください。Docker Desktop が停止中なら起動を試みます。→まだ同梱していません

API は `http://localhost:3002` で、この PC からだけ接続できます。GUI の詳細設定で本文取得エンジンを「Firecrawl セルフホスト」に変更し、API URL を指定します。API キーは `FIRECRAWL_API_KEY` 環境変数から読み込み、GUI の設定ファイルには保存しません。CLI では `--content-fetcher firecrawl` と `--firecrawl-url` を指定できます。

停止スクリプトは Firecrawl のコンテナを停止します。Docker Desktop 自体は他のコンテナに影響しないよう起動したままにします。

アプリは Firecrawl の `POST /v2/scrape` に Markdown 取得を依頼します。セルフホスト手順とサービス構成は [Firecrawl の公式ガイド](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md) を参照してください。

品質評価では一次情報を優先します。一次情報を確認できないテーマでは、二次情報という理由だけで根拠を退けず、独立した発行元による複数の高品質な情報源、記事間の転載関係、主張の一致や矛盾を照合します。最終候補では同一ホストの結果に偏らないよう異なるホストを優先します。登録リスト外のドメインは低品質とはみなさず、中立の信頼度スコアで順位付けして内容評価に委ねます。重み設定は関連度・信頼度・新鮮度の順で、各値は0から1、合計は1にしてください。

調査開始時は入力原文をそのまま調査ボードへ記録してから計画を立てるため、入力整理のLLM呼び出しを行いません。計画では通常の検索と別に、質問に応じた最大5件の公式ソース専用クエリを作成し、各クエリで複数ページの本文を取得して品質評価へ渡します。通常の追加検索でも公式・告知系のクエリで見つけた候補を専用プールへ合流させ、評価と最終回答の出典選択で候補が埋もれにくいようにします。公式検索で見つかったドメインは候補として扱い、自動では信頼登録しません。ユーザーが確認したドメインはGUIの「登録済み公式ドメイン」、`config.json` の `official_domains` 配列、または `SEARCH_OFFICIAL_DOMAINS` で登録できます。登録したドメインとそのサブドメインは一次情報として順位付けします。公式ではないが信頼できる報道・専門サイトはGUIの「信頼済み二次ソース」、`trusted_domains`、または `SEARCH_TRUSTED_DOMAINS` に登録でき、公式より低い信頼度0.80で加点します。内蔵の出版社・企業ドメイン一覧による自動的な公式認定や信頼度の上乗せは行いません。品質評価の後には、同じローカルLLMをメモキーパーとして逐次呼び出し、前回の短いメモと今回の評価差分だけから短いJSONメモを更新します。計画には渡さず、品質評価・最終回答には選定済みURLに紐付く事実・採否・未解決事項だけを最大600文字の研究ブリーフとして渡します。URLに紐付かない自由文や未知のURLは破棄し、研究ブリーフはファイルや次回の調査には保存しません。`config.json` または `config.local.json` の `memo_keeper_enabled`、`memo_keeper_max_chars`、`memo_keeper_max_output_tokens` で制御できます。

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

以下は、直近の実調査で確認された課題と、既存の機能改善案をまとめたロードマップです。公式サイト・公式告知などの一次情報を十分に取得できないケースが確認されました。原因を断定せず、ログとテストで切り分けながら改善します。各項目は未実装の改善候補を含み、完了時はテスト結果と実行ログを確認して更新します。

### 1. 公式ソース検索・根拠の信頼性

- [ ] **公式検索クエリの検証を見直す**: 通常検索向けの関連度判定によって、有用な公式検索クエリが除外されていないか確認する。除外数・採用数とクエリ内容をログに残す。
- [ ] **公式ドメインを検索に活用する**: ユーザーが登録した公式ドメインを利用し、必要に応じて `site:` 検索などドメインを絞った検索を行う。公式ドメインを推測で登録せず、確認済みのものだけを扱う。
- [ ] **公式候補の採用条件を明確にする**: 検索レーンが公式専用であることだけで公式と断定せず、URLのホスト・ページ本文・発行元を確認する。ニュース記事などの混入を検知し、公式候補と一般候補を区別する。
- [ ] **公式ソース不足時の再検索を追加する**: 品質評価で一次情報が不足していると判断した場合、通常検索だけでなく公式検索レーンを追加実行する。検索予算を守り、再検索の条件と回数をログに記録する。
- [ ] **公式ソースの取得状況を可視化する**: 公式検索クエリ数、検索結果数、公式ドメイン一致数、本文取得成功数、評価後の採用数をログに出し、「検索で見つからない」「本文を取得できない」「評価で落ちる」を区別する。
- [ ] **公式情報と二次情報の照合を強化する**: 公式発表で確認できた事実と、ニュース記事・リーク・予想に基づく情報を分けて整理し、公式情報が見つからない場合は未確認であることを明示する。
- [ ] **公式ソース検索の回帰テストを追加する**: 公式ドメインの一致判定、公式候補の保持、検索失敗・本文取得失敗、公式ソース不足時の再検索をモック検索結果で検証する。

### 2. 調査の完了判定・反復制御

- [ ] **目的達成まで公式もしくは通常検索のルートでそれどれ独立した状態で繰り返す方式への任意切り替え**: ユーザーが指定した回数だけ繰り返す方式に加え、調査目的の達成を基準に反復する方式を選択できるようにする。
- [ ] **ユーザーの意図を正確に理解するための　本サーチ前のプリリサーチ機能
### 3. ログ・可観測性

- [ ] **ログの詳細化**: コンテキスト使用量を可視化し、検索・本文取得・評価・追加検索の各段階を追跡しやすくする。

### 4. 設定・操作性

- [ ] **詳細設定の初期値を見直す**: より幅広い用途で使いやすい汎用的な数値に変更する。
- [ ] **詳細設定プリセット機能**: 用途に応じた設定を保存・呼び出しできるようにする。

### 5. 調査ボード・記憶

- [x] **調査ボードの自動整理**: 短いJSONメモを更新するメモキーパーを実装。

### 6. 品質評価・モデル運用

- [ ] **調査終了後のセルフ評価とリトライ機能**: 最終回答を自己評価し、不足があれば条件付きで再調査する。
- [ ] **推奨モデルの選定**: 品質・速度・リソース使用量を踏まえて推奨モデルを整理する。
- [ ] **用途別モデルルーティング**: タスクに応じてモデルを切り替え、軽量化と効率化を図る。
## URL単位の根拠評価

`official_domains` と `trusted_domains` は、ユーザーが明示登録したドメインに対する優先順位付けです。未登録ドメインを一律に信頼または却下するものではありません。品質評価では、本文を取得できた候補ページごとに、質問との関連性、著者・発行主体、編集責任、一次根拠への追跡可能性、独立性、更新性を確認し、最終回答の根拠として採用するかを記録します。

- 取得状態・失敗理由（`fetch_status` / `fetch_error`）、採否・理由・評価値は調査ボードと `[SOURCE_TRACE]` ログで追跡できます。
- 却下されたページは最終回答の引用候補から除外されます。評価が出せなかったページは安全側として候補のまま残し、従来の関連性・多様性選択で扱います。
- 本文ハッシュが同じページの評価は `.cache/source-assessments` に再利用し、本文が変わった場合は再評価します。Gitには含めません。
- `config.json` または `config.local.json` の `source_assessment_cache_enabled`、`source_assessment_cache_dir`、`source_assessment_min_credibility` で調整できます。既定の採用下限は `0.55` です。
### 4Bモデル向けの出力制御

`planner_max_output_tokens`、`evaluation_max_output_tokens`、`final_answer_max_output_tokens` は役割ごとの生成上限です。小型モデルで評価プロンプトの入力と出力がコンテキスト上限を超える場合は、まず本文予算とこれらの上限を下げます。

Qwen系など、プロンプト先頭または末尾の制御トークンに対応するモデルでは、ローカル専用の `config.local.json` で `llm_prompt_prefix` / `llm_prompt_suffix` を設定できます。例: `"llm_prompt_prefix": "/no_think"`。OpenAI互換サーバーにモデル固有値を渡す場合は `llm_extra_body` も使えます。例: `"llm_extra_body": {"enable_thinking": false}`。いずれもモデル名で分岐しない任意設定で、対応しないモデルでは空文字列・空オブジェクトのまま使用します。
### LM Studioのヘッドレスロード設定

GUIの詳細設定では、LM Studioに渡すコンテキスト長を変更できます。既定は `65536` tokens で、設定は `config.local.json` に保存されます。`開始前にLM Studioへモデルを読み込む` を有効にすると、調査開始時にネイティブAPIで選択モデルのロード状態を確認し、コンテキスト長が異なる場合だけその選択モデルを再ロードします。別のロード済みモデルは変更しません。

ロード成功時にLM Studioが返した実際のコンテキスト長を検証します。一致しない場合、調査は開始せずにエラーを表示します。VRAM不足時はコンテキスト長を下げるか、不要なモデルをLM Studio側でアンロードしてください。

## ライセンスとGUI配布

本体は Apache-2.0 です。デスクトップ GUI を配布する場合は、配布物に
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) と
[`LICENSES/LGPL-3.0.txt`](LICENSES/LGPL-3.0.txt) を同梱してください。PySide6/Qt
は LGPL-3.0 の条件で利用しており、Qt の DLL を共有ライブラリとして配布し、
静的リンクしないでください。Firecrawl セルフホスト機能は任意の外部コンポーネントであり、
その本体を製品に同梱する場合は別途 AGPL-3.0 の義務を確認してください。

ビルド後の配布フォルダには、次を実行して通知ファイルをコピーできます。

```powershell
.\scripts\Add-GuiLicenseNotices.ps1 -DistributionPath .\dist\SearchOrchestrator
```
