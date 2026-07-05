# OpenMontage 改善計画書 — research → compose パイプライン全体レビュー

> 作成日: 2026-07-05 | レビュー範囲: `lib/`, `tools/`, `pipeline_defs/`, スキル契約(AGENT_GUIDE.md)
> レビュー観点: 設計上の弱点 / エラーハンドリング / Mac・Windows 二重環境 / fal.ai コスト最適化

---

## 1. 総評

**強み(維持すべき設計)**

- **instruction-driven アーキテクチャの一貫性**: Python はツール+永続化に限定し、オーケストレーションをエージェント側に置く方針が `AGENT_GUIDE.md` → マニフェスト → director スキルまで貫かれている。
- **契約の機械検証**: checkpoint / canonical artifact が JSON Schema で fail-fast 検証される(`lib/checkpoint.py:95-153`)。ステージ間の受け渡しが「スキーマで守られた JSON」であることは、この規模のエージェントシステムでは大きな強み。
- **render_runtime ガバナンス**: `video_compose._render` は runtime 未設定・不正値を即エラーにし、Remotion 失敗時も FFmpeg へ黙って落ちない(`tools/video/video_compose.py:1316-1390`)。
- **Windows 対応の蓄積**: `run_command` の UTF-8 強制と `.cmd` 解決(`tools/base_tool.py:315-345`)、`--props=` 等号形式(`video_compose.py:1700-1705`)、cp1252 向け Unicode スクラブ(`tool_registry.py:22-52`)、atelier のジャンクション対応など、実戦で踏んだ地雷が丁寧に潰されている。

**弱点の要約**

1. **コストガバナンスが「宣言だけ」**: `CostTracker` の estimate→reserve→reconcile、`RetryPolicy`、`idempotency_key` はいずれも契約として宣言されているが、**どのツールの実行パスにも接続されていない**。実質、課金保護はエージェントの善意頼み。
2. **fal.ai 呼び出しコードの 5 重コピペ + 無限ポーリング**: 課金 API のいちばん危険な部分が最も品質の低いコードになっている。
3. **エラーハンドリングの「握りつぶし」パターン**: `except Exception → str(e)` と `except (FileNotFoundError, Exception)` が要所にあり、失敗の根本原因(HTTP ボディ、トレースバック、fal のジョブログ)が失われる。
4. **Windows 対策の適用漏れ**: base_tool で文書化までされた cp1252 対策が、`hyperframes_compose._run_hf` はじめ複数の subprocess 呼び出しに適用されていない。

---

## 2. 指摘事項(重要度順)

### A. コスト・課金ガバナンス

#### A-1. [CRITICAL] fal.ai ポーリングが無期限 `while True`(5 ツールにコピペ)

該当: `tools/video/kling_video.py:166`, `minimax_video.py:159`, `veo_video.py:301`, `seedance_video.py:270`, `grok_video_fal.py:198`

```python
while True:
    time.sleep(5)
    status_resp = requests.get(status_url, headers=headers, timeout=15)
    ...
```

- fal 側のジョブがスタックすると**永久にポーリングし続ける**。締切なし、バックオフなし、キャンセルなし。
- 個々の HTTP リクエストの timeout=15 は「1 回の GET」の締切であり、ループ全体の締切ではない。
- 同じ submit→poll→download コードが 5 ファイルに約 60 行ずつ複製されており、修正が 5 箇所に波及する。

**提案**: `tools/video/_fal_queue.py`(または `_shared.py` 内)に共通クライアントを 1 つ作る。

```python
def run_fal_queue_job(model_path, payload, *, api_key, timeout=900,
                      poll_interval=5.0, max_interval=30.0,
                      request_log_path=None) -> dict:
    # 1. submit → request_id / status_url / response_url を得る
    # 2. request_id を即座に request_log_path (JSON) へ永続化 ← A-2 参照
    # 3. deadline 付きポーリング(指数バックオフ、上限 30s)
    # 4. FAILED 時は logs エンドポイントを取得して error に含める ← B-3 参照
    # 5. TimeoutError には request_id を含める(後から回収可能にする)
```

`poll_heygen`(`_shared.py:386-415`)は既に deadline+バックオフを実装しているので、これを一般化するのが最短。

#### A-2. [CRITICAL] request_id が永続化されず、中断＝課金済みクリップの喪失

fal の queue submit が成功した時点で課金は始まるが、`request_id` はローカル変数にしかない。エージェントの Bash タイムアウトやセッション切断でプロセスが死ぬと、**課金は発生したのに動画 URL を回収する手段がない**。

**提案**: submit 直後に `projects/<project>/artifacts/fal_requests.jsonl` へ `{request_id, model_path, prompt_hash, submitted_at}` を追記。回収用に `fal_queue` ツール(operation: `status` / `collect`)を追加すれば、途中で切れても再取得できる。

#### A-3. [HIGH] CostTracker がツール実行パスに接続されていない

- `tools/cost_tracker.py` の reserve/reconcile はエージェントが手で呼ぶ規約で、`BaseTool.execute` には一切フックがない。
- `ToolResult.cost_usd` は各ツールが `estimate_cost()` の値をそのまま入れている(例: `kling_video.py:212`)。**「実測コストの reconcile」は現状フィクション**であり、fal の実請求とズレても検知できない。
- `CostTracker._approved_tools` はメモリのみで、セッションを跨ぐと承認履歴が消える(`cost_tracker.py:59`)。

**提案**(段階的に):
1. `BaseTool` に `execute_tracked(inputs, tracker)` ラッパーを追加し、estimate→reserve→execute→reconcile を 1 呼び出しに畳む。director スキルには「課金ツールは execute_tracked 経由」と 1 行書くだけで済む。
2. `_approved_tools` を cost_log.json に永続化。
3. fal のレスポンス(またはレスポンスヘッダ)から課金情報が取れるモデルでは実測値を `cost_usd` に入れ、取れないモデルは `cost_basis: "estimate"` をデータに明示する。

#### A-4. [HIGH] clip_cache / idempotency_key が生成ツールに接続されていない

- `tools/video/clip_cache.py` はロック・LRU 逐出・ハードリンクまで実装された完成度の高いキャッシュだが、利用者は `corpus_builder.py` のみ。
- 各生成ツールは `idempotency_key_fields` を宣言している(例: kling は prompt/variant/operation/duration)のに、**同一入力の再実行は素通しで再課金**される。リトライ・やり直しの多いエージェント運用ではここが最大の無駄金ポイント。

**提案**: A-1 の共通クライアントに「submit 前に `idempotency_key()` で clip_cache を引き、ヒットしたらハードリンクで即返す / 生成成功後に ingest する」を組み込む。これだけで再実行・スクリプト修正後の再レンダーが実質無料になる。

#### A-5. [MEDIUM] 単価がコードにハードコード

例: `kling_video.py:107-114`($0.10-0.30/5s)、`seedance_video.py:174`($0.2419/$0.3034)、`grok_video_fal.py` docstring($0.05/s)。fal の価格改定に無言で追随できず、cost_tracker の見積もりが静かに狂う。

**提案**: `config/pricing.yaml`(provider × variant × 単位)に外出しし、`estimate_cost` は共通ローダー経由で参照。価格取得日をコメントではなくデータとして持たせる(`as_of: 2026-07-01`)。プリフライトで「価格データが 90 日以上古い」警告を出せるようになる。

#### A-6. [MEDIUM] RetryPolicy が宣言のみで未実装

`RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])` を宣言するツールが多数あるが、`BaseTool` にも各ツールにもリトライ実行機構がない。レート制限 429 で即失敗 → エージェントが手動で再実行 → 全額再課金、という流れになる。

**提案**: A-3 の `execute_tracked` に retry_policy の解釈を実装(429/timeout のみ、バックオフ付き)。キャッシュ(A-4)と併用すれば安全にリトライできる。

### B. エラーハンドリング

#### B-1. [HIGH] `except Exception → str(e)` による情報消失

- 全 fal ツールの execute(例: `kling_video.py:192-193`)は HTTPError を握りつぶし、**fal が返すエラーボディ(なぜ拒否されたか、モデレーション理由、パラメータエラー)を捨てる**。
- `video_compose.execute`(`video_compose.py:333-334`)も同様で、2500 行のどこで落ちたかトレースバックが消える。

**提案**: 共通ヘルパーで「HTTPError なら `response.status_code` + ボディ先頭 500 文字を error に含める」「予期しない例外は `traceback.format_exc()` の末尾数行を含める」。ToolResult に `error_kind`(auth / rate_limit / provider_rejection / bug)を追加すると、AGENT_GUIDE の「Escalate Blockers Explicitly」(auth なのか provider なのか tool bug なのか)にツール出力が直接answering できる。

#### B-2. [HIGH] checkpoint の全例外サイレントフォールバック

`lib/checkpoint.py:73`:

```python
except (FileNotFoundError, Exception):
    # Graceful fallback: return all known stages in canonical order
    return list(STAGES)
```

`Exception` を含むので実質すべてを握りつぶす。マニフェストの YAML 構文エラーやスキーマ違反があっても、**壊れたパイプライン定義のまま canonical 順で進行してしまう**。「Invalid checkpoints are contract violations and should fail fast」(AGENT_GUIDE)という自らの規約と矛盾。

**提案**: `FileNotFoundError`(未知パイプライン名)のみフォールバック、`jsonschema.ValidationError` / `yaml.YAMLError` は再送出。最低でも warning ログを必須にする。

#### B-3. [MEDIUM] fal FAILED 時に理由を取得しない

`status in ("FAILED", "CANCELLED")` で `"Kling video generation failed"` とだけ返す(`kling_video.py:173-177`)。fal の queue API はジョブログ(`{status_url}?logs=1`)を提供しており、モデレーション拒否かパラメータ不正かが分かる。取得しないため、エージェントは同じ失敗プロンプトを微修正なしに再送しがち = 追加課金。

**提案**: A-1 の共通クライアントで FAILED 時に logs を 1 回取得し error に添付。

#### B-4. [LOW] `get_latest_checkpoint` が mtime 依存

`lib/checkpoint.py:296-299`。git checkout・ファイルコピー・クラウド同期で mtime が変わると「最新」が狂う。checkpoint 本体に timestamp フィールドがあるのだから、内容の timestamp でソートすべき。

#### B-5. [LOW] `_parse_json_output` の波括弧探索

`hyperframes_compose.py:1157-1169` は stdout の最初の `{` と最後の `}` を切り出す方式。バナーに `{}` が含まれると壊れる。行単位で JSON 開始行を探す方が堅牢(実害は未確認のため LOW)。

### C. Mac / Windows 二重環境

#### C-1. [HIGH] `_run_hf` に `encoding="utf-8"` がない

`tools/video/hyperframes_compose.py:1138-1146`。`base_tool.run_command` は「Windows の cp1252 デフォルトが Unicode/emoji 出力で UnicodeDecodeError を起こし、本当のエラーを飲み込む」とコメントで明記して UTF-8 を強制している(`base_tool.py:336-341`)のに、hyperframes CLI 呼び出し(バナーや絵文字を出す典型的な npm ツール)には同じ対策が入っていない。**Windows で HyperFrames レンダーが文字化けまたはクラッシュする再現条件が既知のまま残っている**。

**修正**: `subprocess.run(..., encoding="utf-8", errors="replace")` を追加(1 行)。

#### C-2. [MEDIUM] subprocess 呼び出しの encoding 指定漏れの横断確認

`video_compose._has_audio_stream`(`video_compose.py:357-367`)や `_shared.probe_output` など、`text=True` のみで encoding 未指定の subprocess が点在。ffprobe はファイルパスをエラー出力に含めるため、日本語パス + cp1252 で同じ罠を踏む。**「subprocess を直接呼ばず `run_command` / `_run_hf`(修正後)を使う」規約を PROJECT_CONTEXT.md に明記**し、既存呼び出しを棚卸しする。

#### C-3. [LOW] OS 判定イディオムの混在

`platform.system() == "Windows"`(base_tool, screen_recorder)と `os.name == "nt"`(hyperframes_compose)が混在。`lib/platform_utils.py` に `IS_WINDOWS` / `resolve_cmd()` を置いて統一すると、C-1/C-2 の再発も構造的に防げる。

#### C-4. [参考] 良くできている点(触らない)

- `clip_cache._link_or_copy`: クロスドライブ(C:→D:)の `os.link` 失敗を copy2 へフォールバック(`clip_cache.py:523-543`)。
- atelier のジャンクション(Windows)/シンボリックリンク(Unix)使い分けと「一部の Windows ジャンクションは rmdir が必要」対応(`video_compose.py:743-940`)。
- `screen_recorder` の gdigrab / avfoundation / x11grab 三分岐。
- registry の cp1252 向け Unicode スクラブ。

### D. アーキテクチャ / 設計

#### D-1. [HIGH] `video_compose.py` 2,552 行 — 責務過多

FFmpeg 合成 / Remotion レンダー / HyperFrames ブリッジ / atelier / テーマ生成 / 最終セルフレビュー / 字幕焼き込み / エンコードが 1 クラスに同居。リポジトリ自身の規約(1 ファイル 800 行以下)の 3 倍超。

**分割案**(ツール契約は `video_compose` 1 つのまま、実装をモジュール分割):

```
tools/video/compose/
├── __init__.py          # VideoCompose(BaseTool) — execute + ルーティングのみ
├── ffmpeg_engine.py     # _compose, _burn_subtitles, _overlay, _encode
├── remotion_engine.py   # _remotion_render, _build_theme_from_playbook
├── atelier.py           # _render_via_atelier, _stage_atelier_project
└── final_review.py      # _run_final_review, transcript 比較
```

#### D-2. [MEDIUM] `render_runtime="remotion"` 時の FFmpeg 暗黙ルーティング

`video_compose.py:1391-1412`: runtime に `remotion` がロックされていても、cuts が純動画のみ(`_needs_remotion` が False)だと `_compose`(FFmpeg)へ静かにルーティングされる。コメントは「FFmpeg fallback: only when Remotion is unavailable」だが、実際の条件は**可用性ではなく cut 内容**。トランジション・オーバーレイ・スプリング物理が黙って失われるケースがあり、「Silent runtime swap is forbidden」という自らのハードルールと緊張関係にある。

**提案**: このパスを通る場合は `render_result.data["engine_used"] = "ffmpeg"` と `"engine_downgrade_reason"` を必ず載せ、render_report / final_review で警告として浮上させる(完全禁止にすると純動画カットで Remotion を無駄に通すことになるため、可視化が現実的)。

#### D-3. [MEDIUM] dotenv ローダーが 3 実装

手書きパーサーが `base_tool.py:23-58` と `tool_registry.py:86-116` に完全重複、さらに `lib/env_loader.py` は python-dotenv 使用。挙動差(インラインコメント処理、クォート処理)が将来バグになる。`lib/env_loader.py` に一本化し、他 2 箇所はそれを import する。

#### D-4. [MEDIUM] レジストリの依存チェックにキャッシュがない

`get_status()` が呼ばれるたびに `shutil.which` / `__import__` / env 参照を全実行する。`provider_menu()` は全ツール分を回すため、プリフライトのたびに O(ツール数 × 依存数) のチェックが走る(AGENT_GUIDE も「firehose / 遅い」と自認)。TTL 付き(例: 60 秒)の status キャッシュを `BaseTool.get_status` に入れるだけで体感が大きく変わる。

#### D-5. [LOW] ディレクトリ命名の三重不整合

- AGENT_GUIDE: チェックポイントは `pipelines/<project_id>/`
- `config.yaml` / `config_model.py`: `storage_dir: pipeline`(単数)
- 生成物は `projects/<project-name>/`

エージェントは毎セッションこのドキュメントを読んで行動するため、**ドキュメントと設定の不一致はそのまま実行時の迷いになる**。`pipelines`(複数形)に統一し、config.yaml とガイドを一致させる。

#### D-6. [LOW] BaseTool の可変クラス属性共有

`resource_profile = ResourceProfile()` / `retry_policy = RetryPolicy()` / `supports: dict = {}` はクラスレベルの共有インスタンス。現状ツールは上書き宣言しているので実害はないが、どこかが `self.supports[...] = x` とやった瞬間に全インスタンスへ波及する Python の古典的罠。`__init_subclass__` での防御か、少なくともコメントでの明示を推奨。

#### D-7. [LOW] `web_search` が「幽霊ツール」

`pipeline_defs/cinematic.yaml:64-65` の research ステージは `tools_available: [web_search]` を宣言するが、レジストリに `web_search` ツールは存在しない(エージェント自身の WebSearch 機能への暗黙依存)。プリフライトの「required_tools をレジストリで確認」ルールに従うと research が常に degraded 判定になり得る。マニフェストスキーマに `agent_native_tools` フィールドを分けて宣言するのがクリーン。

---

## 3. fal.ai コスト最適化戦略(まとめ)

現状、コスト防御は「スキル層の運用ルール」(sample sub_stage、retry_multiplier 1.3、コストレンジ提示)に偏っており、**ツール層の機械的防御がゼロ**。以下の順で費用対効果が高い:

| 順位 | 施策 | 対応項目 | 期待効果 |
|------|------|---------|---------|
| 1 | idempotency_key × clip_cache を fal 呼び出しに接続 | A-4 | 再実行・再レンダー時の重複課金をゼロに。エージェント運用では体感 20-40% 削減 |
| 2 | request_id 永続化 + collect 操作 | A-2 | 中断時の「課金済みクリップ喪失」をゼロに |
| 3 | 共通 fal クライアント(deadline/バックオフ/logs) | A-1, B-3 | ハング防止 + 失敗理由の可視化で無駄な再送を削減 |
| 4 | pricing.yaml 外出し + 価格鮮度警告 | A-5 | 見積もり精度の維持。proposal 段階の承認が実勢とズレない |
| 5 | execute_tracked による reserve/reconcile 自動化 | A-3, A-6 | 予算超過の機械的ブロック(cap モードが初めて実効化) |
| 6 | scoring の cost_efficiency 重みを予算残高に連動 | `lib/scoring.py:42`(固定 0.10) | 予算逼迫時に grok_video_fal($0.05/s)等の安価系へ自然に寄る |

補足: `grok_video_fal` の追加(480p $0.05/s)は良いコスト施策。ただし scoring の重みが品質偏重(cost_efficiency 0.10)のため、予算文脈を渡さない限り選ばれにくい。施策 6 とセットで効く。

---

## 4. 実施フェーズ

### Phase 1 — 出血を止める(小さく、即効) ✅ 実施済み (2026-07-05)

1. ✅ C-1: `_run_hf` に `encoding="utf-8", errors="replace"` 追加
2. ✅ B-2: checkpoint のフォールバック条件を `FileNotFoundError` に限定(警告ログ付き)
3. ✅ A-1: 暫定コピペではなく `_shared.submit_and_poll_fal_queue()` を新設し、5 ツールすべてを移行(deadline 900s / 1.5x バックオフ上限 30s / 一時的なポーリング失敗 5 回まで許容)。Timeout メッセージに `status_url` を含め、課金済みジョブの回収経路を確保(A-2 の部分的先取り)
4. ✅ B-1/B-3: submit 拒否時は HTTP ステータス+ボディ先頭 500 文字、FAILED/CANCELLED 時は `?logs=1` のログ末尾 5 件を error に含める

テスト: `tests/tools/test_fal_queue_helper.py`(4 ケース)+ `tests/tools/test_checkpoint_stage_fallback.py`(2 ケース)を追加。`tests/tools` + `tests/contracts` 全 428 件 pass を確認。

### Phase 2 — fal 共通基盤(コスト最適化の本丸) ✅ 実施済み (2026-07-06)

5. ✅ request_id 永続化: `submit_and_poll_fal_queue(request_log_path=...)` が submitted / completed / failed / cancelled / timeout を `fal_requests.jsonl`(出力先と同じディレクトリ)へ JSONL 追記。新設の `fal_queue` ツール(capability=job_recovery)の `status` / `collect` / `list` 操作で、課金済み未回収ジョブを再課金なしで回収・監査できる
6. ✅ clip_cache 接続: `fal_cache_lookup` / `fal_cache_store`(`_shared.py`)を 5 ツール全部に配線。clip_id は `{tool_name}_{idempotency_key}`。同一入力の再実行はキャッシュヒット(`cache_hit: true`, `cost_usd: 0.0`)。`force_regenerate: true` でバイパス、`OPENMONTAGE_CLIP_CACHE=0` で無効化。併せて 5 ツールの `idempotency_key_fields` を全生成関連入力に是正(kling は未宣言だった `generate_audio` もスキーマ宣言+キー追加)
7. ✅ pricing.yaml 外出し: リポジトリ直下 `pricing.yaml`(`as_of` 付き)+ `lib/pricing.py`。yaml 欠損時は各ツールの `_FALLBACK_RATES` に完全フォールバック(挙動不変)。`provider_menu_summary()` が 90 日超の価格鮮度警告を runtime_warnings に出す
8. ✅ 共通クライアント移行は Phase 1 で完了済み

テスト: `test_fal_queue_recovery.py`(9)+ `test_fal_clip_cache.py`(4)+ `test_pricing.py`(6)を追加。全 447 件 pass。

### Phase 3 — ガバナンスの機械化と整理

9. `execute_tracked`(reserve/reconcile/retry の自動化)+ approved_tools 永続化
10. video_compose のモジュール分割(D-1)、engine_used の可視化(D-2)
11. dotenv 一本化(D-3)、status キャッシュ(D-4)、命名統一(D-5)
12. scoring の予算連動重み(施策 6)

### テスト方針

- Phase 2 の共通クライアントは fal をモックした契約テスト(submit 失敗 / FAILED+logs / タイムアウト / キャッシュヒット)を先に `tests/contracts/` へ追加(既存の phase 別契約テストの流儀に合わせる)。
- C 系(Windows)は CI に `windows-latest` ジョブがなければ、最低限 `encoding` 指定の静的チェック(grep ベースの lint スクリプト)を `make` ターゲット化する。

---

## 5. 対象外としたこと

- スキル(Markdown)層の記述品質そのもの(今回はコード⇔契約の整合に絞った)
- Remotion / HyperFrames の各コンポーネント実装(`remotion-composer/src/`)
- ローカル GPU 系ツール(wan / hunyuan / cogvideo)の性能チューニング
