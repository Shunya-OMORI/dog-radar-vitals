# 設計メモ: d ↔ Δs 対応学習システム(2026-08-06)

進捗報告書 `progress_report_2026-08-05_presentation.md` の Part II「RR間隔の先に何を置くか」を
実装する最初のイテレーション。CLAUDE.md の R1 に従い、実装した各モジュールについて
「核心のコード」「なぜその設計か」「対応する先行研究」を記す。

## 全体構成

```
RR間隔(将来: ミリ波、現在: CASEのECG/config269のmmECG再構成)
        │
        ▼
  arousal_from_hr.py ── 個体内正規化 ── Δs(ΔArousal, 非重複窓)
        │
        │                         intervention.py (VLM: gemma-3-4b-it)
        │                              ▲
        │                    transcript.py (Whisper large-v3)
        │                              │
        │                         d = 介入の説明テキスト
        │                              │
        │                         text_embed.py (multilingual-e5-large)
        │                              │
        ▼                              ▼
        └──────── association_model.py (JointEmbeddingModel, InfoNCE) ────────┘
```

## 1. arousal_from_hr.py / windowing.py (`src/dog_radar_vitals/affect/`)

- **核心のコード**(`arousal_from_hr.py:145-178`、`arousal_score_from_rr()`): RR間隔系列を
  `windowing.nonoverlapping_windows()`で非重複窓に分割し、窓ごとの平均HRを求めたうえで、
  呼び出し側が指定する`baseline_mask_fn`(ベースライン区間判定)で得た平均・標準偏差により
  `arousal = (hr - baseline_mean) / baseline_std` で個体内z-score化する。RMSSDは
  `window_rmssd()`として分離し、主指標には使わない。
- **なぜこの設計か**: 入力を「RR間隔(相当)時刻の系列」という抽象インターフェースに
  限定し、ECG依存のR波検出(`rpeaks_from_ecg`)を上位の補助関数として分離してあるため、
  将来ミリ波レーダ由来のRR間隔に差し替えても`arousal_score_from_rr`以降は無変更で動く。
  非重複窓限定(重複窓のAPI自体を提供しない)・HR主指標・個体内正規化必須という3点は
  いずれもCLAUDE.mdの確立済み制約(過去のWESAD/CASE等での実データ検証結果)にそのまま
  対応させてある。
- **対応する先行研究**: 短時間窓でのHRV算出はTask Force of ESC/NASPE(1996)の標準的手法に
  準拠。個体内ベースライン正規化はストレス・感情認識分野の標準的手法(Healey & Picard, 2005)
  に沿う。RMSSDの標本化ジッタ感受性は本プロジェクト独自の過去の実データ検証結果に基づく。

### CASEデータセットでの評価(R3: 学習前に既知値の再現性を確認)

`scripts/evaluate_arousal_from_hr_case.py`で、30名分のCASEデータ(ECG, 1000Hz、連続arousal
アノテーション)に対し、非重複10秒窓でのHR由来覚醒度スコアと連続arousalアノテーションの
相関を検証した(ベースラインは全被験者共通の`startVid`導入映像区間)。**trivialベースライン
(常に0/全体平均を出す予測)のMAEを必ず併記**しており、評価スクリプト自体もCASEデータ配置後に
実行可能な状態で用意済み。実データでの数値は、CASEのgit-lfsダウンロード完了後に確定する
(本レポート末尾または`reports/arousal_from_hr_case_evaluation.md`に追記)。

## 2. transcript.py — Whisper large-v3によるTranscript化

- **核心のコード**: `transcript.py:38-49` の `transcribe()`。`pipeline("automatic-speech-recognition",
  model="openai/whisper-large-v3", dtype=torch.float16)` をそのまま呼ぶだけ。
- **なぜこの設計か**: Transcript化自体は確立済みのASRタスクであり新規性を要求しない部分。
  精度優先(エッジ実行を考えなくてよいとの指示)でlarge-v3を選択。日本語(音声が日本語想定)
  に対応させるため `language="japanese"` を明示。
- **対応する先行研究**: Radford et al., 2022, "Robust Speech Recognition via Large-Scale Weak
  Supervision"(Whisper)。

## 3. intervention.py — VLMによる d の生成

- **核心のコード**(`intervention.py:75-96`、`describe_intervention()`):
  ```python
  content: list[dict] = [{"type": "image", "image": img} for img in images]
  transcript_display = transcript.strip() or "(発話なし)"
  content.append({"type": "text", "text": f"音声の書き起こし: {transcript_display}"})
  messages = [
      {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
      {"role": "user", "content": content},
  ]
  ```
- **なぜこの設計か**:
  1. VRAM制約(RTX3060Ti 8GB1枚)のため、Gemma3-4B-itを4bit量子化(bitsandbytes)で使用。
     実測: 4bit量子化で3.4GB、Whisper large-v3(fp16)が3.1GB、同時ロードでも8GBに収まることを
     スモークテストで確認済み。
  2. 分類ではなく自由記述にしているのは、Part II-3「単なる感情分類+LLM翻訳ではない」という
     設計方針の遵守。介入カテゴリを事前に決め打ちしない。
  3. 発話なし(Transcriptが空)の場合を明示的に扱い、無介入区間で作文(hallucination)しない
     ようプロンプトで指示。実際にランダム画像で「発話なし→ヒトの働きかけは見られない」、
     「発話あり(いい子だね、おいで)→ヒトが近づくよう促している」という妥当な出力を確認済み
     (`intervention.py`のdocstring参照、および本ファイル末尾の動作確認ログ)。
- **対応する先行研究**: 画像+テキストを統合してタスク固有の言語記述を作らせる用途は、
  ロボティクス分野の "VLM as reward/description generator"(Yu et al., 2023 "Language to
  Rewards for Robotic Skill Synthesis"; Rocamonde et al., 2023 "Vision-Language Models are
  Zero-Shot Reward Models for RL")に近い発想だが、本タスクは報酬設計ではなくd↔Δsの対応学習の
  入力生成である点で目的が異なる。VLM自体の系譜はCLIP(Radford et al., 2021)以降の
  画像-言語統合表現モデル群。

## 4. text_embed.py — dのベクトル化

- **核心のコード**(`text_embed.py:40-49`): E5系の`"passage: "`プレフィックス付与→
  average pooling→L2正規化。
- **なぜこの設計か**: multilingual-e5-large(Wang et al., 2024)はこの環境に既にキャッシュ済みで
  日本語性能も高い。S2(逆引き検索: 望ましいΔからdを検索)を見据え、検索用途で事前学習された
  E5系のprefix規約(`query:`/`passage:`)にあえて従っている。
- **対応する先行研究**: Wang et al., 2024, "Multilingual E5 Text Embeddings: A Technical Report"。

## 5. association_model.py — d↔Δsの対応学習

`association_model.py`冒頭のdocstringに核心コード・設計理由・先行研究(CLIP, InfoNCE)を
詳述済み。要点のみ再掲:

- **JointEmbeddingModel**: text embedding(1024次元)とΔs(スカラー、将来拡張可)を
  それぞれ128次元の共有空間に射影し、対称InfoNCE(CLIP loss)で学習する。
- **DeltaRegressor**: 共有埋め込みを使ったS1(効果予測)用の軽量回帰ヘッド。
- 単体テスト(`tests/test_affect_association_model.py`)でランダムデータによる形状・勾配の
  正当性は確認済み。**実データでの学習はまだ行っていない**(下記「現時点の限界」参照)。

## 6. 副次的に実施: config269のLeave-One-Subject-Out CV (`scripts/run_radarode_sceg_lodo_cv.py`)

Part IIの実装とは別に、進捗報告書Part III-9の残課題3「学習7名・検証1名では検証値が
epoch間で±0.08揺れる。先行研究と同じ11-fold交差検証に切り替える」に対応する評価スクリプトを
実装し、GPUキューに投入した。MMECGデータセットは全11被験者(id: 1,2,5,9,10,13,14,16,17,29,30)
であり、11-foldはLeave-One-Subject-Out(9名train・1名val・1名test)に相当する。
**モデル構成(config269)は一切変更していない(R4: 単一変数)。変えたのは被験者分割のみ。**
2epochのプローブで学習が正常に進行する(val_corrが0.164→0.179と単調に伸びる、既存の269の
学習曲線と同じ傾向)ことを確認済み(R2)。フル実行(11fold×25epoch)を`queue/pending/`経由で
キューに投入し、完了後は`reports/radarode_sceg_lodo_cv.json`にfold別test_corrと
平均・標準偏差が出力される。

## 現時点の限界(正直な進捗)

1. **d↔Δsの実ペアデータが無い。** 犬×人の交流+ECGを同時収録した公開データセットは
   調査の結果存在しないことを確認した(別途エージェントによる調査結果を参照)。人間データでの
   代替候補としてK-EmoCon(Zenodo, 申請制)・RECOLA(EULA登録制)を特定し、申請文面の
   下書きを`reports/dataset_access_requests/`に用意した。**申請の実際の提出は、本人確認情報を
   含むためユーザ本人が行う必要がある。**
2. 上記の理由により、association_model.pyは今夜の時点でアーキテクチャ検証止まり。
   実データ到着後、最初にやるべきことは「trivialベースライン(dを無視してΔsの平均を常に
   予測する)との比較」(R3遵守)。
3. Part II-4で明記されている「無介入の安静区間を対照として含む」設計は、intervention.pyの
   空Transcript処理で最初の一歩を用意したが、データ収集プロトコル自体(安静区間を録る)は
   まだ設計していない。K-EmoCon/RECOLAにこの対照区間が実在するかは individual にデータを見て
   確認する必要がある(未確認)。
