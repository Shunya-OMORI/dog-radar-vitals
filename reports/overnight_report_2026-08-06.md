# 夜間作業報告 (2026-08-06)

ユーザ就寝中に実施した3つの作業の報告。

## ① Git整理

- **radarODE-MTL/**: 別研究者(ZYY0844)のリポジトリのため、`origin`を`upstream`にリネームし、
  自分の実験変更は新規の`personal-work`ブランチに分離・コミットした。`data_prepared/`
  (11GB)・`archive_reduced_run/`(577MB)等の大容量生成物は`.gitignore`に追加(誤ってコミット
  しないため)。GitHub上に個人用リモートを作る案もあったが、このマシンで`gh auth login`が
  未実施だったため、まずローカルブランチでの分離のみ実施した。**GitHubに個人アカウントで
  控えを持ちたい場合は`gh auth login`後にリモートを追加してほしい。**
- **dog-radar-vitals/**: 未コミットだった240〜272番台のmmECG実験設定・コード変更をコミット
  (1コミット)。
- **manager-agent/**: dog-mmwave-rri READMEの更新をコミット。
- **push**: dog-radar-vitals・manager-agentともpushを試みたが、WSL環境の
  `git-credential-manager.exe`が動作せず(`Exec format error`、Windows側バイナリをWSLから
  実行しようとして失敗)認証できなかった。**コミット自体はローカルに残っているので作業は
  失われていないが、リモートへの反映は手動で`git push`するか、credential.helperの設定を
  見直してほしい。**

## ② Part II構想の実装(d ↔ Δs 対応学習)

### 調査: 犬×人交流+ECGの公開データセット

犬とヒトの交流を撮影した映像と、覚醒度が推定できる生体信号(ECG等)を同時収録した公開データ
セットは**存在しないことを確認した**(調査結果は本ファイル末尾に添付)。近い候補
(UWB-DVS、犬-飼い主HRV同時計測研究)はいずれも要件を満たさないか非公開。

人間データでの代替候補として**K-EmoCon**(自然な会話+ECG/PPG+映像音声、Zenodo申請制)・
**RECOLA**(対話+ECG/EDA+映像音声、EULA登録制)を特定した。**両方とも本人確認情報を含む
申請が必要なため、私からは送信していない。** 申請文面の下書きを
`reports/dataset_access_requests/`に用意したので、内容を確認して提出してほしい(2〜3分で
終わる作業)。承認され次第、今回実装したパイプラインにそのまま投入できる設計にしてある。

### 実装したもの(`src/dog_radar_vitals/affect/`)

| ファイル | 役割 | 状態 |
|---|---|---|
| `transcript.py` | Whisper large-v3による音声→テキスト | 動作確認済み |
| `intervention.py` | gemma-3-4b-it(4bit量子化)による d=VLM(Transcript,画像) 生成 | 動作確認済み(合成画像+日本語音声で意味の通る出力を確認) |
| `text_embed.py` | multilingual-e5-largeによるdのベクトル化 | 動作確認済み |
| `association_model.py` | d↔Δsの共有埋め込み(InfoNCE)+回帰ヘッド | 単体テストで形状・勾配を確認済み。**実データでの学習は未実施**(ペアデータが無いため) |
| `arousal_from_hr.py` / `windowing.py` | RR間隔→個体内正規化した覚醒度、非重複窓でのΔ計算 | CASEデータで検証(下記) |

設計判断・理由・対応する先行研究は`reports/design_2026-08-06_arousal_intervention_association.md`
に、CLAUDE.md R1に従って核心コード付きで記載した。

### CASEデータセットでの検証結果

CASEデータセット(git-lfs pullが一度エラーで壊れて再取得が必要だったが、最終的に取得完了)で
`arousal_from_hr.py`を検証した。**重要な訂正**: gitlab.com/karan-shr/case_dataset には
`data/interpolated/`・`data/non-interpolated/`の実データ(CSV, video列付き)は含まれておらず
READMEのみだった(過去の記憶にあった情報は古かった)。実データがあった`data/raw/`
(タブ区切り・video列なし)形式に評価スクリプトを合わせ、ベースライン区間は
`metadata/videos_duration.txt`のstartVid区間長(101.5秒、全被験者共通で最初に提示される)を使う
方式に変更した。

**結果(30/30被験者で評価完了)**:

| 指標 | 値 |
|---|---|
| pooled Pearson r | 0.193 (r²=0.037, n_windows=7050) |
| 被験者内 r 中央値 | 0.197 (範囲: -0.168〜0.525) |
| trivial(常に0) MAE | 3508.5 |
| trivial(全体平均) MAE | 3933.6 |
| HR由来スコア(線形較正後) MAE | 3955.7 |

**正直な評価**: 過去の記憶にあったWESADでの参考値R²≈0.55と比べてかなり弱い(r²=0.037)。
さらに、線形較正後のHR由来スコアの点予測MAEは**trivial(常に0)にも負けている**。
相関は弱いながら正の方向にあるが、点予測としては現状trivialベースラインを上回れていない。
この食い違いの原因(CASEの刺激視聴という受動的設計がWESADのTSST等より覚醒variationが小さい
可能性、ベースライン区間の取り方、個体ごとのアノテーションスケールの違いなど)は未調査。
**深追いせず正直な結果として記録した**(`reports/arousal_from_hr_case_evaluation.md`)。
次にやるべきは、この弱さがCASE固有(刺激が受動的で覚醒レンジが狭い)なのか、
arousal_from_hr.py側の問題なのかの切り分け。

### 正直な限界

d↔Δsの実ペアデータが無いため、association_model.pyは今夜の時点でアーキテクチャ検証止まり。
K-EmoCon/RECOLAの承認、または犬データの取得を待って初めて本番の学習に進める。

## ③ GPUを遊ばせない: config269のLODO(11-fold)交差検証

進捗報告書Part III-9の残課題3(「学習7名・検証1名では検証値がepoch間で±0.08揺れる。
11-fold交差検証に切り替える」)にそのまま対応する実験。モデル構成(config269)は変更せず、
被験者分割のみを変えた(R4: 単一変数)。2epochのプローブで動作確認後(R2)、
`queue/pending/010_radarode_sceg_lodo_cv.sh`としてキューに投入、実行中。

<!-- LODO_CV_RESULT_PLACEHOLDER -->

1fold(9名train/1名val/1名test、25epoch)あたり約1.5〜2時間かかっており、11fold全体では
半日以上かかる見込み。**朝の時点でまだ実行中の可能性が高いが、キューは`exp_queue.sh`が
セッション終了後も走り続ける設計なので、そのまま放置して問題ない。** 途中経過は
`reports/radarode_sceg_lodo_cv.json`で確認できる。

## 次にやってほしいこと(優先順)

1. `reports/dataset_access_requests/`の2つの下書きを確認して提出(K-EmoCon, RECOLA)。
2. `git push`(credential-manager修復後、または別マシン経由)でdog-radar-vitals/manager-agent
   のコミットをリモートに反映。
3. LODO CV(`reports/radarode_sceg_lodo_cv.json`)の全fold完了後、平均・分散を確認し
   Part III-9の残課題3の決着をつける。
4. K-EmoCon/RECOLA承認後、`affect/`パイプラインを本番データに接続し、association_model.py
   の実学習とtrivialベースライン比較(R3)に進む。

---

## 付録: データセット調査の詳細

(調査エージェントの報告をそのまま添付)

犬×人交流＋生体信号の公開データセットは存在しないと結論。近い候補を2つ確認したが、
どちらも要件を満たさない: UWB-DVS(Nature Sci Data 2024)は犬単体の生体信号で人との交流が
主眼でない。犬-飼い主HRV同時計測研究(PMC11502769)は交流ありだが非公開(倫理制約で著者への
個別依頼のみ)。

人×人の交流+生体信号データセット比較:

| データセット | 交流性 | ECG/PPG | 映像・音声 | 入手性 | 規模 |
|---|---|---|---|---|---|
| K-EmoCon | ◎ 自然な会話 | ECG+PPG+EDA | あり(21/32名) | Zenodo申請制 | 32名・173分 |
| RECOLA | ◎ 遠隔協調タスク | ECG+EDA | あり | EULA登録制 | 5分×27本 |
| AMIGOS | △ 主に受動視聴 | EEG+ECG+GSR | あり | EULA署名制 | 40名 |
| CASE | × 刺激視聴のみ | ECG,BVP,EMG,EDA等 | なし | 即時公開(git clone) | 30名 |
| SEWA | ◎ ビデオチャット討論 | なし | あり | 申請制 | 398名・44時間 |
| MAHNOB-HCI | × 刺激視聴のみ | EEG+ECG+GSR | あり | 申請制 | 27名 |

CASEは即時利用可能だが被験者自身の映像・音声が無いため、今回はarousal抽出パイプラインの
検証専用に使い、d↔Δs対応学習の検証にはK-EmoCon/RECOLA承認を待つ。
