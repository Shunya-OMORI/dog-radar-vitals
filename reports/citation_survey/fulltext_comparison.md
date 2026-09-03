# 本文確認 — 優先度上位11件の比較（2026-08-25）

対象は `README.md` の「本文を確認する優先順位」に挙げた11件．目的は，我々の主張
(1) パラメータ数0.034Mという軽量性，(2) R波検出とHRV算出の動作点分離，が
「同じ土俵で負けている」状態ではないかを確認すること．

## 取得状況サマリ

| # | 論文 | 掲載 | 本文取得 |
|---|---|---|---|
| 1 | An Onboard Executable Multitask Network Model... | IEEE TIM 2025 | 不可（403）．抄録が数値を網羅しているため実質支障なし |
| 2 | HyperECG | BIBE 2024 | 不可（403，OA版なし） |
| 3 | Frequency Matters | IEEE TIM 2024 | **可**（ETH Research Collection OA） |
| 4 | HEBR | IEEE IoT-J 2026 | 不可（403） |
| 5 | mmCG | IEEE IoT-J 2025 | 不可（403） |
| 6 | Radar HRV Monitoring With Physiological Prior... | IEEE JBHI 2025/2026 | 不可（403）．ただし前身のICASSP 2024版（同一著者グループ）を全文取得 |
| 7 | Robust HRV Monitoring via Massive Radio Sensing | ACM IMWUT 2025 | 不可（ACM DL 403．qeios.comのピアレビューのみ取得，本文なし） |
| 8 | Non-contact seismocardiogram measurement and HRV analysis | Front. Physiol. 2026 | **可**（オープンアクセス誌） |
| 9 | An Adaptive Template Matching Method... | IEEE TIM 2025 | 不可（403） |
| 10 | Monitoring long-term cardiac activity with contactless RF signals | Nature Communications 2024 | **可**（オープンアクセス誌） |
| 11 | Contactless Radar HRV Monitoring Via Deep Spatio-Temporal Modeling | ICASSP 2024 | **可**（著者個人ページ／USTC） |

**11件中，全文または実質的に全文相当（詳細な抄録＋前身論文の全文）でカバーできたのは5件．
残り6件は抄録どまり（すべてIEEE Xploreの403で本文取得不可，OA版も未発見）．**
ただし抄録は査読誌の要求上，主要数値（パラメータ数・誤差指標）を含むことが多く，
比較表の穴は限定的．

## 最重要の発見（先取り）

- **MMECGデータセットを実際に使っている論文は11件中0件．** 「Onboard Executable」が
  「public dataset」と書くのみで名称不明（MMECGの可能性は否定できないが未確認），
  他はすべて独自データセット．したがって本調査は「同じベンチマークで負けているか」
  ではなく「同じ問題設定・目的で，どの程度のコストと引き換えにどの程度の精度を
  達成しているか」の相場観を得るものと位置づけるべき．
- **パラメータ数を明記した論文は1件のみ（Onboard Executable，7.04M）**．
  これは我々の0.034Mの**約207倍**．HDC（HyperECG）や信号処理系（HEBR，mmCG，
  Adaptive Template Matching，Frequency Matters，Nature Comms，Frontiers論文）は
  そもそも学習パラメータを持たない設計であり，「軽量」を謳う論文の多くが
  ニューラルネット自体を持たない（＝我々とは軽量化の手段が異なる）．
- **同一研究グループ（USTC，Chenら2022年MMECG論文の著者ら）の後継研究が，
  データセット規模を上げるほどHRV誤差が悪化する傾向を自ら示している**
  （2026-08-26追記：MMECG原著PDF 1ページ目で著者を確認．Jinbo Chen・Dongheng Zhang・
  Zhi Wu・Fang Zhou・Qibin Sun・Yan Chenの6名．ICASSP 2024にはJinbo Chen・Dongheng Zhang・
  Qibin Sun・Yan Chenが，Nature Comms 2024にはDongheng Zhang・Jinbo Chen・Fang Zhou・
  Qibin Sun・Yan Chenが著者として含まれ，**主要著者が実際に一致する**．
  「同じ大学の別グループ」ではなく同一著者チームの継続研究）：
  ICASSP 2024（自作データ，被験者8名，train6/test2）でRMSSD誤差7.3ms
  → JBHI 2025/2026（外来患者7,150名）でRMSSD誤差16.23ms
  → Nature Communications 2024（外来患者6,222名）でRMSSD誤差53.8ms．
  同じ研究室の同じ設計思想でも，評価対象を臨床規模に拡大すると誤差が
  1桁近く悪化する．小規模・単一環境での「勝敗」の意味は限定的．

---

## 1. An Onboard Executable Multitask Network Model for Bioradar-Based ECG Signal Reconstruction Using High-Fidelity DHD Signals（IEEE TIM 2025）

著者：Fuze Tian, Haojie Zhang, Jie Liu, Jingyu Liu, Mingqi Zhao, Kun Qian, Qinglin Zhao,
Bin Hu, Yoshiharu Yamamoto, Björn W. Schuller．DOI 10.1109/TIM.2025.3617402．

**取得状況**：本文は403で取得不可．抄録に主要数値が網羅されているため実害は小さい．

1. **問題設計**：独自バイオレーダ（I/Qベースバンド，SNR26–119dB）＋高線形arctangent復調で
   Doppler Heartbeat Diagram（DHD）信号を抽出し，軽量U-NetベースのマルチタスクネットでECG
   波形そのものを再構成する（R波位置検出やHRV指標算出とは別の課題設定）．
2. **前処理・前提**：「public dataset」と「own dataset」の2種類で評価しているが，
   public datasetの名称は抄録に記載がなくMMECGかどうか不明．被験者分割方法・
   テスト被験者数の記載なし．
3. **モデルサイズ・指標**：**7.04Mパラメータ**，FLOPs 889.16M，RAM 29.13MB，
   ROM 10.49MB，消費電力379.5mW，推論1.05秒/回．RMSE 0.160/0.330，
   RMAE 0.261/0.400，PCC 95.17%/85.26%（2データセットでそれぞれ）．

**比較**：「onboard executable」を掲げていても7.04Mパラメータであり，
我々の0.034Mの**約207倍**．エッジ実行の主張の中身は「組み込み機器で動く」
ことであって，我々の主張するパラメータ規模の軽量性とは水準が一段違う．

---

## 2. HyperECG: ECG Signal Inference From Radar With Hyperdimensional Computing（BIBE 2024）

著者：Matilda Gaddi, Flavio Ponzina, Fatemeh Asgarinejad, Baris Aksanli, Tajana Rosing
（UC San Diego / San Diego State University）．DOI 10.1109/BIBE63649.2024.10820450．

**取得状況**：本文403，OA版・arXiv版とも未発見．抄録のみ．

1. **問題設計**：Hyperdimensional Computing（HDC）という軽量な代替機械学習手法で
   レーダからECG波形を推論．DLモデルに対する計算コスト削減が主眼で，R波検出や
   HRV指標そのものは扱っていない（連続ECG波形の推論タスク）．
2. **前処理・前提**：データセット名・被験者分割・テスト人数とも抄録に記載なし．
   「on-device model personalization」に言及があり，患者固有の再学習を前提にしている
   可能性がある（＝汎化ではなく個人適応で精度を出している疑い．本文未確認のため断定不可）．
3. **モデルサイズ・指標**：パラメータ数の明記なし．DL SOTA比で**推論23倍・学習36倍高速化**，
   患者個別ファインチューニングで精度**最大68%改善**という相対値のみ．
   RMSSD/SDNN/pNN50などのHRV指標，F1等の絶対誤差は抄録に一切なし．

**比較**：軽量化の手段（HDC）は我々と異なり，量的な比較軸（パラメータ数，HRV誤差の絶対値）
が抄録から得られないため，直接比較は困難．「精度はDL SOTA並み」という定性的な主張のみで，
定量的にどちらが優れるかは判断材料が不足している．

---

## 3. Frequency Matters: Comparative Analysis of Low-Power FMCW Radars for Vital Sign Monitoring（IEEE TIM 2024）

著者：Steven Marty, Andrea Ronco, Federico Pantanella, Kanika Dheman, Michele Magno
（ETH Zurich）．DOI 10.1109/TIM.2024.3381692．**全文取得済み**（ETH Research Collection）．

1. **問題設計**：低電力FMCWレーダ3種（Infineon BGTファミリ，24/60/120GHz）を比較し，
   平均心拍数（HR）・呼吸数（RR）を決定論的信号処理（学習なし，二階微分＋ピーク検出）で
   推定する．**HRVやビート単位のIBIは扱っておらず，我々の目的（RMSSD/SDNN/pNN50）とは
   タスクが異なる**（平均HR/RRの推定に留まる）．
2. **前処理・前提**：健常者24名，着座姿勢，Polar H10 ECG/IMUを正解として使用．
   2分間の記録×2回．訓練を伴わないアルゴリズムのため「訓練・テスト分割」の概念自体が
   存在しない（全24名がそのまま評価対象）．
3. **モデルサイズ・指標**：ニューラルネットを持たない設計．レーダチップ自体の消費電力は
   チャープ蓄積ありで**約26mW**（元は約8mW）——組込み・電池駆動デバイスへの実装を
   明示的に志向．RR誤差はMAE 2 brpm未満（±3.05），HR誤差は60GHz機でMAE 1.8±3.1bpm，
   120GHz機で3.2±5.3bpm，24GHz機は9.0bpm（ノイズプロファイルが悪いため）．

**比較**：「低電力」の中身はレーダハードウェア自体の消費電力（mWオーダー）の話であり，
我々が主張する推論モデルのパラメータ規模の軽量性とは別の軸．またHRV指標を一切
報告していないため，精度面での直接比較対象にはならない．24GHzのような低い搬送周波数では
心拍検出精度が大きく劣化する（MAE 9.0bpm）という点は，我々が使うレーダの周波数選定の
妥当性を検討する上で参考になる．

---

## 4. HEBR: An HRV Noncontact Health Monitoring Method Based on Millimeter-Wave Radar-Enhanced Time–Frequency Concentration Analysis（IEEE IoT-J 2026）

著者：Jinsong Li, Xingguang Li, Yujian Cai, Kaiyao Shi．
DOI 10.1109/JIOT.2026.3661997．**取得状況**：本文403，OA版未発見．抄録のみ．

1. **問題設計**：低SNR環境での時間周波数不確定性を克服するため，動的計画法で
   基本波の高調波支持集合を追跡し，位相整合帯域制限再構成でビート波形を得る
   （純粋な信号処理，学習パラメータなし）．MMECGは使用していない．
   **2026-08-26追記（「本当に機械学習を使っていないか」というユーザ確認に対して）**：
   抄録原文（`citations_raw.json`）を再読したところ，"employs dynamic programming to
   track the fundamental heartbeat component"，"Phase-consistent band-limited
   reconstruction" など，一貫して古典的信号処理の語彙のみで説明されており，
   neural network／deep learning／DNN／training／learn(ed) 等の語は抄録中に一度も
   出てこない．さらに結論部の "HEBR provides a **physically interpretable** and
   readily deployable sensing solution" という一文は，通常ブラックボックス性が
   弱点とされる深層学習手法との対比で使われる言い回しである．**これらから「信号処理
   ベースで学習パラメータを持たない」という判定はかなり確度が高いが，本文（403で
   未取得）を読んだ確証ではなく，抄録からの推測である点は明記しておく．**
2. **前処理・前提**：健常者10名＋心疾患患者1名のレーダデータで主要評価．
   追加で入院患者9名についてはPPG（ECGではない）を正解として汎化を確認——
   **正解データの質が主実験（ECG）と副次実験（PPG）で異なる**点に注意．
   被験者分割やテスト人数の詳細（訓練データという概念自体があるか）は不明．
3. **モデルサイズ・指標**：信号処理ベースのためパラメータ数の概念なし．
   MedAE：IBI 27.93ms，RMSSD 15.37ms，SDNN 7.24ms．
   ビート検出のrecall 88.4%，precision 90.1%，F1 0.893（ECG比）．

**比較**：F1とHRV誤差を同一論文内で両方報告している点は我々の目的（検出とHRVの
動作点分離）に近い題材だが，両者の関係（検出閾値を変えたときHRV誤差がどう動くか）
の分析は抄録からは読み取れない．N=10と小規模である点も割り引いて見る必要がある．

---

## 5. mmCG: Noncontact Millimeter-Wave Cardiography for Heart Rate Variability Monitoring（IEEE IoT-J 2025）

著者：Langcheng Zhao, Rui Lyu, Anfu Zhou, Huadong Ma．DOI 10.1109/JIOT.2025.3573511．
**取得状況**：本文403，OA版未発見．抄録のみ．

1. **問題設計**：心拍の空間局在化（指向性センシングでSNR改善）＋心拍の時間相関を
   利用した動的ピーク探索アルゴリズム（信号処理系，学習の言及なし）．MMECG不使用．
2. **前処理・前提**：被験者数・分割方法とも抄録に記載なし．
3. **モデルサイズ・指標**：「lightweight design」を謳うが定量的なパラメータ数・
   FLOPs等の記載なし．IBI誤差**9.44ms**，既存手法比**51.29%改善**——この
   「既存手法」がChenら2022（MMECG論文，IBI誤差約19ms相当）を指している可能性が高いが，
   抄録からは比較対象の論文名は特定できない．RMSSD/SDNN/pNN50の絶対値は抄録になし．

**比較**：IBI誤差の絶対値（9.44ms）自体はHEBR（27.93ms）やAdaptive Template Matching
（18ms MAE）より良好に見えるが，RMSSD/SDNN/pNN50が報告されていないため，我々の
主張の直接比較材料としては不完全．本文へのアクセスができれば最優先で再確認したい．

---

## 6. Radar HRV Monitoring With Physiological Prior Inspired Deep Neural Networks（IEEE JBHI 2025/2026）

DOI 10.1109/JBHI.2025.3628628．**取得状況**：本文403．ただし**同一テーマ・同一研究グループの
前身であるICASSP 2024論文（下記11番）を全文取得済み**であり，設計思想はほぼ共通と判断できる．

1. **問題設計**：「心拍が全身の体表運動を駆動する」という生理的事前知識に基づき，
   全身のレーダ反射と心拍の時空間関係をハイブリッドDNNでモデル化．さらに心拍の
   自己相似性を用いたデータ拡張でHRV分布を再構成する．MMECG不使用，独自の大規模
   外来患者データセット（N=7,150）で検証——前身のICASSP版（N=8のトイデータ）から
   実運用規模へスケールさせた点が特徴．
2. **前処理・前提**：訓練/テスト分割の詳細は抄録から不明．ただし前身のICASSP版
   （下記11番）では被験者8名中6名訓練・2名テストという極小のホールドアウトだった．
   JBHI版でどう改善されたかは本文未確認．
3. **モデルサイズ・指標**：パラメータ数の記載なし．平均IBI誤差**19.21ms**，
   RMSSD誤差**16.23ms**，SDSD誤差**16.70ms**，pNN50誤差**7.28%**．
   HRVから5種類の心疾患を分類し，ECGベース手法に匹敵する性能と主張．

**比較**：同一研究グループのICASSP 2024版（RMSSD誤差7.3ms，N=8）と比べ，
臨床規模（N=7,150）に拡大した本論文ではRMSSD誤差が2倍以上（16.23ms）に悪化している．
小規模実験での高精度が実運用規模でどれだけ目減りするかを示す好例であり，
我々自身のMMECGでの評価規模・被験者多様性を照らし合わせて評価する必要がある．

---

## 7. Robust HRV Monitoring via Massive Radio Sensing（ACM IMWUT 2025）

著者：Guixin Xu, Jinbo Chen, Haoyu Wang, Yuqin Yuan, Ganlin Zhang, Ziqian Zhang,
Dongheng Zhang, Yang Hu, Qibin Sun, Yan Chen（USTC）．DOI 10.1145/3770711．

**取得状況**：ACM DLは403．qeios.com上の査読コメント（本文ではない）のみ取得できたが，
数値情報は含まれていなかった．**本文取得不可**．

1. **問題設計**（抄録より）：環境変動（部屋のレイアウト・姿勢・周囲物体）に対する
   汎化性能の低さを統計的推定問題として定式化し，大規模な環境多様サンプリング＋
   構造化学習戦略でロバストなHRV推定器を得る．**アーキテクチャの新規性ではなく，
   データ・学習戦略側での頑健性獲得**が主眼——本リポジトリのCLAUDE.mdのR5
   （モデルより先に入力と評価を疑う）と方向性が一致する着眼点．
2. **前処理・前提**：一般環境30名×32環境，臨床では入院患者130名×8病院環境という
   大規模かつ多様な評価．我々のMMECGでの検証規模・環境多様性と比べると，
   環境変動への頑健性評価という観点で明確に上回っている．
3. **モデルサイズ・指標**：パラメータ数の記載なし．「現行SOTA比29.7%改善（一般環境）」
   「41.7%改善（臨床）」という**相対値のみ**で，RMSSD/SDNN/pNN50等の絶対誤差は
   抄録から不明．本文が必要．

**比較**：著者は同じUSTC研究室（Chenら2022年MMECG論文の著者を含む）で，
モデル側ではなく評価環境の多様性・頑健性で差別化する方向性を取っている．
これは我々のCLAUDE.mdの知見（モデル側の工夫はほぼ全滅，前処理・評価側にこそ価値がある）
と符合する重要な傍証だが，絶対的な精度数値が取れておらず，本文確認を優先したい．

---

## 8. Non-contact seismocardiogram measurement and HRV analysis using cardiac beamforming with FMCW radar（Frontiers in Physiology 2026）

著者：Guang Yu, Chenxi Yang, Haobo Li, Chaochao Wang, Xianchao Zhang, Jianqing Li,
Chengyu Liu（東南大学／Dundee大学／嘉興大学）．DOI 10.3389/fphys.2025.1733573．
**全文取得済み**（オープンアクセス誌）．

1. **問題設計**：Capon型ビームフォーミングで心臓方向を特定→修正型差分交叉乗算（MDACM）で
   位相復調→6段ウェーブレットパケット変換（db45）でSCG信号を再構成→
   独自の大動脈弁開放（AO）点検出アルゴリズムでIBI系列を得る．
   **学習パラメータを持たない純粋な信号処理パイプライン**．MMECG不使用．
2. **前処理・前提**：**被験者13名，全員仰臥位，各10分**．重要な点として，
   ビームフォーミング方位の選定に「20秒のテンプレート信号とのDTW距離」を使う
   **被験者内キャリブレーション**方式であり，訓練/テストの被験者分割という概念が
   そもそも存在しない（13名全員が自分自身のテンプレートで評価される，
   leave-one-subject-outではない）．姿勢は仰臥位のみで，複数人シナリオは扱っていない．
3. **モデルサイズ・指標**：信号処理のためパラメータ数の概念なし．
   平均誤差：SDNN **4.11ms**，RMSSD **8.05ms**（Table 2の平均は8.08ms），
   pNN50 **2.15%**．ただし被験者間のばらつきが大きい——Table 2を見るとSDNN誤差は
   被験者9で1.01ms，被験者7で10.84msと**約10倍のレンジ**があり，
   RMSSD誤差も被験者7で0.45ms，被験者6で20.01msと同様に大きくばらつく．
   Bland-Altman分析でも比較5手法中最良（LoA 0.04s）．

**比較**：平均値だけを見ると我々の目標水準に近い高精度に見えるが，
(a) 被験者内キャリブレーション（自分の20秒データをテンプレートに使う）という
極めて有利な前提，(b) 全員仰臥位・静止という制御環境，(c) N=13という小規模，
(d) 被験者間の誤差が最大10倍ばらつく，という4点を踏まえると，
「held-outの新規被験者に対して安定に動く」我々の設定とは前提の有利さが
大きく異なる．平均値の単純比較は避けるべき典型例．

---

## 9. An Adaptive Template Matching Method for Noncontact Cardiac Inter-Beat Interval Detection Using a 120 GHz FMCW Radar（IEEE TIM 2025）

著者：Z. Yang, H. Yang, A. Hu, X. Zhuge, J. Miao．DOI 10.1109/TIM.2025.3545726．
**取得状況**：本文403，OA版未発見．抄録のみ．

1. **問題設計**：カスタム120GHz FMCWレーダ（狭ビームレンズアンテナ）＋
   被験者個別のテンプレートを生成する適応型テンプレートマッチング（相互相関）．
   信号処理ベースで学習パラメータなし．MMECG不使用．
2. **前処理・前提**：「通常環境」での評価人数・被験者分割方法は抄録に記載なし．
   心房細動患者1名での臨床テストに言及あり（症例数はごく少数）．
3. **モデルサイズ・指標**：パラメータ数の概念なし．MAE 18ms，RMSE 24.4ms，
   中央値誤差12.6ms，心拍間隔の**カバレッジ率98.8%**（正常環境下）．
   RMSSD/SDNN/pNN50は抄録に報告なし——生のIBI誤差とカバレッジ率のみ．

**比較**：カバレッジ率98.8%という指標は，我々の「検出とHRVの動作点分離」という
問題意識と関連しうる（検出できなかった心拍がどれだけあるか，という観点）．
ただしRMSSD/SDNN/pNN50が報告されておらず，HRV指標での直接比較はできない．
本文確認できれば，検出率とHRV精度のトレードオフに関する記述がないか確認したい．

---

## 10. Monitoring long-term cardiac activity with contactless radio frequency signals（Nature Communications 2024）

著者：Bin-Bin Zhang, Dongheng Zhang, Yadong Li, Zhi Lu, Jinbo Chen, Haoyu Wang,
Fang Zhou, Yu Pu, Yang Hu, Li-Kun Ma, Qibin Sun, Yan Chen（USTC，
第一附属病院，Zhongke Radar Sensing AI社）．DOI 10.1038/s41467-024-55061-9．
**全文取得済み**（オープンアクセス誌）．

1. **問題設計**：60–64GHzレーダで**10次以上の高調波**という従来「有害」とされてきた
   帯域を逆手に取り，呼吸由来の高調波が急速に減衰する一方で心拍由来の高調波は
   残ることを利用して，ビート周波数パターンを抽出する．**「訓練なし」（without any
   model training）を明言する純粋な信号処理システム**——本文中でも学習ベース手法
   （supplementary比較）に対し「comparable yet slightly inferior」と評しており，
   むしろ学習ベースを上回ったと主張している．MMECG不使用．
2. **前処理・前提**：外来患者評価はN=**6,222**（除外868名を除く7,090名中）——
   非常に大規模だが，各患者は病院での**短時間スナップショット**（詳細な収集時間は
   本文に明記なし，長時間の連続記録ではない）．一方「長期」を謳うもう一方の実験は
   **同一被験者1名**を対象に，2ヶ月間で5回のランダムな夜間睡眠＋連続21夜——
   すなわち**タイトルの「長期モニタリング」の実証はN=1**であり，
   外来患者6,222名の実験は横断的（1回計測）であって長期性を検証したものではない．
   この乖離はタイトルの印象と実験設計の間にギャップがある点として明記に値する．
3. **モデルサイズ・指標**：信号処理ベースでパラメータ数の概念なし．
   外来評価（N=6,222，中央値）：RT-IBI誤差**26.1ms**，RMSSD誤差**53.8ms**，
   SDRR誤差**33.7ms**，pNN50誤差**11.1%**．mmHRV/V2iFiという既存手法比で
   RMSSD/SDRRは**10倍以上改善**したと主張．心疾患4分類（正常/徐脈/頻脈/不整脈）の
   精度は平均83.4%だが，**不整脈の分類精度はECG正解データを使ってすら53.9%**
   （レーダ由来IBIでは47.6%）と低く，平均IBI・分散という単純特徴量に基づく分類の限界を
   著者ら自身が認めている．

**比較**：N=6,222という規模は圧倒的だが，RMSSD誤差53.8msは我々の目標精度から見て
大きい．「長期モニタリング」という論文タイトルの主張の核心（N=1の連続夜間データ）と，
精度評価の主データ（N=6,222だが単発計測）が実は別物である点は，
評価プロトコルを読む上での重要な注意点．

---

## 11. Contactless Radar Heart Rate Variability Monitoring Via Deep Spatio-Temporal Modeling（ICASSP 2024）

著者：Haoyu Wang, Jinbo Chen, Dongheng Zhang, Zhi Lu, Changwei Wu, Yang Hu, Qibin Sun,
Yan Chen（USTC）．DOI 10.1109/ICASSP48485.2024.10447570．
**全文取得済み**（著者個人ページ／USTC Intelligent Perception Lab経由）．

上記6番（JBHI 2025/2026）の前身にあたる会議論文（著者ほぼ同一，コアアイデア同一：
「心拍は全身の体表を駆動する」という物理的事前知識に基づく空間モデリング）．

1. **問題設計**：粗いボクセル信号を全身から抽出し，時空間的な心臓信号表現
   （H×W×L，H=W=9の0.5m×0.5m領域）を構築．TCN（9段，Gated Activation Unit）で
   時間方向を圧縮したのち，Vision Transformer的にパッチ化してTransformer Encoder
   （attention次元32，4ヘッド）で系列モデリングし，最終的にGaussian補間したR波タイミングを
   MLPで回帰．**参考文献[4]としてChenら2022（MMECG原論文，IEEE TMC）を明示的に引用**
   しているが，これは信号処理ベースの先行研究として引用されているだけで，
   **MMECGデータセット自体は使用していない**——著者ら自身が独自に新規収集した
   小規模データセットで評価している．
2. **前処理・前提**：**被験者8名のみ**．着座姿勢，1〜4mの距離で計測（合計約5時間）．
   **訓練6名（75%）／テスト2名（25%）**という極めて小さいホールドアウト——
   我々のMMECGでのテスト被験者数・分割方針と比べて，汎化性能の検証としては
   相当に緩い設定であることに留意．入力はレーダボクセル，正解はECG R波位置から
   ガウス補間した心拍波形であり，正解ECGを入力に混ぜるような明示的なリークは見当たらない．
3. **モデルサイズ・指標**：パラメータ数は明記されていないが，H×W×C×Tout=9×9×128×128の
   特徴量を扱うTCN 9段＋Transformerというアーキテクチャ規模から見て，
   0.034Mよりは大幅に大きいと推定される（本文に定量値なし）．
   全距離・中央値：IBI誤差**12ms**（相対誤差1.5%），RMSSD誤差**7.3ms**，
   SDRR誤差**2.9ms**，pNN50誤差**5.5%**．比較対象のmmHRV（信号処理）は
   IBI誤差35ms，VED（深層生成モデル）は26ms——**深層学習が信号処理を明確に上回る**
   という主張の実験．距離が1mから4mに伸びるとIBI誤差は12ms→24ms程度まで悪化する
   （Table 1）．

**比較**：この論文（およびその発展形であるJBHI版）は，我々と同じ「MMECG原著論文の著者
自身」による後継研究である点が重要．著者らはMMECGデータセットを自分たちの後続研究では
使わず独自データに切り替えており，かつテスト被験者2名という小規模設定でRMSSD誤差7.3ms
という好成績を出している一方，のちにJBHI版でN=7,150に拡大するとRMSSD誤差が16.23msへと
倍増している．**同一グループ・同一設計思想のスケール依存性**として，我々自身のMMECGでの
評価規模（テスト被験者数）を照らし合わせて，過大評価になっていないか点検する材料になる．

---

## まとめ表（取得できた数値の一覧）

| # | 論文 | データセット | N（評価） | 手法 | パラメータ数 | IBI/RT-IBI誤差 | RMSSD誤差 | SDNN/SDRR誤差 | pNN50誤差 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Onboard Executable | 独自＋public（名称不明） | 不明 | 軽量U-Net | **7.04M** | — | — | — | — |
| 2 | HyperECG | 不明 | 不明 | HDC | 不明 | — | — | — | — |
| 3 | Frequency Matters | 独自 | 24 | 決定論的信号処理 | なし（DSP） | HR MAE 1.8–9.0bpm（HRVでない） | — | — | — |
| 4 | HEBR | 独自 | 10+1 | 信号処理 | なし | 27.93ms(MedAE) | 15.37ms | 7.24ms(SDNN) | — |
| 5 | mmCG | 不明 | 不明 | 信号処理 | なし | **9.44ms** | — | — | — |
| 6 | Physiological Prior (JBHI) | 独自（外来7,150名） | 7,150 | ハイブリッドDNN | 不明 | 19.21ms | 16.23ms | 16.70ms(SDSD) | 7.28% |
| 7 | Massive Radio Sensing | 独自（一般30/臨床130） | 30〜130 | DL＋頑健化学習 | 不明 | SOTA比+29.7〜41.7%改善（相対） | — | — | — |
| 8 | 非接触SCG（Frontiers） | 独自（被験者内較正） | 13 | 信号処理 | なし | — | 8.05ms（変動大） | 4.11ms(SDNN) | 2.15% |
| 9 | Adaptive Template Matching | 独自 | 不明 | 信号処理 | なし | 18ms(MAE) | — | — | — |
| 10 | Nature Comms | 独自（外来6,222，長期N=1） | 6,222 | 信号処理（無訓練） | なし | 26.1ms | 53.8ms | 33.7ms(SDRR) | 11.1% |
| 11 | Deep Spatio-Temporal (ICASSP) | 独自（8名，train6/test2） | 2 | TCN+Transformer | 不明（0.034Mより大と推定） | 12ms | 7.3ms | 2.9ms(SDRR) | 5.5% |

---

## 大森への報告用の要点

1. **パラメータ数を明記した唯一の競合（Onboard Executable, TIM 2025）でも7.04M**——
   我々の0.034Mの約207倍．他の「軽量」を謳う論文はニューラルネット自体を持たない
   信号処理系（HEBR, mmCG, Frequency Matters, Nature Comms, Frontiers, Adaptive
   Template Matching）であり，比較の土俵が違う．
2. **MMECGを実際に使っている論文は0件**——「同じ条件で負けている」かどうかを
   直接判定できる論文は今回存在しなかった．
3. **同一研究グループ（MMECG原著者ら）の後継研究がデータ規模拡大とともに誤差悪化を
   自己申告している**（ICASSP RMSSD 7.3ms, N=2テスト → JBHI RMSSD 16.23ms, N=7,150
   → Nature Comms RMSSD 53.8ms, N=6,222）．小規模での高精度の意味を割り引く根拠になる．
4. 6件（HyperECG, HEBR, mmCG, Physiological Prior, Massive Radio Sensing, Adaptive
   Template Matching）はIEEE/ACMの403で本文取得不可．とくにmmCGとMassive Radio
   Sensingは重要な数値（絶対誤差，モデルサイズ）が抄録に欠けており，
   大学図書館経由等での本文入手を推奨する．
