# 大学のアクセス経由で取得をお願いしたい原文（2026-08-25 時点）

自動取得を試して **出版社に拒否された（HTTP 403）もの**だけを挙げる．
`docs/` に PDF を置いてもらえれば，`rules/reading-paper.txt` に沿った要約をこちらで作る．
ファイル名は右端の名前にしてもらえると，こちらのスクリプトがそのまま拾う．

## 優先度 高（要約に本文が要る）

| # | 文献 | 取得先 | 置き場所のファイル名 |
|---|---|---|---|
| A12 | Lipponen & Tarvainen, *J. Med. Eng. Technol.* 43(3):173–181, 2019（Kubios の補正アルゴリズム） | https://www.tandfonline.com/doi/full/10.1080/03091902.2019.1640306 | `Lipponen2019_Kubios_correction.pdf` |
| A21 | Task Force, *Circulation* 93(5):1043–1065, 1996（RMSSD・SDNN・pNN50 の定義の出典） | https://www.ahajournals.org/doi/10.1161/01.CIR.93.5.1043 | `TaskForce1996_HRV_standards.pdf` |
| A4 | Makowski ら, NeuroKit2, *Behavior Research Methods* 53:1689–1696, 2021 | https://link.springer.com/article/10.3758/s13428-020-01516-5 | `NeuroKit2_2021.pdf` |

## 優先度 中

| # | 文献 | 取得先 | 置き場所のファイル名 |
|---|---|---|---|
| A14 | Coast ら, *IEEE T-BME* 37(9):826–836, 1990（HMM による心電図系列復号） | https://ieeexplore.ieee.org/document/58593 | `Coast1990_HMM_arrhythmia.pdf` |
| A22a | Guide to Canine and Feline Electrocardiography, Ch.16 “Heart Rate Variability” | https://onlinelibrary.wiley.com/doi/10.1002/9781119254355.ch16 | `Canine_ECG_ch16_HRV.pdf` |
| A22b | Behavior-related changes in canine heart rate and HRV, *Appl. Anim. Behav. Sci.*, 2025 | https://www.sciencedirect.com/science/article/pii/S0168159125003983 | `Canine_HRV_behavior_2025.pdf` |
| A15 | ANSI/AAMI EC57（許容 150 ms の出どころ） | https://my.aami.org/aamiresources/previewfiles/EC57_1212_preview.pdf ※サーバ証明書が期限切れで自動取得不可 | `AAMI_EC57_preview.pdf` |

## 優先度 低（数値は `02_先行研究_同一データセット.md` に収集済み．本文は表の裏取り用）

A6 の 5 本（RSSRnet／RF-ECG／AirECG／mmJEPA-ECG／CFT-RFcardi）は arXiv になく，
ACM・IEEE の有料版しか見つからなかった．**表に載せる数値は既に収集済みなので，
本文が取れなくても資料は作れる．** 手が空いたときで構わない．

---

## 取得済みの一次資料（`docs/`）

arXiv・EuropePMC・公開ミラーから自動取得できたもの：
CornerNet／Focal Loss／CenterNet／Sensors 2025 FMCW HRV（A10）／Wang 2020 イヌネコ UWB／
Virtanen 2018 乾式電極／Zhao 2025 獣医リモートセンシング／Pan & Tompkins 1985．
EuropePMC 由来のものは `docs/fulltext/*.xml` に**節構造つきの全文**があり，
要約で「Results 第 3 段落」まで出典を特定できる．
