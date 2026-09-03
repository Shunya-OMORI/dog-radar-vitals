# Chen ら (TMC 2022, MMECG) 被引用調査

取得日 2026-08-25．出典 Semantic Scholar API（`paperId 419d1b1c...`）．

## 内訳（抄録による自動振り分け → 本文で確定）

| 区分 | 件数 |
|---|---|
| **被引用 総数** | **158** |
| 抄録なし（要手動確認） | 18 |
| 話題が外れる（レーダ×心臓でない） | 51 |
| ミリ波レーダ × 心臓（段階A） | 89 |
| └ 抄録に MMECG を示唆（段階B） | 10 |
| └ RRI / HRV / F1 のいずれかを報告（段階C） | 27 |
| エッジ実行・軽量化に言及 | 10 |

**注意**：段階B は抄録にデータセット名が出た場合しか拾えないので**過小評価**である．
実際に MMECG を使っているかは本文の実験節を見ないと確定しない．段階C の 27 件が
本文確認の第一優先である．

## 本文を確認する優先順位

### 最優先 — エッジ実行を掲げている（我々の主張(1)の直接の競合）

| 論文 | 掲載 | 確認事項 |
|---|---|---|
| An Onboard Executable Multitask Network Model for Bioradar-Based ECG Reconstruction | IEEE TIM 2025 | **「onboard 実行可能」の中身．パラメータ数・対象ハードウェア** |
| HyperECG: ECG Signal Inference From Radar With Hyperdimensional Computing | BIBE 2024 | HDC は軽量計算の枠組み．モデルサイズと精度 |
| Frequency Matters: Comparative Analysis of Low-Power FMCW Radars | IEEE TIM 2024 | 低電力レーダの比較．我々の前提と合うか |

### 高 — HRV を主目的にしている（我々の指標と直接比較できる）

| 論文 | 掲載 |
|---|---|
| HEBR: An HRV Noncontact Health Monitoring Method Based on mmWave Radar | IEEE IoT-J 2026 |
| mmCG: Noncontact Millimeter-Wave Cardiography for HRV Monitoring | IEEE IoT-J 2025 |
| Radar HRV Monitoring With Physiological Prior Inspired Deep Neural Networks | IEEE JBHI 2025 |
| Robust HRV Monitoring via Massive Radio Sensing | ACM IMWUT 2025 |
| Non-contact seismocardiogram measurement and HRV analysis | Front. Physiol. 2026 |
| An Adaptive Template Matching Method for Noncontact Cardiac IBI | IEEE TIM 2025 |
| Monitoring long-term cardiac activity with contactless RF signals | Nature Communications 2024 |
| Contactless Radar HRV Monitoring Via Deep Spatio-Temporal M... | ICASSP 2024 |

### 中 — MMECG 上で ECG 再構成（R 波誤差の比較対象）

LifWavNet (arXiv 2025) / CNN-BiLSTM (Electronics 2026) / PulseStateNet (ICSIPC 2026) /
DualRateECGNet (ISEAE 2026) / mmJEPA-ECG (AAAI 2026, 取得済) / radarODE (TMC 2024, 取得済)

## 各論文で確認する 3 点（大森の指定）

1. **同じ MMECG で，他が解けない問題をどう解こうとしているか**（アーキテクチャと問題設計）
2. **前処理・前提に有利不利がないか**（例：正解 ECG を入力に入れていないか，
   被験者をどう分割しているか，テスト被験者を何名選んでいるか）
3. **モデルサイズ**（パラメータ数・FLOPs・推論時間）

## 生成物

- `citations_raw.json` — 158 件の生データ（題目・抄録・DOI・PDF リンク）
- `triage.json` — 上記の振り分け結果
- `scripts/triage_citations.py` — 振り分けスクリプト（判定規則はここに書いてある）
- `fulltext_comparison.md`（2026-08-25 追加）— 上記「本文を確認する優先順位」11 件の
  本文比較．11 件中 5 件は全文を取得，6 件は IEEE/ACM の 403 で抄録どまり．
  **MMECG を実際に使っている論文は 11 件中 0 件**，パラメータ数を明記した論文は
  Onboard Executable（7.04 M）のみという結果．`05_提案手法と実装.md` §7.2-D に反映済み
