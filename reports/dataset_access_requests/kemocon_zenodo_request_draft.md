# K-EmoCon (Zenodo) アクセス申請 下書き

申請先: https://doi.org/10.5281/zenodo.3931963 (Zenodo の "Request access" フォーム)

**これは下書きです。私(Claude)が代わりに送信することはしていません。** 氏名・所属・メールアドレスなど
本人確認情報を含むため、内容を確認のうえご自身で送信してください。所属・学年などは
このリポジトリのメモから推測した部分があるので、事実と異なる箇所は修正してください。

## 申請理由(英語, フォームにそのまま貼れる分量)

```
I am a researcher working on physiology-based arousal estimation and its downstream use in
human(-animal) interaction modeling. My current project estimates continuous arousal from
RR-interval-derived heart rate, with per-subject normalization, and I want to associate this
signal with natural-language descriptions of the interpersonal interaction that co-occurred
with it (generated via a vision-language model from video frames and speech transcripts).

K-EmoCon is, to my knowledge, the only public dataset combining (a) naturalistic dyadic
conversation with audio/video, (b) continuous physiological signals (ECG, PPG, EDA) suitable
for arousal extraction, and (c) valence/arousal annotations at multiple time granularities.
I would like to use it to validate a text-to-physiological-change association model before
extending the same pipeline to non-human-language settings (dog-human interaction).

I will use the dataset for non-commercial academic research only, will not attempt to
re-identify participants, and will not redistribute the raw data.
```

## 記入時に埋める/確認する項目
- 氏名、所属機関、メールアドレス(申請者本人のもの)
- 指導教員名(必要な場合、田中聡久研としてよいか要確認)
- 用途の説明は上記英文をベースに調整可

## 承認後にやること
- `data/raw/README.md` の慣習に従い、`dog-radar-vitals/data/raw/kemocon/` 等の
  gitignore対象ディレクトリに配置する。
- ECG/PPGからのarousal抽出は `src/dog_radar_vitals/affect/arousal_from_hr.py`
  (今回CASEで検証済みの実装)をそのまま流用できる設計にしてある。
