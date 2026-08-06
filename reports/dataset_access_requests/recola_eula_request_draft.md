# RECOLA EULA登録 下書き

申請先: https://recola.human-ist.ch/ (End User License Agreement の登録フォーム)

**これも下書きのみです。** EULAは法的な合意書であり、私が代理で同意・送信すべきものでは
ないため、内容確認の上ご自身で登録してください。

## 位置づけ

K-EmoCon(Zenodo)の審査状況次第の保険として申請しておくことを推奨します。RECOLAは
遠隔協調タスク中の対話(交流性あり)+ECG/EDA+映像音声を含み、K-EmoConと似た用途に
使えます。ただし個体数・収録時間はK-EmoConよりやや小さめです。

## 記入時に埋める/確認する項目
- 氏名、所属機関(大学・研究室)、指導教員名、メールアドレス
- 利用目的: 「介入の言語記述と生理指標由来の覚醒度変化の対応学習のための基礎データとして」
  という趣旨で問題ないか
- 研究目的が非商用学術研究であることの確認(EULA上、通常は非商用限定)

## 承認後にやること
- 同様に `data/raw/README.md` の慣習に従いgitignore対象ディレクトリへ配置。
- K-EmoConと同一のパイプライン(affect/arousal_from_hr.py, affect/intervention.py等)を
  流用して前処理する。
