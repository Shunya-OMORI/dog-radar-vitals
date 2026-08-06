"""d ↔ Δs (ΔArousal) の対応を学習するモデル(Part II-1 の骨子そのもの)。

## 核心のコード

    class JointEmbeddingModel(nn.Module):
        def forward(self, text_emb, delta_s):
            z_text = F.normalize(self.text_proj(text_emb), dim=-1)   # d 側
            z_delta = F.normalize(self.delta_proj(delta_s), dim=-1)  # Δs 側
            return z_text, z_delta

    def info_nce_loss(z_text, z_delta, temperature=0.07):
        logits = z_text @ z_delta.T / temperature
        labels = torch.arange(len(z_text), device=z_text.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

(このファイル内、JointEmbeddingModel と info_nce_loss を参照)

## なぜこの設計にしたか

- **回帰1本ではなく、対照学習による共有埋め込み空間を主形式にした。** Part II-2 のシナリオ表
  (S1〜S7)を見ると、「d→Δs」の一方向だけでなく「望ましいΔを指定してdを検索する」(S2)、
  「介入で説明できない残差を取り出す」(S5)など、**双方向の近さ**を使うシナリオが多い。
  回帰モデル(d→Δsのスカラー予測)だけではS2(逆引き検索)ができない。CLIP型の共有埋め込みに
  しておけば、cos類似度でどちら向きにも検索・整合度評価ができ、S5の残差(=どのd候補と対応
  づけても類似度が低い)も自然に定義できる。
- **入出力の次元**: text_emb は multilingual-e5-large の1024次元(text_embed.py)、delta_s は
  当面 ΔArousal のスカラー1次元(将来、個体差コンテキストなどを足す余地を残すため
  `delta_dim` を可変にしている)。共有空間の次元は128に縮約(joint_dim)。パラメータ数を
  絞っているのは、当面のデータ量(数百〜数千サンプル規模を想定)に対して1024次元のまま
  対照学習すると過学習しやすいため。
- **DeltaRegressor(回帰ヘッド)を別途用意した理由**: S1(効果予測: この介入をしたらΔはどの
  程度か、を具体的な数値で答える)は、検索ベースの近似ではなく直接のスカラー予測の方が
  実用上扱いやすい。共有埋め込み(z_text)を凍結して軽い回帰ヘッドを乗せる転移学習構成に
  してあり、対照学習で学んだ表現をS1にも再利用できるようにしている。

## 対応する先行研究

- **CLIP** (Radford et al., 2021, "Learning Transferable Visual Models From Natural Language
  Supervision"): 画像と自然文の共有埋め込みを対称InfoNCEで学習する構成をそのまま
  「介入テキスト」と「生理由来のΔs」の対応づけに転用している。d↔Δs の対応学習は、
  CLIPの「画像↔キャプション」を「テキスト↔スカラー系列」に置き換えたものとみなせる。
- InfoNCE自体は Oord et al., 2018, "Representation Learning with Contrastive Predictive
  Coding" が出典。
- スカラー/低次元の連続値を埋め込み空間に射影して対照学習する構成は、時系列×テキストの
  対照学習(例: CLASP, Zeng et al. 系統)に類例があるが、本タスク固有の設計(Δsが個体内
  正規化済みの覚醒変化量であること、無介入区間を明示的な負例として扱うこと)はここでの
  新規実装。

## 現時点の限界(正直に書く)

d↔Δs のペアデータ(実際の犬または人の交流+生体信号)がまだ無いため、**このモデルは
アーキテクチャと学習ループの検証(ランダムデータでの形状・勾配確認)のみ済んでいる。
実データでの学習・評価はまだ行っていない。** 学習が始められるのは、K-EmoCon等の
アクセス許可が下りてから、または犬データが取得できてから。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class JointEmbeddingModel(nn.Module):
    """d(テキスト埋め込み) と Δs(状態変化) を共有埋め込み空間に射影する。"""

    def __init__(self, text_dim: int = 1024, delta_dim: int = 1, joint_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, joint_dim)
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(delta_dim, hidden), nn.GELU(), nn.Linear(hidden, joint_dim)
        )

    def forward(self, text_emb: torch.Tensor, delta_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_text = F.normalize(self.text_proj(text_emb), dim=-1)
        z_delta = F.normalize(self.delta_proj(delta_s), dim=-1)
        return z_text, z_delta


def info_nce_loss(z_text: torch.Tensor, z_delta: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """対称InfoNCE(CLIP loss)。バッチ内の他サンプルのΔsを負例として使う。

    無介入区間(Δs≈0, dも「何もしていない」)がバッチ内に複数あると、これらが互いに
    正例のように扱われてしまう(Δsが同じ値に潰れるため)点は既知の限界。実データでは
    Δsの量子化・バケット化や、無介入区間専用のサンプリング戦略が必要になる見込み。
    """
    logits = z_text @ z_delta.T / temperature
    labels = torch.arange(len(z_text), device=z_text.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class DeltaRegressor(nn.Module):
    """S1(効果予測)用: 凍結した text embedding から Δs を直接回帰する軽量ヘッド。"""

    def __init__(self, text_dim: int = 1024, hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb)
