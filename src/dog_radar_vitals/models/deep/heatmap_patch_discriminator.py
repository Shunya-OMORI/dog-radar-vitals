"""R波heatmapの敵対的リファインメント用、1D PatchGAN判別器（GAN拡張の本命、フェーズE）。

201-215フェーズのbeatgraph GAN(215)は、PQRSTグラフ(5ノード×2特徴という疎な低次元出力)を
単一のreal/fakeスカラーで判別する判別器を使い、MSE回帰のみ(214)より悪化した。文献調査
（RF2ESG・Doppler-GAN論文、`reports/mmecg_comparison/prior_work_accuracy_comparison.md`
「GANベース手法」参照）では、GANの成功例は密な波形合成やheatmap/スペクトルからのR波検出に
限られる。本モジュールは、pix2pix(Isola et al. 2017)のPatchGAN判別器を1次元化したもので、
単一のグローバルスカラーではなく時系列に沿った局所的なreal/fakeスコア列を出力する
（heatmapの「立ち上がり方・鋭さ」という局所的な形状の妥当性を判定させる方が、勾配の
フィードバックが密になり有効という設計判断）。

条件付けとして、判別器はheatmap単体ではなくRCG入力（レーダ波形、条件c）も受け取り、
[heatmap; conditioning(c)]を連結して判定する（conditional GAN、pix2pixと同じ構成）。
"""
from __future__ import annotations

import torch
from torch import nn


class HeatmapPatchDiscriminator(nn.Module):
    def __init__(self, condition_channels: int = 50, condition_proj_channels: int = 8, channels: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.condition_proj = nn.Conv1d(condition_channels, condition_proj_channels, kernel_size=1)

        layers = []
        in_ch = 1 + condition_proj_channels
        for i in range(n_layers):
            out_ch = channels * (2**i)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_ch) if i > 0 else nn.Identity(),  # 最初の層は正規化しない(pix2pix慣例)
                nn.LeakyReLU(0.2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.patch_head = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, heatmap: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """heatmap: (batch, seq_len)。condition: (batch, seq_len, condition_channels)(レーダ入力)。
        -> (batch, n_patches) の局所real/fakeロジット列。
        """
        h_map = heatmap.unsqueeze(1)  # (batch, 1, seq_len)
        cond = self.condition_proj(condition.transpose(1, 2))  # (batch, condition_proj_channels, seq_len)
        h = torch.cat([h_map, cond], dim=1)
        h = self.conv(h)
        return self.patch_head(h).squeeze(1)
