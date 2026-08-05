"""単一拍SST->ECG回帰のための、意図的に「凝っていない」CNNベースライン。

`radarode_sceg.py`のSCEG(Deformable Conv2d+ODEデコーダ+2分岐融合)と全く同じ
「SST(50, F, T)の単一拍セグメント -> ECG(200)」タスクを、`ecg_cnn1d.py`(201のfeedforward
CNNベースライン)と全く同じ全畳み込み1D CNN・同じハイパーパラメータ(channels=64, n_layers=6,
kernel_size=15, dropout=0.1)で解く。周波数軸はチャネル間で共有した平均poolingのみで潰し、
architecture上の工夫は一切加えない。

この実装の目的は「タスク設計(単一拍単位への再定式化)・前処理(SST変換)・拍境界推定
(コンセンサス検出)を201と同じ凡庸なCNNに適用したとき、SCEGのような凝ったアーキテクチャに
どれだけ迫れるか」を切り分けて検証すること
（`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「タスク設計・前処理・
学習設計の改善」節参照）。SCEGとの差が小さければアーキテクチャは支配的要因ではなく、
差が大きければアーキテクチャ(Deformable Conv2d・ODE形状事前分布等)にも実質的な価値がある
ことになる。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.ecg_cnn1d import ECGWaveformCNN1D
from dog_radar_vitals.models.deep.temporal_alignment import TemporalShiftHead


class SingleCycleCNN(nn.Module):
    """SST(N, n_points, F, T) -> 単一拍ECG(N, out_len)。

    `use_temporal_shift=True`で、`TemporalShiftHead`(学習可能な時間遅延τ補正、
    `temporal_alignment.py`参照)を末尾に追加できる。CNNの中間特徴からτを予測し、
    最終出力波形をτだけシフトする。パラメータ数・epoch時間の増加がごく小さいため、
    この凡庸なCNNをτ補正アルゴリズムの高速な反復検証に使う。
    """

    def __init__(
        self,
        n_points: int = 50,
        channels: int = 64,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1,
        out_len: int = 200,
        use_temporal_shift: bool = False,
        max_shift_frac: float = 0.15,
    ) -> None:
        super().__init__()
        self.out_len = out_len
        self.use_temporal_shift = use_temporal_shift
        # 周波数軸は単純平均poolingで潰す(学習可能な重み付けを一切加えない)。
        self.backbone = ECGWaveformCNN1D(
            in_channels=n_points, channels=channels, n_layers=n_layers, kernel_size=kernel_size, dropout=dropout
        )
        if use_temporal_shift:
            self.shift_head = TemporalShiftHead(in_channels=channels, max_shift_frac=max_shift_frac)

    def forward(self, sst: torch.Tensor, return_tau: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = sst.mean(dim=2)  # (N, n_points, T): 周波数軸を平均で潰す
        h = h.transpose(1, 2)  # (N, T, n_points): ECGWaveformCNN1Dの入力形式に合わせる

        if self.use_temporal_shift:
            out, feat = self.backbone(h, return_features=True)
        else:
            out = self.backbone(h)

        if out.shape[-1] != self.out_len:
            out = nn.functional.interpolate(
                out.unsqueeze(1), size=self.out_len, mode="linear", align_corners=False
            ).squeeze(1)

        if self.use_temporal_shift:
            out, tau = self.shift_head(feat, out)
            if return_tau:
                return out, tau

        return out
