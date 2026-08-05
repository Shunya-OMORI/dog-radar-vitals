"""Chen et al. (2022, RCG2ECG)アーキテクチャの忠実な再現を狙ったモデル。

`docs/`のRadar2ECGとは別物。原著Fig.6の3要素を実装する:

1. **空間時間エンコーダ**: 50点のRCGを点ごとに重み共有1D convで時間ダウンサンプリングし
   (`ecg_spatial_fusion.py`の`PerPointTemporalEncoder`を再利用)、posXYZの線形埋め込みを
   加算した後、50点間でSelf-Attention（Transformer）を行い空間融合する。既存の
   `ECGSpatialFusion`は融合後に点方向平均プーリング+転置畳み込みで直接ECGを回帰する
   フィードフォワード構成だったが、本モデルはプーリング後の特徴を「条件付け特徴h」として
   デコーダに渡す点が異なる。
2. **TCN自己回帰デコーダ**: 原著の式(13) p(X|h)=Π p(xt|x1..xt-1, ht) を再現する。
   条件付け特徴hに加え、1つ前のECGサンプル（μ-law companding後、教師強制時は正解値）を
   入力チャネルとして与える、dilation 2^iのcausal dilated conv1dスタック。訓練時は全時刻を
   並列に処理する教師強制（原著と同じ「訓練は並列、推論は自己回帰」という非対称性）。
3. **μ-law companding + カテゴリカル出力**: 最終層は256クラスのlogitsを出力し、
   `data/mulaw.py`で量子化した正解ビンとのcross-entropyで学習する（MSEが持つ
   「QRSの振幅を潰す」構造的弱点を避けるため。詳細は
   `reports/mmecg_comparison/prior_work_accuracy_comparison.md`「相関0.90再現の第一歩」参照）。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.data.mulaw import N_CLASSES
from dog_radar_vitals.models.deep.ecg_spatial_fusion import PerPointTemporalEncoder


class SpatialTemporalEncoder(nn.Module):
    """RCG(50点)+posXYZ -> 条件付け特徴h (batch, cond_channels, seq_len)。"""

    def __init__(self, n_points: int = 50, embed_dim: int = 32, cond_channels: int = 8, n_heads: int = 4, n_attn_layers: int = 2) -> None:
        super().__init__()
        self.point_encoder = PerPointTemporalEncoder(embed_dim)
        self.pos_embed = nn.Linear(3, embed_dim)

        attn_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True
        )
        self.spatial_attn = nn.TransformerEncoder(attn_layer, num_layers=n_attn_layers)

        # 点方向融合後の特徴(embed_dim, T')を、元の時間解像度・cond_channelsまで転置畳み込みで戻す。
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose1d(embed_dim, cond_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, rcg: torch.Tensor, posxyz: torch.Tensor) -> torch.Tensor:
        """rcg: (batch, seq_len, n_points)。posxyz: (batch, n_points, 3)。-> (batch, cond_channels, seq_len)"""
        batch, seq_len, n_points = rcg.shape
        x = rcg.transpose(1, 2)  # (batch, n_points, seq_len)

        point_feat = self.point_encoder(x)  # (batch, n_points, embed_dim, T')
        t_reduced = point_feat.shape[-1]

        pos_emb = self.pos_embed(posxyz)  # (batch, n_points, embed_dim)
        point_feat = point_feat + pos_emb.unsqueeze(-1)

        h = point_feat.permute(0, 3, 1, 2).reshape(batch * t_reduced, n_points, -1)
        h = self.spatial_attn(h)
        h = h.mean(dim=1)  # 点方向プーリング -> (batch*T', embed_dim)
        h = h.reshape(batch, t_reduced, -1).transpose(1, 2)  # (batch, embed_dim, T')

        cond = self.upsample(h)
        if cond.shape[-1] != seq_len:
            cond = nn.functional.interpolate(cond, size=seq_len, mode="linear", align_corners=False)
        return cond


class _CausalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 2) -> None:
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.pad(x, (self.pad, 0))
        return x + self.act(self.conv(h))


class TCNAutoregressiveDecoder(nn.Module):
    """原著4.2.3節「9 stacks of TCN with dilation factor of 2, receptive field 512」を再現。"""

    def __init__(self, cond_channels: int, hidden: int = 32, n_layers: int = 9, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(cond_channels + 1, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([_CausalConvBlock(hidden, dilation=2**i) for i in range(n_layers)])
        self.head = nn.Conv1d(hidden, n_classes, kernel_size=1)

    def forward(self, cond: torch.Tensor, prev_ecg: torch.Tensor) -> torch.Tensor:
        """cond: (batch, cond_channels, T)。prev_ecg: (batch, 1, T)（1つ前のcompanded ECG値、
        教師強制時は正解を1サンプル右シフトしたもの）。-> logits (batch, n_classes, T)。
        """
        x = self.input_proj(torch.cat([cond, prev_ecg], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class Chen2022Reconstructor(nn.Module):
    def __init__(
        self,
        n_points: int = 50,
        embed_dim: int = 32,
        cond_channels: int = 8,
        decoder_hidden: int = 32,
        n_tcn_layers: int = 9,
    ) -> None:
        super().__init__()
        self.encoder = SpatialTemporalEncoder(n_points, embed_dim, cond_channels)
        self.decoder = TCNAutoregressiveDecoder(cond_channels, decoder_hidden, n_tcn_layers)

    def forward_teacher_forced(self, rcg: torch.Tensor, posxyz: torch.Tensor, prev_ecg_companded: torch.Tensor) -> torch.Tensor:
        """訓練用。prev_ecg_companded: (batch, 1, T)（正解を1サンプル右シフト、教師強制）。
        -> logits (batch, n_classes, T)。
        """
        cond = self.encoder(rcg, posxyz)
        return self.decoder(cond, prev_ecg_companded)

    @torch.no_grad()
    def generate(self, rcg: torch.Tensor, posxyz: torch.Tensor) -> torch.Tensor:
        """推論用。原著同様、1サンプルずつ自己回帰的に生成する。
        -> companding空間の連続値(batch, T)、[-1, 1]。呼び出し側で`mulaw.mu_law_decode`する。
        """
        from dog_radar_vitals.data.mulaw import N_CLASSES, dequantize

        cond = self.encoder(rcg, posxyz)
        batch, _, seq_len = cond.shape
        prev_buffer = torch.zeros(batch, 1, seq_len, device=cond.device)
        generated = torch.zeros(batch, seq_len, device=cond.device)

        for t in range(seq_len):
            logits = self.decoder(cond, prev_buffer)  # causalなのでt以前の入力のみ影響
            bin_t = logits[:, :, t].argmax(dim=1)  # (batch,)
            value_t = dequantize(bin_t, mu=N_CLASSES - 1)
            generated[:, t] = value_t
            if t + 1 < seq_len:
                prev_buffer[:, 0, t + 1] = value_t

        return generated
