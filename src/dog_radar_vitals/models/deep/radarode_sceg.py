"""radarODE(Zhang et al. 2024/2025, arXiv:2408.01672)のSCEG(Single-Cycle ECG Generator)。

原著Fig.5・Table Iの設計思想（backbone→squeezer&encoder→[temporal decoder, ODEデコーダ]の
2分岐→feature fusion）を再現するが、以下の点は簡略化している（詳細は
`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「radarODE再現」節参照）:

- backboneのDeformable Conv2dは、2026-07-28にtorchvisionを導入し`use_deformable=True`
  (既定)で本来設計通り使えるようになった。`torchvision.ops.DeformConv2d`はoffset
  (カーネル位置のずれ)を外部から与える必要があるため、通常Conv2dでoffsetを予測する
  `_DeformConv2dBlock`を追加している(offset予測convはゼロ初期化し、学習開始時は
  通常convと同じ振る舞いから始まるようにした)。`use_deformable=False`で従来の
  通常Conv2dへ戻せる。
- Table Iの厳密なチャネル数・カーネルサイズではなく、我々の入力サイズ(50, F, T)に
  合わせた汎用的な段数のdownsampling/upsamplingを使う。
- ODEデコーダは、x,y(単位円上の位相)のODEを数値積分する代わりに、1心拍を固定長へ
  リサンプル済みという前提を使い、位相θを`linspace(0, 2π, T)`で直接与える
  （α(x,y)=1-sqrt(x²+y²)の安定化項が収束する定常解と等価）。z(ECG値)のODEのみ
  Euler法で数値積分する。P/Q/R/S/Tの既定パラメータはMcSharry et al. (2003)
  "A dynamical model for generating synthetic electrocardiogram signals"の標準値を使う
  （原著radarODEもこのモデル系譜に基づく設計であり、Table IIはこの標準値からの調整値）。
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.ops import DeformConv2d

# McSharry et al. (2003)の標準パラメータ（P, Q, R, S, Tの順）。
_DEFAULT_THETA = torch.tensor([-1.2, -0.15, 0.0, 0.15, 1.5])
_DEFAULT_A = torch.tensor([1.2, -5.0, 30.0, -7.5, 0.75])
_DEFAULT_B = torch.tensor([0.25, 0.1, 0.1, 0.1, 0.4])

# radarODE公式実装(GitHub `ZYY0844/radarODE-MTL`、Projects/radarODE_plus/nets/ODE_solver.py
# の`default_input`)がそのまま使っているデフォルト値(P, Q, R, S, Tの順、a, b, theta[deg])。
# McSharry標準値とは異なり、著者らが実データに合わせて調整した値と見られる(theta_Rが0でなく
# 40度である点が重要: 後述`OfficialECGParameterEstimator`のdocstring参照)。
_OFFICIAL_DEFAULT_A = torch.tensor([5.0, -100.0, 480.0, -120.0, 8.0])
_OFFICIAL_DEFAULT_B = torch.tensor([0.25, 0.1, 0.1, 0.1, 0.4])
_OFFICIAL_DEFAULT_THETA_DEG = torch.tensor([-15.0, 25.0, 40.0, 60.0, 135.0])


class _Conv2dDownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DeformConv2dDownsampleBlock(nn.Module):
    """通常convでoffsetを予測してからDeformConv2dへ渡すdownsampleブロック(原著のDeformable Conv2d相当)。

    offset予測convはゼロ初期化する。これによりoffsetは学習開始時点で全て0となり、
    DeformConv2dは通常のConv2dと数学的に等価な状態から学習を始める(学習が進むにつれて
    カーネルの参照位置が心拍の三角形状の振動パターンに適応的にずれていくことを期待する)。
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: tuple[int, int] = (2, 2), padding: int = 1
    ) -> None:
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_ch, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        self.deform_conv = DeformConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        return self.act(self.bn(self.deform_conv(x, offset)))


class Backbone(nn.Module):
    """SST(N, 50, F, T) -> 潜在特徴マップ(N, C, F', T')。

    `strides`(2026-08-01追加): 各段の(周波数軸stride, 時間軸stride)を個別に指定できる
    (既定None時は全段(2,2)、従来通り)。269/270で「backbone段数を減らすと時間分解能が
    上がり精度が改善する」ことを確認した後、「段数はそのまま(=表現力/チャネル数は維持)に、
    時間軸だけpoolingを緩めて周波数軸だけ従来通り潰す」非対称downsampling(公式実装の
    `DCNResNet`後段(stride=(2,1))と同じ発想)を試すための拡張。
    """

    def __init__(
        self,
        in_channels: int = 50,
        channels: tuple[int, ...] = (128, 256, 512),
        use_deformable: bool = True,
        strides: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        super().__init__()
        if strides is None:
            strides = tuple((2, 2) for _ in channels)
        assert len(strides) == len(channels)
        block_cls = _DeformConv2dDownsampleBlock if use_deformable else _Conv2dDownsampleBlock
        blocks = []
        in_ch = in_channels
        for out_ch, stride in zip(channels, strides):
            blocks.append(block_cls(in_ch, out_ch, stride=tuple(stride)))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SqueezerEncoder(nn.Module):
    """周波数軸をpoolingで潰し(squeeze)、時間方向の潜在特徴(N, C, T')を得る。"""

    def __init__(self, channels: int, out_channels: int, out_time: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, out_time))
        self.proj = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(x).squeeze(2)  # (N, C, out_time)
        return self.proj(h)


class InitialDecoder(nn.Module):
    """temporal/ODE両分岐が共有する初期デコーダ。

    `dropout`(既定0、245/246では未使用)は2026-07-29の「学習設計の改善」検証で追加した
    正則化オプション。245/246はtrain_corrが0.7〜0.99に達する一方val/testは0.2台に留まる
    強い過学習を示したため、正則化が効くかを確認する目的で追加した
    (`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)。
    """

    def __init__(self, in_channels: int, hidden: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalFeatureDecoder(nn.Module):
    """初期デコーダの出力から、時間方向にアップサンプルして単一拍のECG形状を推定する。"""

    def __init__(self, in_channels: int, out_len: int) -> None:
        super().__init__()
        self.out_len = out_len
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return nn.functional.interpolate(h, size=self.out_len, mode="linear", align_corners=False)


class ODEDecoder(nn.Module):
    """McSharry型ECG力学モデルをEuler法で積分し、PQRST形状の事前分布を生成する(式19-20)。

    ## τ(時間遅延)の追加(2026-07-31)

    原著p.7「the aforementioned two obstacles」節の記述:「the parameters estimation part
    contains four linear blocks... to project the latent space... into parameters η, τ」
    「the solution of the ODEs will be shifted to the left with time τ」に基づき、
    η(a, b, theta)と同じ潜在特徴からτを予測し、ODEの位相グリッドθ(t)に位相シフトとして
    加える。過去の実装(2026-07-28〜29)はθ(t)を`linspace(0, 2π, T)`で固定的に与えており、
    τが完全に欠落していた(`reports/mmecg_comparison/prior_work_accuracy_comparison.md`
    「τ専用アーキテクチャ」節参照)。

    256/258/261/262(波形回帰用CNNの後付けτヘッド)や263(τ専用の教師ありオラクル回帰)は、
    いずれも「学習可能なτ」を「疑似ラベルへの回帰」または「自由形状CNN出力への後付け補正」
    として実装しており、原著の設計(τをηと同じ潜在特徴から同時予測し、ODEの強い形状制約の
    下でend-to-endの再構成損失のみで学習する)とは異なっていた。原著の設計をそのまま
    再現することが目的で、疑似ラベルは一切使わない(教師信号は最終的な再構成損失のみ)。
    ゼロ初期化によりτ=0(位相シフトなし)から学習を始める。
    """

    def __init__(
        self, in_channels: int, in_time: int, out_len: int, tau_max_frac: float = 0.15, use_tau: bool = True
    ) -> None:
        super().__init__()
        self.out_len = out_len
        self.tau_max_frac = tau_max_frac
        self.use_tau = use_tau
        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * in_time, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 15),  # a, b, thetaのスケール係数(各5個)、[-1,1]
            nn.Tanh(),
        )
        # τはη(a, b, theta)と同じ潜在特徴xから、独立した小さな線形ヘッドで予測する
        # (原著:「four linear blocks to project the latent space into parameters η, τ」)。
        self.tau_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * in_time, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.tau_net[-2].weight)
        nn.init.zeros_(self.tau_net[-2].bias)
        self.register_buffer("theta_grid", torch.linspace(0, 2 * math.pi, out_len))
        self.register_buffer("default_theta", _DEFAULT_THETA)
        self.register_buffer("default_a", _DEFAULT_A)
        self.register_buffer("default_b", _DEFAULT_B)
        # McSharry型ODEの生の積分結果は振幅オーダーが0.01程度と小さく(a_i, b_iの標準値は
        # 実時間dtでの積分を前提とした値のため)、我々のz-score化ECG(振幅オーダー数)に
        # スケールを合わせるための学習可能な出力ゲイン。実測比に基づき50で初期化する。
        self.output_scale = nn.Parameter(torch.tensor(50.0))

    def forward(self, x: torch.Tensor, return_tau: bool = False):
        batch = x.shape[0]
        scale = self.param_net(x)  # (N, 15), [-1, 1]
        a_scale, b_scale, theta_scale = scale[:, 0:5], scale[:, 5:10], scale[:, 10:15]

        a = self.default_a.unsqueeze(0) * (1.0 + 0.5 * a_scale)  # (N, 5)
        b = (self.default_b.unsqueeze(0) * (1.0 + 0.5 * b_scale)).clamp(min=1e-2)
        theta_ef = self.default_theta.unsqueeze(0) + 0.3 * theta_scale  # (N, 5)

        if self.use_tau:
            tau_frac = self.tau_net(x).squeeze(-1) * self.tau_max_frac  # (N,), 系列長に対する割合
            tau_phase = tau_frac * (2 * math.pi)  # 位相シフト量。正のtau_fracで波形は左(時間的に早く)ずれる。
            theta_t = self.theta_grid.unsqueeze(0).expand(batch, -1) + tau_phase.unsqueeze(1)  # (N, T)
        else:
            tau_frac = torch.zeros(batch, device=x.device)
            theta_t = self.theta_grid.unsqueeze(0).expand(batch, -1)  # (N, T)
        dt = 1.0 / self.out_len

        z = torch.zeros(batch, device=x.device)
        z_trace = []
        for t in range(self.out_len):
            delta = (theta_t[:, t : t + 1] - theta_ef + math.pi) % (2 * math.pi) - math.pi  # (N, 5)
            forcing = (a * delta * torch.exp(-(delta**2) / (2 * b**2))).sum(dim=1)  # (N,)
            dz = -forcing - z
            z = z + dt * dz
            z_trace.append(z)

        z_out = torch.stack(z_trace, dim=1) * self.output_scale
        out = z_out.unsqueeze(1)  # (N, 1, T)
        if return_tau:
            return out, tau_frac
        return out


class OfficialODEDecoder(nn.Module):
    """radarODE公式実装(GitHub `ZYY0844/radarODE-MTL`)のパラメータ推定を忠実に再現したODEデコーダ。

    ## τの正体についての訂正(2026-07-31)

    `ODEDecoder`(上記)は、論文本文「the parameters estimation part... project the latent
    space into parameters η, τ」という記述を字義通りに読み、ηとは別の独立したτヘッドを追加
    していた。しかし著者ら自身が公開している公式実装コード(`nets/ODE_solver.py`の
    `ECGParameterEstimator`)を直接確認したところ、**出力されるのはP/Q/R/S/T各波の
    `(a, b, theta)`×5=15個のみで、独立した「τ」というスカラー変数はコード上どこにも
    存在しなかった**(`shift`/`delay`/`align`でリポジトリ全体を検索しても該当実装なし)。
    つまり論文の「τだけ左にシフトする」という記述は、5つの`theta_i`(各波の位相中心)を
    まとめて動かした場合の効果を説明した言い回しであり、専用ヘッドではないと考えられる。

    このクラスはτ専用ヘッドを持たない。代わりに、公式実装の`ECGParameterEstimator`
    (深い全結合スタック: Linear→Dropout(0.5)→BatchNorm→Tanhを4回、その後さらに縮小して
    最終的に15出力・Tanh)と`scale_output`(`(1 + 1*pred) * default_input`という±100%の
    乗算スケーリング)、および公式の`default_input`定数をそのまま使う。特にtheta_Rの
    デフォルトが0度ではなく40度である点が重要で、乗算スケーリングでも0にならず
    R波の位相を±40度の範囲で動かせる(McSharry標準値のtheta_R=0を乗算スケーリングに
    使うと常に0のままになってしまうバグを、著者らはデフォルト値の選び方で回避している)。
    この「大きな可動域を持つtheta_R」こそが、論文が"τ"と呼ぶ効果の実体だと考えられる。

    入力特徴の次元(`in_channels*in_time`)は我々のbackbone構成に依存するため、公式の
    `8*207`という最初のLinear層の入力次元だけは我々の実際の特徴サイズに合わせて変更している
    (それ以外の層のユニット数・Dropout率・活性化関数は公式実装のまま)。
    """

    def __init__(self, in_channels: int, in_time: int, out_len: int) -> None:
        super().__init__()
        self.out_len = out_len
        feat_dim = in_channels * in_time
        # 公式ECGParameterEstimatorの層構成をそのまま踏襲(最初のLinearの入力次元のみ我々の特徴量に合わせる)。
        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024, bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Linear(1024, 512, bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 256, bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128, bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 16, bias=True),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, 15, bias=True),
            nn.Tanh(),
        )
        self.register_buffer("theta_grid", torch.linspace(0, 2 * math.pi, out_len))
        default_a = _OFFICIAL_DEFAULT_A
        default_b = _OFFICIAL_DEFAULT_B
        default_theta = _OFFICIAL_DEFAULT_THETA_DEG * math.pi / 180.0
        self.register_buffer("default_a", default_a)
        self.register_buffer("default_b", default_b)
        self.register_buffer("default_theta", default_theta)
        # 公式実装はECG値がmV単位(振幅オーダー0.005〜数mV)のまま出力されるが、我々のECGは
        # z-score化されているため、振幅オーダーを合わせるための学習可能な出力ゲインが必要
        # (公式コードにはこのゲインは無いが、GT側の正規化規約が異なる以上、代替なしでは
        # 比較不能なため導入する。既存`ODEDecoder`と同じ理由・同じ初期値)。
        self.output_scale = nn.Parameter(torch.tensor(50.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        pred = self.param_net(x)  # (N, 15), [-1, 1] (公式のscale_output前の生出力)

        a_pred, b_pred, theta_pred = pred[:, 0:5], pred[:, 5:10], pred[:, 10:15]
        # 公式`scale_output`: input_params = (1 + 1*pred) * default_input (±100%の乗算スケール)。
        a = self.default_a.unsqueeze(0) * (1.0 + 1.0 * a_pred)  # (N, 5)
        b = (self.default_b.unsqueeze(0) * (1.0 + 1.0 * b_pred)).clamp(min=1e-2)
        theta_ef = self.default_theta.unsqueeze(0) * (1.0 + 1.0 * theta_pred)  # (N, 5)

        theta_t = self.theta_grid.unsqueeze(0).expand(batch, -1)  # (N, T)
        dt = 1.0 / self.out_len

        z = torch.zeros(batch, device=x.device)
        z_trace = []
        for t in range(self.out_len):
            delta = (theta_t[:, t : t + 1] - theta_ef + math.pi) % (2 * math.pi) - math.pi  # (N, 5)
            forcing = (a * delta * torch.exp(-(delta**2) / (2 * b**2))).sum(dim=1)  # (N,)
            dz = -forcing - z
            z = z + dt * dz
            z_trace.append(z)

        z_out = torch.stack(z_trace, dim=1) * self.output_scale
        return z_out.unsqueeze(1)  # (N, 1, T)


class PretrainedShapeDecoder(nn.Module):
    """McSharry型ODEデコーダの代わりに、実ECGで事前学習した波形デコーダを形状事前分布として使う。

    ユーザ提案の「ECG基盤モデル/事前学習モデル」路線。`scripts/pretrain_ecg_beat_autoencoder.py`で
    Schellenberger(ヒト30名、レーダ非依存の臨床ECG)を使い教師なしで学習した
    `BeatDecoder`を凍結して読み込み、SCEGの初期デコーダ特徴からこの学習済み形状空間への
    写像(線形層)だけを学習する。ODEデコーダが物理モデルに基づく固定的な事前分布なのに対し、
    こちらは実データから学習した(より柔軟だが、対象データが少ないと過学習しうる)事前分布。
    """

    def __init__(self, in_channels: int, in_time: int, pretrained_decoder_path: str, latent_dim: int = 16) -> None:
        super().__init__()
        from dog_radar_vitals.models.deep.ecg_beat_autoencoder import BeatDecoder

        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * in_time, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.beat_decoder = BeatDecoder(latent_dim=latent_dim)
        state = torch.load(pretrained_decoder_path, map_location="cpu")
        decoder_state = {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
        self.beat_decoder.load_state_dict(decoder_state)
        for p in self.beat_decoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_latent(x)
        return self.beat_decoder(z).unsqueeze(1)  # (N, 1, T)


class AnchorHead(nn.Module):
    """radarODE公式実装の"Anchor"補助タスク(R波位置ヒートマップ回帰)を再現する軽量ヘッド。

    公式実装(`nets/anchor_decoder.py`)は4秒複数拍窓・専用の重いdeconvスタックを前提とした
    設計だが、ここでは既存の1心拍タイトクロップパイプラインに載せるため、`initial_decoder`の
    共有特徴(`TemporalFeatureDecoder`と同じ入力)から単純なConv1d+線形補間で
    `out_len`長のheatmapを予測する軽量な代替実装にしている(ユーザとの相談の上、フル
    パイプライン再現ではなく「既存の1心拍パイプラインにAnchor補助タスクだけ追加」という
    中間案を採用: `reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)。
    出力は[0,1]のheatmapなのでsigmoidで抑える(公式の`anchorLoss`は`pred`側に活性化関数を
    課していないが、GTが`tau_pseudo_labels.compute_rpeak_heatmap_labels`により最初から
    [0,1]で作られているため、predも同じ値域に制約する方が学習が安定する)。
    """

    def __init__(self, in_channels: int, out_len: int) -> None:
        super().__init__()
        self.out_len = out_len
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = nn.functional.interpolate(h, size=self.out_len, mode="linear", align_corners=False)
        return torch.sigmoid(h.squeeze(1))  # (N, out_len)


class FeatureFusion(nn.Module):
    """temporal・ODE両分岐の出力を要素積で融合し、Conv1dで最終的な単一拍ECGを出す。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def forward(self, temporal: torch.Tensor, ode: torch.Tensor) -> torch.Tensor:
        product = temporal * ode
        stacked = torch.cat([temporal, ode, product], dim=1)  # (N, 3, T)
        return self.net(stacked).squeeze(1)  # (N, T)


class RadarODESCEG(nn.Module):
    """SST(N, 50, F, T) -> 単一拍ECG(N, T_out)。"""

    def __init__(
        self,
        n_points: int = 50,
        backbone_channels: tuple[int, ...] = (128, 256, 512),
        encoder_out_channels: int = 128,
        encoder_out_time: int = 8,
        out_len: int = 200,
        shape_prior: str = "ode",
        pretrained_decoder_path: str | None = None,
        use_deformable: bool = True,
        dropout: float = 0.0,
        use_tau: bool = True,
        tau_max_frac: float = 0.15,
        use_anchor_head: bool = False,
        backbone_strides: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """`shape_prior`: "ode"(McSharry型力学モデル+独立τヘッド、既定、2026-07-31時点では
        非推奨。`OfficialODEDecoder`のdocstring参照)、"ode_official"(公式実装の
        `ECGParameterEstimator`をそのまま再現、τ専用ヘッド無し)、または"pretrained"
        (`scripts/pretrain_ecg_beat_autoencoder.py`で事前学習した実ECG形状事前分布、
        "pretrained"の場合`pretrained_decoder_path`必須)。`use_deformable`: backboneに
        Deformable Conv2dを使うか(既定True、Falseで従来の通常Conv2d)。`dropout`: 初期デコーダに
        入れる正則化(既定0、245/246は未使用。過学習対策の検証用)。`use_tau`: ODEデコーダに
        原著の時間遅延τを追加するか(既定True、`shape_prior="ode"`の時のみ有効)。
        `use_anchor_head`: 公式実装の"Anchor"補助タスク(R波位置ヒートマップ回帰、
        `AnchorHead`参照)を追加するか(既定False)。`backbone_strides`: `Backbone`の
        `strides`をそのまま渡す(既定None=全段(2,2)、`backbone_channels`と同じ長さの
        (周波数stride, 時間stride)タプル列を指定すると非対称downsamplingにできる)。
        """
        super().__init__()
        self.backbone = Backbone(n_points, backbone_channels, use_deformable=use_deformable, strides=backbone_strides)
        self.squeezer_encoder = SqueezerEncoder(backbone_channels[-1], encoder_out_channels, encoder_out_time)
        self.initial_decoder = InitialDecoder(encoder_out_channels, dropout=dropout)
        self.temporal_decoder = TemporalFeatureDecoder(32, out_len)

        if shape_prior == "ode":
            self.shape_decoder = ODEDecoder(32, encoder_out_time, out_len, tau_max_frac=tau_max_frac, use_tau=use_tau)
        elif shape_prior == "ode_official":
            self.shape_decoder = OfficialODEDecoder(32, encoder_out_time, out_len)
        elif shape_prior == "pretrained":
            if pretrained_decoder_path is None:
                raise ValueError("shape_prior='pretrained' には pretrained_decoder_path が必須")
            self.shape_decoder = PretrainedShapeDecoder(32, encoder_out_time, pretrained_decoder_path)
        else:
            raise ValueError(f"unknown shape_prior: {shape_prior}")

        self.fusion = FeatureFusion()
        self.anchor_head = AnchorHead(32, out_len) if use_anchor_head else None

    def forward(self, sst: torch.Tensor, return_tau: bool = False, return_anchor: bool = False):
        h = self.backbone(sst)
        h = self.squeezer_encoder(h)
        h = self.initial_decoder(h)
        temporal = self.temporal_decoder(h)

        if return_tau and isinstance(self.shape_decoder, ODEDecoder):
            shape, tau = self.shape_decoder(h, return_tau=True)
            out = self.fusion(temporal, shape)
        else:
            shape = self.shape_decoder(h)
            out = self.fusion(temporal, shape)
            tau = None

        anchor = self.anchor_head(h) if (return_anchor and self.anchor_head is not None) else None

        if return_tau or return_anchor:
            return out, tau, anchor
        return out
