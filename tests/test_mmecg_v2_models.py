import torch

from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model
from dog_radar_vitals.models.deep.ecg_spatial_fusion import ECGSpatialFusion
from dog_radar_vitals.models.deep.ecg_spatial_gnn import ECGSpatialGNN
from dog_radar_vitals.models.deep.heatmap_patch_discriminator import HeatmapPatchDiscriminator
from dog_radar_vitals.models.deep.rpeak_spatial_fusion import RPeakSpatialFusion
from dog_radar_vitals.models.deep.rpeak_spatial_gnn import RPeakSpatialGNN

SEQ_LEN = 800
IN_CHANNELS = 50


def test_complex_cnn_v2_forward_shape():
    model = build_ecg_model("ecg_complex_cnn_v2", in_channels=IN_CHANNELS, channels=16, n_blocks=2, kernel_size=5)
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS, dtype=torch.complex64)
    y = model(x)
    assert y.shape == (2, SEQ_LEN)
    assert y.dtype == torch.float32


def test_rpeak_complex_cnn_v2_output_in_unit_range():
    model = build_ecg_model("rpeak_complex_cnn_v2", in_channels=IN_CHANNELS, channels=16, n_blocks=2, kernel_size=5)
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS, dtype=torch.complex64)
    y = model(x)
    assert y.shape == (2, SEQ_LEN)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_conformer_forward_shape():
    model = build_ecg_model(
        "ecg_conformer", in_channels=IN_CHANNELS, d_model=32, conv_kernel_size=5, conv_layers=2,
        n_heads=2, n_transformer_layers=2, dim_feedforward=64, dropout=0.1,
    )
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    y = model(x)
    assert y.shape == (2, SEQ_LEN)


def test_rpeak_conformer_output_in_unit_range():
    model = build_ecg_model(
        "rpeak_conformer", in_channels=IN_CHANNELS, d_model=32, conv_kernel_size=5, conv_layers=2,
        n_heads=2, n_transformer_layers=2, dim_feedforward=64, dropout=0.1,
    )
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    y = model(x)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_unet1d_forward_shape():
    model = build_ecg_model("ecg_unet1d", in_channels=IN_CHANNELS, base_channels=8, depth=3, kernel_size=5)
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    y = model(x)
    assert y.shape == (2, SEQ_LEN)


def test_rpeak_unet1d_output_in_unit_range():
    model = build_ecg_model("rpeak_unet1d", in_channels=IN_CHANNELS, base_channels=8, depth=3, kernel_size=5)
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    y = model(x)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_conv_ncp_forward_shape():
    model = build_ecg_model(
        "ecg_conv_ncp", in_channels=IN_CHANNELS, conv_channels=16, conv_kernel_size=5, conv_layers=2,
        ncp_units=32, ncp_output_dim=4, n_ncp_layers=2, mixed_memory=True,
    )
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    y = model(x)
    assert y.shape == (2, SEQ_LEN)


def test_rpeak_conv_ncp_output_in_unit_range():
    model = build_ecg_model(
        "rpeak_conv_ncp", in_channels=IN_CHANNELS, conv_channels=16, conv_kernel_size=5, conv_layers=2,
        ncp_units=32, ncp_output_dim=4, n_ncp_layers=2, mixed_memory=True,
    )
    x = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    y = model(x)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_spatial_fusion_forward_shape():
    model = ECGSpatialFusion(n_points=IN_CHANNELS, embed_dim=16, n_heads=2, n_attn_layers=1)
    rcg = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    posxyz = torch.randn(2, IN_CHANNELS, 3)
    y = model(rcg, posxyz)
    assert y.shape == (2, SEQ_LEN)


def test_rpeak_spatial_fusion_output_in_unit_range():
    model = RPeakSpatialFusion(n_points=IN_CHANNELS, embed_dim=16, n_heads=2, n_attn_layers=1)
    rcg = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    posxyz = torch.randn(2, IN_CHANNELS, 3)
    y = model(rcg, posxyz)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_spatial_gnn_forward_shape():
    model = ECGSpatialGNN(n_points=IN_CHANNELS, embed_dim=16, k_neighbors=6, n_gat_layers=2)
    rcg = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    posxyz = torch.randn(2, IN_CHANNELS, 3)
    y = model(rcg, posxyz)
    assert y.shape == (2, SEQ_LEN)


def test_rpeak_spatial_gnn_output_in_unit_range():
    model = RPeakSpatialGNN(n_points=IN_CHANNELS, embed_dim=16, k_neighbors=6, n_gat_layers=2)
    rcg = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    posxyz = torch.randn(2, IN_CHANNELS, 3)
    y = model(rcg, posxyz)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_heatmap_patch_discriminator_outputs_patch_sequence():
    disc = HeatmapPatchDiscriminator(condition_channels=IN_CHANNELS, condition_proj_channels=4, channels=8, n_layers=3)
    heatmap = torch.rand(2, SEQ_LEN)
    condition = torch.randn(2, SEQ_LEN, IN_CHANNELS)
    out = disc(heatmap, condition)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert out.shape[1] > 1  # 単一スカラーではなく局所パッチ列であることを確認


def test_unet1d_handles_non_power_of_two_seq_len():
    """window_sec/stride_secの取り方次第でseq_lenが2^depthの倍数でない場合があるため確認する。"""
    model = build_ecg_model("ecg_unet1d", in_channels=IN_CHANNELS, base_channels=8, depth=3, kernel_size=5)
    x = torch.randn(2, 404, IN_CHANNELS)
    y = model(x)
    assert y.shape == (2, 404)
